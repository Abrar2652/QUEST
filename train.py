"""
QUEST — Training pipeline.

Features
--------
* CDL (Confidence Distribution Learning) with SRT and evidential uncertainty.
* Simple self-training without meta-learning (replaces ssCDL's PCDG).
* Per-dataset hyper-parameters matching the ssCDL paper.
* Multi-seed evaluation for error bars  (``--seeds 42,123,456``).
* CSV + .log logging for learning-curve plots and reproducibility.
* Final comparison table against every baseline in Table 2 of the ssCDL paper.
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import QUEST
from data import (
    UKGDataset, get_dataloaders,
    confidence_to_distribution,
)
from evaluate import (
    evaluate_confidence_prediction,
    evaluate_confidence_binned,
    evaluate_link_prediction,
    evaluate_low_confidence,
)


# =====================================================================
# Baseline results from Table 2 of the ssCDL paper (NeurIPS 2025)
# =====================================================================

BASELINES = {
    "nl27k": {
        "UKGElogi":         dict(MSE=0.029, MAE=0.060, WMRR=0.593, H1=0.462),
        "UKGErect":         dict(MSE=0.033, MAE=0.071, WMRR=0.580, H1=0.452),
        "BEUrRE":           dict(MSE=0.089, MAE=0.222, WMRR=0.272, H1=0.117),
        "PASSLEAFDistMult": dict(MSE=0.023, MAE=0.051, WMRR=0.676, H1=0.553),
        "PASSLEAFComplEx":  dict(MSE=0.024, MAE=0.052, WMRR=0.708, H1=0.586),
        "PASSLEAFRotatE":   dict(MSE=0.019, MAE=0.063, WMRR=0.715, H1=0.580),
        "UKGsE":            dict(MSE=0.122, MAE=0.271, WMRR=0.064, H1=0.031),
        "UPGAT":            dict(MSE=0.029, MAE=0.101, WMRR=0.658, H1=0.530),
        "ssCDL":            dict(MSE=0.009, MAE=0.042, WMRR=0.727, H1=0.636),
    },
    "cn15k": {
        "UKGElogi":         dict(MSE=0.246, MAE=0.409, WMRR=0.118, H1=0.072),
        "UKGErect":         dict(MSE=0.202, MAE=0.364, WMRR=0.127, H1=0.060),
        "BEUrRE":           dict(MSE=0.117, MAE=0.283, WMRR=0.138, H1=0.039),
        "PASSLEAFDistMult": dict(MSE=0.216, MAE=0.379, WMRR=0.170, H1=0.078),
        "PASSLEAFComplEx":  dict(MSE=0.231, MAE=0.400, WMRR=0.196, H1=0.086),
        "PASSLEAFRotatE":   dict(MSE=0.094, MAE=0.248, WMRR=0.137, H1=0.037),
        "UKGsE":            dict(MSE=0.103, MAE=0.256, WMRR=0.012, H1=0.002),
        "UPGAT":            dict(MSE=0.149, MAE=0.308, WMRR=0.165, H1=0.078),
        "ssCDL":            dict(MSE=0.034, MAE=0.116, WMRR=0.207, H1=0.133),
    },
}

# =====================================================================
# EXPERIMENTAL TARGETS — 1.5% better than ssCDL (the SOTA to beat)
# =====================================================================
# MSE/MAE  — LOWER  is better → target = ssCDL × 0.985
# WMRR/H@1 — HIGHER is better → target = ssCDL × 1.015
# =====================================================================
QUEST_TARGETS = {
    "nl27k": dict(MSE=0.00887, MAE=0.04137, WMRR=0.73791, H1=0.64554),
    "cn15k": dict(MSE=0.03349, MAE=0.11426, WMRR=0.21011, H1=0.13500),
}


# =====================================================================
# Per-dataset configurations
# =====================================================================

def get_config(name: str) -> dict:
    shared = dict(
        lr=1e-3,
        weight_decay=0.01,         # matches ssCDL
        num_neg=50,
        train_bs=4096,
        eval_bs=8,
        num_bands=8,
        margin=0.1,
        phi=1.0,                   # LP weight; ssCDL uses 0.1 but Dirichlet CDL needs more LP budget
        sigma=0.6,                 # Gaussian σ for CDL
        n_bins=101,
        dropout_high=0.5,          # matches ssCDL FCN architecture
        dropout_low=0.3,
        evidential_lambda=0.005,         # light regularisation; too high fights KL
        grad_clip=1.0,
        # Self-training schedule (matches ssCDL's T_PCDG / T_CDLRL)
        st_warmup=5,               # epoch to start pseudo-label gen
        st_start=30,               # epoch to start using pseudo-labels
    )
    if name == "nl27k":
        return {
            **shared,
            "emb_dim": 128,
            "max_epochs": 500,
            "patience": 50,
            "eval_every": 10,
            "st_threshold": 0.03,  # pseudo-label selection threshold
            "st_weight": 0.7,      # weight for pseudo-labeled loss (w_p)
        }
    if name == "cn15k":
        return {
            **shared,
            "emb_dim": 512,
            "max_epochs": 300,
            "patience": 50,
            "eval_every": 10,
            "st_threshold": 0.015,
            "st_weight": 0.3,
        }
    raise ValueError(f"Unknown dataset: {name}")


# =====================================================================
# Self-training: pseudo-label generation (replaces ssCDL's PCDG)
# =====================================================================

def generate_pseudo_labels(model, train_data, dataset, device,
                           threshold, sigma, n_bins, batch_size=4096):
    """Generate pseudo-labeled negatives using the model's own predictions.

    For each training triple, corrupt the tail to get a negative.
    Run the model on the negative and keep those whose max predicted
    distribution probability exceeds *threshold*.

    Returns list of (h, r, t_neg, pred_conf, pred_dist_numpy).
    """
    model.eval()
    pseudo = []  # (triple_tensor, dist_tensor) pairs
    num_ent = dataset.num_ent

    # Build negative samples in bulk
    negs_h, negs_r, negs_t = [], [], []
    for h, r, t, _ in train_data:
        nt = random.randint(0, num_ent - 1)
        while nt in dataset.hr2t.get((h, r), set()):
            nt = random.randint(0, num_ent - 1)
        negs_h.append(h)
        negs_r.append(r)
        negs_t.append(nt)

    negs_h = torch.tensor(negs_h, dtype=torch.long, device=device)
    negs_r = torch.tensor(negs_r, dtype=torch.long, device=device)
    negs_t = torch.tensor(negs_t, dtype=torch.long, device=device)

    selected_triples = []
    selected_dists   = []

    with torch.no_grad():
        for start in range(0, len(negs_h), batch_size):
            end = min(start + batch_size, len(negs_h))
            bh = negs_h[start:end]
            br = negs_r[start:end]
            bt = negs_t[start:end]

            dist = model.predict_conf_dist(bh, br, bt)  # (B, 101)
            max_prob = dist.max(dim=-1).values           # (B,)
            mask = max_prob >= threshold

            if mask.sum() == 0:
                continue

            # For selected samples, get the expected confidence
            sel_dist = dist[mask]                        # (S, 101)
            sel_conf = model.score_func(sel_dist)        # (S,)

            sel_h = bh[mask].cpu()
            sel_r = br[mask].cpu()
            sel_t = bt[mask].cpu()
            sel_s = sel_conf.cpu()

            for i in range(sel_h.size(0)):
                tri = torch.tensor([sel_h[i], sel_r[i], sel_t[i], sel_s[i]],
                                   dtype=torch.float32)
                # Use the model's predicted distribution as the pseudo target
                d = sel_dist[i].cpu()
                selected_triples.append(tri)
                selected_dists.append(d)

    if not selected_triples:
        return None, None

    return torch.stack(selected_triples), torch.stack(selected_dists)


# =====================================================================
# Helpers
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_table(dataset_name, quest, multi=None):
    bl = BASELINES[dataset_name]
    w = 10
    hdr = f"{'Method':<22} {'MSE':>{w}} {'MAE':>{w}} {'WMRR':>{w}} {'Hits@1':>{w}}"
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    print(f"  RESULTS — {dataset_name.upper()}  "
          f"(CP: lower=better | LP: higher=better)")
    print(f"{'=' * len(hdr)}")
    print(hdr); print(sep)
    for m, r in bl.items():
        print(f"{m:<22} {r['MSE']:{w}.3f} {r['MAE']:{w}.3f} "
              f"{r['WMRR']:{w}.3f} {r['H1']:{w}.3f}")
    print(sep)

    if multi and "std" in multi:
        mu, sd = multi["mean"], multi["std"]
        fmt = lambda k: f"{mu[k]:.3f}±{sd[k]:.3f}"
        print(f"{'QUEST (Ours)':<22} {fmt('MSE'):>{w}} {fmt('MAE'):>{w}} "
              f"{fmt('WMRR'):>{w}} {fmt('Hits@1'):>{w}}")
    else:
        print(f"{'QUEST (Ours)':<22} {quest['MSE']:{w}.3f} {quest['MAE']:{w}.3f} "
              f"{quest['WMRR']:{w}.3f} {quest['Hits@1']:{w}.3f}")
    print(f"{'=' * len(hdr)}")

    ss = bl["ssCDL"]
    ref = multi["mean"] if multi and "mean" in multi else quest
    d_mse  = (ss["MSE"]  - ref["MSE"])  / ss["MSE"]  * 100
    d_mae  = (ss["MAE"]  - ref["MAE"])  / ss["MAE"]  * 100
    d_wmrr = (ref["WMRR"] - ss["WMRR"]) / ss["WMRR"] * 100
    d_h1   = (ref["Hits@1"] - ss["H1"]) / ss["H1"]   * 100
    print(f"  Δ vs ssCDL:  MSE {d_mse:+.1f}%  MAE {d_mae:+.1f}%  "
          f"WMRR {d_wmrr:+.1f}%  Hits@1 {d_h1:+.1f}%")

    tgt = QUEST_TARGETS[dataset_name]
    print(f"\n  Targets (1.5% better than ssCDL):")
    print(f"    {'Metric':<8} {'Ours':>10} {'Target':>10} {'Gap':>10}  Status")
    print(f"    {'-'*50}")
    for key, tkey, lo in [("MSE","MSE",True),("MAE","MAE",True),
                          ("WMRR","WMRR",False),("Hits@1","H1",False)]:
        ours = ref[key]; target = tgt[tkey]
        hit = ours <= target if lo else ours >= target
        gap = (target - ours) if lo else (ours - target)
        print(f"    {key:<8} {ours:>10.4f} {target:>10.4f} {gap:>+10.4f}  "
              f"{'HIT' if hit else 'MISS'}")
    print()


# =====================================================================
# Single-seed training
# =====================================================================

def train_single(dataset_name, data_path, seed=42,
                 quick_test=False, gpu=0,
                 use_cpn=False, use_srt=False):
    set_seed(seed)
    cfg = get_config(dataset_name)
    cfg["use_cpn"] = use_cpn
    cfg["use_srt"] = use_srt
    if quick_test:
        cfg["max_epochs"] = 5
        cfg["eval_every"] = 2
        cfg["st_start"]   = 999   # disable self-training in quick test

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # ---- logging ----
    out_dir = os.path.join("output", dataset_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    _logger = logging.getLogger(f"quest_{dataset_name}_{seed}")
    _logger.setLevel(logging.INFO)
    _logger.handlers.clear()
    _fh = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w")
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(_fh)

    def log(msg="", **kw):
        print(msg, **kw)
        if "end" not in kw or kw["end"] == "\n":
            _logger.info(msg)

    log(f"\n{'=' * 70}")
    log(f"  QUEST  |  {dataset_name}  |  seed={seed}  |  device={device}")
    log(f"{'=' * 70}")
    log(json.dumps(cfg, indent=2))

    # ---- data ----
    ds = UKGDataset(data_path, dataset_name)
    tr_loader, va_loader, te_loader = get_dataloaders(
        ds, cfg["train_bs"], cfg["eval_bs"], cfg["num_neg"],
        sigma=cfg["sigma"], n_bins=cfg["n_bins"],
    )

    # ---- model ----
    model = QUEST(
        num_ent=ds.num_ent, num_rel=ds.num_rel,
        emb_dim=cfg["emb_dim"], num_bands=cfg["num_bands"],
        sigma=cfg["sigma"], n_bins=cfg["n_bins"],
        margin=cfg["margin"], phi=cfg["phi"],
        dropout_high=cfg["dropout_high"], dropout_low=cfg["dropout_low"],
        use_cpn=cfg.get("use_cpn", False),
        use_srt=cfg.get("use_srt", False),
    ).to(device)
    log(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- CPN: set adjacency matrix on model if enabled ----
    if cfg.get("use_cpn", False):
        log("  Building confidence-weighted adjacency for CPN…")
        adj_norm = ds.build_confidence_adjacency(device=device)
        model.adj_norm = adj_norm
        log(f"    Adjacency: {adj_norm._nnz()} non-zero entries")

    # Separate param groups: no weight decay on embeddings (prevents collapse)
    embed_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "emb" in name or "scale" in name or "phase" in name:
            embed_params.append(p)
        else:
            other_params.append(p)
    opt = optim.Adam([
        {"params": embed_params, "weight_decay": 1e-5},   # very light — prevents norm explosion
        {"params": other_params, "weight_decay": cfg["weight_decay"]},
    ], lr=cfg["lr"])

    # ---- CSV header ----
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "cp_loss", "lp_loss", "kl_loss",
            "mse_loss", "evid_loss", "mean_unc", "n_pseudo",
            "val_MSE", "val_MAE", "val_WMRR", "val_Hits@1",
            "lr", "epoch_sec",
        ])

    # ---- training loop ----
    best_composite = float("-inf")
    best_epoch = 0
    patience_ctr = 0
    pseudo_triples = None
    pseudo_dists   = None

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        accum = {k: 0.0 for k in
                 ["total","cp","lp","kl","mse","evid","mean_unc"]}
        n_batch = 0
        n_pseudo = 0

        # --- self-training: generate pseudo-labels ---
        if epoch >= cfg["st_start"] and epoch % 5 == 0:
            pt, pd = generate_pseudo_labels(
                model, ds.train_data, ds, device,
                threshold=cfg["st_threshold"],
                sigma=cfg["sigma"], n_bins=cfg["n_bins"],
            )
            if pt is not None:
                pseudo_triples = pt.to(device)
                pseudo_dists   = pd.to(device)
                n_pseudo = len(pt)
                log(f"    [self-train] generated {n_pseudo} pseudo-labels "
                    f"(threshold={cfg['st_threshold']})")
            else:
                pseudo_triples = pseudo_dists = None

        for pos, neg, cdist in tr_loader:
            pos   = pos.to(device)
            neg   = neg.to(device)
            cdist = cdist.to(device)

            opt.zero_grad()

            # Determine pseudo-label weight
            wp = cfg["st_weight"] if (epoch >= cfg["st_start"]
                                      and pseudo_triples is not None) else 0.0

            loss, info = model.compute_loss(
                pos, neg, cdist,
                pseudo_triples=pseudo_triples,
                pseudo_dists=pseudo_dists,
                wp=wp,
                epoch=epoch,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

            for k in accum:
                accum[k] += info.get(k, 0.0)
            n_batch += 1

        dt = time.time() - t0
        for k in accum:
            accum[k] /= max(n_batch, 1)

        if epoch % 5 == 0 or epoch == 1:
            log(f"  E{epoch:>4d}/{cfg['max_epochs']}  "
                f"loss={accum['total']:.4f}  kl={accum['kl']:.4f}  "
                f"mse={accum['mse']:.4f}  lp={accum['lp']:.4f}  "
                f"evid={accum['evid']:.4f}  unc={accum['mean_unc']:.4f}  "
                f"({dt:.1f}s)")

        # ---- validation ----
        vr = {}
        if (epoch % cfg["eval_every"] == 0) or epoch == 1:
            cp = evaluate_confidence_prediction(model, va_loader, device)
            vr["MSE"], vr["MAE"] = cp["MSE"], cp["MAE"]
            log(f"    val CP  MSE={cp['MSE']:.6f}  MAE={cp['MAE']:.6f}")

            if not quick_test:
                lp = evaluate_link_prediction(
                    model, va_loader, ds, device, cfg["eval_bs"]
                )
                vr["WMRR"], vr["Hits@1"] = lp["WMRR"], lp["Hits@1"]
                log(f"    val LP  WMRR={lp['WMRR']:.4f}  "
                    f"Hits@1={lp['Hits@1']:.4f}")

        # ---- checkpointing (composite = WMRR - MSE) ----
        if "MSE" in vr:
            wmrr_v = vr.get("WMRR", 0.0)
            comp = wmrr_v - vr["MSE"]
            if comp > best_composite:
                best_composite = comp
                best_epoch = epoch
                patience_ctr = 0
                torch.save(
                    dict(epoch=epoch, state_dict=model.state_dict(),
                         opt=opt.state_dict(), cfg=cfg,
                         val_mse=vr["MSE"], val_wmrr=wmrr_v),
                    os.path.join(out_dir, "best_model.pt"),
                )
                log(f"    *** best (MSE={vr['MSE']:.6f}  "
                    f"WMRR={wmrr_v:.4f}) ***")
            else:
                patience_ctr += 1

        # ---- CSV ----
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{accum['total']:.6f}", f"{accum['cp']:.6f}",
                f"{accum['lp']:.6f}", f"{accum['kl']:.6f}",
                f"{accum['mse']:.6f}", f"{accum['evid']:.6f}",
                f"{accum['mean_unc']:.6f}", n_pseudo,
                f"{vr.get('MSE','')}", f"{vr.get('MAE','')}",
                f"{vr.get('WMRR','')}", f"{vr.get('Hits@1','')}",
                f"{opt.param_groups[0]['lr']:.8f}", f"{dt:.2f}",
            ])

        if patience_ctr >= cfg["patience"] and not quick_test:
            log(f"\n  Early stopping at epoch {epoch}")
            break

    # ==================================================================
    # TEST
    # ==================================================================
    log(f"\n{'=' * 70}")
    log(f"  TEST EVALUATION  (best model from epoch {best_epoch})")
    log(f"{'=' * 70}")

    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log("  Confidence prediction …")
    cp = evaluate_confidence_prediction(model, te_loader, device)
    log(f"    MSE = {cp['MSE']:.6f}")
    log(f"    MAE = {cp['MAE']:.6f}")

    np.savez_compressed(
        os.path.join(out_dir, "test_confidence_predictions.npz"),
        predictions=cp["predictions"], targets=cp["targets"],
        uncertainties=cp["uncertainties"],
    )
    binned = evaluate_confidence_binned(cp["predictions"], cp["targets"])
    with open(os.path.join(out_dir, "test_confidence_binned.json"), "w") as f:
        json.dump(binned, f, indent=2)
    log("    Binned MAE by confidence decile:")
    for c, m, n in zip(binned["bin_centers"], binned["bin_mae"],
                       binned["bin_count"]):
        log(f"      [{c:.2f}]  MAE={m:.4f}  (n={n})")

    if not quick_test:
        log("  Link prediction …")
        lp = evaluate_link_prediction(model, te_loader, ds, device,
                                      cfg["eval_bs"])
        log(f"    WMRR   = {lp['WMRR']:.4f}")
        log(f"    Hits@1 = {lp['Hits@1']:.4f}")
        log(f"    Hits@3 = {lp['Hits@3']:.4f}")
        log(f"    Hits@10= {lp['Hits@10']:.4f}")
        np.savez_compressed(
            os.path.join(out_dir, "test_link_predictions.npz"),
            ranks=lp["ranks"], confidences=lp["confidences"],
        )
    else:
        lp = dict(WMRR=0.0, **{f"Hits@{k}": 0.0 for k in (1, 3, 10)})

    if not quick_test:
        log("  Low-confidence analysis …")
        lc = evaluate_low_confidence(model, te_loader, device, threshold=0.5)
        log(f"    MAE(low) = {lc['MAE_low']:.6f}  ({lc['count']} triples)")
    else:
        lc = dict(MAE_low=0.0, count=0)

    results = dict(
        MSE=cp["MSE"], MAE=cp["MAE"],
        WMRR=lp["WMRR"], **{f"Hits@{k}": lp[f"Hits@{k}"] for k in (1, 3, 10)},
        MAE_low=lc["MAE_low"], seed=seed, best_epoch=best_epoch,
        binned_analysis=binned,
    )
    with open(os.path.join(out_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log(f"\n  All outputs saved to {out_dir}/")
    log(f"    train.log / training_log.csv / final_results.json")
    log(f"    test_confidence_predictions.npz / test_confidence_binned.json")
    if not quick_test:
        log(f"    test_link_predictions.npz")

    print_table(dataset_name, results)
    return results


# =====================================================================
# Multi-seed
# =====================================================================

def aggregate(runs):
    keys = ["MSE", "MAE", "WMRR", "Hits@1"]
    mean = {k: np.mean([r[k] for r in runs]) for k in keys}
    std  = {k: np.std([r[k]  for r in runs]) for k in keys}
    return dict(mean=mean, std=std, runs=runs)


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser("QUEST training")
    ap.add_argument("--dataset", default="nl27k", choices=["nl27k", "cn15k"])
    ap.add_argument("--data_path", default="dataset")
    ap.add_argument("--seeds", default="42",
                    help="Comma-separated seeds (e.g. 42,123,456)")
    ap.add_argument("--quick_test", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpn", action="store_true",
                    help="Enable Confidence Propagation Network")
    ap.add_argument("--srt", action="store_true",
                    help="Enable Spectral Relational Transform")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []

    for seed in seeds:
        res = train_single(args.dataset, args.data_path, seed=seed,
                           quick_test=args.quick_test, gpu=args.gpu,
                           use_cpn=args.cpn, use_srt=args.srt)
        all_results.append(res)

    if len(seeds) > 1:
        agg = aggregate(all_results)
        print(f"\n{'#' * 70}")
        print(f"  AGGREGATED  ({len(seeds)} seeds)")
        print(f"{'#' * 70}")
        for k in ["MSE", "MAE", "WMRR", "Hits@1"]:
            print(f"  {k:>7s}:  {agg['mean'][k]:.4f} ± {agg['std'][k]:.4f}")
        print_table(args.dataset, all_results[0], multi=agg)
        out = os.path.join("output", args.dataset, "aggregated_results.json")
        with open(out, "w") as f:
            json.dump(agg, f, indent=2, default=float)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
