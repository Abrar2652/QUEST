"""
Evaluation for QUEST.

Matches the **exact** evaluation protocol of the ssCDL paper (NeurIPS 2025)
and the unKR reference implementation:

  - Confidence Prediction : MSE, MAE
    → predicted via CDL (101-bin distribution → expected value → normalise)
  - Link Prediction       : WMRR, Hits@1  (tail-only, filtered)
    → double-argsort rank computation (same as unKR)

All functions return raw per-sample arrays for figure generation.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Confidence prediction  (CDL-based evaluation)
# ---------------------------------------------------------------------------

def evaluate_confidence_prediction(model, eval_loader, device):
    """MSE / MAE using CDL: distribution → expected value → normalise.

    This matches ssCDL's ``conf_predict_two`` evaluation exactly:
      raw = Σ(pred_dist × [0, 0.01, …, 1.0])
      normalised = (raw − lower_bound) × 0.9 / (upper − lower) + 0.1
    """
    model.eval()
    preds, targets, uncs = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            tri = batch[0].to(device) if isinstance(batch, (list, tuple)) \
                else batch.to(device)
            h, r, t, s = (tri[:, 0].long(), tri[:, 1].long(),
                          tri[:, 2].long(), tri[:, 3])

            # CDL prediction → normalised confidence
            dist = model.predict_conf_dist(h, r, t)
            raw  = model.score_func(dist)
            pred = model.normalize_conf(raw)

            # Uncertainty from distribution entropy
            _, unc, _ = model.predict_uncertainty(h, r, t)

            preds.append(pred.cpu())
            targets.append(s.cpu())
            uncs.append(unc.cpu())

    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    uncs    = torch.cat(uncs).numpy()

    return dict(
        MSE=float(np.mean((preds - targets) ** 2)),
        MAE=float(np.mean(np.abs(preds - targets))),
        predictions=preds, targets=targets, uncertainties=uncs,
    )


# ---------------------------------------------------------------------------
# Confidence-binned analysis  (Figure 3-style)
# ---------------------------------------------------------------------------

def evaluate_confidence_binned(predictions, targets, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    mae_bins, mse_bins, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (targets >= lo) & (targets < hi) if hi < 1.0 \
            else (targets >= lo) & (targets <= hi)
        n = mask.sum()
        counts.append(int(n))
        if n > 0:
            err = np.abs(predictions[mask] - targets[mask])
            mae_bins.append(float(err.mean()))
            mse_bins.append(float((err ** 2).mean()))
        else:
            mae_bins.append(float("nan"))
            mse_bins.append(float("nan"))
    return dict(bin_edges=edges.tolist(), bin_centers=centers.tolist(),
                bin_mae=mae_bins, bin_mse=mse_bins, bin_count=counts)


# ---------------------------------------------------------------------------
# Link prediction  (tail-only, filtered, double-argsort)
# ---------------------------------------------------------------------------

def evaluate_link_prediction(model, eval_loader, dataset, device, eval_bs=8):
    """Tail prediction, filtered, double-argsort ranks (matches unKR)."""
    model.eval()
    all_ranks, all_confs = [], []
    total = len(eval_loader.dataset)
    done  = 0

    with torch.no_grad():
        for batch in eval_loader:
            tri = batch[0].to(device) if isinstance(batch, (list, tuple)) \
                else batch.to(device)
            h, r, t, s = (tri[:, 0].long(), tri[:, 1].long(),
                          tri[:, 2].long(), tri[:, 3])
            B = h.size(0)

            scores = model.score_all_tails(h, r)

            for i in range(B):
                hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
                for kt in dataset.hr2t.get((hi, ri), set()):
                    if kt != ti:
                        scores[i, kt] = -1e9

            b_range = torch.arange(B, device=scores.device)
            ranks = 1 + torch.argsort(
                torch.argsort(scores, dim=1, descending=True),
                dim=1, descending=False,
            )[b_range, t]

            all_ranks.append(ranks.cpu())
            all_confs.append(s.cpu())
            done += B
            if done % max(1, total // 20) < B:
                print(f"\r    LP eval: {done}/{total}  "
                      f"({100.0 * done / total:.0f}%)", end="", flush=True)

    print(f"\r    LP eval: {total}/{total}  (100%)")

    ranks = torch.cat(all_ranks).float().numpy()
    confs = torch.cat(all_confs).float().numpy()
    rr = 1.0 / ranks

    return dict(
        WMRR=float(np.sum(confs * rr) / np.sum(confs)),
        MRR=float(np.mean(rr)),
        **{f"Hits@{k}": float(np.mean(ranks <= k)) for k in (1, 3, 10)},
        ranks=ranks, confidences=confs, num_queries=len(ranks),
    )


# ---------------------------------------------------------------------------
# Low-confidence analysis  (Section 5.5)
# ---------------------------------------------------------------------------

def evaluate_low_confidence(model, eval_loader, device, threshold=0.5):
    model.eval()
    p_all, t_all = [], []
    with torch.no_grad():
        for batch in eval_loader:
            tri = batch[0].to(device) if isinstance(batch, (list, tuple)) \
                else batch.to(device)
            h, r, t, s = (tri[:, 0].long(), tri[:, 1].long(),
                          tri[:, 2].long(), tri[:, 3])
            mask = s < threshold
            if mask.sum() == 0:
                continue
            dist = model.predict_conf_dist(h[mask], r[mask], t[mask])
            raw  = model.score_func(dist)
            pred = model.normalize_conf(raw)
            p_all.append(pred.cpu())
            t_all.append(s[mask].cpu())
    if not p_all:
        return dict(MAE_low=float("nan"), count=0)
    p = torch.cat(p_all).numpy()
    t = torch.cat(t_all).numpy()
    return dict(MAE_low=float(np.mean(np.abs(p - t))), count=len(p))
