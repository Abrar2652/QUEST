"""
Figure 2: Training dynamics — QUEST vs ablations vs ssCDL.

2 datasets (NL27k, CN15k) × 3 metrics (MAE, MSE, WMRR) = 6 subplots.
Shows:
  * Divergence at PCDG activation (ep30)
  * QUEST's pre-PCDG-only graph-reg fix stabilizes vs ablations
  * Approach to / surpass of ssCDL published baseline (horizontal dashed)
"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

QUEST_DIR = Path("/nas/home/jahin/QUEST")

# Published ssCDL baselines from the paper
SSCDL = {
    "nl27k": {"MAE": 0.042, "MSE": 0.009, "WMRR": 0.727},
    "cn15k": {"MAE": 0.116, "MSE": 0.034, "WMRR": 0.207},
}

# Which log file corresponds to which (dataset, config)
LOGS = {
    ("cn15k", "Full QUEST"): "logs_quest_cn15k_v1_def.txt",
    ("cn15k", "No spectral"): "logs_ablation_cn15k_noSpec.txt",
    ("cn15k", "No graph reg"): "logs_ablation_cn15k_noReg.txt",
    ("nl27k", "No spectral"): "logs_ablation_nl27k_noSpec.txt",
    ("nl27k", "No graph reg"): "logs_ablation_nl27k_noReg.txt",
}

# NL27k full QUEST was a finished run — we don't have its per-epoch log
# anymore but we have the final best-ckpt values. Add as a single marker.
NL27K_FULL_FINAL = {
    "epoch": 499,
    "MAE": 0.04121,
    "MSE": 0.00952,
    "WMRR": 0.7360,
}


def parse_log(path: Path):
    """Extract (epoch -> {MAE, MSE, WMRR}) from a training log file.

    Accepts either `<path>.txt` or `<path>.txt.gz` (compressed after runs).
    """
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            import gzip
            raw = gzip.open(gz, "rt", errors="ignore").read()
        else:
            return {}
    else:
        raw = path.read_text(errors="ignore")
    # Progress-bar lines carry the *previous* epoch's validation values.
    # We want the real end-of-epoch validation rows (100% complete).
    pattern = re.compile(
        r"Epoch (\d+): 100%\|.*?Eval_MAE=([\d.]+).*?Eval_MSE=([\d.]+).*?"
        r"Eval_wmrr=([\d.]+)"
    )
    per_ep = {}
    for m in pattern.finditer(raw.replace("\r", "\n")):
        ep = int(m.group(1))
        per_ep[ep] = {
            "MAE": float(m.group(2)),
            "MSE": float(m.group(3)),
            "WMRR": float(m.group(4)),
        }
    return per_ep


def load_all():
    data = {}
    for (dataset, cfg), fname in LOGS.items():
        path = QUEST_DIR / fname
        data.setdefault(dataset, {})[cfg] = parse_log(path)
    return data


def plot(data):
    datasets = ["nl27k", "cn15k"]
    metrics = ["MAE", "MSE", "WMRR"]
    metric_labels = {"MAE": "MAE (↓)", "MSE": "MSE (↓)", "WMRR": "WMRR (↑)"}

    colors = {
        "Full QUEST": "#d62728",          # red
        "No spectral": "#1f77b4",         # blue
        "No graph reg": "#2ca02c",        # green
    }
    ssCDL_color = "#7f7f7f"

    fig, axes = plt.subplots(
        len(datasets), len(metrics),
        figsize=(10, 5.5), sharex="col",
    )

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Plot each config
            for cfg, per_ep in data.get(dataset, {}).items():
                if not per_ep:
                    continue
                epochs = sorted(per_ep.keys())
                vals = [per_ep[e][metric] for e in epochs]
                ax.plot(epochs, vals, label=cfg,
                        color=colors.get(cfg, "k"), linewidth=1.5)

            # NL27k full QUEST: single endpoint marker (we no longer have
            # per-epoch log because the run finished and was cleaned).
            if dataset == "nl27k":
                ax.scatter(
                    [NL27K_FULL_FINAL["epoch"]],
                    [NL27K_FULL_FINAL[metric]],
                    marker="*", s=120,
                    color=colors["Full QUEST"],
                    label="Full QUEST (final)",
                    zorder=5,
                )

            # ssCDL reference
            y = SSCDL[dataset][metric]
            ax.axhline(y, color=ssCDL_color, linestyle="--",
                       linewidth=1.2, label="ssCDL (paper)")

            # PCDG activation marker
            ax.axvline(30, color="black", linestyle=":",
                       linewidth=0.8, alpha=0.5)

            ax.set_ylabel(metric_labels[metric])
            if i == 0:
                ax.set_title(f"{dataset.upper()}: {metric_labels[metric]}")
            else:
                ax.set_title(f"{dataset.upper()}: {metric_labels[metric]}")
            if i == len(datasets) - 1:
                ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)

    # Gather handles/labels from ALL axes (some configs only appear on
    # some subplots), dedup while preserving order
    seen = set()
    dedup = []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                dedup.append((h, l))
    fig.legend(
        [h for h, _ in dedup], [l for _, l in dedup],
        loc="upper center", ncol=len(dedup),
        bbox_to_anchor=(0.5, 1.01), frameon=False,
    )
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_pdf = QUEST_DIR / "figure2_training_dynamics.pdf"
    out_png = QUEST_DIR / "figure2_training_dynamics.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    data = load_all()
    for ds in data:
        for cfg, per_ep in data[ds].items():
            print(f"  {ds}/{cfg}: {len(per_ep)} epochs, "
                  f"last={max(per_ep) if per_ep else 'n/a'}")
    plot(data)
