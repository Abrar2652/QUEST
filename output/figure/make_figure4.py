"""
Figure 4: Xavier vs Spectral initialization — 2D t-SNE.

Shows that the QUEST spectral init already encodes structural information
(from confidence-weighted graph Laplacian) *before any training*, while
Xavier init is essentially noise.  Coloring by node degree highlights that
nodes with similar centrality cluster together in the spectral-init space.

Layout: 2 datasets (rows) x 2 init methods (cols).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

QUEST_DIR = Path("/nas/home/jahin/QUEST")
TORCH_SEED = 321


def load_triples(dataset):
    """Load (h, r, t, s) triples, building string->int entity IDs on the fly."""
    path = QUEST_DIR / "dataset" / dataset / "train.tsv"
    ent2id = {}
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            h_s, _, t_s, s_s = parts
            h = ent2id.setdefault(h_s, len(ent2id))
            t = ent2id.setdefault(t_s, len(ent2id))
            s = float(s_s)
            triples.append((h, None, t, s))
    num_ent = len(ent2id)
    return triples, num_ent


def xavier_init(num_ent, dim):
    torch.manual_seed(TORCH_SEED)
    emb = nn.Embedding(num_ent, dim)
    nn.init.xavier_uniform_(emb.weight.data)
    return emb.weight.detach().numpy()


def spectral_init(triples, num_ent, dim):
    """Same computation as in SSCDL_QUEST.apply_spectral_init."""
    h = np.array([t[0] for t in triples])
    t = np.array([t[2] for t in triples])
    s = np.array([t[3] for t in triples])
    rows = np.concatenate([h, t])
    cols = np.concatenate([t, h])
    vals = np.concatenate([s, s])

    A = coo_matrix((vals, (rows, cols)), shape=(num_ent, num_ent)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-10)
    D = diags(deg)
    L = D - A
    L_reg = L + diags(np.full(num_ent, 1e-5))

    k = min(dim + 1, num_ent - 2)
    eigenvalues, eigenvectors = eigsh(
        L_reg, k=k, sigma=0.0, which="LM", tol=1e-4, maxiter=2000
    )
    init = eigenvectors[:, 1:dim + 1]  # skip trivial eigenvalue
    std_target = np.sqrt(2.0 / (num_ent + dim))
    init = init / (np.std(init) + 1e-10) * std_target
    return init, deg


def reduce_tsne(emb, sample_size=3000, seed=TORCH_SEED):
    """t-SNE projection to 2D, subsampled for speed."""
    n = emb.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, sample_size, replace=False)
    else:
        idx = np.arange(n)
    tsne = TSNE(n_components=2, random_state=seed,
                perplexity=30, max_iter=750, init="pca", verbose=0)
    xy = tsne.fit_transform(emb[idx])
    return xy, idx


def make_figure():
    datasets = [
        ("nl27k", 128),
        ("cn15k", 512),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8))

    for i, (dataset, dim) in enumerate(datasets):
        print(f"Processing {dataset} (dim={dim})...")
        triples, num_ent = load_triples(dataset)
        print(f"  {num_ent} entities, {len(triples)} triples")

        # Xavier init
        xavier = xavier_init(num_ent, dim)
        # Spectral init + node degrees (for coloring)
        spec, deg = spectral_init(triples, num_ent, dim)

        # t-SNE reductions (subsample for speed)
        print(f"  t-SNE on Xavier...")
        xy_x, idx = reduce_tsne(xavier, sample_size=2500)
        print(f"  t-SNE on Spectral...")
        xy_s, _ = reduce_tsne(spec, sample_size=2500)
        color = np.log1p(deg[idx])  # log-scale degree for visual contrast

        for j, (xy, label) in enumerate([(xy_x, "Xavier init"),
                                          (xy_s, "Spectral init")]):
            ax = axes[i, j]
            sc = ax.scatter(
                xy[:, 0], xy[:, 1], c=color,
                s=6, cmap="viridis", alpha=0.7,
                edgecolors="none",
            )
            ax.set_title(f"{dataset.upper()} — {label}")
            ax.set_xticks([])
            ax.set_yticks([])

            # Only add colorbar on the right side of each row
            if j == 1:
                cbar = fig.colorbar(sc, ax=axes[i, :].ravel().tolist(),
                                     fraction=0.03, pad=0.02,
                                     shrink=0.85)
                cbar.set_label("log(1 + degree)", rotation=270, labelpad=12)

    # (suptitle intentionally omitted — caption carries the message)

    out_pdf = QUEST_DIR / "figure4_embedding_viz.pdf"
    out_png = QUEST_DIR / "figure4_embedding_viz.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    make_figure()
