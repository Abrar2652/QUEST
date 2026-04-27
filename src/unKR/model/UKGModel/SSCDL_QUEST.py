"""
QUEST: Graph-Regularized ssCDL for Uncertain Knowledge Graph Completion.

Minimal modifications to ssCDL:
  1. Confidence-weighted spectral initialization for entity embeddings
  2. Graph smoothness regularizer (L_smooth) added to training loss
  3. Everything else (CDL, meta self-training, FCN1/FCN2, evaluation) unchanged

The forward pass is IDENTICAL to ssCDL. We only change init_emb() and add
a helper method for computing the graph regularizer.
"""

import torch
import torch.nn as nn
import numpy as np
from .SSCDL import ssCDL


class ssCDL_QUEST(ssCDL):
    """ssCDL with graph-aware initialization and graph smoothness regularizer.

    The forward() method is NOT overridden — ssCDL's training dynamics
    are preserved exactly.  Only init_emb() is changed to provide
    spectral initialization, and graph_smoothness_loss() is added for
    the LitModel to call during training.
    """

    def __init__(self, args):
        super().__init__(args)

        # Store edges for graph regularizer (set by runner before training)
        self.register_buffer("edge_h", torch.zeros(1, dtype=torch.long))
        self.register_buffer("edge_t", torch.zeros(1, dtype=torch.long))
        self.register_buffer("edge_s", torch.zeros(1, dtype=torch.float32))
        self.n_edges = 0

        # Regularizer weight (can be tuned)
        self.lambda_smooth = getattr(args, "lambda_smooth", 0.01)
        # Whether to run the (expensive) spectral eigen-init on startup
        self.use_spectral_init = getattr(args, "use_spectral_init", True)

    def set_graph_edges(self, train_triples, device="cuda:0"):
        """Precompute edge tensors for graph smoothness regularizer.

        Called once by the runner before training starts.
        """
        hs, ts, ss = [], [], []
        for h, r, t, s in train_triples:
            hs.append(h); ts.append(t); ss.append(s)
        self.edge_h = torch.tensor(hs, dtype=torch.long, device=device)
        self.edge_t = torch.tensor(ts, dtype=torch.long, device=device)
        self.edge_s = torch.tensor(ss, dtype=torch.float32, device=device)
        self.n_edges = len(hs)
        print(f"  QUEST: loaded {self.n_edges} edges for graph regularizer")

    def init_emb(self):
        """Initialize entity embeddings using confidence-weighted spectral
        decomposition of the graph Laplacian, if training data is available.

        Falls back to Xavier uniform (same as ssCDL) if spectral init fails.
        """
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)

        # Spectral init will be applied after set_graph_edges() is called
        # (entity count might not match config at __init__ time)

    def apply_spectral_init(self):
        """Apply spectral initialization using precomputed graph edges.

        Called by the runner after set_graph_edges().
        """
        if self.n_edges == 0:
            print("  QUEST: no edges loaded, skipping spectral init")
            return

        try:
            from scipy.sparse import coo_matrix, diags
            from scipy.sparse.linalg import eigsh

            num_ent = self.args.num_ent
            emb_dim = self.args.emb_dim

            # Build confidence-weighted adjacency (undirected)
            h = self.edge_h.cpu().numpy()
            t = self.edge_t.cpu().numpy()
            s = self.edge_s.cpu().numpy()

            rows = np.concatenate([h, t])
            cols = np.concatenate([t, h])
            vals = np.concatenate([s, s])

            A = coo_matrix((vals, (rows, cols)),
                           shape=(num_ent, num_ent)).tocsr()
            deg = np.array(A.sum(axis=1)).flatten()
            deg = np.maximum(deg, 1e-10)
            D = diags(deg)
            L = D - A

            # Compute smallest eigenvectors.  For real KG Laplacians,
            # which="SM" uses inverse iterations that often stall.
            # We use shift-invert (sigma=0) which factors L once then
            # finds eigenvalues nearest sigma — much faster and robust.
            # Regularize L a bit to avoid exact singularity at sigma=0.
            import time
            k = min(emb_dim + 1, num_ent - 2)
            L_reg = L + diags(np.full(num_ent, 1e-5))
            t0 = time.time()
            print(f"  QUEST: starting spectral init (n={num_ent}, k={k})...",
                  flush=True)
            eigenvalues, eigenvectors = eigsh(
                L_reg, k=k, sigma=0.0, which="LM", tol=1e-4, maxiter=2000
            )
            print(f"  QUEST: spectral eigendecomposition took "
                  f"{time.time() - t0:.1f}s", flush=True)

            init = eigenvectors[:, 1:emb_dim + 1]  # skip first

            # Scale to match Xavier uniform range
            fan = num_ent + emb_dim
            std_target = np.sqrt(2.0 / fan)
            init = init / (np.std(init) + 1e-10) * std_target

            self.ent_emb.weight.data.copy_(
                torch.tensor(init, dtype=torch.float32)
            )
            print(f"  QUEST: spectral init applied "
                  f"(eigenvalues[1:5] = {eigenvalues[1:5].round(4)})")
        except Exception as e:
            print(f"  QUEST: spectral init failed ({e}), using Xavier")

    def graph_smoothness_loss(self, sample_size=2048):
        """Confidence-weighted graph smoothness regularizer.

        L_smooth = mean_i [ s_i × ||e_{h_i} - e_{t_i}||² ]

        Encourages entities connected by high-confidence edges to have
        similar embeddings.  Provides structural awareness without
        changing ssCDL's forward pass.
        """
        if self.n_edges == 0:
            return torch.tensor(0.0, device=self.ent_emb.weight.device)

        # Sample random edges for efficiency
        n = min(sample_size, self.n_edges)
        idx = torch.randint(0, self.n_edges, (n,),
                            device=self.edge_h.device)

        h_emb = self.ent_emb(self.edge_h[idx])
        t_emb = self.ent_emb(self.edge_t[idx])
        confs = self.edge_s[idx]

        dist_sq = ((h_emb - t_emb) ** 2).sum(dim=-1)
        return (confs * dist_sq).mean()
