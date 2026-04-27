"""
Data loading and sampling for QUEST.

Reads the same tab-separated format as unKR:  head \\t relation \\t tail \\t confidence

Adds Gaussian target distribution generation for CDL (Confidence Distribution
Learning), matching the ssCDL preprocessing exactly.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Gaussian target distribution  (same as ssCDL Section 3.2)
# ---------------------------------------------------------------------------

def confidence_to_distribution(conf: float, sigma: float = 0.6,
                               n_bins: int = 101) -> np.ndarray:
    """Convert scalar confidence → 101-bin Gaussian distribution.

    This is the core of CDL: a confidence of 0.78 becomes a Gaussian
    N(0.78, σ²) discretised over [0, 1] with n_bins points, normalised
    to sum to 1.  This introduces supervision from neighbouring
    confidence values (0.76, 0.77, 0.79, …).
    """
    x = np.linspace(0, 1, n_bins)
    pdf = stats.norm(conf, sigma).pdf(x)
    return (pdf / pdf.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKGDataset:
    """Load and pre-process an Uncertain Knowledge Graph dataset."""

    def __init__(self, data_path: str, dataset_name: str):
        self.root = os.path.join(data_path, dataset_name)
        self.name = dataset_name

        self.entity2id:   dict[str, int] = {}
        self.relation2id: dict[str, int] = {}
        self.id2entity:   dict[int, str] = {}
        self.id2relation: dict[int, str] = {}

        train_raw = self._load_raw("train.tsv")
        val_raw   = self._load_raw("val.tsv")
        test_raw  = self._load_raw("test.tsv")

        self._build_vocab(train_raw + val_raw + test_raw)

        self.train_data = self._to_numeric(train_raw)
        self.val_data   = self._to_numeric(val_raw)
        self.test_data  = self._to_numeric(test_raw)

        self.num_ent = len(self.entity2id)
        self.num_rel = len(self.relation2id)

        self.all_triples: set[tuple[int, int, int]] = set()
        self.hr2t: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.rt2h: dict[tuple[int, int], set[int]] = defaultdict(set)
        for h, r, t, _ in self.train_data + self.val_data + self.test_data:
            self.all_triples.add((h, r, t))
            self.hr2t[(h, r)].add(t)
            self.rt2h[(r, t)].add(h)

        confs = [c for _, _, _, c in self.train_data]
        print(f"Dataset : {dataset_name}")
        print(f"  #Ent={self.num_ent}  #Rel={self.num_rel}")
        print(f"  Train={len(self.train_data)}  Val={len(self.val_data)}  "
              f"Test={len(self.test_data)}")
        print(f"  Conf  [{min(confs):.3f}, {max(confs):.3f}]  "
              f"mean={np.mean(confs):.3f}  std={np.std(confs):.3f}")

    def _load_raw(self, fname):
        data = []
        with open(os.path.join(self.root, fname)) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    data.append((parts[0], parts[1], parts[2], float(parts[3])))
        return data

    def _build_vocab(self, raw):
        ents, rels = set(), set()
        for h, r, t, _ in raw:
            ents.update([h, t]); rels.add(r)
        for i, e in enumerate(sorted(ents)):
            self.entity2id[e] = i; self.id2entity[i] = e
        for i, r in enumerate(sorted(rels)):
            self.relation2id[r] = i; self.id2relation[i] = r

    def _to_numeric(self, raw):
        return [(self.entity2id[h], self.relation2id[r],
                 self.entity2id[t], s) for h, r, t, s in raw]

    def build_confidence_adjacency(self, device="cpu"):
        """Build sparse confidence-weighted adjacency for CPN.

        A[i,j] = sum of confidences of triples connecting entity i and j
        (undirected, from training data only).  Row-normalised so each
        entity's neighbors sum to 1.

        Returns: sparse (num_ent, num_ent) tensor on *device*.
        """
        from collections import defaultdict
        edge_weights = defaultdict(float)
        for h, r, t, s in self.train_data:
            edge_weights[(h, t)] += s
            edge_weights[(t, h)] += s   # undirected

        rows, cols, vals = [], [], []
        for (i, j), w in edge_weights.items():
            rows.append(i); cols.append(j); vals.append(w)

        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.tensor(vals, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(idx, val, (self.num_ent, self.num_ent))

        # Row-normalise
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-10)
        inv_deg = 1.0 / deg
        # D^{-1} A  via scaling each value by 1/degree_of_row
        inv_deg_vals = inv_deg[rows]
        adj_norm = torch.sparse_coo_tensor(
            idx, val * inv_deg_vals.numpy(), (self.num_ent, self.num_ent)
        ).coalesce().to(device)

        return adj_norm


# ---------------------------------------------------------------------------
# Train dataset — returns (triple, neg_tails, conf_distribution)
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """Training set with negative sampling and precomputed CDL targets."""

    def __init__(self, data, num_ent, num_neg=50, sigma=0.6,
                 n_bins=101, hr2t=None):
        self.data    = data
        self.num_ent = num_ent
        self.num_neg = num_neg
        self.hr2t    = hr2t or defaultdict(set)

        # Precompute Gaussian distributions for all training triples
        print(f"  Precomputing {len(data)} CDL distributions (σ={sigma})…")
        self.conf_dists = [
            confidence_to_distribution(s, sigma, n_bins) for _, _, _, s in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, r, t, s = self.data[idx]
        existing = self.hr2t.get((h, r), set())
        negs = []
        while len(negs) < self.num_neg:
            n = random.randint(0, self.num_ent - 1)
            if n not in existing:
                negs.append(n)
        return (
            torch.tensor([h, r, t, s], dtype=torch.float32),
            torch.tensor(negs, dtype=torch.long),
            torch.from_numpy(self.conf_dists[idx]),
        )


# ---------------------------------------------------------------------------
# Eval dataset — same as before
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        h, r, t, s = self.data[idx]
        return torch.tensor([h, r, t, s], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data-loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(dataset, train_bs=4096, eval_bs=8, num_neg=50,
                    sigma=0.6, n_bins=101, num_workers=0):
    train_ds = TrainDataset(
        dataset.train_data, dataset.num_ent, num_neg,
        sigma=sigma, n_bins=n_bins, hr2t=dataset.hr2t,
    )
    val_ds  = EvalDataset(dataset.val_data)
    test_ds = EvalDataset(dataset.test_data)

    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, batch_size=train_bs, shuffle=True,
                   drop_last=True, **kw),
        DataLoader(val_ds,  batch_size=eval_bs, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **kw),
    )
