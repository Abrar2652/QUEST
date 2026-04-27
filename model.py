"""
QUEST model for Uncertain Knowledge Graph Completion.

The model is built in layers that can be toggled for ablation:

  BASE    — CDL (101-bin softmax) + margin LP.  This is ssCDL without meta-
            learning.  Must reach MSE ≈ 0.010 on NL27k to verify correctness.

  + CPN   — Confidence-weighted Propagation Network on entity embeddings.
            Novel: structural confidence propagation for UKG.

  + SRT   — Spectral Relational Transform on head entity.

  + EVID  — Dirichlet evidential CDL for uncertainty quantification.

Each layer is controlled by constructor flags so ablations are trivial.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


# ---------------------------------------------------------------------------
# Confidence Propagation Network  (Phase 2 — novel)
# ---------------------------------------------------------------------------

class ConfidencePropagationNetwork(nn.Module):
    """Propagate information through the UKG weighted by triple confidence.

    Uses Personalized-PageRank-style iterative propagation:
        Z^{(k+1)} = (1-α) X  +  α A_norm Z^{(k)}
    where A_norm is the confidence-weighted, row-normalised adjacency.

    After K iterations, Z encodes K-hop structural context weighted by
    confidence — high-confidence paths contribute more than low-confidence.
    """

    def __init__(self, emb_dim: int, alpha: float = 0.85, K: int = 5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable teleport
        self.K = K
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, entity_emb, adj_norm):
        """
        entity_emb : (num_ent, D)
        adj_norm   : sparse (num_ent, num_ent) confidence-weighted row-normalised
        """
        X = self.proj(entity_emb)
        Z = X
        a = torch.sigmoid(self.alpha)  # keep in (0,1)
        for _ in range(self.K):
            Z = (1 - a) * X + a * torch.sparse.mm(adj_norm, Z)
        return self.norm(Z)


# ---------------------------------------------------------------------------
# Spectral Relational Transform  (Phase 2 — enhancement)
# ---------------------------------------------------------------------------

class SpectralRelationalTransform(nn.Module):
    def __init__(self, emb_dim: int, num_bands: int = 8):
        super().__init__()
        assert emb_dim % num_bands == 0
        self.num_bands = num_bands
        self.band_dim  = emb_dim // num_bands
        self.spectral_proj = nn.Parameter(
            torch.empty(num_bands, self.band_dim, self.band_dim)
        )
        for i in range(num_bands):
            nn.init.orthogonal_(self.spectral_proj.data[i])

    def forward(self, entity_emb, rel_scale, rel_phase):
        B = entity_emb.size(0)
        bands = entity_emb.view(B, self.num_bands, self.band_dim)
        spectral = torch.einsum("bkd,kde->bke", bands, self.spectral_proj)
        modulated = spectral * rel_scale.unsqueeze(-1) * torch.tanh(rel_phase)
        recon = torch.einsum(
            "bkd,ked->bke", modulated, self.spectral_proj.transpose(-1, -2)
        )
        return recon.reshape(B, -1)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _expected_gaussian(conf, sigma, n_bins=101):
    x = np.linspace(0, 1, n_bins)
    pdf = stats.norm(conf, sigma).pdf(x)
    return float(np.sum(x * (pdf / pdf.sum())))


# ---------------------------------------------------------------------------
# QUEST model
# ---------------------------------------------------------------------------

class QUEST(nn.Module):
    def __init__(
        self,
        num_ent: int,
        num_rel: int,
        emb_dim: int = 128,
        sigma: float = 0.6,
        n_bins: int = 101,
        margin: float = 0.1,
        phi: float = 0.1,
        dropout_high: float = 0.5,
        dropout_low: float = 0.3,
        # Feature flags for ablation
        use_cpn: bool = False,
        use_srt: bool = False,
        num_bands: int = 8,
        cpn_alpha: float = 0.85,
        cpn_K: int = 5,
    ):
        super().__init__()
        self.num_ent = num_ent
        self.emb_dim = emb_dim
        self.n_bins  = n_bins
        self.margin  = margin
        self.phi     = phi
        self.use_cpn = use_cpn
        self.use_srt = use_srt

        # ---- entity / relation embeddings ----
        self.ent_emb = nn.Embedding(num_ent, emb_dim)
        self.rel_emb = nn.Embedding(num_rel, emb_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ---- CPN (optional, Phase 2) ----
        if use_cpn:
            self.cpn = ConfidencePropagationNetwork(emb_dim, cpn_alpha, cpn_K)
            self.adj_norm = None  # set by train.py before training

        # ---- SRT (optional, Phase 2) ----
        if use_srt:
            band_dim = emb_dim // num_bands
            self.rel_scale = nn.Embedding(num_rel, num_bands)
            self.rel_phase = nn.Embedding(num_rel, num_bands * band_dim)
            nn.init.ones_(self.rel_scale.weight)
            nn.init.zeros_(self.rel_phase.weight)
            self.srt = SpectralRelationalTransform(emb_dim, num_bands)

        # ---- FCN1: confidence distribution (101-bin softmax) ----
        in_dim = 3 * emb_dim
        self.fcn_conf = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(dropout_high),
            nn.Linear(1024, 512),    nn.ReLU(), nn.Dropout(dropout_low),
            nn.Linear(512, n_bins),
        )

        # ---- FCN2: link prediction ----
        self.fcn_rank = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(dropout_high),
            nn.Linear(1024, 512),    nn.ReLU(), nn.Dropout(dropout_low),
            nn.Linear(512, 1),
        )

        # ---- multi-task weighting (Kendall et al.) ----
        self.log_sigma1 = nn.Parameter(torch.zeros(1))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))

        # ---- CDL buffers ----
        self.register_buffer("conf_weights", torch.linspace(0, 1, n_bins))
        lb = _expected_gaussian(0.1, sigma, n_bins)
        ub = _expected_gaussian(1.0, sigma, n_bins)
        self.register_buffer("lower_bound", torch.tensor(lb))
        self.register_buffer("upper_bound", torch.tensor(ub))

    # ------------------------------------------------------------------
    # Entity embeddings (optionally enriched by CPN)
    # ------------------------------------------------------------------

    def _get_entity_emb(self):
        """Return entity embeddings, optionally enriched by CPN."""
        emb = self.ent_emb.weight
        if self.use_cpn and self.adj_norm is not None:
            emb = self.cpn(emb, self.adj_norm)
        return emb

    def _get_head_emb(self, head_ids, rel_ids, all_emb):
        """Return head embeddings, optionally with SRT."""
        h = all_emb[head_ids]
        if self.use_srt:
            rs = self.rel_scale(rel_ids)
            rp = self.rel_phase(rel_ids).view(
                -1, self.srt.num_bands, self.srt.band_dim)
            h = self.srt(h, rs, rp)
        return h

    # ------------------------------------------------------------------
    # Feature construction:  concat(h, r, t)
    # ------------------------------------------------------------------

    def _features(self, head_ids, rel_ids, tail_emb, all_emb=None):
        if all_emb is None:
            all_emb = self._get_entity_emb()
        h = self._get_head_emb(head_ids, rel_ids, all_emb)
        r = self.rel_emb(rel_ids)
        if tail_emb.dim() == 2:
            return torch.cat([h, r, tail_emb], dim=-1)
        B, N, _ = tail_emb.shape
        return torch.cat([
            h.unsqueeze(1).expand(-1, N, -1),
            r.unsqueeze(1).expand(-1, N, -1),
            tail_emb,
        ], dim=-1)

    # ------------------------------------------------------------------
    # Confidence prediction (softmax CDL)
    # ------------------------------------------------------------------

    def predict_conf_dist(self, head_ids, rel_ids, tail_ids):
        all_emb = self._get_entity_emb()
        feat = self._features(head_ids, rel_ids, all_emb[tail_ids], all_emb)
        return F.softmax(self.fcn_conf(feat), dim=-1)

    def score_func(self, dist):
        return (dist * self.conf_weights).sum(dim=-1)

    def normalize_conf(self, raw):
        return ((raw - self.lower_bound) * 0.9
                / (self.upper_bound - self.lower_bound) + 0.1)

    def predict_uncertainty(self, head_ids, rel_ids, tail_ids):
        """Uncertainty from distribution entropy."""
        dist = self.predict_conf_dist(head_ids, rel_ids, tail_ids)
        raw = self.score_func(dist)
        pred = self.normalize_conf(raw)
        entropy = -(dist * torch.log(dist + 1e-10)).sum(-1)
        max_entropy = np.log(self.n_bins)
        uncertainty = entropy / max_entropy  # normalised [0, 1]
        return pred, uncertainty, entropy

    # ------------------------------------------------------------------
    # Link prediction
    # ------------------------------------------------------------------

    def predict_rank_score(self, head_ids, rel_ids, tail_ids,
                           apply_sigmoid=True):
        all_emb = self._get_entity_emb()
        feat = self._features(head_ids, rel_ids, all_emb[tail_ids], all_emb)
        if feat.dim() == 3:
            B, N, D = feat.shape
            out = self.fcn_rank(feat.view(-1, D)).squeeze(-1).view(B, N)
        else:
            out = self.fcn_rank(feat).squeeze(-1)
        return torch.sigmoid(out) if apply_sigmoid else out

    def score_all_tails(self, head_ids, rel_ids, chunk_size=500):
        all_emb = self._get_entity_emb()
        h = self._get_head_emb(head_ids, rel_ids, all_emb)
        r = self.rel_emb(rel_ids)
        B = h.size(0)
        parts = []
        for s in range(0, self.num_ent, chunk_size):
            e = min(s + chunk_size, self.num_ent)
            t_c = all_emb[s:e]
            feat = torch.cat([
                h.unsqueeze(1).expand(-1, e - s, -1),
                r.unsqueeze(1).expand(-1, e - s, -1),
                t_c.unsqueeze(0).expand(B, -1, -1),
            ], dim=-1)
            scores = torch.sigmoid(
                self.fcn_rank(feat.view(-1, feat.size(-1)))
            ).squeeze(-1).view(B, -1)
            parts.append(scores)
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, pos_triples, neg_tails, conf_dists,
                     pseudo_triples=None, pseudo_dists=None, wp=0.0,
                     **kwargs):
        h = pos_triples[:, 0].long()
        r = pos_triples[:, 1].long()
        t = pos_triples[:, 2].long()
        s = pos_triples[:, 3]

        # ========== CDL: softmax confidence distribution ==========
        pred_dist = self.predict_conf_dist(h, r, t)

        # KL(target || predicted) — manual computation matching ssCDL
        # ssCDL uses sum but divides by batch implicitly through sigma.
        # We use per-sample mean for stable gradients.
        pred_clamped = torch.clamp(pred_dist, min=1e-10)
        conf_clamped = torch.clamp(conf_dists, min=1e-10)
        loss_kl = (conf_clamped * (torch.log(conf_clamped) - torch.log(pred_clamped))).sum(-1).mean()

        # MSE on expected confidence
        loss_mse = F.mse_loss(self.score_func(pred_dist), s)

        loss_cp = loss_kl + loss_mse

        # Pseudo-labeled CP (Eq. 6)
        if pseudo_triples is not None and pseudo_dists is not None and wp > 0:
            ph = pseudo_triples[:, 0].long()
            pr = pseudo_triples[:, 1].long()
            pt = pseudo_triples[:, 2].long()
            ppred = self.predict_conf_dist(ph, pr, pt)
            ppred_c = torch.clamp(ppred, min=1e-10)
            pdist_c = torch.clamp(pseudo_dists, min=1e-10)
            pkls = (pdist_c * (torch.log(pdist_c) - torch.log(ppred_c))).sum(-1).mean()
            pmse = F.mse_loss(self.score_func(ppred), pseudo_triples[:, 3])
            loss_cp = loss_cp + wp * (pkls + pmse)

        # ========== LP: margin ranking ==========
        pos_rk = self.predict_rank_score(h, r, t, apply_sigmoid=False)
        neg_rk = self.predict_rank_score(h, r, neg_tails, apply_sigmoid=False)
        loss_lp = (
            s.unsqueeze(1) * F.relu(self.margin + neg_rk - pos_rk.unsqueeze(1))
        ).mean()

        # ========== Multi-task weighting (Eq. 5) ==========
        s1 = torch.exp(self.log_sigma1)
        s2 = torch.exp(self.log_sigma2)
        total = (
            (1.0 / (2 * s1**2)) * loss_cp + self.log_sigma1
            + (self.phi / (2 * s2**2)) * loss_lp + self.log_sigma2
        )

        return total, dict(
            total=total.item(), cp=loss_cp.item(), lp=loss_lp.item(),
            kl=loss_kl.item(), mse=loss_mse.item(), evid=0.0,
            mean_unc=0.0, mean_evidence=0.0,
        )
