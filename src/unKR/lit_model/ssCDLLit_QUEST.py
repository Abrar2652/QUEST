"""
QUEST LitModel — inherits ssCDLLitModel and adds graph smoothness regularizer.

The graph regularizer is applied ONLY during the pre-PCDG phase
(epochs < 30).  Once PCDG semi-supervised training activates at
epoch 30, the smoothness term is disabled to avoid interfering with
the meta self-training dynamics.  The spectral initialization has
already seeded the embeddings with structural information by then.

Rationale: on dense UKGs like CN15k, continuous graph smoothing after
PCDG kicks in destabilizes training (MSE exploded from 0.035 to 0.198
between ep29 and ep39 in a direct test).  Early-phase graph reg +
persistent spectral init gives us the benefit without the instability.
"""

from .ssCDLLit import ssCDLLitModel


class ssCDLLitModel_QUEST(ssCDLLitModel):
    """ssCDLLitModel + spectral init + graph smoothness (pre-PCDG only)."""

    # PCDG starts at epoch 30 in ssCDLLitModel; disable smoothness there.
    PCDG_START_EPOCH = 30

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Add graph smoothness to CDL-RL step only before PCDG activates."""

        # Run the original ssCDL training step
        loss = super().training_step(batch, batch_idx, optimizer_idx)

        if loss is None:
            return None

        # Graph regularizer for CDL-RL (optimizer 0) before PCDG kicks in
        if optimizer_idx == 0 and self.epoch_num < self.PCDG_START_EPOCH:
            L_smooth = self.model.graph_smoothness_loss()
            loss = loss + self.model.lambda_smooth * L_smooth

        return loss
