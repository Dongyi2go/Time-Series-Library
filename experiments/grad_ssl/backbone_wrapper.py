"""
Backbone wrappers for self-supervised gradient feature extraction.

Each wrapper wraps a backbone (Informer / LightTS / TimesNet) and adds:
  - grad_proj : Linear(H -> D, bias=False)   shape [D, H]
  - recon_head: Linear(D -> C)

The gradient of grad_proj.weight after a backward pass is used as a
per-sample feature (row-wise L2 norm -> [D] vector).
"""

import copy
import os
import sys

import torch
import torch.nn as nn

# Allow imports from repo root when running this file directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.Informer import Model as InformerModel  # noqa: E402
from models.LightTS import Model as LightTSModel  # noqa: E402
from models.TimesNet import Model as TimesNetModel  # noqa: E402


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class _SelfSupervisedWrapper(nn.Module):
    """Abstract base: backbone + grad_proj + recon_head."""

    def __init__(self, backbone: nn.Module, hidden_dim: int, enc_in: int,
                 grad_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        # grad_proj.weight: [D, H]  -- gradients here form the feature
        self.grad_proj = nn.Linear(hidden_dim, grad_dim, bias=False)
        self.recon_head = nn.Linear(grad_dim, enc_in)
        self.hidden_dim = hidden_dim
        self.enc_in = enc_in
        self.grad_dim = grad_dim

    def get_hidden(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Return backbone hidden states h [B, T, H]."""
        raise NotImplementedError

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Return reconstructed series x_hat [B, T, C]."""
        h = self.get_hidden(x_enc)       # [B, T, H]
        proj = self.grad_proj(h)         # [B, T, D]
        x_hat = self.recon_head(proj)    # [B, T, C]
        return x_hat


# ---------------------------------------------------------------------------
# Informer wrapper  (H = d_model)
# ---------------------------------------------------------------------------

class InformerSSL(_SelfSupervisedWrapper):
    """Informer backbone for self-supervised masked autoencoding."""

    def __init__(self, configs, grad_dim: int = 128):
        cfg = copy.copy(configs)
        cfg.task_name = "imputation"
        backbone = InformerModel(cfg)
        super().__init__(backbone, cfg.d_model, cfg.enc_in, grad_dim)

    def get_hidden(self, x_enc: torch.Tensor) -> torch.Tensor:
        enc_out = self.backbone.enc_embedding(x_enc, None)      # [B, T, d_model]
        enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None)
        return enc_out                                           # [B, T, d_model]


# ---------------------------------------------------------------------------
# TimesNet wrapper  (H = d_model)
# ---------------------------------------------------------------------------

class TimesNetSSL(_SelfSupervisedWrapper):
    """TimesNet backbone for self-supervised masked autoencoding."""

    def __init__(self, configs, grad_dim: int = 128):
        cfg = copy.copy(configs)
        cfg.task_name = "imputation"
        cfg.pred_len = 0  # imputation does not need pred_len
        backbone = TimesNetModel(cfg)
        super().__init__(backbone, cfg.d_model, cfg.enc_in, grad_dim)

    def get_hidden(self, x_enc: torch.Tensor) -> torch.Tensor:
        enc_out = self.backbone.enc_embedding(x_enc, None)      # [B, T, d_model]
        for i in range(self.backbone.layer):
            enc_out = self.backbone.layer_norm(self.backbone.model[i](enc_out))
        return enc_out                                           # [B, T, d_model]


# ---------------------------------------------------------------------------
# LightTS wrapper  (H = enc_in, i.e. the channel dimension of the output)
# ---------------------------------------------------------------------------

class LightTSSSL(_SelfSupervisedWrapper):
    """LightTS backbone for self-supervised masked autoencoding.

    LightTS.encoder() returns [B, seq_len, enc_in]; we treat enc_in as H.
    """

    def __init__(self, configs, grad_dim: int = 128):
        cfg = copy.copy(configs)
        cfg.task_name = "imputation"
        cfg.pred_len = cfg.seq_len   # imputation keeps the same length
        backbone = LightTSModel(cfg)
        # The output of LightTS.encoder has enc_in channels
        super().__init__(backbone, cfg.enc_in, cfg.enc_in, grad_dim)
        # Keep original seq_len for slicing (LightTS may pad internally)
        self._orig_seq_len = configs.seq_len

    def get_hidden(self, x_enc: torch.Tensor) -> torch.Tensor:
        out = self.backbone.encoder(x_enc)    # [B, pred_len, enc_in]
        # pred_len == original seq_len; if LightTS padded internally it still
        # returns exactly pred_len time steps.
        return out                            # [B, T, enc_in]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

BACKBONE_CLASSES = {
    "Informer": InformerSSL,
    "LightTS": LightTSSSL,
    "TimesNet": TimesNetSSL,
}


def build_wrapper(model_name: str, configs, grad_dim: int = 128) -> _SelfSupervisedWrapper:
    """Instantiate a self-supervised wrapper by model name."""
    if model_name not in BACKBONE_CLASSES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(BACKBONE_CLASSES)}")
    return BACKBONE_CLASSES[model_name](configs, grad_dim=grad_dim)
