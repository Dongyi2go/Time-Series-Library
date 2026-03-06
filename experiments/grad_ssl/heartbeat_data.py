"""
Heartbeat UEA dataset loading utilities.

Provides:
  - HeartbeatDataset: thin wrapper around UEAloader for TRAIN / TEST splits.
  - iter_samples(): yields (x [1,T,C], pad_mask [1,T], label [1]) for each
    sample in fixed (deterministic) order.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_provider.data_loader import UEAloader  # noqa: E402
from data_provider.uea import collate_fn  # noqa: E402


class HeartbeatDataset:
    """Load one split (TRAIN or TEST) of the Heartbeat UEA dataset.

    Parameters
    ----------
    root_path : str
        Directory that contains ``Heartbeat_TRAIN.ts`` and ``Heartbeat_TEST.ts``.
        If the files are not found locally, the loader will attempt to download
        them from HuggingFace (requires ``huggingface_hub`` and internet access).
    split : str
        ``'TRAIN'`` or ``'TEST'``.
    """

    def __init__(self, root_path: str, split: str = "TRAIN"):
        args = argparse.Namespace(
            model_id="Heartbeat",
            augmentation_ratio=0,
        )
        self._uea = UEAloader(args, root_path=root_path, flag=split)
        self.max_seq_len: int = self._uea.max_seq_len
        self.enc_in: int = self._uea.feature_df.shape[1]
        self.num_class: int = len(self._uea.class_names)
        self.class_names = self._uea.class_names
        self.n_samples: int = len(self._uea)

    # ------------------------------------------------------------------
    # Batch DataLoader (for pretraining)
    # ------------------------------------------------------------------

    def get_loader(self, batch_size: int = 32, max_len: int = None,
                   shuffle: bool = False) -> DataLoader:
        """Return a DataLoader that pads sequences to ``max_len``."""
        if max_len is None:
            max_len = self.max_seq_len
        return DataLoader(
            self._uea,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_len),
            num_workers=0,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # Per-sample iterator (for feature extraction at batch_size=1)
    # ------------------------------------------------------------------

    def iter_samples(self, max_len: int = None):
        """Iterate over all samples in fixed index order.

        Yields
        ------
        x_pad  : torch.FloatTensor  [1, max_len, C]
        pad_mask: torch.FloatTensor [1, max_len]   (1=real, 0=padded)
        label  : torch.LongTensor  [1]
        """
        if max_len is None:
            max_len = self.max_seq_len

        for idx in range(self.n_samples):
            x_raw, label_raw = self._uea[idx]   # x_raw: [T_i, C], label_raw: [1]
            T, C = x_raw.shape

            # Pad or clip to max_len
            x_pad = torch.zeros(max_len, C, dtype=torch.float32)
            real_len = min(T, max_len)
            x_pad[:real_len] = x_raw[:real_len].float()

            pad_mask = torch.zeros(max_len, dtype=torch.float32)
            pad_mask[:real_len] = 1.0

            # Add batch dimension
            yield (
                x_pad.unsqueeze(0),           # [1, T, C]
                pad_mask.unsqueeze(0),         # [1, T]
                label_raw.long().view(1),      # [1]
            )


def load_heartbeat(root_path: str, max_len: int = None):
    """Convenience function: load both splits and return dataset objects.

    Returns
    -------
    train_ds, test_ds : HeartbeatDataset
    max_len : int  (max of train and test max_seq_len if not provided)
    """
    train_ds = HeartbeatDataset(root_path, split="TRAIN")
    test_ds = HeartbeatDataset(root_path, split="TEST")
    if max_len is None:
        max_len = max(train_ds.max_seq_len, test_ds.max_seq_len)
    return train_ds, test_ds, max_len
