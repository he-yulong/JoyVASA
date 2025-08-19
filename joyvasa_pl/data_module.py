from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random


class MotionAudioWindowDataset(Dataset):
    """
    Expects each item to provide:
      - X0: (T, Dm) clean motion seq from LivePortrait motion encoder
      - A:  (T, Da) audio features from wav2vec2
    We sample a window with Wpre past + Wcur current frames.
    """

    def __init__(self, items, Wpre=25, Wcur=100, t_max=1000, start_tokens=True):
        self.items = items
        self.Wpre, self.Wcur = Wpre, Wcur
        self.t_max = t_max
        self.start_tokens = start_tokens

    def __len__(self): return len(self.items)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        X0, A = self.items[idx]  # tensors: (T, Dm), (T, Da)
        T = min(X0.shape[0], A.shape[0])

        # Random truncation for robustness to varying speech lengths (paper §3.3.1). :contentReference[oaicite:4]{index=4}
        end = random.randint(self.Wcur, T)  # inclusive
        start = max(0, end - (self.Wpre + self.Wcur))
        X0w = X0[start:end]  # (Wpre+Wcur, Dm) possibly shorter at beginning
        Aw = A[start:end]  # (Wpre+Wcur, Da)

        # Left pad with learnable A_start / X_start at model side; here we just mark masks
        # Split: [-Wpre:0] for motion cond; [0:Wcur] for current window
        return {
            "X0": X0w,  # clean motion for whole window
            "A": Aw,  # audio for whole window
            "mask": torch.ones(len(X0w)).bool(),
            "t": torch.randint(0, self.t_max, (1,), dtype=torch.long)
        }


class MotionAudioDataModule(pl.LightningDataModule):
    def __init__(self, train_items, val_items, batch_size=16, num_workers=8, Wpre=25, Wcur=100, t_max=1000):
        super().__init__()
        self.train_items, self.val_items = train_items, val_items
        self.bs, self.nw = batch_size, num_workers
        self.Wpre, self.Wcur, self.t_max = Wpre, Wcur, t_max

    def setup(self, stage=None):
        self.train_ds = MotionAudioWindowDataset(self.train_items, self.Wpre, self.Wcur, self.t_max)
        self.val_ds = MotionAudioWindowDataset(self.val_items, self.Wpre, self.Wcur, self.t_max)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True, num_workers=self.nw, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False, num_workers=self.nw, pin_memory=True)
