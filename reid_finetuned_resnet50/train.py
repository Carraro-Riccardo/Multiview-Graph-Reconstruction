"""Fine-tune ReIDResNet50 with NT-Xent contrastive loss on hydraulic component pairs.

Usage:
    python reid_finetuned_resnet50/train.py [--epochs 50] [--batch 32] [--lr 1e-4]

Output:
    reid_finetuned_resnet50/weights/best.pt   ← best val-loss checkpoint
    reid_finetuned_resnet50/weights/last.pt   ← final epoch checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from reid_finetuned_resnet50.model import ReIDResNet50  # noqa: E402

WEIGHTS_DIR  = Path(__file__).parent / "weights"
PAIRS_FILE   = Path(__file__).parent / "pairs.json"
RANDOM_SEED  = 42
PATIENCE     = 10   # early stopping patience (epochs without val improvement)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs: list[dict], transform, root: Path):
        self.pairs     = pairs
        self.transform = transform
        self.root      = root

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        img_a = self.transform(Image.open(self.root / p["img_a"]).convert("RGB"))
        img_b = self.transform(Image.open(self.root / p["img_b"]).convert("RGB"))
        label = torch.tensor(p["label"], dtype=torch.float32)
        return img_a, img_b, label


# ── Loss: NT-Xent on pairs ────────────────────────────────────────────────────

class NTXentPairLoss(nn.Module):
    """NT-Xent adapted for labelled pairs.

    For each sample in a batch, positives are other samples with label=1
    sharing the same anchor; negatives are samples with label=0 or
    cross-batch negatives.

    Simpler formulation: supervised contrastive on pairs.
      loss = mean over positive pairs of:
               -sim(a,b)/τ + log Σ_j exp(sim(a, z_j)/τ)
    where j ranges over all other samples in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.τ = temperature

    def forward(
        self,
        emb_a: torch.Tensor,   # (B, D)
        emb_b: torch.Tensor,   # (B, D)
        labels: torch.Tensor,  # (B,) 0/1
    ) -> torch.Tensor:
        B = emb_a.size(0)
        # Stack all embeddings: first half = a, second half = b
        embs   = torch.cat([emb_a, emb_b], dim=0)           # (2B, D)
        labels2 = torch.cat([labels, labels], dim=0)        # (2B,)

        # Cosine sim matrix
        sim = embs @ embs.T / self.τ                        # (2B, 2B)

        # Mask self-comparisons
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(mask_self, -1e9)

        # For each anchor i, positives are j where:
        #   (i < B and j = i+B or i >= B and j = i-B) AND label==1
        # i.e. the paired sample with the same label=1
        loss = torch.tensor(0.0, device=sim.device)
        n_pos = 0
        for i in range(B):
            if labels[i].item() != 1:
                continue
            j = i + B   # paired b-embedding
            numerator   = sim[i, j]
            denominator = torch.logsumexp(sim[i], dim=0)
            loss += denominator - numerator
            n_pos += 1

        if n_pos == 0:
            return torch.tensor(0.0, device=sim.device, requires_grad=True)
        return loss / n_pos


# ── Training loop ─────────────────────────────────────────────────────────────

def train(epochs: int = 30, batch_size: int = 32, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pairs
    pairs = json.loads(PAIRS_FILE.read_text())
    random.shuffle(pairs)
    split     = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]
    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")

    model     = ReIDResNet50().to(device)

    # ── augmentation for train (reid-specific) ────────────────────────────
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),        # partial occlusion / zoom
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.4, contrast=0.4,        # lighting variation
                      saturation=0.3, hue=0.08),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),   # blur / focus variation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),        # random patch occlusion
    ])
    val_transform = ReIDResNet50.get_transform()            # no augmentation for val

    criterion = NTXentPairLoss(temperature=0.07)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = PairDataset(train_pairs, train_transform, ROOT)
    val_ds   = PairDataset(val_pairs,   val_transform,   ROOT)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    WEIGHTS_DIR.mkdir(exist_ok=True)
    best_val_loss  = float("inf")
    epochs_no_imp  = 0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        t_loss = 0.0
        for img_a, img_b, lbl in train_dl:
            img_a, img_b, lbl = img_a.to(device), img_b.to(device), lbl.to(device)
            emb_a = model(img_a)
            emb_b = model(img_b)
            loss  = criterion(emb_a, emb_b, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        # ── val ──
        model.eval()
        v_loss = 0.0
        sim_pos, sim_neg = [], []
        with torch.no_grad():
            for img_a, img_b, lbl in val_dl:
                img_a, img_b, lbl = img_a.to(device), img_b.to(device), lbl.to(device)
                emb_a = model(img_a)
                emb_b = model(img_b)
                v_loss += criterion(emb_a, emb_b, lbl).item()
                sims = (emb_a * emb_b).sum(dim=1).cpu().numpy()
                for s, l in zip(sims, lbl.cpu().numpy()):
                    (sim_pos if l == 1 else sim_neg).append(float(s))

        n_train = max(len(train_dl), 1)
        n_val   = max(len(val_dl),   1)
        mean_pos = np.mean(sim_pos) if sim_pos else float("nan")
        mean_neg = np.mean(sim_neg) if sim_neg else float("nan")

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={t_loss/n_train:.4f}  val_loss={v_loss/n_val:.4f} | "
            f"sim_pos={mean_pos:+.3f}  sim_neg={mean_neg:+.3f}  "
            f"gap={mean_pos-mean_neg:+.3f}"
        )

        torch.save(model.state_dict(), WEIGHTS_DIR / "last.pt")
        cur_val = v_loss / n_val
        if cur_val < best_val_loss:
            best_val_loss = cur_val
            epochs_no_imp = 0
            torch.save(model.state_dict(), WEIGHTS_DIR / "best.pt")
            print(f"  → best checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to {WEIGHTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=1e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
