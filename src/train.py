"""Training loop with early stopping. Device-aware: picks CUDA, MPS, or CPU."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """x is (n, T, C); we transpose to (n, C, T) for Conv1d."""
    x_t = torch.from_numpy(np.transpose(x, (0, 2, 1))).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    checkpoint_path: str | None = None


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_model(model: nn.Module, train_xy: tuple[np.ndarray, np.ndarray],
                val_xy: tuple[np.ndarray, np.ndarray], cfg: TrainConfig | None = None,
                device: torch.device | None = None) -> tuple[nn.Module, History]:
    cfg = cfg or TrainConfig()
    device = device or pick_device()
    model = model.to(device)

    train_loader = make_loader(*train_xy, batch_size=cfg.batch_size, shuffle=True)
    val_loader = make_loader(*val_xy, batch_size=cfg.batch_size, shuffle=False)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = History()
    best_val_acc = -1.0
    best_state = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses, accs = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            accs.append(_accuracy(logits, yb))
        train_loss = float(np.mean(losses))
        train_acc = float(np.mean(accs))

        model.eval()
        losses, accs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                losses.append(loss_fn(logits, yb).item())
                accs.append(_accuracy(logits, yb))
        val_loss = float(np.mean(losses))
        val_acc = float(np.mean(accs))

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch} (best val_acc {best_val_acc:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if cfg.checkpoint_path:
            Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, cfg.checkpoint_path)

    return model, history


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device | None = None,
            batch_size: int = 256) -> np.ndarray:
    device = device or pick_device()
    model = model.to(device).eval()
    x_t = torch.from_numpy(np.transpose(x, (0, 2, 1))).float()
    out = []
    for i in range(0, len(x_t), batch_size):
        batch = x_t[i:i + batch_size].to(device)
        out.append(model(batch).softmax(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
