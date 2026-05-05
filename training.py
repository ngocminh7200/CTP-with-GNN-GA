"""
Full training pipeline for the GNN scheduler.

Usage:
    python run_training.py

Outputs (under ./results/):
    best_model.pt      — checkpoint with best validation loss
    history.json       — per-epoch metrics
    loss_curve.png     — train vs val loss
    accuracy_curve.png — val ts/room/both accuracy
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from GNN import GNN
from training_test_GNN import (
    train_one_epoch,
    evaluate,
    extract_mappings_from_edges,
    build_room_local_targets,
)


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    cases: list = field(default_factory=lambda: [50, 100, 150, 200, 300, 400,500,800,1000])
    data_dir: str = "TrainingData"
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac is implied = 1 - train - val

    # Model
    hidden_channels: int = 64
    nb_layers_conv: int = 4
    nb_layers_mlp: int = 4
    nb_heads: int = 8
    dropout_conv: float = 0.0
    dropout_mlp: float = 0.0

    # Training

    max_epochs: int = 100
    patience: int = 10           # early-stop if val loss doesn't improve for this many epochs
    lr: float = 1e-3
    weight_decay: float = 1e-5

    num_workers: int = 4  # 4 is sane on most desktops; bump to 8 if no OOM
    batch_size: int = 8  # CHANGED: 4 -> 16. RTX 4070 has 12 GB; should fit.

    # Misc
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "results"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Make training reproducible across reruns."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_combined_dataset(cases: list, data_dir: str) -> list:
    """Concatenate all dataset_<size>.pt files into one list of HeteroData graphs."""
    dataset = []
    for case in cases:
        path = Path(data_dir) / f"dataset_{case}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing training file: {path}")
        dataset += torch.load(path, weights_only=False)
    return dataset


def split_dataset(dataset: list, train_frac: float, val_frac: float, seed: int):
    """Random 80/10/10 split (or whatever fractions are configured)."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)


def detect_feature_dims(g) -> dict:
    """Read input feature dimensions from a single sample graph."""
    return {
        "course_feature_dim":    g["course"].x.size(1),
        "professor_feature_dim": g["professor"].x.size(1),
        "room_feature_dim":      g["Room"].x.size(1),
        "program_feature_dim":   g["program"].x.size(1),
        "timeslot_feature_dim":  g["timeslot"].x.size(1),
    }


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

def plot_loss(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (CE_ts + CE_room) per course")
    ax.set_title("Training and validation loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(history["val_ts_acc"]) + 1))
    ax.plot(epochs, history["val_ts_acc"],   label="timeslot",     linewidth=1.5)
    ax.plot(epochs, history["val_room_acc"], label="room",         linewidth=1.5)
    ax.plot(epochs, history["val_both_acc"], label="both correct", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Validation accuracy by prediction head")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main(cfg: TrainConfig) -> None:
    set_seeds(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {cfg.device}")
    print(f"Output dir: {out_dir.resolve()}")

    # ---- Data ----
    print("\n--- Loading data ---")
    dataset = load_combined_dataset(cfg.cases, cfg.data_dir)
    print(f"Loaded {len(dataset)} graphs from sizes {cfg.cases}")

    train_set, val_set, test_set = split_dataset(
        dataset, cfg.train_frac, cfg.val_frac, cfg.seed,
    )
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # NOTE: train shuffles, val/test do not. shuffle in val/test would not
    # change the metrics but would make per-epoch numbers harder to compare.
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0,


    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,

    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,


    )

    # ---- Model ----
    sample = dataset[0]
    feature_dims = detect_feature_dims(sample)
    num_timeslots = sample["timeslot"].x.size(0)
    print(f"\n--- Building model ---")
    print(f"Feature dims: {feature_dims}")
    print(f"num_timeslots: {num_timeslots}")

    model = GNN(
        num_timeslots=num_timeslots,
        hidden_channels=cfg.hidden_channels,
        nb_layers_conv=cfg.nb_layers_conv,
        nb_layers_mlp=cfg.nb_layers_mlp,
        nb_heads=cfg.nb_heads,
        dropout_conv=cfg.dropout_conv,
        dropout_mlp=cfg.dropout_mlp,
        **feature_dims,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ---- Training loop ----
    print(f"\n--- Training (max {cfg.max_epochs} epochs, patience {cfg.patience}) ---")
    history = {
        "train_loss":    [],
        "val_loss":      [],
        "val_ts_acc":    [],
        "val_room_acc":  [],
        "val_both_acc":  [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    ckpt_path = out_dir / "best_model.pt"

    header = f"{'Epoch':>5} | {'Train':>9} | {'Val':>9} | {'TS':>6} | {'Room':>6} | {'Both':>6} | {'Note'}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_loss, val_ts, val_room, val_both = evaluate(model, val_loader, cfg.device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ts_acc"].append(val_ts)
        history["val_room_acc"].append(val_room)
        history["val_both_acc"].append(val_both)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "feature_dims": feature_dims,
                "num_timeslots": num_timeslots,
                "config": asdict(cfg),
            }, ckpt_path)
            note = "✓ saved"
        else:
            epochs_without_improvement += 1
            note = f"({epochs_without_improvement}/{cfg.patience})"

        print(f"{epoch:>5} | {train_loss:>9.4f} | {val_loss:>9.4f} | "
              f"{val_ts:>6.2%} | {val_room:>6.2%} | {val_both:>6.2%} | {note}")

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stop: no improvement for {cfg.patience} epochs.")
            break

    print(f"\nBest epoch: {best_epoch}, val_loss = {best_val_loss:.4f}")

    # ---- Final test evaluation on best checkpoint ----
    print("\n--- Final test set evaluation (best checkpoint) ---")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_ts, test_room, test_both = evaluate(model, test_loader, cfg.device)
    print(f"Test loss:           {test_loss:.4f}")
    print(f"Test TS accuracy:    {test_ts:.2%}")
    print(f"Test Room accuracy:  {test_room:.2%}")
    print(f"Test Both accuracy:  {test_both:.2%}")

    # ---- Save history & test results ----
    summary = {
        "config": asdict(cfg),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_ts_acc": test_ts,
        "test_room_acc": test_room,
        "test_both_acc": test_both,
        "history": history,
    }
    with open(out_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nHistory saved to {out_dir / 'history.json'}")

    # ---- Plots ----
    plot_loss(history, out_dir / "loss_curve.png")
    plot_accuracy(history, out_dir / "accuracy_curve.png")
    print(f"Plots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)