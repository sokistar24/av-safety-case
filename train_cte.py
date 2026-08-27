"""
train_cte.py
Train the TaxiNet-style CNN to predict continuous cte from camera images.

Reports MAE (the paper's headline metric for cte; TaxiNet achieved MAE 1.185) and,
after training, a first DISCRETIZED accuracy: bin predictions and ground truth into
the 6 states (via the shared bin_cte) and measure agreement. That preview tells us
whether the model is good enough for the confusion-matrix / PRISM analysis to be
meaningful.

Run (from av/, with the collected data present):
    python train_cte.py --data_dir data_cte/generated_track --epochs 30

Outputs:
    results_cte/cte_model.pth        best-val weights (driving test + confusion matrix use this)
    results_cte/cte_history.json     loss / MAE curves + final discretized accuracy
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets, bin_cte, STATES, STATE_NAMES
from cte_model import TaxiNetCTE, count_parameters

SEED = 42


def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (MSE, MAE) and arrays of (true, pred) cte for discretization."""
    model.eval()
    se = ae = n = 0.0
    trues, preds = [], []
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        out = model(imgs)
        se += ((out - targets) ** 2).sum().item()
        ae += (out - targets).abs().sum().item()
        n += targets.numel()
        trues.append(targets.cpu().numpy().ravel())
        preds.append(out.cpu().numpy().ravel())
    return se / n, ae / n, np.concatenate(trues), np.concatenate(preds)


def discretized_accuracy(trues, preds):
    """Bin both with the shared bin_cte and report state-agreement accuracy."""
    ts = np.array([bin_cte(x) for x in trues])
    ps = np.array([bin_cte(x) for x in preds])
    return float((ts == ps).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cte/generated_track")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="results_cte")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, info = build_cte_datasets(args.data_dir, seed=SEED)
    print(f"Data: total={info['total']}  train={info['train']}  val={info['val']}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TaxiNetCTE(dropout=args.dropout).to(device)
    # initialize LazyLinear before optimizer sees params
    with torch.no_grad():
        model(torch.zeros(2, 3, 80, 160, device=device))
    print(f"Model parameters: {count_parameters(model)/1e6:.3f} M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = (device.type == "cuda")

    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_disc_acc": []}
    best_mae = float("inf")
    ckpt = os.path.join(args.out_dir, "cte_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(imgs)
                loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * imgs.size(0)

        train_loss = running / len(train_ds)
        val_mse, val_mae, trues, preds = evaluate(model, val_loader, device)
        disc_acc = discretized_accuracy(trues, preds)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_disc_acc"].append(disc_acc)

        marker = ""
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt)
            marker = "  <- best, saved"
        elif not os.path.exists(ckpt):
            torch.save(model.state_dict(), ckpt)

        print(f"Epoch {epoch:2d}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_MSE {val_mse:.4f}  val_MAE {val_mae:.4f}  "
              f"disc_acc {disc_acc*100:.1f}%{marker}")

    history["best_val_mae"] = best_mae
    with open(os.path.join(args.out_dir, "cte_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"Done. Best val MAE: {best_mae:.4f}  (paper's TaxiNet cte MAE was 1.185)")
    print(f"Final discretized accuracy: {history['val_disc_acc'][-1]*100:.1f}%")
    print(f"Saved weights -> {ckpt}")


if __name__ == "__main__":
    main()
