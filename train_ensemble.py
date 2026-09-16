"""
train_ensemble.py
Train K independently initialised members for the Deep-Ensemble guard.

Deep Ensembles obtain disagreement from independently trained parameter solutions,
not from stochastic masks within one model (paper Sec. 3.4, Eq. 8-9). Two things
must therefore be held fixed across members and one must vary:

  FIXED  the train/val split (--split_seed, default 42, the same value every other
         script in the pipeline uses). Calibration and the guarded abstraction are
         estimated on the held-out set, so every member must have been trained
         without those frames. A per-member split would leak training frames into
         the calibration set for some members and invalidate tau_DE.
  FIXED  architecture, data, epochs, optimiser, augmentation policy - identical to
         train_cte.py, so the MC-Dropout and Deep-Ensemble arms differ only in the
         uncertainty mechanism.
  VARIES weight initialisation, batch shuffling order, and the augmentation draw
         (all keyed to --init_seed per member).

Dropout is kept as a training regulariser at the same rate as the single model, but
members are EVALUATED deterministically (model.eval(), see de_uncertainty.py). If
members were evaluated with dropout active the arm would measure a dropout-ensemble
hybrid and the comparison could not attribute a result to either mechanism.

Resumable: a member whose checkpoint already exists is skipped unless --force, so
training can be interrupted and restarted, or K extended later without retraining
what is already on disk. The manifest is rewritten after every member.

Run (from av/):
    python train_ensemble.py --K 10
    python train_ensemble.py --K 10 --epochs 30 --data_dir data_cte/generated_track

Outputs:
    results_de/member_00.pth ... member_{K-1}.pth   best-val weights per member
    results_de/ensemble_manifest.json               seeds, best MAE, disc acc per member
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets
from cte_model import TaxiNetCTE, count_parameters
from train_cte import evaluate, discretized_accuracy

SPLIT_SEED = 42          # must match every other script; do not vary per member
BASE_INIT_SEED = 1000    # member m gets BASE_INIT_SEED + m


def set_member_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_one(member, seed, args, device):
    """Train a single member. Returns a manifest record."""
    ckpt = os.path.join(args.out_dir, f"member_{member:02d}.pth")

    set_member_seed(seed)

    # Split is keyed to SPLIT_SEED and is identical for every member.
    train_ds, val_ds, info = build_cte_datasets(args.data_dir, seed=args.split_seed)
    # Augmentation draws are keyed to the MEMBER seed, so members see different
    # flip/brightness realisations of the same underlying training frames.
    train_ds.rng = np.random.default_rng(seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TaxiNetCTE(dropout=args.dropout).to(device)
    with torch.no_grad():                       # initialise LazyLinear before AdamW
        model(torch.zeros(2, 3, 80, 160, device=device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = (device.type == "cuda")

    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_disc_acc": []}
    best_mae = float("inf")
    best_epoch = 0
    t0 = time.time()

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
            best_mae, best_epoch = val_mae, epoch
            torch.save(model.state_dict(), ckpt)
            marker = "  <- best, saved"
        elif not os.path.exists(ckpt):
            torch.save(model.state_dict(), ckpt)

        print(f"  [m{member:02d}] epoch {epoch:2d}/{args.epochs}  "
              f"train_loss {train_loss:.4f}  val_MAE {val_mae:.4f}  "
              f"disc_acc {disc_acc*100:.1f}%{marker}")

    mins = (time.time() - t0) / 60.0
    print(f"  [m{member:02d}] done in {mins:.1f} min  best val MAE {best_mae:.4f} "
          f"(epoch {best_epoch})  -> {ckpt}")

    return {"member": member, "seed": seed, "ckpt": os.path.basename(ckpt),
            "best_val_mae": best_mae, "best_epoch": best_epoch,
            "final_disc_acc": history["val_disc_acc"][-1],
            "epochs": args.epochs, "dropout": args.dropout,
            "split_seed": args.split_seed, "data_dir": args.data_dir,
            "n_train": info["train"], "n_val": info["val"],
            "minutes": round(mins, 2), "history": history}


def write_manifest(path, args, records):
    maes = [r["best_val_mae"] for r in records]
    payload = {
        "data_dir": args.data_dir,
        "split_seed": args.split_seed,
        "base_init_seed": args.base_seed,
        "epochs": args.epochs,
        "dropout_train": args.dropout,
        "eval_mode": "deterministic (model.eval(); dropout OFF at inference)",
        "K_trained": len(records),
        "member_mae_mean": float(np.mean(maes)) if maes else None,
        "member_mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else None,
        "member_mae_min": float(np.min(maes)) if maes else None,
        "member_mae_max": float(np.max(maes)) if maes else None,
        "members": records,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cte/generated_track")
    ap.add_argument("--out_dir", default="results_de")
    ap.add_argument("--K", type=int, default=10,
                    help="number of members to train (train the largest K you are "
                         "willing to; ensemble_size_sweep.py picks the final K)")
    ap.add_argument("--base_seed", type=int, default=BASE_INIT_SEED,
                    help="member m uses init seed base_seed + m")
    ap.add_argument("--split_seed", type=int, default=SPLIT_SEED,
                    help="train/val split seed; MUST match the rest of the pipeline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="retrain members whose checkpoint already exists")
    args = ap.parse_args()

    if args.split_seed != SPLIT_SEED:
        print(f"WARNING: --split_seed {args.split_seed} differs from the pipeline "
              f"value {SPLIT_SEED}. The held-out set will not match mc_val.csv and "
              f"the MC/DE comparison will be invalid.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training {args.K} members  |  split_seed {args.split_seed} (fixed)  |  "
          f"init seeds {args.base_seed}..{args.base_seed + args.K - 1}")

    manifest_path = os.path.join(args.out_dir, "ensemble_manifest.json")
    records = []
    if os.path.exists(manifest_path) and not args.force:
        with open(manifest_path) as f:
            records = json.load(f).get("members", [])

    done = {r["member"] for r in records}
    for m in range(args.K):
        ckpt = os.path.join(args.out_dir, f"member_{m:02d}.pth")
        if os.path.exists(ckpt) and m in done and not args.force:
            print(f"[m{m:02d}] checkpoint exists, skipping (use --force to retrain)")
            continue
        seed = args.base_seed + m
        print(f"\n[m{m:02d}] init seed {seed}")
        rec = train_one(m, seed, args, device)
        records = [r for r in records if r["member"] != m] + [rec]
        records.sort(key=lambda r: r["member"])
        write_manifest(manifest_path, args, records)   # rewrite after every member

    print("\n" + "-" * 62)
    print(f"{'member':>7} {'seed':>6} {'best val MAE':>13} {'disc acc':>10}")
    for r in records:
        print(f"{r['member']:>7} {r['seed']:>6} {r['best_val_mae']:>13.4f} "
              f"{r['final_disc_acc']*100:>9.1f}%")
    maes = [r["best_val_mae"] for r in records]
    if len(maes) > 1:
        print(f"\nMember MAE spread: mean {np.mean(maes):.4f}  sd {np.std(maes, ddof=1):.4f}  "
              f"range [{min(maes):.4f}, {max(maes):.4f}]")
        print("A near-zero spread means the members converged to near-identical "
              "solutions; their disagreement would then be small by construction and "
              "the ensemble guard would inherit that. Report this spread.")
    print(f"\nManifest -> {manifest_path}")
    print("Next:  python de_uncertainty.py")


if __name__ == "__main__":
    main()
