"""
shift_distance.py
Direct statistical evidence of distribution shift between the training track and
a shifted track, measured three independent ways (simplest -> strongest):

 1. PIXEL: Jensen-Shannon divergence (base 2, in [0,1]) between grayscale
    intensity histograms of the two image sets.
 2. FEATURE / Mahalanobis: fit a Gaussian to the model's 10-d penultimate-layer
    features on in-distribution validation images; report the Mahalanobis
    distance distribution for ID vs shifted samples and the AUROC of the
    distance as an OOD detector (the standard Lee et al. 2018 baseline).
 3. FEATURE / divergence: symmetrized KL between Gaussians fitted to the two
    feature sets (closed form).

Also correlates Mahalanobis distance with the MC-dropout std where the per-image
files exist — two independent detectors agreeing triangulates the shift claim.

Run:  python shift_distance.py --shift_dir data_cte/generated_road
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets, load_cte_dataframe, CTEDataset
from cte_model import TaxiNetCTE


def extract_features(model, loader, device, hook_layer):
    feats = []
    grabbed = {}

    def hook(_m, _i, out):
        grabbed["f"] = out.detach()

    h = hook_layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
            feats.append(grabbed["f"].cpu().numpy())
    h.remove()
    return np.concatenate(feats, axis=0)


def grayscale_histogram(loader, bins=64, max_images=500):
    vals = []
    n = 0
    for imgs, _ in loader:
        g = imgs.mean(dim=1).numpy().ravel()   # images are [0,1]; mean over channels
        vals.append(g)
        n += imgs.shape[0]
        if n >= max_images:
            break
    v = np.concatenate(vals)
    hist, _ = np.histogram(v, bins=bins, range=(0, 1), density=False)
    p = hist.astype(float) + 1e-9
    return p / p.sum()


def js_divergence(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gaussian_sym_kl(x, y, eps=1e-6):
    k = x.shape[1]
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False) + eps * np.eye(k)
    cy = np.cov(y, rowvar=False) + eps * np.eye(k)
    icx, icy = np.linalg.inv(cx), np.linalg.inv(cy)
    _, ldx = np.linalg.slogdet(cx)
    _, ldy = np.linalg.slogdet(cy)
    d = my - mx
    kl_xy = 0.5 * (np.trace(icy @ cx) + d @ icy @ d - k + (ldy - ldx))
    d2 = mx - my
    kl_yx = 0.5 * (np.trace(icx @ cy) + d2 @ icx @ d2 - k + (ldx - ldy))
    return float(0.5 * (kl_xy + kl_yx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift_dir", required=True)
    ap.add_argument("--id_dir", default="data_cte/generated_track")
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--out_dir", default="results_cte")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--id_mc_csv", default="results_cte/mc_val.csv")
    ap.add_argument("--shift_mc_csv", default="results_cte/mc_shift.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ID = the held-out validation split (same seed as everywhere else)
    _, id_ds, info = build_cte_datasets(args.id_dir, seed=42)
    id_loader = DataLoader(id_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sh_df = load_cte_dataframe(args.shift_dir)
    sh_ds = CTEDataset(sh_df, augment=False)
    sh_loader = DataLoader(sh_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"ID: {len(id_ds)} val images ({args.id_dir})  |  "
          f"shifted: {len(sh_ds)} images ({args.shift_dir})")

    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, 80, 160, device=device))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # ---- 1) pixel-level JS ----
    p_id = grayscale_histogram(id_loader)
    p_sh = grayscale_histogram(sh_loader)
    js_pix = js_divergence(p_id, p_sh)
    print(f"\n1) PIXEL  Jensen-Shannon (grayscale hist, base 2): {js_pix:.4f}  "
          f"(0 = identical, 1 = disjoint)")

    # ---- 2) feature-space Mahalanobis ----
    hook_layer = model.fc[7]   # ELU after the 10-unit dense layer -> 10-d features
    f_id = extract_features(model, id_loader, device, hook_layer)
    f_sh = extract_features(model, sh_loader, device, hook_layer)
    mu = f_id.mean(0)
    cov = np.cov(f_id, rowvar=False) + 1e-6 * np.eye(f_id.shape[1])
    icov = np.linalg.inv(cov)

    def maha(F):
        d = F - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, icov, d))

    m_id, m_sh = maha(f_id), maha(f_sh)
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(
        np.concatenate([np.zeros(len(m_id)), np.ones(len(m_sh))]),
        np.concatenate([m_id, m_sh])))
    print(f"\n2) FEATURE  Mahalanobis distance (10-d penultimate layer, Gaussian fit on ID):")
    print(f"   ID      : median {np.median(m_id):7.2f}   mean {m_id.mean():7.2f}")
    print(f"   shifted : median {np.median(m_sh):7.2f}   mean {m_sh.mean():7.2f}")
    print(f"   AUROC of distance as OOD detector: {auroc:.4f}  (0.5 = chance, 1.0 = perfect)")

    # ---- 3) feature-space symmetrized KL ----
    skl = gaussian_sym_kl(f_id, f_sh)
    print(f"\n3) FEATURE  symmetrized KL between Gaussian fits: {skl:.2f} nats")

    # ---- bonus: Mahalanobis vs MC-dropout std ----
    corr_note = {}
    for name, path, m in (("ID", args.id_mc_csv, m_id), ("shifted", args.shift_mc_csv, m_sh)):
        if os.path.exists(path):
            stds = [float(r["mc_std"]) for r in csv.DictReader(open(path))]
            if len(stds) == len(m):
                c = float(np.corrcoef(m, np.array(stds))[0, 1])
                corr_note[name] = c
                print(f"   corr(Mahalanobis, MC std) on {name}: {c:.3f}")
            else:
                print(f"   ({name}: mc csv length {len(stds)} != features {len(m)}; skipped)")

    out = {"pixel_js_bits": js_pix,
           "mahalanobis": {"id_median": float(np.median(m_id)),
                           "shift_median": float(np.median(m_sh)),
                           "id_mean": float(m_id.mean()),
                           "shift_mean": float(m_sh.mean()),
                           "auroc": auroc},
           "feature_sym_kl_nats": skl,
           "corr_maha_mcstd": corr_note,
           "id_dir": args.id_dir, "shift_dir": args.shift_dir}
    jp = os.path.join(args.out_dir, "shift_distance.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {jp}")


if __name__ == "__main__":
    main()
