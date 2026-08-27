"""
donkey_drive_cte.py
Closed-loop driving test for the trained CTE model.

This is the "how well does it actually drive?" test (Test B). The model perceives
cte from each camera frame; a simple PROPORTIONAL controller turns that predicted
cte into steering; the car drives; we log laps completed and how well it stays on
the road (using the sim's GROUND-TRUTH cte, so we can compare predicted vs. true).

Perception -> controller split (like TaxiNet Fig. 1):
    cte_pred = model(image)                       # perception
    steer = -Kp * (cte_pred - lane_offset)        # controller (proportional)
    env.step([steer, throttle])                   # actuation

Live preprocessing is imported from cte_dataset so it is byte-identical to training.

Run (load the track in the sim first, model already trained):
    python donkey_drive_cte.py --env donkey-generated-track-v0 --laps 3

Logs results_cte/drive_<track>_log.csv with per-frame:
    time_s, cte_true, cte_pred, steer, throttle, speed, lap, hit
and prints a summary: laps completed, mean |true cte|, time off-road, etc.
"""

import os
import csv
import time
import argparse
import numpy as np
import cv2
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401

from cte_dataset import CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H, ROAD_EDGE, bin_cte
from cte_model import TaxiNetCTE


def preprocess(obs):
    """RGB frame from sim -> model input tensor. IDENTICAL to cte_dataset._load_image."""
    img = obs[CROP_TOP:CROP_BOTTOM, :, :]          # same crop as training
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="gym env id matching the loaded track")
    ap.add_argument("--track", default=None, help="label for the log file (defaults from env)")
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--laps", type=int, default=3, help="stop after this many laps")
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--out_dir", default="results_cte")
    # controller + speed
    ap.add_argument("--kp", type=float, default=0.5, help="proportional gain on cte error")
    ap.add_argument("--lane_offset", type=float, default=-0.3,
                    help="target cte (Framing B: keep the lane, not the centerline)")
    ap.add_argument("--throttle", type=float, default=0.3)
    ap.add_argument("--show", action="store_true", help="show the camera window")
    args = ap.parse_args()

    track = args.track or args.env.replace("donkey-", "").replace("-v0", "")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load trained model
    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, IMG_H, IMG_W, device=device))  # init LazyLinear
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded model: {args.ckpt}")

    conf = {"exe_path": "remote", "host": args.host, "port": args.port}
    print(f"Connecting to sim on {args.host}:{args.port} as '{args.env}' ...")
    env = gym.make(args.env, conf=conf)
    obs, info = env.reset()
    print(f"Connected. Driving track '{track}' with P-controller (Kp={args.kp}).")

    log_path = os.path.join(args.out_dir, f"drive_{track}_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["time_s", "cte_true", "cte_pred", "steer", "throttle",
                     "speed", "lap", "hit"])

    frame = 0
    start_lap = info.get("lap_count", 0)
    off_road_frames = 0
    abs_cte_sum = 0.0
    t0 = time.time()
    laps_done = 0

    try:
        while True:
            cte_true = float(info.get("cte", 0.0))
            speed = float(info.get("speed", 0.0))
            hit = info.get("hit", "none")
            lap = int(info.get("lap_count", 0)) - start_lap

            # --- perception: model predicts cte from the frame ---
            x = preprocess(obs).to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device.type == "cuda")):
                    cte_pred = float(model(x).item())

            # --- controller: proportional steering from predicted cte ---
            steer = -args.kp * (cte_pred - args.lane_offset)
            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = args.throttle

            # --- metrics on TRUE cte ---
            abs_cte_sum += abs(cte_true)
            if abs(cte_true) > ROAD_EDGE:
                off_road_frames += 1

            writer.writerow([f"{time.time()-t0:.3f}", f"{cte_true:.5f}",
                             f"{cte_pred:.5f}", f"{steer:.4f}", f"{throttle:.3f}",
                             f"{speed:.4f}", lap, hit])
            log_file.flush()

            if args.show:
                cv2.imshow("camera", cv2.cvtColor(
                    cv2.resize(obs, (320, 240)), cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if frame % 30 == 0:
                err = cte_pred - cte_true
                print(f"t={time.time()-t0:5.1f}s lap={lap} cte_true={cte_true:+.2f} "
                      f"cte_pred={cte_pred:+.2f} err={err:+.2f} steer={steer:+.2f}",
                      end="\r")

            obs, reward, terminated, truncated, info = env.step(
                np.array([steer, throttle], dtype=np.float32))
            frame += 1

            if lap >= args.laps:
                laps_done = lap
                print(f"\nCompleted {args.laps} laps.")
                break
            if frame >= args.max_frames:
                laps_done = lap
                print(f"\nReached max_frames ({args.max_frames}).")
                break
            if terminated or truncated:
                laps_done = lap
                print(f"\nEpisode ended (cte_true={cte_true:+.2f}, hit={hit}) "
                      f"after {lap} lap(s), {frame} frames.")
                break
    except KeyboardInterrupt:
        laps_done = int(info.get("lap_count", 0)) - start_lap
        print("\nInterrupted.")
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        env.close()
        elapsed = time.time() - t0
        mean_abs_cte = abs_cte_sum / max(frame, 1)
        off_road_pct = off_road_frames / max(frame, 1) * 100
        print("=" * 60)
        print(f"DRIVING SUMMARY ({track}):")
        print(f"  Laps completed:        {laps_done}")
        print(f"  Frames driven:         {frame}  ({elapsed:.0f}s)")
        print(f"  Mean |true cte|:       {mean_abs_cte:.3f}  (lower = better centered)")
        print(f"  Time off-road:         {off_road_pct:.1f}%  (|cte|>{ROAD_EDGE})")
        print(f"  Log saved ->           {log_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
