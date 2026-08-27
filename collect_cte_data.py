"""
collect_cte_data.py  (v2: appends across runs, multiple collection modes)

Autonomous data collector for CTE-prediction training.

KEY CHANGE FROM v1: this APPENDS across reruns instead of overwriting. Run it as
many times as you like on the same track and it keeps accumulating frames (image
numbering continues from the last run; CSV is appended; header written only once).
So you can build a large dataset over several sessions, and a crash/interrupt never
loses previously collected data.

Collection modes (--mode):
  autopilot_lane : drive the lane naturally (P-control toward the car's LANE
                   position, not centerline), with occasional SYMMETRIC excursions
                   to both sides so all cte bins fill. Realistic "mostly correct
                   driving + some recovery". DEFAULT.
  sweep          : deliberately sweep the steering offset back and forth so the car
                   crosses the full cte band evenly. Less realistic, best coverage.
  manual         : you drive with W/A/S/D (needs the camera window focused); logs
                   image + true cte. Human driving data.

Per frame it saves:
  - image -> data_cte/<track>/IMG/frame_XXXXXX.jpg   (model INPUT)
  - row   -> data_cte/<track>/cte_log.csv            (image, cte, speed, steer, lap, track, mode)

Run (load the matching track in the sim first):
  python collect_cte_data.py --track generated_track --env donkey-generated-track-v0 --laps 10
  # run again to add MORE data to the same track:
  python collect_cte_data.py --track generated_track --env donkey-generated-track-v0 --laps 10
"""

import os
import csv
import glob
import time
import argparse
import numpy as np
import cv2
import gymnasium as gym
import gym_donkeycar  # noqa: F401


def next_frame_index(img_dir):
    """Continue numbering after the highest existing frame_XXXXXX.jpg (append-safe)."""
    existing = glob.glob(os.path.join(img_dir, "frame_*.jpg"))
    if not existing:
        return 0
    nums = []
    for p in existing:
        base = os.path.basename(p)
        try:
            nums.append(int(base.replace("frame_", "").replace(".jpg", "")))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--mode", default="autopilot_lane",
                    choices=["autopilot_lane", "sweep", "manual"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=30000)
    ap.add_argument("--out_root", default="data_cte")
    # autopilot params
    ap.add_argument("--kp", type=float, default=0.95, help="P-gain toward lane pos")
    ap.add_argument("--target_speed", type=float, default=0.1)
    ap.add_argument("--lane_offset", type=float, default=-0.3,
                    help="the car's natural lane cte (Framing B: drive the lane, "
                         "not the centerline). Autopilot steers toward THIS, not 0.")
    # excursion / sweep params
    ap.add_argument("--excursion_every", type=int, default=180,
                    help="autopilot_lane: push off-lane every N frames")
    ap.add_argument("--excursion_mag", type=float, default=0.0)
    ap.add_argument("--sweep_period", type=int, default=200,
                    help="sweep mode: frames per full left-right sweep cycle")
    ap.add_argument("--sweep_mag", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.track)
    img_dir = os.path.join(out_dir, "IMG")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cte_log.csv")

    # APPEND-SAFE: continue frame numbering, append to CSV, header only if new file.
    start_index = next_frame_index(img_dir)
    file_is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    log_file = open(csv_path, "a", newline="")
    writer = csv.writer(log_file)
    if file_is_new:
        writer.writerow(["image", "cte", "speed", "steer_cmd", "lap", "track", "mode"])
    print(f"Appending: image numbering starts at frame_{start_index:06d} "
          f"(existing frames kept).")

    if args.mode == "manual":
        print("MANUAL mode: focus the 'camera' window; W/A/S/D to drive, q to stop.")

    conf = {"exe_path": "remote", "host": args.host, "port": args.port}
    print(f"Connecting to sim on {args.host}:{args.port} as '{args.env}' ...")
    env = gym.make(args.env, conf=conf)
    obs, info = env.reset()
    print("Connected. Track:", args.track, "| Mode:", args.mode)

    rng = np.random.default_rng()
    idx = start_index
    n_this_run = 0
    excursion = 0.0
    m_steer, m_throttle = 0.0, 0.0
    start_lap = info.get("lap_count", 0)
    t0 = time.time()

    try:
        while True:
            cte = float(info.get("cte", 0.0))
            speed = float(info.get("speed", 0.0))
            lap = int(info.get("lap_count", 0)) - start_lap

            # ---- choose action by mode ----
            if args.mode == "autopilot_lane":
                # steer toward the LANE position (cte = lane_offset), not centerline
                if n_this_run % args.excursion_every == 0:
                    # symmetric push: alternate/ random both directions to fill bins
                    excursion = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * args.excursion_mag
                steer = -args.kp * (cte - args.lane_offset) + excursion
                throttle = args.target_speed
            elif args.mode == "sweep":
                # deterministic sinusoidal sweep across the band for even coverage
                phase = 2 * np.pi * (n_this_run % args.sweep_period) / args.sweep_period
                steer = -args.kp * (cte - args.lane_offset) + args.sweep_mag * np.sin(phase)
                throttle = args.target_speed
            else:  # manual
                steer, throttle = m_steer, m_throttle

            steer = float(np.clip(steer, -1.0, 1.0))

            # ---- save frame ----
            img_name = f"frame_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_name),
                        cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            writer.writerow([os.path.join("IMG", img_name), f"{cte:.5f}",
                             f"{speed:.4f}", f"{steer:.4f}", lap, args.track, args.mode])
            log_file.flush()
            idx += 1
            n_this_run += 1

            # ---- show camera (needed for manual key capture; harmless otherwise) ----
            frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imshow("camera", cv2.resize(frame_bgr, (320, 240)))
            k = cv2.waitKey(1) & 0xFF
            if args.mode == "manual":
                if k == ord('a'): m_steer = max(-1.0, m_steer - 0.1)
                elif k == ord('d'): m_steer = min(1.0, m_steer + 0.1)
                elif k == ord('w'): m_throttle = min(1.0, m_throttle + 0.05)
                elif k == ord('s'): m_throttle = max(0.0, m_throttle - 0.05)
                elif k == ord(' '): m_steer = 0.0
                elif k == ord('q'): break
            else:
                if k == ord('q'): break

            if n_this_run % 50 == 0:
                print(f"run_frames={n_this_run:5d}  total_idx={idx:5d}  lap={lap}  "
                      f"cte={cte:+.3f}  speed={speed:5.2f}  steer={steer:+.3f}", end="\r")

            obs, reward, terminated, truncated, info = env.step(
                np.array([steer, throttle], dtype=np.float32))

            if lap >= args.laps:
                print(f"\nReached {args.laps} laps this run. Stopping.")
                break
            if n_this_run >= args.max_frames:
                print(f"\nHit max_frames ({args.max_frames}). Stopping.")
                break
            if terminated or truncated:
                print(f"\nEpisode ended (cte={cte:+.2f}, hit={info.get('hit')}). Resetting.")
                obs, info = env.reset()
                start_lap = info.get("lap_count", 0) - lap
                excursion = 0.0
    except KeyboardInterrupt:
        print("\nInterrupted — data so far is saved.")
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        env.close()
        # report cumulative total in the CSV
        total = 0
        try:
            with open(csv_path) as f:
                total = sum(1 for _ in f) - 1  # minus header
        except Exception:
            pass
        print(f"\nThis run added {n_this_run} frames in {time.time()-t0:.0f}s.")
        print(f"CUMULATIVE dataset size for {args.track}: {total} frames.")
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
