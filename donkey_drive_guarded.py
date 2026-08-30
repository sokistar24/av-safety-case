"""
donkey_drive_guarded.py  (v2: bidirectional — handover AND hand-back)
Closed-loop driving with the MC-Dropout run-time guard.

Per frame: mean, std = MC-dropout(model, frame, n passes)
  MODEL mode:
    std <= thr -> CERTIFIED: model steers; fail counter resets
    std >  thr -> hold last certified steering; fail += 1
    fail == M  -> HANDOVER to fail-safe
  FALLBACK mode (after handover):
    --fallback oracle: ground-truth P-controller drives; --fallback stop: halt
    std is still monitored: after R consecutive PASSES (--handback R, default 30)
    control is HANDED BACK to the model (hysteresis: R >> 1 prevents chattering).
    --handback 0 disables hand-back (absorbing, matching the formal m2 model).

Guard calibration (threshold, M) loads from results_cte/guarded_alpha.json
(fitted in-distribution; never tuned on shifted data).

Run:
  home track : python donkey_drive_guarded.py --env donkey-generated-track-v0 --laps 3 --show
  shifted    : python donkey_drive_guarded.py --env donkey-generated-roads-v0 --show --fallback oracle
"""

import os
import csv
import json
import time
import argparse
import numpy as np
import cv2
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401

from cte_dataset import CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H, ROAD_EDGE
from cte_model import TaxiNetCTE, mc_dropout_predict


def preprocess(obs):
    img = obs[CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--track", default=None)
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--guard_json", default="results_cte/guarded_alpha.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--out_dir", default="results_cte")
    ap.add_argument("--kp", type=float, default=0.95)
    ap.add_argument("--fallback_kp", type=float, default=None,
                    help="proportional gain for the fail-safe controller "
                         "(default: same as --kp)")
    ap.add_argument("--fallback_kd", type=float, default=0.0,
                    help="derivative gain for the fail-safe controller. A P-only "
                         "fail-safe oscillates around the target even with perfect "
                         "state knowledge; damping it separates 'handover cannot "
                         "help' from 'this fail-safe is under-designed'.")
    ap.add_argument("--lane_offset", type=float, default=-0.3)
    ap.add_argument("--throttle", type=float, default=0.1)
    ap.add_argument("--n_mc", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--handback", type=int, default=30,
                    help="consecutive PASSES required to hand control back to the "
                         "model (0 = never hand back, absorbing like the m2 DTMC)")
    ap.add_argument("--fallback", choices=["oracle", "stop"], default="oracle")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    track = args.track or args.env.replace("donkey-", "").replace("-v0", "")
    with open(args.guard_json) as f:
        guard = json.load(f)
    thr = args.threshold if args.threshold is not None else float(guard["threshold"])
    M = args.M if args.M is not None else int(guard.get("M", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, IMG_H, IMG_W, device=device))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    fb_kp = args.fallback_kp if args.fallback_kp is not None else args.kp
    print(f"Guard: std <= {thr:.4f}, M={M} fails -> handover, "
          f"R={args.handback} passes -> hand-back, fallback={args.fallback}")
    print(f"Fail-safe controller: kp={fb_kp}, kd={args.fallback_kd}"
          if args.fallback == "oracle" else "Fail-safe: stop")

    conf = {"exe_path": "remote", "host": args.host, "port": args.port}
    print(f"Connecting to sim on {args.host}:{args.port} as '{args.env}' ...")
    env = gym.make(args.env, conf=conf)
    obs, info = env.reset()
    print(f"Connected. GUARDED driving on '{track}'.")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"guarded_drive_{track}_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["time_s", "cte_true", "cte_pred", "mc_std", "certified",
                     "fail_count", "pass_count", "mode", "steer", "throttle",
                     "speed", "lap", "hit"])

    handed_over = False
    fail_count = 0
    pass_count = 0
    last_cert_steer = 0.0
    prev_err = None
    n_handover = n_handback = 0
    first_handover = None
    stats = {"MODEL": {"n": 0, "cert": 0, "abscte": [], "off": 0},
             "FALLBACK": {"n": 0, "cert": 0, "abscte": [], "off": 0}}
    frame = 0
    start_lap = info.get("lap_count", 0)
    t0 = time.time()
    laps_done = 0

    try:
        while True:
            cte_true = float(info.get("cte", 0.0))
            speed = float(info.get("speed", 0.0))
            hit = info.get("hit", "none")
            lap = int(info.get("lap_count", 0)) - start_lap
            now = time.time() - t0

            x = preprocess(obs).to(device)
            mean, std = mc_dropout_predict(model, x, n_samples=args.n_mc)
            cte_pred = float(mean.item())
            u = float(std.item())
            passed = u <= thr

            # ---- guard state machine (bidirectional) ----
            if not handed_over:
                if passed:
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= M:
                        handed_over = True
                        n_handover += 1
                        pass_count = 0
                        if first_handover is None:
                            first_handover = (now, frame)
                        print(f"\n{'='*62}\nHANDOVER #{n_handover} at t={now:.1f}s "
                              f"frame={frame} ({M} consecutive fails, std>{thr:.3f}). "
                              f"Fail-safe: {args.fallback.upper()}\n{'='*62}")
            else:
                if args.handback > 0:
                    pass_count = pass_count + 1 if passed else 0
                    if pass_count >= args.handback:
                        handed_over = False
                        n_handback += 1
                        fail_count = 0
                        # seed continuity: keep current fallback steering as last certified
                        print(f"\n{'-'*62}\nHAND-BACK #{n_handback} at t={now:.1f}s "
                              f"frame={frame} ({args.handback} consecutive passes). "
                              f"Model resumes control.\n{'-'*62}")

            # ---- action by mode ----
            if not handed_over:
                mode = "MODEL"
                if passed:
                    steer = float(np.clip(-args.kp * (cte_pred - args.lane_offset), -1, 1))
                    last_cert_steer = steer
                else:
                    steer = last_cert_steer
                throttle = args.throttle
            else:
                mode = "FALLBACK"
                if args.fallback == "oracle":
                    err = cte_true - args.lane_offset
                    derr = (err - prev_err) if prev_err is not None else 0.0
                    steer = float(np.clip(-(fb_kp * err + args.fallback_kd * derr), -1, 1))
                    last_cert_steer = steer
                    throttle = args.throttle
                else:
                    steer, throttle = 0.0, 0.0
            prev_err = cte_true - args.lane_offset

            s = stats[mode]
            s["n"] += 1
            s["cert"] += int(passed)
            s["abscte"].append(abs(cte_true))
            s["off"] += int(abs(cte_true) > ROAD_EDGE)

            writer.writerow([f"{now:.3f}", f"{cte_true:.5f}", f"{cte_pred:.5f}",
                             f"{u:.5f}", int(passed), fail_count, pass_count, mode,
                             f"{steer:.4f}", f"{throttle:.3f}", f"{speed:.4f}", lap, hit])
            log_file.flush()

            if args.show:
                disp = cv2.cvtColor(cv2.resize(obs, (320, 240)), cv2.COLOR_RGB2BGR)
                if handed_over:
                    color = (0, 0, 255)
                    status = f"HANDED OVER -> {args.fallback} (r={pass_count}/{args.handback})"
                elif fail_count > 0:
                    color = (0, 165, 255)
                    status = f"CHECK FAILING {fail_count}/{M}"
                else:
                    color = (0, 200, 0)
                    status = "CERTIFIED (model driving)"
                cv2.rectangle(disp, (0, 0), (319, 239), color, 3)
                cv2.putText(disp, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.putText(disp, f"std {u:.3f} thr {thr:.3f}", (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(disp, f"pred {cte_pred:+.2f} true {cte_true:+.2f}", (8, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow("guarded drive", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if frame % 30 == 0:
                print(f"t={now:5.1f}s lap={lap} mode={mode:8s} std={u:.3f} "
                      f"i={fail_count}/{M} r={pass_count}/{args.handback} "
                      f"pred={cte_pred:+.2f} true={cte_true:+.2f}", end="\r")

            obs, reward, terminated, truncated, info = env.step(
                np.array([steer, throttle], dtype=np.float32))
            frame += 1

            if lap >= args.laps:
                laps_done = lap
                print(f"\nCompleted {args.laps} laps.")
                break
            if frame >= args.max_frames:
                laps_done = lap
                break
            if handed_over and args.fallback == "stop" and args.handback == 0 \
                    and (now - first_handover[0]) > 8:
                print("\nStopped safely post-handover; ending run.")
                break
            if terminated or truncated:
                laps_done = lap
                print(f"\nEpisode ended (cte_true={cte_true:+.2f}, hit={hit}).")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        env.close()
        print("=" * 62)
        print(f"GUARDED DRIVING SUMMARY ({track}, fallback={args.fallback}, "
              f"R={args.handback}):")
        print(f"  Handovers: {n_handover}   Hand-backs: {n_handback}"
              + (f"   first handover t={first_handover[0]:.1f}s frame={first_handover[1]}"
                 if first_handover else "   (guard never fired)"))
        for mname, s in stats.items():
            if s["n"]:
                print(f"  {mname:8s}: {s['n']:5d} frames, {s['cert']/s['n']*100:5.1f}% certified, "
                      f"mean|cte| {np.mean(s['abscte']):.2f}, off-road {s['off']/s['n']*100:.1f}%")
        print(f"  Laps completed: {laps_done}")
        print(f"  Log -> {log_path}")
        print("=" * 62)


if __name__ == "__main__":
    main()
