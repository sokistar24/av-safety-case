"""
donkey_drive_de.py
Live closed-loop driving with the Deep-Ensemble run-time guard. The ensemble
counterpart of donkey_drive_guarded.py, with the same guard state machine, the same
fail-safe, and the same log schema, so DE and MC runs go into the same Table 8.

Per frame the K members are evaluated deterministically (dropout OFF):
    yhat_DE = mean over members      -> steers, when certified
    u_DE    = sd over members, ddof=1 -> guard score
    pass iff u_DE <= tau_DE
  MODEL mode:    pass -> model steers, fail counter resets
                 fail -> hold last certified steering, fail += 1
                 fail == M -> HANDOVER to the fail-safe
  FALLBACK mode: PD controller on ground-truth cte (idealised, as in the paper)
                 --handback R > 0 returns control after R consecutive passes;
                 default 0 = absorbing, matching the verified m2 DTMC.

DIFFERENCES FROM donkey_drive_guarded.py, all deliberate:
  --handback defaults to 0, not 30. The verified model has an absorbing handover
      state; a non-zero default silently drives a different system from the one
      that was model-checked.
  fail-safe defaults to the paper's final PD (Kp = 5, Kd = 5), not the superseded
      proportional-only fallback.
  log filenames carry a timestamp, so repeated runs of the same condition no
      longer overwrite each other. `light` re-randomises illumination per episode
      and needs repeats to support any claim about it.
  --unguarded runs the ensemble with no guard at all. This is the direct test of
      the DE m1 prediction (under light, m1 ~ 1.7e-04: the vehicle should stay on
      the road unguarded). Do not skip it - it is the cleanest falsification of
      the abstraction available.
  the summary reports |cte| AT HANDOVER and the frame of first off-road, which are
      the quantities Table 8 actually compares (MC arm: 7.9 m at M=10 vs 0.3 m at
      M=3). Previously these had to be recovered from the log by hand.

CONDITION SELECTION. trees, light and cones are simulator toggles on the SAME env
(donkey-generated-track-v0), so the gym side cannot tell them apart. --cond is what
names the run: it picks the env, routes the log to results_de/<cond>/live/, and
prints the sim settings to confirm before connecting. Set the simulator first.

Run (from av/, simulator already listening and set to the condition):
  light, unguarded (the m1 test):
    python donkey_drive_de.py --cond cond1_light --laps 3 --unguarded --show
  light, guarded, verified policy:
    python donkey_drive_de.py --cond cond1_light --laps 3 --show
  detection-latency policy check:
    python donkey_drive_de.py --cond cond1_light --laps 3 --M 3 --show
  severe shift:
    python donkey_drive_de.py --cond generated_road --max_frames 6000 --show

Repeat each condition several times; `light` especially, because randomLight
re-randomises per episode and the condition is not reproducible run to run.
Use --tag rep2, rep3 ... to keep repeats apart at a glance.
"""

import os
import csv
import json
import glob
import time
import argparse
import numpy as np
import cv2
import torch
import gymnasium as gym
import gym_donkeycar  # noqa: F401  # noqa

from cte_dataset import CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H, ROAD_EDGE
from cte_model import TaxiNetCTE


# The condition is a SIMULATOR SETTING, not an env name: trees, light and cones all
# run on donkey-generated-track-v0 and are indistinguishable from the gym side. The
# --cond flag is therefore what identifies the run, routes the log to the matching
# results_de/<cond>/ folder, and drives the pre-run checklist below. Settings are
# taken from conditions.md; the seed matters because it fixes the road geometry.
CONDITIONS = {
    "baseline_run":   {"env": "donkey-generated-track-v0",  "seed": "20432814",
                       "toggles": {"trees": "off", "light": "off", "cones": "off"}},
    "cond1_trees":    {"env": "donkey-generated-track-v0",  "seed": "20432814",
                       "toggles": {"trees": "ON",  "light": "off", "cones": "off"}},
    "cond1_light":    {"env": "donkey-generated-track-v0",  "seed": "20432814",
                       "toggles": {"trees": "off", "light": "ON",  "cones": "off"}},
    "cond1_cones":    {"env": "donkey-generated-track-v0",  "seed": "20432814",
                       "toggles": {"trees": "off", "light": "off", "cones": "ON"}},
    "generated_road": {"env": "donkey-generated-roads-v0",  "seed": None,
                       "toggles": {"trees": "off", "light": "off", "cones": "off"}},
    "mini_monaco":    {"env": "donkey-minimonaco-track-v0", "seed": None,
                       "toggles": {"trees": "off", "light": "off", "cones": "off"}},
}


def confirm_condition(cond, env, skip):
    """Print the simulator settings this run claims to be, and pause."""
    spec = CONDITIONS[cond]
    print("\n" + "=" * 62)
    print(f"CONDITION: {cond}")
    print(f"  env    : {env}")
    if spec["seed"]:
        print(f"  seed   : {spec['seed']}   (road geometry; must match the dataset)")
    tog = "  ".join(f"{k}={v}" for k, v in spec["toggles"].items())
    print(f"  sim    : {tog}")
    print("  These are set in the simulator UI and CANNOT be verified from here.")
    print("  A mismatch produces a correctly-named log containing the wrong condition.")
    print("=" * 62)
    if not skip:
        try:
            input("Press Enter when the simulator is set as above (Ctrl-C to abort)... ")
        except EOFError:
            pass


def render_panel(obs, status, color, l1, l2, w=480, h=360):
    """Annotated panel, identical in layout to the MC script's."""
    disp = cv2.cvtColor(cv2.resize(obs, (w, h)), cv2.COLOR_RGB2BGR)
    sc = w / 320.0
    cv2.rectangle(disp, (0, 0), (w - 1, h - 1), color, int(round(3 * sc)))
    cv2.putText(disp, status, (int(8 * sc), int(20 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, color, max(1, int(round(2 * sc))))
    cv2.putText(disp, l1, (int(8 * sc), int(40 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (255, 255, 255), max(1, int(round(sc))))
    cv2.putText(disp, l2, (int(8 * sc), int(58 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (255, 255, 255), max(1, int(round(sc))))
    return disp


def preprocess(obs):
    img = obs[CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def load_members(ckpt_dir, K, device):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "member_*.pth")))
    if not paths:
        raise SystemExit(f"No member_*.pth in {ckpt_dir}. Run train_ensemble.py first.")
    if K is not None:
        paths = paths[:K]
    models = []
    for p in paths:
        m = TaxiNetCTE().to(device)
        with torch.no_grad():
            m(torch.zeros(2, 3, IMG_H, IMG_W, device=device))
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()                       # deterministic: dropout OFF
        models.append(m)
    return models


@torch.no_grad()
def ensemble_step(models, x):
    """One frame -> (mean, sd, per-member predictions)."""
    preds = np.array([float(m(x).item()) for m in models])
    return float(preds.mean()), float(preds.std(ddof=1)), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS),
                    help="which evaluation condition the simulator is set to. This "
                         "identifies the run and routes output to results_de/<cond>/live/")
    ap.add_argument("--env", default=None,
                    help="gym env id (default: the one this condition uses)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the simulator-settings confirmation prompt")
    ap.add_argument("--ckpt_dir", default="results_de")
    ap.add_argument("--guard_json", default="results_de/de_calibration.json")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--out_dir", default=None,
                    help="default: results_de/<cond>/live/")
    ap.add_argument("--tag", default="",
                    help="extra label for the log filename, e.g. 'rep2'")
    ap.add_argument("--kp", type=float, default=0.95,
                    help="proportional gain of the perception-based controller")
    ap.add_argument("--fallback_kp", type=float, default=5.0,
                    help="fail-safe PD proportional gain (paper: 5)")
    ap.add_argument("--fallback_kd", type=float, default=5.0,
                    help="fail-safe PD derivative gain (paper: 5)")
    ap.add_argument("--lane_offset", type=float, default=-0.3)
    ap.add_argument("--throttle", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override tau_DE (default: frozen value from --guard_json)")
    ap.add_argument("--M", type=int, default=10,
                    help="consecutive failures before handover (verified: 10)")
    ap.add_argument("--handback", type=int, default=0,
                    help="consecutive passes to hand control back; 0 = absorbing, "
                         "matching the verified m2 DTMC")
    ap.add_argument("--fallback", choices=["oracle", "stop"], default="oracle")
    ap.add_argument("--unguarded", action="store_true",
                    help="disable the guard entirely: the ensemble drives throughout. "
                         "This is the live test of the m1 prediction.")
    ap.add_argument("--log_members", action="store_true",
                    help="log every member's prediction per frame (wider csv)")
    ap.add_argument("--snap_dir", default=None,
                    help="save annotated PNG panels at the first certified / failing / "
                         "handover frame (appendix figures)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    spec = CONDITIONS[args.cond]
    env_id = args.env or spec["env"]
    track = args.cond
    out_dir = args.out_dir or os.path.join("results_de", args.cond, "live")
    if not os.path.isdir(os.path.join("results_de", args.cond)):
        print(f"NOTE: results_de/{args.cond}/ does not exist yet - the offline "
              f"evaluation for this condition has not been run.")

    thr = args.threshold
    if thr is None:
        if not os.path.exists(args.guard_json):
            raise SystemExit(f"{args.guard_json} not found; pass --threshold.")
        with open(args.guard_json) as f:
            calib = json.load(f)
        thr = calib.get("threshold")
        if thr is None:
            raise SystemExit("No 'threshold' in the calibration json; pass --threshold.")
        if calib.get("premise_holds_at_all_percentiles") and \
           str(calib.get("selection_mode", "")).startswith("selection rule"):
            print("WARNING: this threshold came from the degenerate selection rule. "
                  "Re-run de_uncertainty.py --pct before using it live.")
    thr = float(thr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_members(args.ckpt_dir, args.K, device)
    K = len(models)
    print(f"Device: {device}   ensemble K = {K}")
    if args.unguarded:
        print("UNGUARDED run: the ensemble drives every frame, the guard never acts. "
              "u_DE is still logged for comparison.")
    else:
        print(f"Guard: u_DE <= {thr:.4f}, M = {args.M} -> handover, "
              f"hand-back R = {args.handback}"
              f"{' (absorbing)' if args.handback == 0 else ''}, "
              f"fallback = {args.fallback}")
        if args.fallback == "oracle":
            print(f"Fail-safe PD on ground-truth cte: Kp = {args.fallback_kp}, "
                  f"Kd = {args.fallback_kd}")

    confirm_condition(args.cond, env_id, args.yes)

    conf = {"exe_path": "remote", "host": args.host, "port": args.port}
    print(f"Connecting to sim on {args.host}:{args.port} as '{env_id}' ...")
    env = gym.make(env_id, conf=conf)
    obs, info = env.reset()
    print(f"Connected. Driving '{track}'.")

    os.makedirs(out_dir, exist_ok=True)
    if args.snap_dir:
        os.makedirs(args.snap_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode_tag = "unguarded" if args.unguarded else f"M{args.M}"
    if not args.unguarded and args.handback > 0:
        mode_tag += f"_hb{args.handback}"
    parts = [p for p in ["de", args.cond, mode_tag, args.tag, stamp] if p]
    log_path = os.path.join(out_dir, "_".join(parts) + ".csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    header = ["time_s", "frame", "cte_true", "cte_pred", "u_de", "certified",
              "fail_count", "pass_count", "mode", "steer", "throttle",
              "speed", "lap", "hit"]
    if args.log_members:
        header += [f"m{k}" for k in range(K)]
    writer.writerow(header)

    handed_over = False
    fail_count = pass_count = 0
    last_cert_steer = 0.0
    prev_err = None
    n_handover = n_handback = 0
    first_handover = None          # (t, frame, |cte|)
    cf_fail = 0                    # counterfactual consecutive failures (unguarded runs)
    cf_trigger = None              # (t, frame, |cte|) when M consecutive would be hit
    first_offroad = None           # (t, frame, mode)
    snapped = set()
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
            cte_pred, u, member_preds = ensemble_step(models, x)
            passed = u <= thr        # always the guard verdict, even when --unguarded:
                                     # in an unguarded run this is the counterfactual
                                     # ("would the guard have rejected this frame?")

            # ---- counterfactual guard trace (unguarded runs) ----
            if args.unguarded:
                cf_fail = 0 if passed else cf_fail + 1
                if cf_fail >= args.M and cf_trigger is None:
                    cf_trigger = (now, frame, abs(cte_true))
                    print(f"\n[counterfactual] the guard would have handed over here: "
                          f"t={now:.1f}s frame={frame} |cte|={abs(cte_true):.2f} m")

            # ---- guard state machine ----
            if not args.unguarded:
                if not handed_over:
                    if passed:
                        fail_count = 0
                    else:
                        fail_count += 1
                        if fail_count >= args.M:
                            handed_over = True
                            n_handover += 1
                            pass_count = 0
                            if first_handover is None:
                                first_handover = (now, frame, abs(cte_true))
                            print(f"\n{'='*62}\nHANDOVER #{n_handover} at t={now:.1f}s "
                                  f"frame={frame}  |cte|={abs(cte_true):.2f} m "
                                  f"({args.M} consecutive fails, u>{thr:.3f})."
                                  f"\n{'='*62}")
                else:
                    if args.handback > 0:
                        pass_count = pass_count + 1 if passed else 0
                        if pass_count >= args.handback:
                            handed_over = False
                            n_handback += 1
                            fail_count = 0
                            print(f"\n{'-'*62}\nHAND-BACK #{n_handback} at t={now:.1f}s "
                                  f"frame={frame}\n{'-'*62}")

            # ---- action ----
            if not handed_over:
                mode = "MODEL"
                if passed or args.unguarded:
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
                    steer = float(np.clip(
                        -(args.fallback_kp * err + args.fallback_kd * derr), -1, 1))
                    last_cert_steer = steer
                    throttle = args.throttle
                else:
                    steer, throttle = 0.0, 0.0
            prev_err = cte_true - args.lane_offset

            off = abs(cte_true) > ROAD_EDGE
            if off and first_offroad is None:
                first_offroad = (now, frame, mode)
                print(f"\n*** OFF ROAD at t={now:.1f}s frame={frame} "
                      f"cte={cte_true:+.2f} while {mode} held control ***")

            s = stats[mode]
            s["n"] += 1
            s["cert"] += int(passed)
            s["abscte"].append(abs(cte_true))
            s["off"] += int(off)

            row = [f"{now:.3f}", frame, f"{cte_true:.5f}", f"{cte_pred:.5f}",
                   f"{u:.5f}", int(passed), fail_count, pass_count, mode,
                   f"{steer:.4f}", f"{throttle:.3f}", f"{speed:.4f}", lap, hit]
            if args.log_members:
                row += [f"{p:.5f}" for p in member_preds]
            writer.writerow(row)
            log_file.flush()

            # ---- status string, shared by the window and the snapshots ----
            if args.unguarded:
                color, status = (200, 200, 0), "UNGUARDED (ensemble driving)"
                event = "unguarded"
            elif handed_over:
                hb = f" (r={pass_count}/{args.handback})" if args.handback > 0 else ""
                color, status = (0, 0, 255), f"HANDED OVER -> {args.fallback}{hb}"
                event = "handover"
            elif fail_count > 0:
                color, status = (0, 165, 255), f"CHECK FAILING {fail_count}/{args.M}"
                event = "failing"
            else:
                color, status = (0, 200, 0), "CERTIFIED (ensemble driving)"
                event = "certified"
            l1 = f"u_DE {u:.3f} thr {thr:.3f}"
            l2 = f"pred {cte_pred:+.2f} true {cte_true:+.2f}"

            if args.snap_dir and event not in snapped:
                snapped.add(event)
                panel = render_panel(obs, status, color, l1, l2)
                pth = os.path.join(args.snap_dir,
                                   f"de_{args.cond}_{mode_tag}_{event}.png")
                cv2.imwrite(pth, panel)
                print(f"\n[panel] {event} -> {pth}")

            if args.show:
                cv2.imshow("DE guarded drive",
                           render_panel(obs, status, color, l1, l2, 320, 240))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if frame % 30 == 0:
                print(f"t={now:5.1f}s lap={lap} mode={mode:8s} u={u:.3f} "
                      f"i={fail_count}/{args.M} pred={cte_pred:+.2f} "
                      f"true={cte_true:+.2f}", end="\r")

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

        summary = {
            "condition": args.cond, "env": env_id, "sim_settings": spec,
            "estimator": "deep_ensemble", "K": K,
            "unguarded": args.unguarded, "threshold": thr, "M": args.M,
            "handback": args.handback, "fallback": args.fallback,
            "fallback_kp": args.fallback_kp, "fallback_kd": args.fallback_kd,
            "frames": frame, "laps_completed": laps_done,
            "handovers": n_handover, "handbacks": n_handback,
            "first_handover": ({"t_s": first_handover[0], "frame": first_handover[1],
                                "abs_cte": first_handover[2]} if first_handover else None),
            "first_offroad": ({"t_s": first_offroad[0], "frame": first_offroad[1],
                               "mode": first_offroad[2]} if first_offroad else None),
            "counterfactual_handover": ({"t_s": cf_trigger[0], "frame": cf_trigger[1],
                                         "abs_cte": cf_trigger[2]} if cf_trigger else None),
            "log": log_path, "timestamp": stamp,
        }
        for mname, s in stats.items():
            summary[mname] = {
                "frames": s["n"],
                "certified_pct": (s["cert"] / s["n"] * 100) if s["n"] else None,
                "mean_abs_cte": float(np.mean(s["abscte"])) if s["n"] else None,
                "offroad_pct": (s["off"] / s["n"] * 100) if s["n"] else None,
            }
        sp = log_path.replace(".csv", "_summary.json")
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)

        print("=" * 62)
        print(f"DE DRIVING SUMMARY ({track}, "
              f"{'UNGUARDED' if args.unguarded else f'M={args.M}'}, "
              f"handback={args.handback}):")
        if first_handover:
            print(f"  Handovers: {n_handover}   hand-backs: {n_handback}")
            print(f"  First handover: t={first_handover[0]:.1f}s  "
                  f"frame={first_handover[1]}  |cte|={first_handover[2]:.2f} m")
        elif not args.unguarded:
            print("  Guard never fired.")
        if first_offroad:
            print(f"  First off-road: t={first_offroad[0]:.1f}s  "
                  f"frame={first_offroad[1]}  under {first_offroad[2]} control")
        else:
            print("  Never left the road.")
        if args.unguarded:
            if cf_trigger:
                print(f"  Counterfactual: the guard would have fired at frame "
                      f"{cf_trigger[1]} (|cte|={cf_trigger[2]:.2f} m)"
                      + (f", {first_offroad[1] - cf_trigger[1]} frames "
                         f"{'before' if cf_trigger[1] < first_offroad[1] else 'after'} "
                         f"the first off-road frame" if first_offroad else ""))
            else:
                print(f"  Counterfactual: the guard would NOT have reached {args.M} "
                      f"consecutive failures in this run.")
        for mname, s in stats.items():
            if s["n"]:
                print(f"  {mname:8s}: {s['n']:5d} frames, "
                      f"{s['cert']/s['n']*100:5.1f}% "
                      f"{'would-be certified' if args.unguarded else 'certified'}, "
                      f"mean|cte| {np.mean(s['abscte']):.2f}, "
                      f"off-road {s['off']/s['n']*100:.1f}%")
        print(f"  Laps completed: {laps_done}")
        print(f"  Log     -> {log_path}")
        print(f"  Summary -> {sp}")
        if args.snap_dir and snapped:
            print(f"  Panels  -> {args.snap_dir} ({', '.join(sorted(snapped))})")
        print("=" * 62)


if __name__ == "__main__":
    main()
