"""
donkey_manual_test.py
Manual keyboard-drive test for the Donkey / SDSandbox simulator.

PURPOSE: confirm the full pipeline works BEFORE any model is involved —
that we can connect to the running sim, receive camera frames, send
[steer, throttle] actions, and read the cte / speed / hit telemetry.

PREREQ: the simulator must ALREADY be running with a track loaded
(select a track from the SDSandbox menu first, e.g. generated_track or
mountain_track). This script connects to it over port 9091; it does NOT
launch the sim (exe_path="remote").

CONTROLS (focus the small OpenCV "camera" window, not the sim window):
    a / d : steer left / right
    w / s : throttle up / down
    space : re-center steering
    r     : reset the episode (car back to start)
    q     : quit

Every frame is logged to donkey_manual_log.csv (time, steer, throttle, cte,
speed, hit) so even this manual drive produces usable telemetry and confirms
cte is being received.

Install first:  pip install gymnasium gym_donkeycar opencv-python numpy
"""

import csv
import time
import argparse
import numpy as np
import cv2
import gymnasium as gym
import gym_donkeycar  # noqa: F401  (registers the donkey-* envs on import)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="donkey-generated-track-v0",
                    help="gym env id; must match the track loaded in the sim")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 127.0.0.1 (IPv4) not 'localhost' — on Windows "
                         "localhost may resolve to IPv6 ::1, which the sim isn't on")
    ap.add_argument("--log", default="donkey_manual_log.csv")
    args = ap.parse_args()

    # exe_path="remote" => attach to the ALREADY-RUNNING sim, don't launch one.
    # host forced to IPv4 so we don't hit the ::1 IPv6 "connection refused" trap.
    conf = {"exe_path": "remote", "host": args.host, "port": args.port}
    print(f"Connecting to sim on {args.host}:{args.port} as env '{args.env}' ...")
    env = gym.make(args.env, conf=conf)

    obs, info = env.reset()
    print("Connected. Telemetry keys available:", list(info.keys()))

    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["time_s", "steer", "throttle", "cte", "speed", "hit"])
    t0 = time.time()

    steer, throttle = 0.0, 0.0
    print("Driving. Focus the 'camera' window and use W/A/S/D, space, r, q.")

    try:
        while True:
            # obs is the camera frame (RGB). Show it (convert to BGR for OpenCV).
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("camera", frame)

            k = cv2.waitKey(20) & 0xFF
            if k == ord('a'):
                steer = max(-1.0, steer - 0.1)
            elif k == ord('d'):
                steer = min(1.0, steer + 0.1)
            elif k == ord('w'):
                throttle = min(1.0, throttle + 0.05)
            elif k == ord('s'):
                throttle = max(0.0, throttle - 0.05)
            elif k == ord(' '):
                steer = 0.0
            elif k == ord('r'):
                obs, info = env.reset()
                steer, throttle = 0.0, 0.0
                print("Episode reset.")
                continue
            elif k == ord('q'):
                break

            obs, reward, terminated, truncated, info = env.step(
                np.array([steer, throttle], dtype=np.float32))

            cte = info.get("cte", float("nan"))
            speed = info.get("speed", float("nan"))
            hit = info.get("hit", "?")
            writer.writerow([f"{time.time() - t0:.3f}", f"{steer:.3f}",
                             f"{throttle:.3f}", f"{cte:.4f}", f"{speed:.4f}", hit])
            log_file.flush()
            print(f"cte={cte:+.2f}  speed={speed:5.2f}  hit={hit}  "
                  f"steer={steer:+.2f}  thr={throttle:.2f}", end="\r")

            # In this sim, exceeding max_cte (~5.0) or a crash ends the episode.
            if terminated or truncated:
                print("\nEpisode ended (off-track or crash). Pressing r resets, q quits.")
                # keep the window alive so you can read the final state / reset
    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        env.close()
        print(f"\nSaved telemetry -> {args.log}")


if __name__ == "__main__":
    main()
