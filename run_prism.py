"""
run_prism.py
Sweep the horizon N on the generated lane-keeping DTMC, invoking PRISM with the
Java incantation confirmed working on this machine in the previous (tinysafe)
project:

    java -Djava.library.path=C:\\prism-4.10.1\\lib -cp "C:\\prism-4.10.1\\lib\\*"
         prism.PrismCL <model> <props> -const N=<n> -explicit

-Djava.library.path points Java at PRISM's native DLLs; -explicit selects the
pure-Java engine that needs no DLLs at all (our DTMC is tiny, so it's plenty).
Together they are the belt-and-braces combo that made PRISM run on Windows before.

Parses PRISM's "Result: <p>" per N and writes results_cte/prism_results.csv —
the data for the paper's Fig.3-style curve: P[F cte=-1] vs horizon N.

Run (after make_prism_model.py has generated the model):
    python run_prism.py
Options:
    --ns 5,10,...     horizons to check
    --java <path>     full path to java.exe if 'java' isn't on PATH in this env
    --lib  <path>     PRISM lib folder (default C:\\prism-4.10.1\\lib)
    --no-explicit     use PRISM's default engines instead of -explicit
"""

import argparse
import csv
import os
import re
import subprocess

DEFAULT_LIB = r"C:\prism-4.10.1\lib"
RESULT_RE = re.compile(r"Result:\s*([0-9.eE+\-]+)")


def parse_result(output):
    m = RESULT_RE.search(output)
    return float(m.group(1)) if m else None


def run_one(java, lib, model, props, n, explicit=True, timeout=180):
    cmd = [java, f"-Djava.library.path={lib}", "-cp", os.path.join(lib, "*"),
           "prism.PrismCL", model, props, "-const", f"N={n}"]
    if explicit:
        cmd.append("-explicit")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    val = parse_result(out)
    if val is None:
        raise RuntimeError(
            f"No 'Result:' in PRISM output for N={n}. Last output:\n{out[-2000:]}")
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"results_cte/lanekeep.prism")
    ap.add_argument("--props", default=r"results_cte/lanekeep.props")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--java", default="java")
    ap.add_argument("--ns", default="5,10,15,20,25,30,35,40,45,50")
    ap.add_argument("--no-explicit", action="store_true")
    ap.add_argument("--out", default=r"results_cte/prism_results.csv")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    print(f"PRISM lib: {args.lib}")
    print(f"Model: {args.model}  Props: {args.props}")
    print(f"{'N':>4s}  {'P[F cte=-1]':>14s}")
    rows = []
    for n in ns:
        val = run_one(args.java, args.lib, args.model, args.props, n,
                      explicit=not args.no_explicit)
        rows.append((n, val))
        print(f"{n:>4d}  {val:>14.6g}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "p_offroad"])
        w.writerows(rows)
    print(f"\nSaved curve -> {args.out}")


if __name__ == "__main__":
    main()
