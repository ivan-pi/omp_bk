#!/usr/bin/env python3
"""advisor_table.py -- per-DoF work of GPU kernels from Advisor survey CSVs.

usage:  scripts/profile/advisor_table.py <survey.csv>:<dofs per call> ...

Each CSV is `advisor --report=survey --gpu --show-all-columns --format=csv`
of a project from advisor_gpu.sh; dofs per call is the problem size of the
run (every kernel of a run is called the same number of times, warm-up
included).  Prints, per kernel, the instrumented counts divided by DoFs:
floating-point and integer operations and SLM bytes read and written, and
the integer share of all operations.  Not reported: Advisor's kernel times
(rounded to ms), the sampled instruction counts (missing for some kernels
on DG2), DRAM (GTI, left at 0 on DG2 by Advisor 2026.0) and L3 (only part
of the traffic counted).
"""

import csv
import re
import sys


def num(s):
    s = s.replace(",", "").rstrip("s%")
    try:
        return float(s)
    except ValueError:
        return 0.0


def short(name):
    name = re.sub(r"\(anonymous namespace\)::|\$omp\$offloading:\d+|float, |\(int\)|\(bool\)", "", name)
    name = re.sub(r"_ZTSN12_GLOBAL__N_1\d*", "", name)
    return name[:58]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(f"{'kernel':58s} {'flop':>6s} {'intop':>6s} {'int %':>6s} {'SLM rd':>6s} {'SLM wr':>6s}   (per DoF; SLM in bytes)")
    for arg in sys.argv[1:]:
        path, dofs = arg.rsplit(":", 1)
        dofs = float(dofs)
        lines = open(path).read().splitlines()
        start = [k for k, line in enumerate(lines) if line.startswith('"ID"')][0]
        rows = [r for r in csv.reader(lines[start:]) if r]
        head = rows[0]
        idx = {}
        for k, h in enumerate(head):
            idx.setdefault(h, k)
        for r in rows[1:]:
            if len(r) < len(head) or "Outside" in r[1] or "zeCommand" in r[1] or "Symbol" in r[1]:
                continue
            calls = num(r[idx["Calls"]])
            if calls == 0:
                continue
            n = calls * dofs
            fl = 1e9 * num(r[idx["GFLOP"]]) / n
            it = 1e9 * num(r[idx["GINTOP"]]) / n
            print(f"{short(r[idx['Kernel']]):58s} {fl:6.1f} {it:6.1f} {100 * it / (fl + it):5.0f}% "
                  f"{1e9 * num(r[idx['SLM Read, GB']]) / n:6.1f} {1e9 * num(r[idx['SLM Write, GB']]) / n:6.1f}")


if __name__ == "__main__":
    main()
