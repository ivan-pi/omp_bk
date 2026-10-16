#!/usr/bin/env python3
"""Profile the BK1 float-float kernels with Nsight Compute and print the numbers
that matter for an issue-bound kernel, side by side per order.

For every order and kernel the script runs ncu on the driver with an explicit
metric list (duration, SM and memory throughput, issue-slot utilisation,
eligible and active warps per scheduler, achieved occupancy, registers and
the occupancy limits, executed instructions, pipe utilisation of fma, fp64,
alu and lsu, shared-memory load and store instructions and bank conflicts,
and the warp stall reasons in cycles per issued instruction), skipping the
driver's warm-up launches, and prints one table per nq with a column per
kernel.  The metric list is checked against ncu --query-metrics first and
metrics that ncu does not list are left out; --drop leaves out more.  The
CSV that ncu prints is read with the csv module, nothing else is parsed.

    cuda/scripts/ncu_stats.py --orders 7 8                      # MassOperatorFF and MassOperatorFF2, nq = 9, 10
    cuda/scripts/ncu_stats.py --orders 8 --kernels MassOperatorFF2 --csv ncu.csv
    cuda/scripts/ncu_stats.py --binary ./BK1_ff --kernels regex:PlaneLoopILi10ELi1E --skip 0 --orders 8

--kernels takes ncu kernel names (the function name without template
arguments, exact match) or regex:... patterns, which is what the nvc++
kernels of BK1_ff need (regex:PlaneLoop matches the OpenMP loop kernel).
ncu locks the clocks to base by default, so its durations are not the
drivers' timings; compare its numbers with each other.
"""

import argparse
import csv
import io
import os
import subprocess
import sys

STALLS = ['barrier', 'long_scoreboard', 'short_scoreboard', 'mio_throttle', 'math_pipe_throttle',
          'wait', 'not_selected', 'selected', 'no_instruction', 'lg_throttle', 'dispatch_stall',
          'imc_miss', 'misc', 'drain', 'membar', 'branch_resolving', 'sleeping', 'tex_throttle']

# (metric, short label); the stall metrics are appended below
METRICS = [
    ('gpu__time_duration.sum', 'duration'),
    ('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'SM throughput %'),
    ('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'memory throughput %'),
    ('sm__inst_issued.avg.pct_of_peak_sustained_active', 'issue slots busy %'),
    ('smsp__issue_active.avg.per_cycle_active', 'issue active / cycle / sched'),
    ('smsp__warps_eligible.avg.per_cycle_active', 'eligible warps / sched'),
    ('smsp__warps_active.avg.per_cycle_active', 'active warps / sched'),
    ('sm__warps_active.avg.pct_of_peak_sustained_active', 'achieved occupancy %'),
    ('launch__registers_per_thread', 'registers / thread'),
    ('launch__occupancy_limit_registers', 'block limit: registers'),
    ('launch__occupancy_limit_shared_mem', 'block limit: shared mem'),
    ('launch__occupancy_limit_warps', 'block limit: warps'),
    ('launch__waves_per_multiprocessor', 'waves / SM'),
    ('smsp__inst_executed.sum', 'instructions executed'),
    ('smsp__inst_executed.avg.per_cycle_active', 'IPC / sched'),
    ('sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active', 'pipe fma %'),
    ('sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active', 'pipe fp64 %'),
    ('sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active', 'pipe alu %'),
    ('sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active', 'pipe lsu %'),
    ('smsp__sass_inst_executed_op_shared_ld.sum', 'shared loads'),
    ('smsp__sass_inst_executed_op_shared_st.sum', 'shared stores'),
    ('smsp__sass_inst_executed_op_global_ld.sum', 'global loads'),
    ('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 'shared bank conflicts'),
    ('smsp__average_warp_latency_per_inst_issued.ratio', 'warp cycles / issued inst'),
]
METRICS += [(f'smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio', f'stall {s}') for s in STALLS]


def known_metrics(ncu):
    """Base names ncu --query-metrics lists (first word of each line with a
    '__' in it), or None when the query fails."""
    try:
        out = subprocess.run([ncu, '--query-metrics'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, check=False).stdout
    except OSError as e:
        print(f'cannot run {ncu}: {e}', file=sys.stderr)
        return None
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if parts and '__' in parts[0]:
            names.add(parts[0])
    return names or None


def parse_csv(text):
    """{metric name: (value, unit)} from ncu --csv output for one kernel launch.
    ncu prints its CSV with a header line that starts with "ID"; lines before
    it (==PROF== messages) are skipped."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith('ID,')), None)
    if start is None:
        return {}
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    values = {}
    for row in reader:
        name = row.get('Metric Name')
        if not name:
            continue
        raw = (row.get('Metric Value') or '').replace(',', '')
        try:
            value = float(raw)
        except ValueError:
            value = raw
        values[name] = (value, row.get('Metric Unit') or '')
    return values


def profile(args, kernel, p, metrics):
    """Run ncu for one kernel and order; {metric: (value, unit)}, empty on failure."""
    cmd = [args.ncu, '--csv', '--kernel-name', kernel, '--launch-skip', str(args.skip),
           '--launch-count', '1', '--metrics', ','.join(metrics),
           args.binary, str(p), str(args.nelmt), str(args.ntests)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f'ncu failed for {kernel} p={p} (exit {proc.returncode}):', file=sys.stderr)
        print((proc.stderr + proc.stdout).strip()[-2000:], file=sys.stderr)
        print('a metric this ncu rejects can be left out with --drop NAME', file=sys.stderr)
        return {}
    if args.raw:
        os.makedirs(args.raw, exist_ok=True)
        with open(os.path.join(args.raw, f'{kernel.replace(":", "_")}_p{p}.csv'), 'w') as fh:
            fh.write(proc.stdout)
    return parse_csv(proc.stdout)


def fmt(value, unit):
    if isinstance(value, str):
        return value
    if unit in ('%',):
        return f'{value:.1f}'
    if abs(value) >= 1e6:
        return f'{value / 1e6:.3f}M'
    if abs(value) >= 1000:
        return f'{value:,.0f}'
    return f'{value:.2f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binary', default='./BK1_ff_cuda_plain')
    ap.add_argument('--kernels', nargs='+', default=['MassOperatorFF', 'MassOperatorFF2'],
                    help='ncu kernel names or regex:... (default: both CUDA kernels)')
    ap.add_argument('--orders', nargs='+', type=int, default=[7, 8], help='orders p (default 7 8)')
    ap.add_argument('--nelmt', type=int, default=200000)
    ap.add_argument('--ntests', type=int, default=1)
    ap.add_argument('--skip', type=int, default=3, help='launches of the kernel to skip (the warm-up; default 3)')
    ap.add_argument('--ncu', default='ncu')
    ap.add_argument('--csv', metavar='FILE', help='write every metric to this CSV')
    ap.add_argument('--raw', metavar='DIR', help='keep the ncu CSV of every run in this directory')
    ap.add_argument('--top-stalls', type=int, default=6, help='stall reasons to print per column (default 6)')
    ap.add_argument('--drop', nargs='*', default=[], metavar='METRIC', help='metrics to leave out')
    args = ap.parse_args()

    known = known_metrics(args.ncu)
    metrics = [m for m, _ in METRICS if m not in args.drop]
    if known is not None:
        metrics = [m for m in metrics if m.split('.')[0] in known]
    dropped = sorted({m for m, _ in METRICS} - set(metrics))
    if dropped:
        print('left out: ' + ', '.join(dropped), file=sys.stderr)
    labels = dict(METRICS)

    results = {}   # (p, kernel) -> {metric: (value, unit)}
    for p in args.orders:
        for kernel in args.kernels:
            results[(p, kernel)] = profile(args, kernel, p, metrics)

    rows_out = []
    for p in args.orders:
        nq = p + 2
        cols = [k for k in args.kernels if results.get((p, k))]
        if not cols:
            print(f'p={p} nq={nq}: no results')
            continue
        width = max(len(k) for k in cols) + 2
        print(f'\n== p={p} nq={nq} ' + ' '.join(f'{k:>{width}}' for k in cols))
        for metric, label in METRICS:
            if label.startswith('stall'):
                continue
            if not any(metric in results[(p, k)] for k in cols):
                continue
            unit = next((results[(p, k)][metric][1] for k in cols if metric in results[(p, k)]), '')
            line = f'{label:<30}'
            for k in cols:
                v = results[(p, k)].get(metric)
                line += f'{fmt(*v) if v else "-":>{width}}'
            print(line + (f'  [{unit}]' if unit else ''))
        # the stall reasons, largest first by the first column
        first = results[(p, cols[0])]
        stalls = [(m, l) for m, l in METRICS if l.startswith('stall') and m in first]
        stalls.sort(key=lambda ml: -first[ml[0]][0] if isinstance(first[ml[0]][0], float) else 0.0)
        for metric, label in stalls[:args.top_stalls]:
            line = f'{label:<30}'
            for k in cols:
                v = results[(p, k)].get(metric)
                line += f'{fmt(*v) if v else "-":>{width}}'
            print(line + '  [cycles / issue]')
        for k in cols:
            for metric, (value, unit) in results[(p, k)].items():
                rows_out.append({'p': p, 'nq': nq, 'kernel': k, 'metric': metric,
                                 'label': labels.get(metric, ''), 'value': value, 'unit': unit})
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['p', 'nq', 'kernel', 'metric', 'label', 'value', 'unit'])
            w.writeheader()
            w.writerows(rows_out)
    return 0 if rows_out else 1


if __name__ == '__main__':
    sys.exit(main())
