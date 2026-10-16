#!/usr/bin/env python3
"""Profile the BK1 float-float kernels with Nsight Compute and print the numbers
that matter for an issue-bound kernel, side by side per order.

For every order and kernel the script has ncu write a report (.ncu-rep, kept
for ncu-ui) with a fixed metric list, skipping the driver's warm-up launches,
then reads the reports through Nsight Compute's Python Report Interface,
the ncu_report module: installed with `pip install ncu-report`, or taken
from extras/python of the Nsight Compute installation (found next to the ncu
binary, or given with --python-path or NCU_PYTHON_PATH), and prints one
table per nq with a column per kernel.  Nothing is parsed:
every number is action.metric_by_name(name).value() of the report.

    cuda/scripts/ncu_stats.py --orders 7 8                       # MassOperatorFF and MassOperatorFF2 at nq = 9, 10
    cuda/scripts/ncu_stats.py --orders 8 --kernels MassOperatorFF2 --csv ncu.csv
    cuda/scripts/ncu_stats.py --binary ./BK1_ff --kernels regex:PlaneLoopILi10ELi1E --skip 0 --orders 8
    cuda/scripts/ncu_stats.py --load ncu_reports/*.ncu-rep       # tabulate existing reports (any metric set)

The metrics: duration, SM and memory throughput, issue-slot utilisation,
eligible and active warps per scheduler, achieved occupancy, registers and
the occupancy limits, executed instructions, pipe utilisation of fma, fp64,
alu and lsu, shared-memory load and store instructions and bank conflicts,
and the warp stall reasons in cycles per issued instruction.  The list is
checked against ncu --query-metrics first and names this ncu does not list
are left out (--drop leaves out more); with --load, whatever a report holds
is shown.  --kernels takes ncu kernel names (the function name without
template arguments) or regex:... patterns, which the nvc++ kernels of
BK1_ff need.  ncu locks the clocks to base by default, so its durations are
not the drivers' timings; compare its numbers with each other.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

STALLS = ['barrier', 'long_scoreboard', 'short_scoreboard', 'mio_throttle', 'math_pipe_throttle',
          'wait', 'not_selected', 'selected', 'no_instruction', 'lg_throttle', 'dispatch_stall',
          'imc_miss', 'misc', 'drain', 'membar', 'branch_resolving', 'sleeping', 'tex_throttle']

# (metric, label); the stall metrics are appended below
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
LABELS = dict(METRICS)


def import_ncu_report(ncu, python_path):
    """The ncu_report module: the installed package (pip install ncu-report)
    unless --python-path / NCU_PYTHON_PATH, extras/python next to the
    (resolved) ncu binary, NVHPC's profilers/Nsight_Compute (NVHPC_ROOT) or a
    nsight-compute-* directory of a toolkit (CUDA_HOME) holds one, which then
    comes first."""
    candidates = []
    if python_path:
        candidates.append(python_path)
    if os.environ.get('NCU_PYTHON_PATH'):
        candidates.append(os.environ['NCU_PYTHON_PATH'])
    exe = shutil.which(ncu)
    if exe:
        real = os.path.dirname(os.path.realpath(exe))
        candidates.append(os.path.join(real, 'extras', 'python'))
        candidates.extend(glob.glob(os.path.join(os.path.dirname(real), 'nsight-compute-*', 'extras', 'python')))
    for root in ('NVHPC_ROOT', 'NVHPC', 'CUDA_HOME', 'CUDA_PATH'):
        if os.environ.get(root):
            candidates.append(os.path.join(os.environ[root], 'profilers', 'Nsight_Compute', 'extras', 'python'))
            candidates.extend(glob.glob(os.path.join(os.environ[root], 'nsight-compute-*', 'extras', 'python')))
    candidates.extend(glob.glob('/opt/nvidia/nsight-compute/*/extras/python'))
    candidates.extend(glob.glob('/usr/local/cuda/nsight-compute-*/extras/python'))
    for path in candidates:
        if os.path.isfile(os.path.join(path, 'ncu_report.py')):
            sys.path.insert(0, path)
            break
    try:
        import ncu_report  # pylint: disable=import-outside-toplevel
        return ncu_report
    except ImportError as e:
        print(f'cannot import ncu_report ({e}); looked in: {", ".join(candidates) or "nothing"}', file=sys.stderr)
        print('pip install ncu-report, or give the extras/python directory of your Nsight Compute with '
              '--python-path or NCU_PYTHON_PATH', file=sys.stderr)
        return None


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


def collect(args, kernel, p, metrics):
    """Run ncu into a report file; returns its path or None."""
    os.makedirs(args.reports, exist_ok=True)
    stem = os.path.join(args.reports, f'{kernel.replace(":", "_")}_p{p}')
    cmd = [args.ncu, '-o', stem, '-f', '--kernel-name', kernel, '--launch-skip', str(args.skip),
           '--launch-count', '1', '--metrics', ','.join(metrics),
           args.binary, str(p), str(args.nelmt), str(args.ntests)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not os.path.isfile(stem + '.ncu-rep'):
        print(f'ncu failed for {kernel} p={p} (exit {proc.returncode}):', file=sys.stderr)
        print(proc.stdout.strip()[-2000:], file=sys.stderr)
        print('a metric this ncu rejects can be left out with --drop NAME', file=sys.stderr)
        return None
    return stem + '.ncu-rep'


def read_report(ncu_report, path):
    """[(kernel name, {metric: (value, unit)})] for every action of a report."""
    report = ncu_report.load_report(path)
    actions = []
    for r in range(report.num_ranges()):
        rng = report.range_by_idx(r)
        for a in range(rng.num_actions()):
            action = rng.action_by_idx(a)
            values = {}
            for name in action.metric_names():
                metric = action.metric_by_name(name)
                try:
                    unit = metric.unit()
                except Exception:   # older interfaces without unit()
                    unit = ''
                values[name] = (metric.value(), unit)
            actions.append((action.name(), values))
    return actions


def fmt(value, unit):
    if not isinstance(value, (int, float)):
        return str(value)
    if unit == '%':
        return f'{value:.1f}'
    if abs(value) >= 1e6:
        return f'{value / 1e6:.3f}M'
    if abs(value) >= 1000:
        return f'{value:,.0f}'
    return f'{value:.2f}'


def print_table(title, columns, top_stalls):
    """columns: [(column label, {metric: (value, unit)})]."""
    width = max(len(label) for label, _ in columns) + 2
    print(f'\n== {title} ' + ' '.join(f'{label:>{width}}' for label, _ in columns))
    first = columns[0][1]
    ordered = [(m, l) for m, l in METRICS if not l.startswith('stall')]
    stalls = [(m, l) for m, l in METRICS if l.startswith('stall') and m in first]
    stalls.sort(key=lambda ml: -float(first[ml[0]][0]) if isinstance(first[ml[0]][0], (int, float)) else 0.0)
    for metric, label in ordered + stalls[:top_stalls]:
        if not any(metric in values for _, values in columns):
            continue
        unit = next((values[metric][1] for _, values in columns if metric in values), '')
        line = f'{label:<30}'
        for _, values in columns:
            v = values.get(metric)
            line += f'{fmt(*v) if v else "-":>{width}}'
        print(line + (f'  [{unit}]' if unit else ''))


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
    ap.add_argument('--python-path', metavar='DIR', help='the extras/python directory holding ncu_report')
    ap.add_argument('--reports', default='ncu_reports', metavar='DIR', help='where the .ncu-rep files go')
    ap.add_argument('--load', nargs='*', metavar='REPORT', help='only tabulate these existing reports')
    ap.add_argument('--csv', metavar='FILE', help='write every metric to this CSV')
    ap.add_argument('--top-stalls', type=int, default=6, help='stall reasons to print per column (default 6)')
    ap.add_argument('--drop', nargs='*', default=[], metavar='METRIC', help='metrics to leave out')
    args = ap.parse_args()

    ncu_report = import_ncu_report(args.ncu, args.python_path)
    if ncu_report is None:
        return 1

    tables = []     # (title, [(column label, values)])
    if args.load:
        # one table, a column per report (its first action)
        columns = []
        for path in args.load:
            for _, values in read_report(ncu_report, path)[:1]:
                columns.append((os.path.basename(path).replace('.ncu-rep', ''), values))
        if columns:
            tables.append(('reports', columns))
    else:
        metrics = [m for m, _ in METRICS if m not in args.drop]
        known = known_metrics(args.ncu)
        if known is not None:
            # launch__ metrics are launch attributes, not listed by --query-metrics
            metrics = [m for m in metrics if m.startswith('launch__') or m.split('.')[0] in known]
        dropped = sorted({m for m, _ in METRICS} - set(metrics))
        if dropped:
            print('left out: ' + ', '.join(dropped), file=sys.stderr)
        for p in args.orders:
            columns = []
            for kernel in args.kernels:
                path = collect(args, kernel, p, metrics)
                if path:
                    for _, values in read_report(ncu_report, path)[:1]:
                        columns.append((kernel, values))
            if columns:
                tables.append((f'p={p} nq={p + 2}', columns))

    rows_out = []
    for title, columns in tables:
        print_table(title, columns, args.top_stalls)
        for label, values in columns:
            for metric, (value, unit) in values.items():
                rows_out.append({'table': title, 'kernel': label, 'metric': metric,
                                 'label': LABELS.get(metric, ''), 'value': value, 'unit': unit})
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['table', 'kernel', 'metric', 'label', 'value', 'unit'])
            w.writeheader()
            w.writerows(rows_out)
    return 0 if tables else 1


if __name__ == '__main__':
    sys.exit(main())
