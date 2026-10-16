#!/usr/bin/env python3
"""Run the BK1 float-float drivers over a range of orders and tabulate them.

Runs ./BK1_ff_cuda_plain (both CUDA kernels, --cuda) and/or ./BK1_ff (the
OpenMP kernels, --omp) for each order, parses the rows they print, writes one
CSV line per kernel and run, and prints a table of GDoF/s by kernel and nq.
Every ok/FAIL line and the identical check of the CUDA driver are checked;
the exit status is 1 if any run failed.

    cuda/scripts/sweep.py --cuda ./BK1_ff_cuda_plain --omp ./BK1_ff --orders 2-8 --csv sweep.csv
    cuda/scripts/sweep.py --cuda ./BK1_ff_cuda_plain --repeat 2 --nelmt 1000000
    BK_RANDOM=1 cuda/scripts/sweep.py --omp ./BK1_ff --orders 1-8 --nelmt 4096 --ntests 1

CSV columns: driver, kernel, p, nq, nelmt, batch, run, gdofs, gbs, norm,
error, ok.  The OpenMP driver's "ff-plane x2" style labels give kernel
ff-plane with batch 2.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

HEADER = re.compile(r'nelmt = (\d+) nq = (\d+) data = (\w+)(?:.* batch = (\d+))?')
ROW = re.compile(r'^(\S.*?)\s+GDoF/s =\s*(\S+) GB/s =\s*(\S+) norm =\s*(\S+)')
OK = re.compile(r'^(ok|FAIL): (.*)')
ERRORS = re.compile(r'= ([0-9.eE+-]+|nan|inf)')
FLOAT_ERROR = re.compile(r'^float\s+rel\. error vs double = (\S+)')
IDENTICAL = re.compile(r'against float-float: (identical|.*)')
LABEL_BATCH = re.compile(r'^(.*?)\s+x(\d+)$')


def parse_orders(spec):
    orders = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            orders.extend(range(int(a), int(b) + 1))
        else:
            orders.append(int(part))
    return orders


def parse_run(text, driver, p, run):
    """The rows of one driver run as dicts, and whether every check passed."""
    nelmt = nq = ''
    batch = ''
    rows = []
    ok_values = []
    float_error = ''
    passed = True
    for line in text.splitlines():
        m = HEADER.search(line)
        if m and not rows:
            nelmt, nq = int(m.group(1)), int(m.group(2))
            batch = int(m.group(4)) if m.group(4) else ''
            continue
        m = ROW.match(line)
        if m:
            label, gdofs, gbs, norm = m.groups()
            label = label.strip()
            kernel, row_batch = label, batch
            mb = LABEL_BATCH.match(label)
            if mb:
                kernel, row_batch = mb.group(1), int(mb.group(2))
            elif driver == 'omp' and label in ('ff-teams', 'ff-plane', 'ff-loop'):
                row_batch = 1
            rows.append({'driver': driver, 'kernel': kernel, 'p': p, 'nq': nq, 'nelmt': nelmt,
                         'batch': row_batch, 'run': run, 'gdofs': float(gdofs), 'gbs': float(gbs),
                         'norm': norm, 'error': '', 'ok': ''})
            continue
        m = FLOAT_ERROR.match(line)
        if m:
            float_error = m.group(1)
            continue
        m = OK.match(line)
        if m:
            passed = passed and (m.group(1) == 'ok')
            ok_values.extend(ERRORS.findall(m.group(2).split('(tol')[0]))
            continue
        m = IDENTICAL.search(line)
        if m:
            passed = passed and (m.group(1) == 'identical')
    # errors: float's own line, then the ok values in row order for the rest
    checked = [r for r in rows if r['kernel'] not in ('double', 'float')]
    if len(ok_values) == len(checked):
        for r, e in zip(checked, ok_values):
            r['error'] = e
    for r in rows:
        if r['kernel'] == 'float':
            r['error'] = float_error
        r['ok'] = 'ok' if passed else 'FAIL'
    return rows, passed


def run_driver(binary, p, args, driver, run):
    cmd = [binary, str(p), str(args.nelmt), str(args.ntests)]
    if driver == 'cuda' and args.batch:
        cmd.append(str(args.batch))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if args.verbose:
        sys.stdout.write(proc.stdout)
    rows, passed = parse_run(proc.stdout, driver, p, run)
    if proc.returncode != 0:
        passed = False
    if not rows:
        print(f'{binary} {p}: no rows parsed (exit {proc.returncode})', file=sys.stderr)
        passed = False
    return rows, passed


def table(rows, value='gdofs'):
    """Best value per (driver, kernel, batch) and nq."""
    best = {}
    nqs = sorted({r['nq'] for r in rows})
    for r in rows:
        key = (r['driver'], r['kernel'], r['batch'])
        best.setdefault(key, {})
        best[key][r['nq']] = max(best[key].get(r['nq'], 0.0), r[value])
    width = 24
    print(f"{value + ' by nq':<{width}}" + ''.join(f'{nq:>8}' for nq in nqs))
    for key in sorted(best, key=lambda k: (k[0], k[1], str(k[2]))):
        label = f"{key[0]}:{key[1]}" + (f" x{key[2]}" if key[2] not in ('', 1) else '')
        print(f'{label:<{width}}' + ''.join(
            f"{best[key][nq]:>8.2f}" if nq in best[key] else f"{'':>8}" for nq in nqs))


def spread(rows):
    """Largest run-to-run spread of gdofs per (driver, kernel, batch, nq), in percent."""
    groups = {}
    for r in rows:
        groups.setdefault((r['driver'], r['kernel'], r['batch'], r['nq']), []).append(r['gdofs'])
    worst = 0.0
    for key, values in groups.items():
        if len(values) > 1 and max(values) > 0:
            s = 100.0 * (max(values) - min(values)) / max(values)
            if s > worst:
                worst = s
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cuda', metavar='BIN', help='the CUDA driver (BK1_ff_cuda_plain)')
    ap.add_argument('--omp', metavar='BIN', help='the OpenMP driver (BK1_ff)')
    ap.add_argument('--orders', default='2-8', help='orders p, e.g. 2-8 or 1,3,5 (default 2-8)')
    ap.add_argument('--nelmt', type=int, default=1000000)
    ap.add_argument('--ntests', type=int, default=5)
    ap.add_argument('--batch', type=int, default=0, help='elements per block for the CUDA driver (0: default)')
    ap.add_argument('--repeat', type=int, default=1, help='runs per order')
    ap.add_argument('--csv', metavar='FILE', help='write every row to this CSV')
    ap.add_argument('--verbose', action='store_true', help='echo the drivers\' output')
    args = ap.parse_args()
    if not args.cuda and not args.omp:
        ap.error('give --cuda and/or --omp')

    rows = []
    all_ok = True
    for p in parse_orders(args.orders):
        for run in range(1, args.repeat + 1):
            for driver, binary in (('cuda', args.cuda), ('omp', args.omp)):
                if binary:
                    r, ok = run_driver(binary, p, args, driver, run)
                    rows.extend(r)
                    all_ok = all_ok and ok
                    for row in r:
                        print(f"{driver} p={p} nq={row['nq']} {row['kernel']:<12} batch={row['batch']!s:<3} "
                              f"GDoF/s={row['gdofs']:8.3f} error={row['error']} {row['ok']}")
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['driver', 'kernel', 'p', 'nq', 'nelmt', 'batch', 'run',
                                               'gdofs', 'gbs', 'norm', 'error', 'ok'])
            w.writeheader()
            w.writerows(rows)
    print()
    if rows:
        table(rows)
        if args.repeat > 1:
            print(f'largest run-to-run spread: {spread(rows):.1f}%')
    print('random data' if os.environ.get('BK_RANDOM') else 'constant data', end=', ')
    print('all checks ok' if all_ok else 'FAIL: a check failed')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
