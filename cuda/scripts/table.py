#!/usr/bin/env python3
"""Pivot the drivers' CSV (BK_CSV) into GDoF/s by kernel and nq.

    cuda/scripts/table.py sweep.csv            # best GDoF/s per kernel and nq
    cuda/scripts/table.py sweep.csv gbs        # another column: gbs, error
    cuda/scripts/table.py sweep.csv gdofs min  # worst run instead of the best

A kernel is a (driver, kernel, batch) triple, so "ff-loop" and "ff-loop x14"
are separate lines.  With repeated runs the last line gives the largest
run-to-run spread of GDoF/s of any kernel and order, in percent.
"""

import csv
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    column = sys.argv[2] if len(sys.argv) > 2 else 'gdofs'
    pick = min if (len(sys.argv) > 3 and sys.argv[3] == 'min') else max
    with open(sys.argv[1], newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print('no rows')
        return 1
    values = {}      # (driver, kernel, batch) -> {nq: [values]}
    for r in rows:
        if r[column] == '':
            continue
        key = (r['driver'], r['kernel'], r['batch'])
        values.setdefault(key, {}).setdefault(int(r['nq']), []).append(float(r[column]))
    nqs = sorted({nq for per in values.values() for nq in per})
    print(f"{column + ' by nq':<26}" + ''.join(f'{nq:>9}' for nq in nqs))
    for key in sorted(values):
        driver, kernel, batch = key
        label = f'{driver}:{kernel}'
        per = values[key]
        cells = []
        for nq in nqs:
            if nq in per:
                v = pick(per[nq])
                cells.append(f'{v:>9.3g}' if column == 'error' else f'{v:>9.2f}')
            else:
                cells.append(' ' * 9)
        print(f'{label:<26}' + ''.join(cells))
    spread = 0.0
    for per in values.values():
        for vals in per.values():
            if len(vals) > 1 and max(vals) > 0:
                spread = max(spread, 100.0 * (max(vals) - min(vals)) / max(vals))
    if spread > 0:
        print(f'largest run-to-run spread of {column}: {spread:.1f}%')
    failed = sum(1 for r in rows if r['ok'] != 'ok')
    print(f'{len(rows)} rows, {failed} failed checks')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
