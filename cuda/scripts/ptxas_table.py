#!/usr/bin/env python3
"""Tabulate a ptxas resource report (nvcc -Xptxas -v, or nvc++ -gpu=ptxinfo).

Reads the compiler output from the files given or from stdin and prints one
line per kernel: name, arithmetic type (nvc++'s double/float kernels), nq,
batch, registers, stack frame, spill stores, spill loads.  The kernel names
of this repository are recognised in both nvcc's mangling (_ZN2bk14MassOperatorFF
ILi10EEE...) and nvc++'s (nvkernel_..._SumFactorizationFFPlaneLoopILi10ELi1E...);
anything else is printed with its mangled name.

    make -B BK1_ff_cuda_plain NVCCEXTRA='-Xptxas -v' 2>&1 | cuda/scripts/ptxas_table.py
    nvc++ ... -gpu=ptxinfo -c BK1_ff.cpp 2>&1 | cuda/scripts/ptxas_table.py --csv regs.csv
"""

import argparse
import csv
import re
import sys

NAME = re.compile(
    r'(MassOperatorFF2|MassOperatorFF|SumFactorizationFFPlaneLoop|SumFactorizationFFPlane'
    r'|SumFactorizationFFTeams|SumFactorizationFF|SumFactorization)'
    r'I(?:([df]))?Li(\d+)E(?:Li(\d+)E)?')
FUNCTION = re.compile(r"(?:Compiling entry function '|Function properties for )(\S+?)'?\s*$")
PROPERTIES = re.compile(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads')
REGISTERS = re.compile(r'Used (\d+) registers')


def describe(mangled):
    """(name, type, nq, batch) of a kernel of this repository, else (mangled, '', '', '')."""
    m = NAME.search(mangled)
    if m is None:
        return mangled, '', '', ''
    name, kind, nq, batch = m.groups()
    kind = {'d': 'double', 'f': 'float'}.get(kind, '')
    return name, kind, int(nq), (int(batch) if batch else '')


def parse(lines):
    """The entry functions of a ptxas log in order of their 'Used N registers' lines."""
    rows = []
    current = None
    props = {}
    for line in lines:
        m = FUNCTION.search(line)
        if m:
            current = m.group(1)
            continue
        m = PROPERTIES.search(line)
        if m and current:
            props[current] = tuple(int(x) for x in m.groups())
            continue
        m = REGISTERS.search(line)
        if m and current:
            stack, stores, loads = props.get(current, ('', '', ''))
            name, kind, nq, batch = describe(current)
            rows.append({'kernel': name, 'type': kind, 'nq': nq, 'batch': batch,
                         'registers': int(m.group(1)), 'stack': stack,
                         'spill_stores': stores, 'spill_loads': loads, 'mangled': current})
            current = None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='compiler logs (default: stdin)')
    ap.add_argument('--csv', metavar='FILE', help='also write the table as CSV')
    ap.add_argument('--all', action='store_true', help='include kernels not recognised by name')
    args = ap.parse_args()

    lines = []
    if args.files:
        for f in args.files:
            with open(f) as fh:
                lines.extend(fh.readlines())
    else:
        lines = sys.stdin.readlines()

    rows = parse(lines)
    if not args.all:
        rows = [r for r in rows if r['nq'] != '']
    rows.sort(key=lambda r: (r['kernel'], r['type'], r['nq'] if r['nq'] != '' else 0, r['batch'] if r['batch'] != '' else 0))

    print(f"{'kernel':<28} {'type':<6} {'nq':>3} {'batch':>5} {'regs':>5} {'stack':>6} {'spill_st':>8} {'spill_ld':>8}")
    for r in rows:
        print(f"{r['kernel']:<28} {r['type']:<6} {r['nq']:>3} {r['batch']!s:>5} {r['registers']:>5} "
              f"{r['stack']!s:>6} {r['spill_stores']!s:>8} {r['spill_loads']!s:>8}")
    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['kernel', 'type', 'nq', 'batch', 'registers', 'stack',
                                               'spill_stores', 'spill_loads', 'mangled'])
            w.writeheader()
            w.writerows(rows)
    if not rows:
        print('no ptxas lines found (nvcc: -Xptxas -v; nvc++: -gpu=ptxinfo)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
