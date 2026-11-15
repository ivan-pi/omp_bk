#!/usr/bin/env python3
"""bk3_slm_pad.py -- strides of the local boxes of the BK3 sf kernel
(sycl/bk3_sycl.cpp, openmp/offload/bk3_offload.cpp) that keep its access
patterns free of SLM bank conflicts.

A box element (e, a, b, c) sits at e * se + a * sa + b * sb + c.  Each step
of the kernel has work-item t = (e, x, y) access one fixed index and vary
two (copy in/out: (a, b, c) < nm, flattened); within a message of L lanes
(8 for SIMD8, 16 for SIMD16) the cost is the largest number of distinct
addresses that fall in one of 16 banks of 4 bytes.  Prints, per nq, the
current layout (sb = nq or nq + 1 for even nq) and the cheapest strides
(ties: least memory), with the summed conflict degree over all steps.

usage:  scripts/study/bk3_slm_pad.py [nq ...]     (default 3..10)
"""
import itertools
import sys

BANKS = 16


def patterns(nq, nm):
    """(name, lane extents, address of (e, x, y) as (a, b, c) with a fixed index)"""
    return [
        ("copy", None, None),
        ("1/9 (i,j)", (nm, nm), lambda x, y: (x, y, 1)),
        ("2/8 (i,r)", (nm, nq), lambda x, y: (x, 1, y)),
        ("3/C/7 (q,r)", (nq, nq), lambda x, y: (1, x, y)),
        ("A/E (p,r)", (nq, nq), lambda x, y: (x, 1, y)),
        ("B/F (p,q)", (nq, nq), lambda x, y: (x, y, 1)),
    ]


def degree(addrs):
    banks = {}
    for a in set(addrs):
        banks.setdefault(a % BANKS, set()).add(a)
    return max(len(v) for v in banks.values())


def cost(nq, se, sa, sb, lanes, batch=4):
    nm = nq - 1
    total = 0
    for name, ext, f in patterns(nq, nm):
        if ext is None:   # coalesced copy of nm^3 values per element
            pts = [(e, i, j, k) for e in range(batch) for i in range(nm) for j in range(nm) for k in range(nm)]
        else:
            pts = [(e,) + f(x, y) for e in range(batch) for x in range(ext[0]) for y in range(ext[1])]
        addrs = [e * se + a * sa + b * sb + c for e, a, b, c in pts]
        for g in range(0, len(addrs), lanes):
            total += degree(addrs[g:g + lanes])
    return total


def main():
    nqs = [int(a) for a in sys.argv[1:]] or list(range(3, 11))
    print(" nq  lanes  current (sb,sa,se) cost  ->  best (sb,sa,se) cost   memory/element")
    for nq in nqs:
        pc = nq + 1 if nq % 2 == 0 else nq
        cur = (pc, nq * pc, nq * nq * pc)
        for lanes in (8, 16):
            c0 = cost(nq, cur[2], cur[1], cur[0], lanes)
            best = None
            for sb in range(nq, nq + 9):
                for sa in range(nq * sb, nq * sb + 17):
                    for se in range(nq * sa, nq * sa + 17):
                        c = cost(nq, se, sa, sb, lanes)
                        key = (c, se)
                        if best is None or key < best[0]:
                            best = (key, (sb, sa, se))
            (c1, se), strides = best
            print(f"{nq:3d}  {lanes:5d}  {str(cur):18s} {c0:5d}  ->  {str(strides):18s} {c1:5d}   {cur[2]} -> {se}")


if __name__ == "__main__":
    main()
