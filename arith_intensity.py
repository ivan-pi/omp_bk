#!/usr/bin/env python3
"""Arithmetic intensity of the BK kernels -- closed-form model and an
experiment on whether the contraction order matters.

This mirrors, in a few lines of Python, what ``arith_intensity.cpp`` measures
by instrumenting the real C++ kernels. The two agree exactly (the C++ tool
asserts its measurements against these same formulas), so this file is the
convenient place to explore parameter ranges and answer "does the order of the
contractions change the flop count?".

Conventions (identical to the C++ tool):
  * nq quadrature points per direction; for BK1/BK3, nm = nq-1 modes, p = nq-2.
  * a*b counts as 2 flops (one multiply + one add); a hardware FMA is still 2.
  * bytes = streaming DRAM traffic per element = read `in` + write `out` +
    read the geometric factors. Basis matrices are cached/reused -> excluded;
    scratch is register/local memory -> excluded.
  * bytes use 4 (float), the type the kernels are instantiated with. Using
    double doubles the bytes and halves the intensity.
"""

from itertools import permutations

BYTES_PER_SCALAR = 4  # float


# --------------------------------------------------------------------------
# Closed-form flop counts (adds, muls) per element. A contraction step
# `acc = 0; for n<L: acc += a*b` over S outputs costs L*S adds and L*S muls.
# --------------------------------------------------------------------------
def bk1_counts(nq):
    m, q = nq - 1, nq
    shared = 2 * (q * m**3 + q**2 * m**2 + m * q**3)
    return shared, shared + q**3            # (adds, muls); +q^3 muls from JxW


def bk3_counts(nq):
    m, q = nq - 1, nq
    interp = 2 * (q * m**3 + q**2 * m**2 + m * q**3)
    return interp + 6 * q**4 + 6 * q**3, interp + 6 * q**4 + 9 * q**3


def bk5_counts(nq):
    q = nq
    return 6 * q**4 + 6 * q**3, 6 * q**4 + 9 * q**3


def bk1_bytes(nq):
    m, q = nq - 1, nq
    return BYTES_PER_SCALAR * (2 * m**3 + q**3)        # in + out + JxW


def bk3_bytes(nq):
    m, q = nq - 1, nq
    return BYTES_PER_SCALAR * (2 * m**3 + 6 * q**3)    # in + out + G(6)


def bk5_bytes(nq):
    q = nq
    return BYTES_PER_SCALAR * (2 * q**3 + 6 * q**3)    # in + out + G(6)


def bk1_dofs(nq):
    return (nq - 1) ** 3


bk3_dofs = bk1_dofs


def bk5_dofs(nq):
    return nq**3


# Order axis p = 1..16. BK1/BK3 over-integrate with nq = p+2; BK5 collocates
# at the p+1 GLL nodes, so nq = p+1.
KERNELS = {
    "BK1": (bk1_counts, bk1_bytes, bk1_dofs, range(3, 19), 2),  # nq = p+2
    "BK3": (bk3_counts, bk3_bytes, bk3_dofs, range(3, 19), 2),  # nq = p+2
    "BK5": (bk5_counts, bk5_bytes, bk5_dofs, range(2, 18), 1),  # nq = p+1
}


def print_tables():
    for name, (counts, nbytes, ndofs, nq_range, p_off) in KERNELS.items():
        print(f"\n{name}  (nq = p+{p_off})")
        print(f"{'p':>4}{'nq':>4}{'flops':>12}{'bytes':>9}"
              f"{'flop/DoF':>11}{'flop/byte':>12}{'byte/flop':>12}")
        for nq in nq_range:
            a, mul = counts(nq)
            flops = a + mul
            b = nbytes(nq)
            d = ndofs(nq)
            print(f"{nq - p_off:>4}{nq:>4}{flops:>12}{b:>9}"
                  f"{flops / d:>11.3f}{flops / b:>12.3f}{b / flops:>12.3f}")


# --------------------------------------------------------------------------
# Does the contraction order matter?
#
# The forward interpolation contracts one Cartesian direction at a time,
# turning that direction's `m` modes into `q` points. When direction d is
# contracted, the directions already contracted contribute a factor q each and
# those not yet contracted contribute m each; the reduction length is m[d]:
#
#     cost(step) = 2 * m[d] * q[d] * prod(q over done) * prod(m over pending)
#
# so the total over a permutation depends on the intermediate tensor sizes,
# which depend on the order -- unless every direction has the same (m, q).
# --------------------------------------------------------------------------
def forward_interp_flops(m, q, order):
    total, done = 0, set()
    for d in order:
        out = q[d]
        for a in range(3):
            if a == d:
                continue
            out *= q[a] if a in done else m[a]
        total += 2 * m[d] * out       # reduction length m[d], counted as 2 flops
        done.add(d)
    return total


def order_experiment():
    print("\nDoes the contraction order matter? (forward interpolation)\n"
          "Total flops for each of the 6 direction orderings:")
    cases = {
        "isotropic   m=(4,4,4) q=(5,5,5)": ([4, 4, 4], [5, 5, 5]),
        "anisotropic m=(2,4,6) q=(3,5,7)": ([2, 4, 6], [3, 5, 7]),
    }
    for label, (m, q) in cases.items():
        vals = {order: forward_interp_flops(m, q, order)
                for order in permutations(range(3))}
        lo, hi = min(vals.values()), max(vals.values())
        print(f"\n  {label}")
        for order, v in vals.items():
            print(f"    order {order} -> {v:>7} flops")
        if lo == hi:
            print(f"    => all identical ({lo}); order is irrelevant.")
        else:
            best = min(vals, key=vals.get)
            print(f"    => range {lo}..{hi} "
                  f"({100*(hi-lo)/lo:.1f}% spread); best order {best}.")


if __name__ == "__main__":
    print("Per-element arithmetic intensity of the BK kernels "
          f"(bytes: float = {BYTES_PER_SCALAR}; a*b = 2 flops)")
    print_tables()
    order_experiment()
