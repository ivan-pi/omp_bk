// scan_genx.cpp -- inclusive and exclusive prefix sums (scans) in CM.
//
// The serial definitions (as in Intel's "OpenMP SIMD for Inclusive/Exclusive
// Scans", which vectorizes them with `#pragma omp simd reduction(inscan,+:x)`):
//   inclusive:  x += a[i];  out[i] = x;     out = {a0, a0+a1, a0+a1+a2, ...}
//   exclusive:  out[i] = x;  x += a[i];     out = {0,  a0,    a0+a1,    ...}
//
// 1. scan16_demo: the in-register pattern on one vector (a[i] = i, 16 lanes).
//    Hillis-Steele log-step scan: log2(N) shifted adds on select regions;
//    exclusive = inclusive shifted right by one lane. Region SIZES are template
//    arguments, so the steps unroll through `if constexpr` recursion
//    (`#pragma unroll` does not make a loop counter a constant expression).
//
// 2. Full-array scan of n ints, "reduce-then-scan" (traffic: read n, read n,
//    write n):
//    scan_p1_reduce   one thread per SCAN_CHUNK elements: chunk sums
//    scan_p2_offsets  one thread: exclusive scan of the chunk sums (carry
//                     between 128-wide register blocks)
//    scan_p3_scan     one thread per chunk: carry = its offset; for each
//                     128-element block: load, in-register inclusive scan,
//                     add carry, store inclusive or exclusive, carry = last.
//    No tail masking: reads past the end of a stateful buffer return 0 (add
//    nothing) and writes past the end are dropped.
//
// copy_i32: read + write with the same access pattern -- the bandwidth
// reference kernel for scan_p3_scan.
#include <cm/cm.h>

#ifndef SCAN_CHUNK
#define SCAN_CHUNK 1024 // elements per thread in passes 1 and 3 (best of 1024..8192, measured on an Arc A770)
#endif
constexpr int CHUNK = SCAN_CHUNK;
constexpr int B = 128; // elements per register block (4 x 128-byte block reads)
static_assert(CHUNK % B == 0, "a chunk must be whole blocks");

// ---- in-register scans ----------------------------------------------------------

// One Hillis-Steele step per power of two S < N: v[i] += v[i - S] for i >= S.
template <int N, int S>
_GENX_ inline void scan_steps(vector_ref<int, N> v) {
  if constexpr (S < N) {
    vector<int, N> shifted = 0;
    shifted.template select<N - S, 1>(S) = v.template select<N - S, 1>(0);
    v += shifted;
    scan_steps<N, 2 * S>(v);
  }
}

template <int N>
_GENX_ inline vector<int, N> inclusive_scan(vector<int, N> v) {
  scan_steps<N, 1>(v);
  return v;
}

// Exclusive from inclusive: shift right by one lane, 0 in front.
template <int N>
_GENX_ inline vector<int, N> exclusive_from_inclusive(vector_ref<int, N> inc) {
  vector<int, N> exc = 0;
  exc.template select<N - 1, 1>(1) = inc.template select<N - 1, 1>(0);
  return exc;
}

// Block read/write of N ints (N a multiple of 32) at element `at`.
template <int N>
_GENX_ inline void read_ints(SurfaceIndex buf, uint at, vector_ref<int, N> v) {
#pragma unroll
  for (int r = 0; r < N / 32; ++r) {
    read(buf, (at + 32 * r) * sizeof(int), v.template select<32, 1>(32 * r));
  }
}

template <int N>
_GENX_ inline void write_ints(SurfaceIndex buf, uint at, vector_ref<int, N> v) {
#pragma unroll
  for (int r = 0; r < N / 32; ++r) {
    write(buf, (at + 32 * r) * sizeof(int), v.template select<32, 1>(32 * r));
  }
}

// ---- 1. the article's example, in registers ------------------------------------------

// OUT[0..16) = inclusive scan of 0..15, OUT[16..32) = exclusive scan.
extern "C" _GENX_MAIN_ void scan16_demo(SurfaceIndex OUT [[type("buffer_t")]]) {
  vector<int, 16> a;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    a(i) = i;
  }
  vector<int, 32> both;
  vector<int, 16> inc = inclusive_scan<16>(a);
  both.select<16, 1>(0) = inc;
  both.select<16, 1>(16) = exclusive_from_inclusive<16>(inc);
  write(OUT, 0, both.select<32, 1>(0));
}

// ---- 2. full-array scan -------------------------------------------------------------------

extern "C" _GENX_MAIN_ void scan_p1_reduce(SurfaceIndex A [[type("buffer_t")]],
                                           SurfaceIndex SUMS [[type("buffer_t")]]) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  vector<int, B> acc = 0;
#pragma unroll
  for (int b = 0; b < CHUNK / B; ++b) {
    vector<int, B> v;
    read_ints<B>(A, t * CHUNK + b * B, v);
    acc += v;
  }
  write(SUMS, t, 0u, cm_sum<int>(acc));
}

// Single thread: OFFS[t] = SUMS[0] + ... + SUMS[t-1] for t < nchunks.
// Latency-bound (each block needs the previous carry), so each round trip
// fetches P2_READ sums -- 16 block reads in flight -- before scanning them
// as 128-wide register blocks.
constexpr int P2_READ = 4 * B;
extern "C" _GENX_MAIN_ void scan_p2_offsets(SurfaceIndex SUMS [[type("buffer_t")]],
                                            SurfaceIndex OFFS [[type("buffer_t")]],
                                            uint nchunks) {
  int carry = 0;
  for (uint at = 0; at < nchunks; at += P2_READ) {
    vector<int, P2_READ> s;
    read_ints<P2_READ>(SUMS, at, s); // past nchunks: zeros (out of bounds) or ignored
#pragma unroll
    for (int k = 0; k < P2_READ / B; ++k) {
      vector<int, B> inc = inclusive_scan<B>(s.select<B, 1>(k * B));
      vector<int, B> exc = exclusive_from_inclusive<B>(inc);
      exc += carry;
      write_ints<B>(OFFS, at + k * B, exc);
      carry += inc(B - 1);
    }
  }
}

extern "C" _GENX_MAIN_ void scan_p3_scan(SurfaceIndex A [[type("buffer_t")]],
                                         SurfaceIndex OFFS [[type("buffer_t")]],
                                         SurfaceIndex OUT [[type("buffer_t")]],
                                         uint exclusive) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);

  // This chunk's starting offset (scalar gather of one element).
  vector<uint, 1> zero = 0;
  vector<int, 1> off;
  read(OFFS, t, zero, off);
  int carry = off(0);

#pragma unroll
  for (int b = 0; b < CHUNK / B; ++b) {
    uint at = t * CHUNK + b * B;
    vector<int, B> v;
    read_ints<B>(A, at, v);
    vector<int, B> inc = inclusive_scan<B>(v);
    vector<int, B> res;
    if (exclusive) {
      res = exclusive_from_inclusive<B>(inc);
    } else {
      res = inc;
    }
    res += carry;
    write_ints<B>(OUT, at, res);
    carry += inc(B - 1);
  }
}

// ---- bandwidth reference --------------------------------------------------------------------

extern "C" _GENX_MAIN_ void copy_i32(SurfaceIndex A [[type("buffer_t")]],
                                     SurfaceIndex OUT [[type("buffer_t")]]) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
#pragma unroll
  for (int b = 0; b < CHUNK / B; ++b) {
    vector<int, B> v;
    read_ints<B>(A, t * CHUNK + b * B, v);
    write_ints<B>(OUT, t * CHUNK + b * B, v);
  }
}

// ---- 3. single-pass decoupled look-back scan --------------------------------------
//
// One kernel, traffic = read n + write n (a copy), after Merrill & Garland,
// "Single-pass Parallel Prefix Scan with Decoupled Look-back" (2016).
//
//   ticket  : chunks are numbered in the order threads START (atomic counter),
//             so a chunk only ever waits on chunks whose threads are already
//             running -- no deadlock whatever the hardware scheduling order.
//   per chunk (LB_CHUNK ints kept in registers):
//     1. load, scan in registers (128-wide blocks with carry), total = last;
//     2. publish AGG[c] = total, then FLAG[c] = 1 (chunk 0 publishes its
//        inclusive prefix directly: INC[0], FLAG[0] = 2);
//     3. look back c-1, c-2, ...: FLAG 0 -> spin; FLAG 1 -> add AGG and
//        continue; FLAG 2 -> add INC and stop;
//     4. publish INC[c] = prefix + total, FLAG[c] = 2; write the output.
//   STATE layout (uints, zeroed by the host before every run):
//     [0] ticket counter, [1]/[2] error report (chunk+1 / awaited chunk+1 when
//     a spin exceeded LB_MAX_SPIN), then FLAG[nchunks], AGG[nchunks], INC[nchunks] at
//     element offsets 16, 16 + S, 16 + 2S (S = stride argument).
//
// Coherence: GPU L1 caches are not coherent between threads, so every flag
// and value access is an ATOMIC (they are performed at L3/memory): reads are
// `atomic OR 0`, writes `atomic XCHG`, and a fence orders "value before flag".
//
// RESULT measured on an Arc A770 (64M ints): correct, but 28.6 ms with LB_CHUNK=2048 and
// -double-GRF (90 ms with 512) vs 2.03 ms for reduce-then-scan above -- keep
// using reduce-then-scan here. Why: the legacy cm_fence(CM_GLOBAL_COHERENT_FENCE)
// becomes `fence.ugm.evict.gpu` + `fence.tgm.evict.gpu` on DG2 (GPU scope,
// L1 eviction, twice) per publish and per look-back window, and every step
// is an atomic round trip; with thousands of chunks in flight most
// predecessors only have an aggregate, so look-backs are long. A newer cmc
// with the LSC cm_fence<LSC_UGM, LSC_FENCE_OP_NONE, LSC_SCOPE_GPU> (no evict)
// might change that -- untested.
// Lessons kept in the code: (a) a separate `if (c == 0)` publish path with
// constant-offset atomics never became visible to chunk 1 while it polled
// (GPU hang); one uniform path fixed it (root cause not identified);
// (b) cm_fbh returns the count of LEADING ZEROS; (c) bound every spin
// (LB_MAX_SPIN) and report, so a bug cannot hang the GPU.
#ifndef LB_MAX_SPIN
#define LB_MAX_SPIN (1u << 22) // safety: give up (and report) instead of hanging the GPU
#endif
#ifndef LB_W
#define LB_W 16 // look-back window: predecessors examined per round trip
#endif
#ifndef LB_CHUNK
#define LB_CHUNK 512 // ints per chunk, held in registers (512 ints = 64 GRFs)
#endif
constexpr int LB = LB_CHUNK;
static_assert(LB_W >= 1 && LB_W <= 16, "window masks are built with 32-bit shifts: keep LB_W <= 16");
static_assert(LB % B == 0, "a look-back chunk must be whole blocks");

_GENX_ inline uint atomic_load(SurfaceIndex s, uint at) {
  vector<uint, 1> off = at;
  vector<uint, 1> zero = 0;
  vector<uint, 1> old;
  write_atomic<ATOMIC_OR, uint>(s, off, zero, old); // returns the current value
  return old(0);
}

_GENX_ inline void atomic_store(SurfaceIndex s, uint at, uint value) {
  vector<uint, 1> off = at;
  vector<uint, 1> v = value;
  vector<uint, 1> old;
  write_atomic<ATOMIC_XCHG, uint>(s, off, v, old);
}

extern "C" _GENX_MAIN_ void scan_lookback(SurfaceIndex A [[type("buffer_t")]],
                                          SurfaceIndex OUT [[type("buffer_t")]],
                                          SurfaceIndex STATE [[type("buffer_t")]],
                                          uint stride, uint exclusive) {
  // 0. Ticket: this thread's chunk number, in start order.
  vector<uint, 1> ctr = 0;
  vector<uint, 1> one = 1;
  vector<uint, 1> ticket;
  write_atomic<ATOMIC_ADD, uint>(STATE, ctr, one, ticket);
  uint c = ticket(0);
  const uint FLAG = 16, AGG = 16 + stride, INC = 16 + 2 * stride;

  // 1. Load the chunk and scan it in registers.
  vector<int, LB> v; // becomes the chunk's inclusive scan
  read_ints<LB>(A, c * LB, v);
  int carry = 0;
#pragma unroll
  for (int k = 0; k < LB / B; ++k) {
    vector_ref<int, B> blk = v.select<B, 1>(k * B);
    blk = inclusive_scan<B>(blk);
    blk += carry;
    carry = blk(B - 1);
  }
  int total = carry;

  // 2. Publish the aggregate (flag 1).
  atomic_store(STATE, AGG + c, uint(total));
  cm_fence(CM_GLOBAL_COHERENT_FENCE);
  atomic_store(STATE, FLAG + c, 1u);

  // 3. Look back for the prefix (chunk 0 has none), LB_W predecessors per
  //    round trip: one SIMD-LB_W atomic read of their flags; if every flag
  //    above the nearest "2" is set, one more of their aggregates, summed in
  //    registers. Lanes before chunk 0 count as "2" with value 0.
  int prefix = 0;
  uint spins = 0;
  vector<uint, LB_W> lane;
#pragma unroll
  for (int i = 0; i < LB_W; ++i) {
    lane(i) = i;
  }
  int hi = int(c) - 1; // highest predecessor not yet accounted for
  while (hi >= 0) {
    int lo = hi - (LB_W - 1); // window [lo, hi]; lanes with lo + i < 0 are virtual
    vector<int, LB_W> idx = lo + vector<int, LB_W>(lane);
    vector<ushort, LB_W> real = idx >= 0;
    vector<uint, LB_W> at = 0;
    at.merge(vector<uint, LB_W>(idx), real);

    vector<uint, LB_W> zero = 0;
    vector<uint, LB_W> flag = 2; // virtual lanes: "inclusive, value 0"
    vector<uint, LB_W> got;
    write_atomic<ATOMIC_OR, uint>(real, STATE, at + FLAG, zero, got);
    flag.merge(got, real);

    uint has_inc = cm_pack_mask(flag == 2u);
    uint missing = cm_pack_mask(flag == 0u);
    // Highest set bit = nearest predecessor with a prefix. NB: for unsigned
    // input cm_fbh returns the count of LEADING ZEROS, not the bit index.
    int k = has_inc ? 31 - int(cm_fbh(has_inc)) : -1;
    uint above = k >= 0 ? (0xFFFFFFFFu << (k + 1)) : 0xFFFFFFFFu;
    if (missing & above & ((1u << LB_W) - 1u)) {      // someone above k has not published
      if (++spins > LB_MAX_SPIN) {
        atomic_store(STATE, 1, c + 1);
        atomic_store(STATE, 2, uint(hi) + 1);
        break;
      }
#ifdef LB_PAUSE
      cm_pause(LB_PAUSE);
#endif
      continue; // re-read the same window
    }
    cm_fence(CM_GLOBAL_COHERENT_FENCE); // flags before values

    vector<uint, LB_W> agg = 0;
    vector<ushort, LB_W> use = real & (lane > uint(k));
    if (k < 0) {
      use = real;
    }
    write_atomic<ATOMIC_OR, uint>(use, STATE, at + AGG, zero, got);
    agg.merge(got, use);
    prefix += cm_sum<int>(agg.format<int>());
    if (k >= 0) {
      if (idx(k) >= 0) {
        prefix += int(atomic_load(STATE, INC + uint(idx(k))));
      }
      break;
    }
    hi = lo - 1;
  }

  // 4. Publish this chunk's inclusive prefix (flag 2).
  atomic_store(STATE, INC + c, uint(prefix + total));
  cm_fence(CM_GLOBAL_COHERENT_FENCE);
  atomic_store(STATE, FLAG + c, 2u);

  // 5. Write the output (inclusive, or exclusive = shifted by one element).
  int before = prefix; // value preceding each block's first element
#pragma unroll
  for (int k = 0; k < LB / B; ++k) {
    vector<int, B> res;
    if (exclusive) {
      res.select<B - 1, 1>(1) = v.select<B - 1, 1>(k * B);
      res(0) = 0;
      res += prefix;
      res(0) = before;
      before = prefix + v(k * B + B - 1);
    } else {
      res = v.select<B, 1>(k * B) + prefix;
    }
    write_ints<B>(OUT, c * LB + k * B, res);
  }
}
