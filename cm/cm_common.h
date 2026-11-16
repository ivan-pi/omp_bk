// cm_common.h -- memory helpers and the copy yardstick shared by the C for
// Metal kernels of cm/ (bk1_cm_genx.cpp, bk5_cm_genx.cpp).

#pragma once

#include <cm/cm.h>

#ifndef STORE_LANES
#define STORE_LANES 16   // lanes of the scattered writes (unaligned stores)
#endif
static_assert(STORE_LANES == 8 || STORE_LANES == 16 || STORE_LANES == 32, "scattered write width");

constexpr int round4(const int n) { return (n + 3) / 4 * 4; }

// --------------------------------------------------------------------------
// memory
// --------------------------------------------------------------------------

// N floats from byte offset `base` into v[OFF, OFF + N), with OWord block
// reads of 32, 16, 8 or 4 floats (N a multiple of 4); ALIGNED: base is 16 B
// aligned, otherwise 4 B (DWALIGNED).
template <bool ALIGNED, int OFF, int N, int TOT>
_GENX_ inline void load_blocks(SurfaceIndex buf, uint base, vector_ref<float, TOT> v)
{
    static_assert(N % 4 == 0, "whole OWords");
    if constexpr (N > 0) {
        constexpr int K = (N >= 32) ? 32 : (N >= 16) ? 16 : (N >= 8) ? 8 : 4;
        if constexpr (ALIGNED) {
            read(buf, base + OFF * 4, v.template select<K, 1>(OFF));
        } else {
            read(DWALIGNED(buf), base + OFF * 4, v.template select<K, 1>(OFF));
        }
        load_blocks<ALIGNED, OFF + K, N - K, TOT>(buf, base, v);
    }
}

// N floats from v[OFF, OFF + N) to byte offset `base` (16 B aligned), OWord
// block writes of 32, 16, 8 or 4 floats
template <int OFF, int N, int TOT>
_GENX_ inline void store_blocks(SurfaceIndex buf, uint base, vector_ref<float, TOT> v)
{
    static_assert(N % 4 == 0, "whole OWords");
    if constexpr (N > 0) {
        constexpr int K = (N >= 32) ? 32 : (N >= 16) ? 16 : (N >= 8) ? 8 : 4;
        write(buf, base + OFF * 4, v.template select<K, 1>(OFF));
        store_blocks<OFF + K, N - K, TOT>(buf, base, v);
    }
}

// N floats from v[OFF, OFF + N) to element offset `base` (any alignment),
// scattered writes of at most STORE_LANES lanes
template <int OFF, int N, int TOT>
_GENX_ inline void store_scattered(SurfaceIndex buf, uint base, vector_ref<float, TOT> v)
{
    if constexpr (N > 0) {
        constexpr int K = (N >= STORE_LANES) ? STORE_LANES : (N >= 16) ? 16 : (N >= 8) ? 8 : (N >= 4) ? 4
                          : (N >= 2) ? 2 : 1;
        if constexpr (K == 1) {
            write(buf, base + OFF, 0u, v(OFF));
        } else {
            vector<uint, K> lane;
#pragma unroll
            for (int l = 0; l < K; ++l) {
                lane(l) = l;
            }
            vector<float, K> d = v.template select<K, 1>(OFF);
            write(buf, base + OFF, lane, d);
        }
        store_scattered<OFF + K, N - K, TOT>(buf, base, v);
    }
}

// --------------------------------------------------------------------------
// memory yardstick: copy COPY_CHUNK floats per thread
// --------------------------------------------------------------------------
constexpr int COPY_CHUNK = 256;

extern "C" _GENX_MAIN_ void copy_cm(SurfaceIndex src [[type("buffer_t")]],
                                    SurfaceIndex dst [[type("buffer_t")]])
{
    const uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    vector<float, COPY_CHUNK> c;
    load_blocks<true, 0, COPY_CHUNK, COPY_CHUNK>(src, t * (COPY_CHUNK * 4), c);
    store_blocks<0, COPY_CHUNK, COPY_CHUNK>(dst, t * (COPY_CHUNK * 4), c.select_all());
}
