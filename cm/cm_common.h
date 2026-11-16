// cm_common.h -- memory helpers and the copy_cm bandwidth kernel shared by
// the C for Metal kernels of cm/ (bk1_cm_genx.cpp, bk5_cm_genx.cpp).

#pragma once

#include <cm/cm.h>

constexpr int STORE_LANES = 16;   // lanes of the scattered writes (32: no faster)

constexpr int round4(const int n) { return (n + 3) / 4 * 4; }
constexpr int round8(const int n) { return (n + 7) / 8 * 8; }

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

// N floats from v[H, H + N) to element offset base + H, v(0) being 16 B
// aligned at element offset `base`: OWord block writes of the aligned middle,
// scattered writes of the unaligned head and tail
template <int H, int N, int TOT>
_GENX_ inline void store_span(SurfaceIndex buf, uint base, vector_ref<float, TOT> v)
{
    static_assert(H >= 0 && H < 4, "head offset inside the first OWord");
    constexpr int B0 = (H == 0) ? 0 : 4;   // aligned middle [B0, B1)
    constexpr int B1 = (H + N) / 4 * 4;
    store_scattered<H, B0 - H, TOT>(buf, base, v);
    store_blocks<B0, B1 - B0, TOT>(buf, base * 4, v);
    store_scattered<B1, H + N - B1, TOT>(buf, base, v);
}

// SLM block copies of 32, 16, 8 or 4 floats: v[OFF, OFF + N) <-> byte
// offset base + (OFF - V0) * 4 of the SLM buffer `slm` (the handle of
// cm_slm_alloc: an offset added to the handle itself is ignored, every
// thread then hits the buffer's start; it goes in the offset argument)
template <bool WRITE, int V0, int OFF, int N, int TOT>
_GENX_ inline void slm_blocks(uint slm, uint base, vector_ref<float, TOT> v)
{
    if constexpr (N > 0) {
        constexpr int K = (N >= 32) ? 32 : (N >= 16) ? 16 : (N >= 8) ? 8 : 4;
        if constexpr (WRITE) {
            cm_slm_block_write(slm, base + (OFF - V0) * 4, v.template select<K, 1>(OFF));
        } else {
            cm_slm_block_read(slm, GENX_NONE, base + (OFF - V0) * 4, v.template select<K, 1>(OFF));
        }
        slm_blocks<WRITE, V0, OFF + K, N - K, TOT>(slm, base, v);
    }
}

// --------------------------------------------------------------------------
// copy_cm: copy COPY_CHUNK floats per thread, the bandwidth reference of the hosts
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
