// amx.h -- Apple AMX instruction macros with a software emulation fallback.
//
//   #include "amx.h"
//   AMX_SET(); AMX_LDX(op); AMX_LDY(op); AMX_FMA32(op); AMX_STZ(op); AMX_CLR();
//
// On Apple Silicon (arm64, macOS) the AMX_* macros emit the real instructions
// (encodings from https://github.com/corsix/amx, MIT).  Anywhere else -- or on
// Apple Silicon when AMX_EMULATE is defined before including this header --
// they call a software model of the same instructions operating on a
// thread-local register file, so kernels can be developed and checked on any
// machine.  AMX_HW is 1 when the real hardware is used, 0 otherwise.
//
// Emulated subset (operand bitfields as documented by corsix/amx):
//   ldx ldy ldz stx sty stz     64-byte moves, pair (bit 62) and quad (bit 60);
//                               pair/quad addresses must be 128-byte aligned
//   fma32 fma64 fms32 fms64     matrix and vector mode, X/Y byte offsets,
//                               ALU select (bits 27-29), lane enables (32-46);
//                               f16 operand widening (bits 60/61) not supported
//   extrx extry                 X <- Y and Y <- X register copies
//   extrh extrv                 Z row / Z column -> X or Y, f32 and f64 lanes,
//                               both the 26=0 and 26=1 encodings
//   set clr                     tracked; using AMX while disabled aborts,
//                               as it would (SIGILL) on hardware
// Everything else (ldzi, fma16, mac16, vecint, vecfp, matint, matfp, genlut)
// aborts with a message when emulated.
//
// Operand encoders for the emulated subset are in namespace amx::op and work
// identically for hardware and emulation.

#pragma once
#include <cstdint>
#include <cstddef>

#if defined(__APPLE__) && defined(__aarch64__) && !defined(AMX_EMULATE)
#  define AMX_HW 1
#else
#  define AMX_HW 0
#endif

// ---------------------------------------------------------------------------
// Operand encoders (hardware and emulation)
// ---------------------------------------------------------------------------
namespace amx {
namespace op {

constexpr uint64_t PTR_MASK = (uint64_t(1) << 56) - 1;

inline uint64_t ptr(const void* p) { return uint64_t(uintptr_t(p)) & PTR_MASK; }

// ldx / ldy / stx / sty: X or Y register 0-7. pair: two consecutive registers
// (memory must be 128-byte aligned); quad (loads only, M2+): four registers.
inline uint64_t ldxy(int reg, const void* p, bool pair = false, bool quad = false) {
    return (uint64_t(reg & 7) << 56) | ptr(p) | (pair || quad ? uint64_t(1) << 62 : 0)
         | (quad ? uint64_t(1) << 60 : 0);
}
inline uint64_t stxy(int reg, void* p, bool pair = false) {
    return (uint64_t(reg & 7) << 56) | ptr(p) | (pair ? uint64_t(1) << 62 : 0);
}
// ldz / stz: Z row 0-63.
inline uint64_t ldz(int row, const void* p, bool pair = false) {
    return (uint64_t(row & 63) << 56) | ptr(p) | (pair ? uint64_t(1) << 62 : 0);
}
inline uint64_t stz(int row, void* p, bool pair = false) {
    return (uint64_t(row & 63) << 56) | ptr(p) | (pair ? uint64_t(1) << 62 : 0);
}

// Lane-enable field (7-bit form used by fma/fms and the 26=0 extr variants):
//   mode 0: value 0 all, 1 odd lanes, 2 even lanes, >=3 none
//   mode 1: only lane #value      mode 2: first value lanes (0 = all)
//   mode 3: last value lanes (0 = all)
inline uint64_t enable7(int mode, int value) { return (uint64_t(mode & 3) << 5) | uint64_t(value & 31); }
constexpr uint64_t ENABLE_ALL = 0;

// fma32 / fms32 / fma64 / fms64.
//   matrix mode:  z[row 4j+ztile (f32) | 8j+ztile (f64)][i] (+)= x[i] * y[j]
//   vector mode:  z[zrow][i] (+)= x[i] * y[i]
//   xoff, yoff : byte offsets into the 512-byte circular X / Y register files
//   alu        : 0 x*y+z, 1 x*y, 2 x+z, 3 x, 4 y+z, 5 y, 6 z, 7 zero
//                (fms: z-x*y, -x*y, z-x, -x, z-y, -y, z, -0)
inline uint64_t fma(int zrow, int xoff, int yoff, int alu = 0, bool vector = false,
                    uint64_t xenable = ENABLE_ALL, uint64_t yenable = ENABLE_ALL) {
    return (vector ? uint64_t(1) << 63 : 0)
         | (xenable << 41) | (yenable << 32)
         | (uint64_t(alu & 7) << 27)
         | (uint64_t(zrow & 63) << 20)
         | (uint64_t(xoff & 0x1ff) << 10)
         | uint64_t(yoff & 0x1ff);
}

// extrx: X[xreg] = Y[yreg];  extry: Y[yreg] = X[xreg]  (whole registers)
inline uint64_t extrx_copy(int xreg, int yreg) {
    return (uint64_t(1) << 27) | (uint64_t(yreg & 7) << 20) | (uint64_t(xreg & 7) << 16);
}
inline uint64_t extry_copy(int yreg, int xreg) {
    return (uint64_t(1) << 27) | (uint64_t(xreg & 7) << 20) | (uint64_t(yreg & 7) << 6);
}

// extrh (26=1 encoding): Z row -> 64 bytes at X or Y byte offset dst_off.
// extrv (26=1 encoding): Z column -> X or Y.  For f32 the column is
//   y[j] = z[4j + tile][col], selected by zcol = 4*col + tile (0-63);
//   for f64 y[j] = z[8j + tile][col] with zcol = 8*col + tile.
// f64: set f64 = true.  Use AMX_EXTRX for extrh, AMX_EXTRY for extrv.
inline uint64_t extrh(int zrow, int dst_off, bool to_y, bool f64 = false) {
    return (uint64_t(1) << 63) | (uint64_t(f64 ? 1 : 8) << 11) | (uint64_t(1) << 26)
         | (uint64_t(zrow & 63) << 20) | (to_y ? uint64_t(1) << 10 : 0) | uint64_t(dst_off & 0x1ff);
}
inline uint64_t extrv(int zcol, int dst_off, bool to_y, bool f64 = false) {
    return (uint64_t(1) << 63) | (uint64_t(f64 ? 1 : 8) << 11) | (uint64_t(1) << 26)
         | (uint64_t(zcol & 63) << 20) | (to_y ? uint64_t(1) << 10 : 0) | uint64_t(dst_off & 0x1ff);
}

} // namespace op
} // namespace amx

// ---------------------------------------------------------------------------
// Hardware: instruction encodings from corsix/amx aarch64.h
// ---------------------------------------------------------------------------
#if AMX_HW

#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")
#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)  AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

// ---------------------------------------------------------------------------
// Emulation
// ---------------------------------------------------------------------------
#else

#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <cstring>

namespace amx {
namespace emu {

union Reg {
    uint8_t  u8[64];
    uint32_t u32[16];
    float    f32[16];
    double   f64[8];
};

struct State {
    alignas(64) Reg x[8];
    alignas(64) Reg y[8];
    alignas(64) Reg z[64];
    bool enabled = false;
};

inline State& state() {
    static thread_local State s{};
    return s;
}

[[noreturn]] inline void fail(const char* what) {
    std::fprintf(stderr, "amx emulation: %s\n", what);
    std::abort();
}
inline void check_enabled() {
    if (!state().enabled) {
        fail("instruction used while AMX is disabled (missing AMX_SET)");
    }
}

inline void set() { state().enabled = true; }
inline void clr() { state().enabled = false; }

// --- 64-byte moves ---------------------------------------------------------
// A single register may use any address; a pair or quad transfer must be
// 128-byte aligned (corsix/amx ldst.md), which is enforced here so that a
// kernel checked under emulation also meets the hardware's requirement.
inline void check_multi_align(const char* what, uint64_t opnd) {
    if ((opnd & (uint64_t(1) << 62)) && (opnd & op::PTR_MASK & 127)) {
        std::fprintf(stderr, "amx emulation: %s pair/quad transfer at address %% 256 = %u\n",
                     what, unsigned(opnd & 255));
        fail("pair/quad load or store with an address that is not 128-byte aligned");
    }
}
inline void ld_common(Reg* regs, uint64_t opnd, unsigned regmask) {
    check_enabled();
    check_multi_align("load", opnd);
    const unsigned rn = (opnd >> 56) & regmask;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(uintptr_t(opnd & op::PTR_MASK));
    std::memcpy(regs + rn, src, 64);
    if (opnd & (uint64_t(1) << 62)) {
        const unsigned n = (regmask <= 15 && (opnd & (uint64_t(1) << 60))) ? 4 : 2;
        for (unsigned k = 1; k < n; ++k) {
            std::memcpy(regs + ((rn + k) & regmask), src + 64 * k, 64);
        }
    }
}
inline void st_common(const Reg* regs, uint64_t opnd, unsigned regmask) {
    check_enabled();
    check_multi_align("store", opnd);
    const unsigned rn = (opnd >> 56) & regmask;
    uint8_t* dst = reinterpret_cast<uint8_t*>(uintptr_t(opnd & op::PTR_MASK));
    std::memcpy(dst, regs + rn, 64);
    if (opnd & (uint64_t(1) << 62)) {
        std::memcpy(dst + 64, regs + ((rn + 1) & regmask), 64);
    }
}
inline void ldx(uint64_t o) { ld_common(state().x, o, 7); }
inline void ldy(uint64_t o) { ld_common(state().y, o, 7); }
inline void ldz(uint64_t o) { ld_common(state().z, o, 63); }
inline void stx(uint64_t o) { st_common(state().x, o, 7); }
inline void sty(uint64_t o) { st_common(state().y, o, 7); }
inline void stz(uint64_t o) { st_common(state().z, o, 63); }

// --- helpers ---------------------------------------------------------------
// 64-byte window at a byte offset into a 512-byte circular register file
inline void window(void* dst, const Reg* regs, unsigned off) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(regs);
    uint8_t* d = static_cast<uint8_t*>(dst);
    for (unsigned b = 0; b < 64; ++b) {
        d[b] = base[(off + b) & 511];
    }
}
inline void window_store(Reg* regs, unsigned off, const uint8_t* src, uint64_t byte_enable) {
    uint8_t* base = reinterpret_cast<uint8_t*>(regs);
    for (unsigned b = 0; b < 64; ++b) {
        if ((byte_enable >> b) & 1) {
            base[(off + b) & 511] = src[b];
        }
    }
}

// Lane enable -> per-byte mask. `field` is the 7- or 9-bit enable field,
// `g` the lane width in bytes.  Mirrors corsix/amx parse_writemask.
inline uint64_t lane_mask(uint32_t field, unsigned g, unsigned bits) {
    const uint32_t mode = (bits >= 9) ? (field >> 6) & 7 : (field >> 5) & 3;
    uint32_t val = field;
    if (mode != 0) {
        val *= g;
    }
    val &= 0x3f;
    const uint64_t all = ~uint64_t(0);
    switch (mode) {
    case 0: {
        if (val == 1 || val == 2) {
            uint64_t m = ~(all << g) << (g & -(val & 1));   // odd (1) / even (2) lanes
            for (unsigned gg = g; (gg <<= 1) < 64;) {
                m |= m << gg;
            }
            return m;
        }
        return (val < (bits >= 9 ? 6u : 3u)) ? all : 0;
    }
    case 1: return (~(all << g)) << val;                   // only lane #N
    case 2: return (val == 0) ? all : ~(all << val);       // first N lanes (0 = all)
    case 4: return ~(all << val);                          // first N lanes
    case 3: return (val == 0) ? all : ~(all >> val);       // last N lanes (0 = all)
    case 5: return ~(all >> val);                          // last N lanes
    default: return 0;
    }
}

// --- fma / fms ---------------------------------------------------------------
template <typename T> inline T alu_fma(T x, T y, T z, unsigned mode) {
    switch (mode) {
    case 1: return x * y;
    case 2: return z + x;
    case 3: return x;
    case 4: return z + y;
    case 5: return y;
    case 6: return z;
    case 7: return T(0);
    default: return std::fma(x, y, z);
    }
}
template <typename T> inline T alu_fms(T x, T y, T z, unsigned mode) {
    switch (mode) {
    case 1: return std::fma(-x, y, T(-0.0));
    case 2: return z - x;
    case 3: return -x;
    case 4: return z - y;
    case 5: return -y;
    case 6: return z;
    case 7: return T(-0.0);
    default: return std::fma(-x, y, z);
    }
}

// Results are produced as on hardware with FPCR.DN set: any NaN becomes the
// default quiet NaN (payloads are not propagated).
template <typename T> inline T default_nan(T v) { return std::isnan(v) ? std::numeric_limits<T>::quiet_NaN() : v; }

template <typename T, bool SUB> inline void fma_impl(uint64_t o) {
    check_enabled();
    if (sizeof(T) == 4 && (o & ((uint64_t(1) << 61) | (uint64_t(1) << 60)))) {
        fail("fma32/fms32 with f16 operands is not emulated");
    }
    constexpr unsigned N = 64 / sizeof(T);          // lanes
    State& s = state();
    const unsigned yoff = o & 0x1ff, xoff = (o >> 10) & 0x1ff, zrow = (o >> 20) & 63;
    const unsigned mode = (o >> 27) & 7;
    const bool vector = (o >> 63) & 1;
    const uint64_t xen = lane_mask((o >> 41) & 0x7f, sizeof(T), 7);
    const uint64_t yen = lane_mask((o >> 32) & 0x7f, sizeof(T), 7);
    T x[N], y[N];
    window(x, s.x, xoff);
    window(y, s.y, yoff);
    for (unsigned i = 0; i < N; ++i) {
        if (!((xen >> (i * sizeof(T))) & 1)) {
            continue;
        }
        if (vector) {
            T& z = reinterpret_cast<T*>(s.z[zrow].u8)[i];
            z = default_nan(SUB ? alu_fms(x[i], y[i], z, mode) : alu_fma(x[i], y[i], z, mode));
        } else {
            for (unsigned j = 0; j < N; ++j) {
                if (!((yen >> (j * sizeof(T))) & 1)) {
                    continue;
                }
                T& z = reinterpret_cast<T*>(s.z[j * sizeof(T) + (zrow & (sizeof(T) - 1))].u8)[i];
                z = default_nan(SUB ? alu_fms(x[i], y[j], z, mode) : alu_fma(x[i], y[j], z, mode));
            }
        }
    }
}
inline void fma32(uint64_t o) { fma_impl<float,  false>(o); }
inline void fms32(uint64_t o) { fma_impl<float,  true >(o); }
inline void fma64(uint64_t o) { fma_impl<double, false>(o); }
inline void fms64(uint64_t o) { fma_impl<double, true >(o); }

// --- extr ------------------------------------------------------------------
// Decode the lane-width mode (bit 63 and bits 11-14) for the 26=1 variants.
inline unsigned extr_lane_bytes(uint64_t o) {
    const unsigned key = (unsigned((o >> 63) & 1) << 4) | unsigned((o >> 11) & 0xf);
    switch (key) {
    case 24: return 4;   // f32 -> f32
    case 17: return 8;   // f64 -> f64
    case 8:  return 4;   // i32 -> i32 (same data movement)
    default: fail("extrh/extrv lane-width mode not emulated (only f32/f64/i32)");
    }
}

// opcode 8: extrx (X<-Y copy) or extrh (Z row -> X/Y)
inline void extrx(uint64_t o) {
    check_enabled();
    State& s = state();
    const unsigned zrow = (o >> 20) & 63;
    if (o & (uint64_t(1) << 26)) {                           // extrh, 26=1
        const unsigned g = extr_lane_bytes(o);
        Reg* dst = (o & (uint64_t(1) << 10)) ? s.y : s.x;
        const uint32_t field = (o >> 32) & 0x1ff;
        uint8_t buf[64];
        std::memcpy(buf, s.z[zrow].u8, 64);
        if (field == 3) {
            std::memset(buf, 0, 64);
        }
        window_store(dst, o & 0x1ff, buf, lane_mask(field, g, 9));
    } else if (o & (uint64_t(1) << 27)) {                    // extrx copy
        std::memcpy(s.x + ((o >> 16) & 7), s.y + (zrow & 7), 64);
    } else {                                                 // extrh, 26=0 -> X
        unsigned g = 8 >> ((o >> 28) & 3);
        uint64_t en = ~uint64_t(0);
        if (g == 1) {
            g = 2;
            en = 0x5555555555555555ull;
        }
        en &= lane_mask((o >> 41) & 0x7f, g, 7);
        window_store(s.x, (o >> 10) & 0x1ff, s.z[zrow].u8, en);
    }
}

// opcode 9: extry (Y<-X copy) or extrv (Z column -> X/Y)
inline void extry(uint64_t o) {
    check_enabled();
    State& s = state();
    const unsigned zcol = (o >> 20) & 63;
    if (o & (uint64_t(1) << 27)) {                           // extry copy
        std::memcpy(s.y + ((o >> 6) & 7), s.x + (zcol & 7), 64);
        return;
    }
    unsigned g;
    Reg* dst;
    uint64_t en;
    uint32_t field;
    if (o & (uint64_t(1) << 26)) {                           // extrv, 26=1
        g = extr_lane_bytes(o);
        dst = (o & (uint64_t(1) << 10)) ? s.y : s.x;
        field = (o >> 32) & 0x1ff;
        en = lane_mask(field, g, 9);
    } else {                                                 // extrv, 26=0 -> Y
        g = 8 >> ((o >> 28) & 3);
        en = ~uint64_t(0);
        if (g == 1) {
            g = 2;
            en = 0x5555555555555555ull;
        }
        field = (o >> 32) & 0x7f;
        en &= lane_mask(field, g, 7);
        dst = s.y;
    }
    // column: lane j <- z[(j*g) | (zcol & (g-1))] bytes [zcol & -g, +g)
    uint8_t buf[64];
    for (unsigned j = 0; j < 64; j += g) {
        const unsigned row = (j & ~(g - 1)) | (zcol & (g - 1));
        std::memcpy(buf + j, s.z[row].u8 + (zcol & ~(g - 1)), g);
    }
    if (field == 3) {
        std::memset(buf, 0, 64);
    }
    window_store(dst, o & 0x1ff, buf, en);
}

inline void unsupported(uint64_t) { fail("instruction not emulated (ldzi/stzi/fma16/fms16/mac16/vecint/vecfp/matint/matfp/genlut)"); }

// Debug helpers
inline void dump_z_f32(int tile, std::FILE* f = stdout) {
    const State& s = state();
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            std::fprintf(f, "%10.4g ", s.z[4 * j + (tile & 3)].f32[i]);
        }
        std::fputc('\n', f);
    }
}

} // namespace emu
} // namespace amx

#define AMX_LDX(gpr)    ::amx::emu::ldx((uint64_t)(gpr))
#define AMX_LDY(gpr)    ::amx::emu::ldy((uint64_t)(gpr))
#define AMX_STX(gpr)    ::amx::emu::stx((uint64_t)(gpr))
#define AMX_STY(gpr)    ::amx::emu::sty((uint64_t)(gpr))
#define AMX_LDZ(gpr)    ::amx::emu::ldz((uint64_t)(gpr))
#define AMX_STZ(gpr)    ::amx::emu::stz((uint64_t)(gpr))
#define AMX_LDZI(gpr)   ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_STZI(gpr)   ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_EXTRX(gpr)  ::amx::emu::extrx((uint64_t)(gpr))
#define AMX_EXTRY(gpr)  ::amx::emu::extry((uint64_t)(gpr))
#define AMX_FMA64(gpr)  ::amx::emu::fma64((uint64_t)(gpr))
#define AMX_FMS64(gpr)  ::amx::emu::fms64((uint64_t)(gpr))
#define AMX_FMA32(gpr)  ::amx::emu::fma32((uint64_t)(gpr))
#define AMX_FMS32(gpr)  ::amx::emu::fms32((uint64_t)(gpr))
#define AMX_MAC16(gpr)  ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_FMA16(gpr)  ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_FMS16(gpr)  ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_SET()       ::amx::emu::set()
#define AMX_CLR()       ::amx::emu::clr()
#define AMX_VECINT(gpr) ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_VECFP(gpr)  ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_MATINT(gpr) ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_MATFP(gpr)  ::amx::emu::unsupported((uint64_t)(gpr))
#define AMX_GENLUT(gpr) ::amx::emu::unsupported((uint64_t)(gpr))

#endif // AMX_HW
