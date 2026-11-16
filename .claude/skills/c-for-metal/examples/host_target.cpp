// host_target.cpp -- print the target macros a CM binary was compiled with
// (target_genx.cpp), compare them with the GPU it runs on, and check the
// target-specialized copy. SYCL host (cm_sycl.hpp).
//
//   host_target target_genx.{spv,bin,cpp}
//
// With a current cmc the kernel reports the IP version it was compiled for,
// which must match the device's major.minor (the revision may differ). The
// 2021 SDK cmc reports 0.0.0 and only CM_GENX -- of its -mcpu target (TGLLP
// = 1200 even when ocloc then finalizes for DG2), so CM_GENX-based
// specializations there describe the -mcpu, not the GPU. The device's IP
// version comes from cmsycl::ip_version (SYCL's architecture enum), the same
// word as OpenCL's CL_DEVICE_IP_VERSION_INTEL.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_target.cpp -lze_loader -lOpenCL -o host_target
#include "cm_sycl.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s target_genx.{spv,bin,cpp}\n", argv[0]);
    return 2;
  }
  try {
    // Queue on the GPU, its IP version, and the CM program.
    sycl::queue q = cmsycl::make_queue();
    const sycl::device dev = q.get_device();
    const unsigned ip = cmsycl::ip_version(dev);
    const unsigned major = ip >> 22;
    const unsigned minor = (ip >> 14) & 0xff;
    const unsigned rev = ip & 0x3f;
    std::printf("device IP %u.%u.%u (0x%x)\n", major, minor, rev, ip);
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);

    // target_info: the macros the binary was compiled with.
    int32_t *info = sycl::malloc_shared<int32_t>(8, q);
    for (int i = 0; i < 8; ++i) {
      info[i] = -1;
    }
    sycl::kernel target_info = prog.kernel("target_info");
    cmsycl::launch(q, target_info, 1, 1, info).wait();
    std::printf("compiled for: CM_GENX %d, IP %d.%d.%d, CM_HAS_LSC %d, CM_HAS_DPAS %d, "
                ">= 12.55 %d, C++%d\n",
                info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7]);
    size_t bad = 0;
    if (info[1] != 0) { // current cmc: the binary's target must be this GPU
      bad += unsigned(info[1]) != major || unsigned(info[2]) != minor;
    }
    bad += info[4] != 0 && !cmsycl::has_lsc(dev); // LSC code on a GPU without LSC
    cmsycl::report("target matches the device", 1, bad);
    const bool lsc = info[4] != 0;
    sycl::free(info, q);

    // copy_best: 256 floats per CM thread.
    const size_t n = size_t(1) << 22;
    static_assert(n % 256 == 0, "copy_best copies 256 floats per thread");
    float *x = sycl::malloc_shared<float>(n, q);
    float *y = sycl::malloc_shared<float>(n, q);
    for (size_t i = 0; i < n; ++i) {
      x[i] = float(i % 1009) * 0.25f;
      y[i] = 0.0f;
    }
    sycl::kernel copy = prog.kernel("copy_best");
    cmsycl::launch(q, copy, n / 256, 1, x, y).wait();

    // Check the copy.
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
      diff += y[i] != x[i];
    }
    bool ok = cmsycl::report(lsc ? "copy_best (LSC path)" : "copy_best (legacy path)", n, diff);
    sycl::free(x, q);
    sycl::free(y, q);
    return bad == 0 && ok ? 0 : 1;
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
