// host_traits.cpp -- runs traits_demo of traits_genx.cpp through SYCL
// (cm_sycl.hpp) and prints what the descriptor-configured kernel saw.
//
//   host_traits [traits_genx.{spv,bin,cpp}]   (default: traits_genx.spv)
//
//
// Build: icpx -fsycl -O2 -fp-model=precise host_traits.cpp -lze_loader -lOpenCL -o host_traits
#include "cm_sycl.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
  try {
    // Queue on the GPU and the CM program.
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, argc > 1 ? argv[1] : "traits_genx.spv");
    sycl::kernel k = prog.kernel("traits_demo");

    // One CM thread writes 8 ints.
    int32_t *v = sycl::malloc_shared<int32_t>(8, q);
    for (int i = 0; i < 8; ++i) {
      v[i] = -1;
    }
    cmsycl::launch(q, k, 1, 1, v).wait();

    // Report and check.
    std::printf("from the GPU: m=%d k=%d sizeof(T)=%d tile elements=%d __cplusplus/100=%d\n", v[0], v[1], v[2],
                v[3], v[4]);
    bool ok = v[0] == 16 && v[1] == 4 && v[2] == 4 && v[3] == 256 && v[4] == 2017;
    sycl::free(v, q);
    return cmsycl::report("descriptor-configured kernel", 5, ok ? 0 : 1) ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
