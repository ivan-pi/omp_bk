// repro.cpp -- minimal reproducer: a tinytc kernel whose local memory
// (alloca ... local) exceeds the device's local memory fails at kernel
// creation with only "Error occured in compute runtime" (status 18); the
// cause is not reported.  Built with SYCL + tinytc, compiled for the device
// at run time (tinytc::create_core_info(device)).
//
//   ./repro <N floats of local memory> [<N> ...]
//
// For each N: compiles the kernel below (one work-group of 16 work-items
// writes 1.0 to buf[0] of a local array of N floats, reads it back and
// stores it to out[0]), launches it and checks out[0] == 1.

#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>
#include <tinytc/version.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

const char* kernel_src = R"(
func @slm(%out: memref<f32x?>)
attributes{subgroup_size=16,work_group_size=[16,1]}
{
    %buf = alloca : memref<f32x$N,local>
    %c0 = constant 0 : index
    %x = constant 1.0 : f32
    store %x, %buf[%c0]
    barrier.local
    %y = load %buf[%c0] : f32
    store %y, %out[%c0]
}
)";

int run(sycl::queue& q, const long n)
{
    const std::string code = "$N = " + std::to_string(n) + "\n" + kernel_src;
    std::printf("N = %ld floats (%ld bytes of local memory): ", n, 4 * n);
    std::fflush(stdout);
    try {
        auto info = tinytc::create_core_info(q.get_device());
        auto ctx = tinytc::create_compiler_context();
        tinytc::set_error_reporter(
            ctx.get(), [](char const* what, const tinytc_location_t*, void*) { std::cerr << what << std::endl; });
        auto prog = tinytc::parse_string(code, ctx.get());
        auto bin = tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), bin.get());
        auto k = tinytc::create_kernel(bundle, "slm");
        float* out = sycl::malloc_shared<float>(1, q);
        out[0] = 0.0f;
        const std::int64_t n_out = 1;
        q.submit([&](sycl::handler& h) {
             h.set_args(out, n_out);
             h.parallel_for(tinytc::get_execution_range(k, sycl::range<3u>{1u, 1u, 1u}), k);
         }).wait_and_throw();
        const float v = out[0];
        sycl::free(out, q);
        std::printf("ran, out[0] = %g (%s)\n", v, (v == 1.0f) ? "ok" : "WRONG");
        return (v == 1.0f) ? 0 : 1;
    } catch (const tinytc::status& st) {
        std::printf("tinytc error %d: %s\n", static_cast<int>(st), tinytc::to_string(st));
    } catch (const sycl::exception& e) {
        std::printf("SYCL exception: %s\n", e.what());
    }
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    sycl::queue q{sycl::gpu_selector_v};
    const auto dev = q.get_device();
    std::printf("device: %s, driver %s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("local memory per work-group (sycl::info::device::local_mem_size): %zu bytes\n",
                static_cast<std::size_t>(dev.get_info<sycl::info::device::local_mem_size>()));
    std::printf("tinytc %s\n", TINYTC_VERSION_DESCRIPTION);
    int fails = 0;
    for (int a = 1; a < argc; ++a) {
        fails += run(q, std::atol(argv[a]));
    }
    return fails;
}
