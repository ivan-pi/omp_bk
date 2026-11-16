// hello_world.cpp -- complete host program for the CM runtime ("tasking
// layer"): CmDevice -> CmProgram -> CmKernel -> CmThreadSpace -> CmTask ->
// CmQueue -> CmEvent, the flow of Intel's "C for Metal - The Precious Metal"
// article, plus an output CmBuffer so the result can be checked.
//
// Two builds of the same source (see run_cmrt.sh):
//   * hardware (default): the kernel is precompiled by cmc into a vISA
//     container `hello_world_genx.isa` (cmc -mcpu=<gen> hello_world_genx.cpp)
//     and loaded with CmDevice::LoadProgram; link with -ligfxcmrt (from the
//     Intel media driver, libigfxcmrt; needs a GPU the media driver supports).
//   * emulation (-DCMRT_EMU): the kernel source is compiled into this
//     executable by the host C++ compiler and runs on the CPU; link with
//     -ligfxcmrt_emu -lcm from the CM SDK. CM_KERNEL_FUNCTION(name) then
//     passes the kernel's address as well as its name.
//
// For new code prefer SYCL ESIMD, or a CM kernel launched through OpenCL /
// Level Zero / SYCL (see ../host_ocl.cpp); this API is the legacy path.
#include "cm_rt.h"

#ifdef CMRT_EMU
#include "hello_world_genx.cpp" // emulation: the kernel is ordinary C++
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Every CM runtime call returns CM_SUCCESS (0) or a negative error code.
#define cm_result_check(call)                                                  \
  do {                                                                         \
    int result_ = (call);                                                      \
    if (result_ != CM_SUCCESS) {                                               \
      std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #call,     \
                   result_);                                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static std::string load_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

int main() {
  const int thread_width = 8;
  const int thread_height = 4;
  const int nthreads = thread_width * thread_height;

  // CmDevice creation.
  CmDevice *p_cm_device = nullptr;
  UINT version = 0;
#ifdef CMRT_EMU
  cm_result_check(::CreateCmDevice(p_cm_device, version));
#else
  // Hardware: CreateCmDevice(dev, ver) without a VADisplay fails with
  // CM_NULL_POINTER (-90) when the default GPU is not supported -- e.g. on a
  // machine with an Arc A-series (DG2) card, which the media driver's CM
  // runtime does not support. Enumerate the adapters and take the first one
  // that works (or force one with CM_ADAPTER=<index>).
  {
    uint32_t count = 0;
    cm_result_check(::GetCmSupportedAdapters(count));
    const char *force = std::getenv("CM_ADAPTER");
    int result = CM_FAILURE;
    for (uint32_t a = 0; a < count && result != CM_SUCCESS; ++a) {
      if (force && a != uint32_t(std::atoi(force))) {
        continue;
      }
      char desc[256] = {};
      uint32_t device_id = 0;
      uint32_t out_size = 0;
      ::QueryCmAdapterInfo(a, Description, desc, sizeof desc, &out_size);
      ::QueryCmAdapterInfo(a, DeviceId, &device_id, sizeof device_id, &out_size);
      uint32_t v = 0;
      result = ::CreateCmDeviceFromAdapter(p_cm_device, v, a);
      std::printf("adapter %u: %s (device 0x%x) -> %s\n", a, desc, device_id,
                  result == CM_SUCCESS ? "CmDevice created" : "not supported");
      version = v;
    }
    cm_result_check(result);
  }
#endif
  std::printf("CM runtime version %u\n", version);

  // CmProgram: the precompiled kernel ISA (hardware) or a placeholder
  // (emulation, where the kernel is linked into this executable).
  CmProgram *p_program = nullptr;
#ifdef CMRT_EMU
  std::string isa_code(16, '\0');
#else
  std::string isa_code = load_file("hello_world_genx.isa");
  if (isa_code.empty()) {
    std::fprintf(stderr, "hello_world_genx.isa not found (build it with cmc)\n");
    return 1;
  }
#endif
  cm_result_check(p_cm_device->LoadProgram(const_cast<char *>(isa_code.data()),
                                           isa_code.size(), p_program));

  // CmKernel by name (hardware) or by name and address (emulation).
  CmKernel *p_kernel = nullptr;
  cm_result_check(p_cm_device->CreateKernel(p_program, CM_KERNEL_FUNCTION(hello_world),
                                            p_kernel));

  // Thread space: thread_width x thread_height threads, one kernel instance
  // each; the kernel reads its coordinates with get_thread_origin_x/y().
  CmThreadSpace *p_thread_space = nullptr;
  cm_result_check(p_cm_device->CreateThreadSpace(thread_width, thread_height,
                                                 p_thread_space));

  // Output buffer: one uint per thread, pre-filled with a sentinel.
  CmBuffer *p_out = nullptr;
  cm_result_check(p_cm_device->CreateBuffer(nthreads * sizeof(unsigned), p_out));
  std::vector<unsigned> host(nthreads, 0xDEADBEEFu);
  cm_result_check(p_out->WriteSurface(reinterpret_cast<unsigned char *>(host.data()),
                                      nullptr));
  SurfaceIndex *p_out_index = nullptr;
  cm_result_check(p_out->GetIndex(p_out_index));

  // Kernel arguments: by value for scalars, a SurfaceIndex for surfaces.
  cm_result_check(p_kernel->SetKernelArg(0, sizeof(thread_width), &thread_width));
  cm_result_check(p_kernel->SetKernelArg(1, sizeof(SurfaceIndex), p_out_index));

  // Kernel printf needs a print buffer on the GPU (in emulation it is plain
  // stdio). Without it, printf in the kernel silently prints nothing.
#ifndef CMRT_EMU
  cm_result_check(p_cm_device->InitPrintBuffer());
#endif

  // CmTask: a container of kernels that are enqueued together.
  CmTask *p_task = nullptr;
  cm_result_check(p_cm_device->CreateTask(p_task));
  cm_result_check(p_task->AddKernel(p_kernel));

  // CmQueue and Enqueue: execution starts here.
  CmQueue *p_queue = nullptr;
  cm_result_check(p_cm_device->CreateQueue(p_queue));
  CmEvent *p_event = nullptr;
  cm_result_check(p_queue->Enqueue(p_task, p_event, p_thread_space));

  // Wait for the task, then read the buffer back (ReadSurface waits on the event).
  cm_result_check(p_event->WaitForTaskFinished());
  cm_result_check(p_out->ReadSurface(reinterpret_cast<unsigned char *>(host.data()),
                                     p_event));
#ifndef CMRT_EMU
  cm_result_check(p_cm_device->FlushPrintBuffer()); // kernel printf output -> stdout
#endif

  // Check: element i must hold i, so every thread ran exactly once.
  int bad = 0;
  for (int i = 0; i < nthreads; ++i) {
    bad += host[i] != unsigned(i);
  }
  std::printf("hello_world: %d threads, %d mismatches -> %s\n", nthreads, bad,
              bad ? "FAIL" : "PASS");

  // Clean up (DestroyCmDevice would also release everything it owns).
  cm_result_check(p_queue->DestroyEvent(p_event));
  cm_result_check(p_cm_device->DestroyTask(p_task));
  cm_result_check(p_cm_device->DestroySurface(p_out));
  cm_result_check(p_cm_device->DestroyThreadSpace(p_thread_space));
  cm_result_check(p_cm_device->DestroyKernel(p_kernel));
  cm_result_check(p_cm_device->DestroyProgram(p_program));
  cm_result_check(::DestroyCmDevice(p_cm_device));
  return bad ? 1 : 0;
}
