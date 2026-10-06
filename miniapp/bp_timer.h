#ifndef BP_TIMER_H
#define BP_TIMER_H

#include <chrono>

namespace bp {

// Accumulating wall-clock timer for the phase breakdown.
struct Stopwatch {
    using clock = std::chrono::steady_clock;
    double seconds = 0.0;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    void stop() { seconds += std::chrono::duration<double>(clock::now() - t0).count(); }
};

} // namespace bp

#endif // BP_TIMER_H
