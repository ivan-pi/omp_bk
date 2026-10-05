#ifndef BK_ACC_COMMON_H
#define BK_ACC_COMMON_H

#include <cstdint>
#include <random>
#include <vector>

#include "../bk_common.h"

namespace bk {

// Fill v with uniform values in [-1, 1) from a fixed seed, so that two builds
// of the same driver (device and host) see identical data and their output
// norms can be compared; constant data cannot detect a transposed index.
template <typename T>
void fill_random(std::vector<T>& v, const std::uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    for (T& x : v) {
        x = dist(gen);
    }
}

// BK_RANDOM=1 selects the random test data; unset or 0 keeps the constant
// data of the OpenMP drivers (in = 3, JxW = 1, G = 2).
inline bool random_data_requested()
{
    const auto value = get_env("BK_RANDOM");
    return value.has_value() && *value != "0";
}

} // namespace bk

#endif // BK_ACC_COMMON_H
