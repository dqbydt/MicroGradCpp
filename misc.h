#ifndef MISC_H
#define MISC_H

#include <random>
#include <ranges>
#include <algorithm>

namespace misc {

    template <class T>
    inline T sqr(const T& x) { return x*x; }

    // Insert space before +ve vals for pretty-printing
    constexpr const char* sgnspc(double d) { return (d >= 0)? " ":""; };

    inline double rand_uniform_m1_1() {
        // One RNG per thread
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> urd(-1.0, 1.0);
        return urd(gen);
    }

}

// Pythonic range(..) helpers
// https://gemini.google.com/app/d1b289da7ecf1c88
namespace py {
    // 1. range(stop) --> 0 to stop-1
    // py::range(5)	returns {0, 1, 2, 3, 4}
    auto range(int stop) {
        return std::views::iota(0, std::max(0, stop));
    }

    // 2. range(start, stop) --> start to stop-1
    // py::range(5,10)	returns {5, 6, 7, 8, 9}
    auto range(int start, int stop) {
        return std::views::iota(start, std::max(start, stop));
    }

    // 3. range(start, stop, step) --> start to stop-1 by step
    // py::range(0,10,2) returns {0, 2, 4, 6, 8}
    // Note: ONLY SUPPORTS FWD RANGES!
    // For backward range, e.g. range(9, -1, -1) --> 9, 8, 7...0
    // use for (int i : py::range(0, 10) | std::views::reverse) { ... }
    // See discussion of any_view in the GG chat for why -ve stepsize
    // support had to be dropped.
    auto range(int start, int stop, int step) {
        // Handle Python's empty range behavior for invalid steps
        if (step <= 0) return std::views::iota(0, 0) | std::views::stride(1);

        return std::views::iota(start, std::max(start, stop))
               | std::views::stride(step);
    }

    // 4. enumerate -- Returns a view of tuples: {index, value}
    // Use thus: for (auto&& [i, letter] : py::enumerate(alphabet))
    // Search the GG chat for viewable_range to see why this is the only
    // way to specify this template:
    template<std::ranges::viewable_range R>
    auto enumerate(R&& r) {
        return std::views::enumerate(std::forward<R>(r));
    }

}


#endif // MISC_H
