#ifndef NN_H
#define NN_H

#include <random>
#include <ranges>

#include "value.h"

class Neuron {

public:
    // Creates a Neuron with "nin" inputs, init to random weights and bias
    Neuron(size_t nin) : b{Value{rand_uniform_m1_1()}} {
        for ([[maybe_unused]] auto _ : std::views::iota(0u, nin)) {
            w.emplace_back(Value{rand_uniform_m1_1()});
        }
    }

    Value operator()(std::ranges::input_range auto&& x) const {
        auto act = std::ranges::fold_left(
                        std::views::zip_transform(std::multiplies<>{}, w, x),
                        b + 0.0,    // prvalue to work around deleted copy. std::move(b)?
                        std::plus<>{}
                    );
        auto out = act.tanh();
        return out;
    }

private:
    std::vector<Value> w;
    Value b;

    inline double rand_uniform_m1_1() {
        // One RNG per thread
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> urd(-1.0, 1.0);
        return urd(gen);
    }
};

#endif // NN_H
