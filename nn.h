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

    // Allows external injection of w_b's/b for testing/comparison with PT
    // Works with vector<double>, array<double, N>, initializer_list<double>, span<const double>
    // Treats all but the last as weights; last as bias.
    explicit Neuron(std::ranges::input_range auto&& w_b)
        requires std::ranges::sized_range<decltype(w_b)>
    {
        const auto n = std::ranges::size(w_b);

        // First n-1 values from the input_range are weights
        w = w_b
            | std::views::take(n-1)
            | std::views::transform([](auto&& v) { return Value(std::forward<decltype(v)>(v)); })
            | std::ranges::to<std::vector>();

        // Last value is the bias
        b = Value{*std::prev(std::ranges::end(w_b))};
    }

    Value operator()(std::ranges::input_range auto&& x) const {
        auto act =  std::ranges::fold_left(
                        std::views::zip_transform(std::multiplies<>{}, w, x),
                        b + 0.0,    // prvalue to work around deleted copy. std::move(b)?
                        std::plus<>{}
                    );
        auto out = act.tanh();
        return out;
    }

    // To allow Pythonic out = n({1.0, 2.0, 3.0}) syntax -- looks like even though
    // std::initializer_list<T> has .begin() and .end(), it does not model
    // std::ranges::input_range in the current (Nov 2025) standard library implementations.
    // Workaround is to convert the initializer_list into a vector and call
    // original operator() with the vector.
    Value operator()(std::initializer_list<double> ild) const {
        return operator()(std::vector(ild));
    }

    friend std::ostream& operator<<(std::ostream& os, const Neuron& n) {
        os << "Neuron(\n ";
        for (const auto& w : n.w) os << w << " ";
        os << "bias=" << n.b << ")\n";
        return os;
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
