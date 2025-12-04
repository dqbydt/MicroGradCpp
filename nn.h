#ifndef NN_H
#define NN_H

#include <random>
#include <ranges>

#include "value.h"

class Neuron {

public:
    // Creates a Neuron with "nin" inputs, init to random weights and bias
    explicit Neuron(size_t nin) : nin(nin), params(nin+1) {
        for (auto& p : params) p = Value{rand_uniform_m1_1()};
    }

    // Allows external injection of w_b's/b for testing/comparison with PT
    // Works with vector<double>, array<double, N>, initializer_list<double>, span<const double>
    // Treats all but the last as weights; last as bias.
    explicit Neuron(std::ranges::input_range auto&& w_b)
        requires std::ranges::sized_range<decltype(w_b)>
    {
        const auto n = std::ranges::size(w_b);
        assert(n == nin+1);

        params.reserve(n);

        for (auto&& p : w_b) {
            params.emplace_back(Value{std::forward<decltype(p)>(p)});
        }
    }

    Value operator()(std::ranges::input_range auto&& x) const {
        // zip_transform terminates at the end of the shortest input range.
        // Here that is x, which should be 1 less than the length of the params vec
        auto act =  std::ranges::fold_left(
                        std::views::zip_transform(std::multiplies<>{}, params, x),
                        params.back() + 0.0,    // prvalue to work around deleted copy
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
        auto nin = n.params.size()-1;
        auto wv = n.params | std::views::take(nin);
        os << "Neuron(\n ";
        for (const auto& w : wv) os << w << " ";
        os << "bias=" << n.params.back() << ")\n";
        return os;
    }

    std::span<const Value> parameters() const { return params; }
    std::span<Value>       parameters()       { return params; }

private:
    std::vector<Value> params;
    size_t nin;

    inline double rand_uniform_m1_1() {
        // One RNG per thread
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> urd(-1.0, 1.0);
        return urd(gen);
    }
};

#endif // NN_H
