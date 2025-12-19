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

    // Allows external injection of w_b's/b for testing/comparison with PyTorch
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
    std::vector<Value> params;  // Last = bias, rest = weights
    size_t nin;

    inline double rand_uniform_m1_1() {
        // One RNG per thread
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> urd(-1.0, 1.0);
        return urd(gen);
    }
};

// Layer: each L has a number of Neurons. They are not connected to each other
// but are fully connected to the input. So a layer of Ns is just a set of Ns
// evaluated independently.
// Layer params: nin = # of inputs to each N in the layer, = # of Ns in the prev L
// (each N from prev L is connected to each N of this one. So # of inputs for each
// N in this layer = # of N's in prev layer).
// nout = # of outputs from this L, which is the same as # of N's in this L (since
// each N has a single output).
class Layer {
public:
    explicit Layer(size_t nin, size_t nout) : nin_(nin), nout_(nout) {
        neurons.reserve(nout);
        for ([[maybe_unused]] auto _ : std::views::iota(0u, nout)) {
            neurons.emplace_back(Neuron{nin});
        }
    }

    // operator() for L calls this L with inputs x, which applies the
    // input to each N in this layer. There are nout N's, so this will
    // produce a lazy transform_view of nout output Values which are the
    // result of calling each N with the inputs x. This does not actually
    // perform the operation! That will be done only when the Values are
    // extracted/materialized.
    // NOTE: capturing inputs by ref, so inputs need to live at least
    // as long as the Layer.
    auto operator()(std::ranges::input_range auto&& x) const {
        return neurons | std::views::transform([&](const auto& n) { return n(x); });
    }

    // Same overload as for Neuron to allow init_list to be passed
    auto operator()(std::initializer_list<double> ild) const {
        return operator()(std::vector(ild));
    }

    // parameters() — flat view of all Neurons' params
    auto parameters() const {
        return neurons
               | std::views::transform([](const auto& n) { return n.parameters(); })
               | std::views::join;
    }

    size_t nin()    const { return nin_;  }
    size_t nout()   const { return nout_; }

private:
    std::vector<Neuron> neurons;
    size_t nin_;
    size_t nout_;
};


// MLP: Layers feed into each other sequentially. MLP takes ctor args nin, which is
// the number of inputs to the structure, and an init_list of nouts, which defines
// the sizes of the Layers in the MLP.
class MLP {
public:
    MLP(size_t nin, std::initializer_list<size_t> nouts) {
        layers.reserve(nouts.size());
        // Combine input params into a layer_sizes vec
        std::vector layer_sizes = {nin};
        layer_sizes.insert(layer_sizes.end(), nouts.begin(), nouts.end());

        // Want to iterate 0 to layer_sizes-1, or equivalently, 0 to nouts
        for (auto i : std::views::iota(0u, nouts.size())) {
            std::println("MLP: Creating Layer({},{})", layer_sizes[i], layer_sizes[i+1]);
            layers.emplace_back(Layer{layer_sizes[i], layer_sizes[i+1]});
        }
    }

    // Iterate over each layer successively, feeding in output of
    // last into input of next
    auto operator()(std::ranges::input_range auto&& x) const {
        for (auto& layer : layers) {
            x = layer(x) | std::ranges::to<std::vector<Value>>();
        }
        return x;
    }

    // Here we must convert the init_list to vec<Value> to enable it to be
    // used in the input_range operator() above
    auto operator()(std::initializer_list<double> ild) const {
        return operator()(std::vector(ild)
                          | std::views::transform([](auto d) { return Value{d}; })
                          | std::ranges::to<std::vector<Value>>());
    }

    // parameters() — flat view of all Layers' params
    auto parameters() const {
        return layers
               | std::views::transform([](const auto& l) { return l.parameters(); })
               | std::views::join;
    }

//private:
    std::vector<Layer> layers;
};

#endif // NN_H
