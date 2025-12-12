#ifndef PTNN_H
#define PTNN_H

#include <torch/torch.h>
#include <iostream>
#include <print>

struct TorchNeuron : torch::nn::Module {
    TorchNeuron(int64_t nin)
        : linear(register_module("linear", torch::nn::Linear(nin, 1)))
    {
        // Match your exact initialization: uniform(-1, 1)
        torch::nn::init::uniform_(linear->weight, -1.0, 1.0);
        torch::nn::init::uniform_(linear->bias,   -1.0, 1.0);
    }

    torch::Tensor operator()(const torch::Tensor& x) {
        // x: [nin] or [batch, nin]
        auto act = linear(x);           // does w·x + b
        return torch::tanh(act);        // exactly your .tanh()
    }

    // Helper: print parameters like your Neuron
    friend std::ostream& operator<<(std::ostream& os, const TorchNeuron& n) {
        os << "TorchNeuron(\n";
        os << "  w = [";
        for (int i = 0; i < n.linear->weight.size(1); ++i) {
            os << n.linear->weight[0][i].item<double>();
            if (i + 1 < n.linear->weight.size(1)) os << ", ";
        }
        os << "],\n";
        os << "  b = " << n.linear->bias[0].item<double>() << "\n";
        os << ")\n";
        return os;
    }

    // This is a fully-connected Neuron layer. Same as my
    // Neuron class except vectorized over batch dim
    // Py: layer = nn.Linear(in_features=nin, out_features=1, bias=True)
    torch::nn::Linear linear{nullptr};
};

// Courtesy Grok
struct TorchMLP : torch::nn::Module {
    // Holds the layers in order
    torch::nn::Sequential net;

    TorchMLP() = default;

    // Constructor: same signature as your MLP
    TorchMLP(size_t nin, std::initializer_list<size_t> layer_sizes) {
        std::vector<size_t> sizes = {nin};
        sizes.insert(sizes.end(), layer_sizes.begin(), layer_sizes.end());

        // Build sequential: Linear → tanh → Linear → tanh → ...
        for (size_t i=0; i+1 < sizes.size(); ++i) {
            int64_t in  = static_cast<int64_t>(sizes[i]);
            int64_t out = static_cast<int64_t>(sizes[i+1]);

            std::println("TorchMLP: Creating Layer({},{})", in, out);

            // Create Linear layer
            auto linear = register_module(
                "fc" + std::to_string(i),
                torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(true))
            );

            // Initialize weights and bias with uniform(-1, 1) — exactly like your micrograd
            torch::nn::init::uniform_(linear->weight, -1.0, 1.0);
            torch::nn::init::uniform_(linear->bias,   -1.0, 1.0);

            // Add Linear → tanh
            net->push_back(linear);
            net->push_back(torch::nn::Functional(torch::tanh));
        }

        // Remove the last tanh if you want raw logits (optional)
        // net->pop_back();  // uncomment if you want no final activation
    }

    torch::Tensor forward(torch::Tensor x) {
        return net->forward(x);
    }

    // Convenience: allow calling like tmlp(input_tensor)
    torch::Tensor operator()(torch::Tensor x) {
        return forward(x);
    }
};

#endif // PTNN_H
