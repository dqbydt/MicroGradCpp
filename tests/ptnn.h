#ifndef PTNN_H
#define PTNN_H

#include <torch/torch.h>
#include <iostream>

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

#endif // PTNN_H
