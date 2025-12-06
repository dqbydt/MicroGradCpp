#include <torch/torch.h>

#include <iostream>
#include <format>

#include "value.h"
#include "nn.h"

int main()
{
    // inputs x1, x2
    Value x1{2.0, "x1"};
    Value x2{0.0, "x2"};

    // Weights w1, w2
    Value w1{-3.0, "w1"};
    Value w2{1.0, "w2"};

    // Bias
    // Strange value selected to ensure gradients come out well-formed
    Value b{6.8813735, "b"};

    // Intermediate nodes
    Value x1w1 = x1 * w1; x1w1.label() = "x1w1";
    Value x2w2 = x2 * w2; x2w2.label() = "x2w2";

    // x1*w1 + x2*w2 + b
    Value x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label() = "x1w1 + x2w2";
    // n is the raw cell body activation w/o the activation fn applied
    Value n = x1w1x2w2 + b; n.label() = 'n';

    // Final output of the neuron
    //Value e = (2*n).exp(); e.label() = 'e';
    //Value L = (e-1)/(e+1); L.label() = 'L';
    //Value L = n.tanh(); L.label() = 'L';
    Value L = n.relu(); L.label() = 'L';
    L.grad() = 1.0;

    // Perform the backward pass
    L.backward();

    std::cout << L << n << b << x1w1x2w2 << x1w1 << x2w2 << x1 << w1 << x2 << w2;
    std::cout.flush();

    // w1.grad = 1.0. So inc w1 will cause L to inc
    w1 = w1 + 0.001;
    L = (x1*w1 + x2*w2 + b).relu();
    std::cout << L; std::cout.flush();
    L.backward();
    std::cout << L << w1; std::cout.flush();

    w1 = w1 + 0.001;
    L = (x1*w1 + x2*w2 + b).relu();
    std::cout << L; std::cout.flush();
    L.backward();
    std::cout << L << w1; std::cout.flush();

    Neuron nn(3);
    auto out = nn({1.0, 2.0, 3.0});
    std::cout << "-----------\n";
    std::cout << nn;
    std::cout << "Neuron output: " << out << "\n";
    out.backward();

    std::cout << "Layer test\n";
    Layer l{2,3};   // 2 inputs, 3 neurons (so 3 outputs)
    auto outs = l({1,0, 2.0});
    for (const auto& lout : outs) {
        std::cout << lout;
    }

    return 0;
}
