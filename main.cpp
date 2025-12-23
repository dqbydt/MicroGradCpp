#include <torch/torch.h>

#include <iostream>
#include <format>

#include "value.h"
#include "nn.h"

namespace misc {

template <class T>
inline T sqr(T x) { return x*x; }

}

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
    std::println("-----------");
    std::cout << nn;
    std::cout << "Neuron output: " << out << "\n";
    out.backward();

    std::println("Layer test");
    Layer l{2,3};   // 2 inputs, 3 neurons (so 3 outputs)
    auto outs = l({1,0, 2.0});
    for (const auto& lout : outs) {
        std::cout << lout;
    }

    MLP mlp{3, {4,4,1}};
    auto outvals = mlp({2.0, 3.0, -1.0});
    for (const auto& v : outvals) {
        std::cout << "MLP output: " << v << "\n";
    }

    std::println("# of params = {}", mlp.num_params());

    // auto does not work here: the outer {} deduce init_list, but the
    // inner elements are also braced-init-lists (which have no type).
    // So the only way around is explicit typing.
    std::initializer_list<std::initializer_list<double>> xs = {
        {2.0,  3.0, -1.0},
        {3.0, -1.0,  0.5},
        {0.5,  1.0,  1.0},
        {1.0,  1.0, -1.0},
    };

    auto ypred = xs | std::views::transform([&](auto& x){ return mlp(x)[0].data(); });

    for (const auto& v : ypred) {
        std::cout << "MLP ypred: " << v << "\n";
    }

    std::initializer_list<double> ys = { 1.0, -1.0, -1.0, 1.0 };
    auto yloss = std::views::zip_transform(
                    [](auto ygt, auto yout){ return misc::sqr(yout-ygt); },
                    ys, ypred);

    for (const auto& v : yloss) {
        std::cout << "MLP yloss: " << v << "\n";
    }


    return 0;
}
