#include <torch/torch.h>

#include <iostream>
#include <format>

#include "value.h"
#include "nn.h"
#include "misc.h"

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

    std::println(" {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}\n {}", L, n, b, x1w1x2w2, x1w1, x2w2, x1, w1, x2, w2);

    // w1.grad = 1.0. So inc w1 will cause L to inc
    w1 = w1 + 0.001;
    L = (x1*w1 + x2*w2 + b).relu();
    std::println("L = {}", L);
    L.backward();
    std::println("After backward: L = {}, w1 = {}", L, w1);

    w1 = w1 + 0.001;
    L = (x1*w1 + x2*w2 + b).relu();
    std::println("After w1 nudge: L = {}", L);
    L.backward();
    std::println("Next backward: L = {}, w1 = {}", L, w1);

    Neuron nn(3);
    auto out = nn({1.0, 2.0, 3.0});
    std::println("-----------");
    std::cout << nn;
    std::println("Neuron output: out = {}", out);
    out.backward();

    std::println("\nLayer test");
    std::println("-----------");
    Layer l{2,3};   // 2 inputs, 3 neurons (so 3 outputs)
    auto outs = l({1.0, 2.0});
    for (const auto& lout : outs) {
        std::println("{}", lout);
    }

    MLP mlp{3, {4,4,1}};
    auto outvals = mlp({2.0, 3.0, -1.0});
    for (const auto& v : outvals) {
        std::println("MLP output: {}", v);
    }

    std::println("# of params = {}", mlp.num_params());

    // auto does not work here: the outer {} deduce init_list, but the
    // inner elements are also braced-init-lists (which have no type).
    // So the only way around is explicit typing.
    // HUH! Looks like init_list<init_list> does not guarantee lifetime
    // for the inner list! See https://gemini.google.com/app/db7f033f1290bccf
    // However in all testing it seems to stay alive with no ASan violations.
    std::initializer_list<std::initializer_list<double>> xs = {
    //std::vector<std::vector<double>> xs = {
        {2.0,  3.0, -1.0},
        {3.0, -1.0,  0.5},
        {0.5,  1.0,  1.0},
        {1.0,  1.0, -1.0},
    };

    // This is the set of desired outputs (the ground truth)
    std::array ys = { 1.0, -1.0, -1.0, 1.0 };

    std::vector<Value> ypred;

    for (auto i : py::range(15)) {

        // 1. Forward pass:
        // ----------------
        // MLP output for each input is a vector<Value> (one Value corresponding
        // to each output Neuron). For a set of inputs, the output is a
        // vector<vector<Value>>. We are picking out the 0th element for our
        // single-output MLP(4,4,1), so ypred is a vector<Value>.
        ypred = xs
                | std::views::transform([&](auto& x){ return std::move(mlp(x)[0]); })
                | std::ranges::to<std::vector>();

        // Calculate the square loss (this is still a lazy view, note!)
        // Also this fails to compile unless you have a ref on the yout, bc
        // copies have been disabled on the Value class.
        auto yloss = std::views::zip_transform(
            [](auto ygt, auto& yout){ return misc::sqr(yout-ygt); },
            ys,
            ypred);

        // Finally collapse yloss into the total scalar loss. This "loss" Value
        // object carries the entire history of the forward pass.
        Value loss = std::ranges::fold_left(yloss, Value{0.0}, std::plus<>{});

        // 2. Backward pass:
        // ------------------
        loss.backward();

        // 3. Param update:
        // ----------------
        for (const auto& p : mlp.parameters()) {
            p.data() += -0.01*p.grad();
        }

        std::println("Epoch {}: loss = {:.3f}", i, loss.data());
    }

    // Note the double colon format spec! Reqd because you need to use a colon
    // for the range, then a colon for the elements
    std::println("Final ypred vals: {::.3f}", ypred | std::views::transform([](const auto& v) { return v.data();} ));

    return 0;
}
