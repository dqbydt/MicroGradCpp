#include <torch/torch.h>

#include <iostream>
#include <format>

#include "value.h"
#include "nn.h"
#include "misc.h"

void dynamic_demo(const MLP &mlp, std::span<double> ys, const auto &xs);
void static_demo (const MLP &mlp, std::span<double> ys, auto&& xs_Vals);

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
    //std::initializer_list<std::initializer_list<double>> xs = {
    std::vector<std::vector<double>> xs = {
        {2.0,  3.0, -1.0},
        {3.0, -1.0,  0.5},
        {0.5,  1.0,  1.0},
        {1.0,  1.0, -1.0},
    };

    // This is the set of desired outputs (the ground truth)
    std::array ys = { 1.0, -1.0, -1.0, 1.0 };

    // Sacrificial labeled input Value vec<vec>. We std::move this into
    // the MLP to compile the graph with static input nodes
    std::vector<std::vector<Value>> xs_Vals(xs.size());

    for (const auto& [i, row] : py::enumerate(xs)) {
        xs_Vals[i].reserve(row.size());
        for (const auto& [j, col] : py::enumerate(row)) {
            xs_Vals[i].emplace_back(Value{xs[i][j], std::format("ip_{}_{}", i, j)});
        }
    }

    // Uncomment one of the two below:
    // ------------------------------
    // Static: Freezes the graph built during first dummy fwd pass and
    // uses that in subsequent iterations.
    // Note: xs_Vals gets consumed in this function!
    static_demo(mlp, ys, std::move(xs_Vals));

    // Dynamic: Builds a new graph in each fwd pass. Only param nodes
    // are retained.
    //dynamic_demo(mlp, ys, xs);

    return 0;
}


// static_demo: freezes the graph built during first dummy fwd pass and
// uses that in subsequent iterations.
// Note: xs_Vals auto&& since we need to move out of it
void static_demo(const MLP& mlp, std::span<double> ys, auto&& xs_Vals)
{
    // MLP output for each input is a vector<Value> (one Value corresponding
    // to each output Neuron). For a set of inputs, the output is a
    // vector<vector<Value>>. We are picking out the 0th element for our
    // single-output MLP(4,4,1), so ypred is a vector<Value>.
    auto ypred = xs_Vals
                 | std::views::transform([&](auto&& x){ return std::move(mlp(std::move(x))[0]); })
                 | std::ranges::to<std::vector>();

    // Data view of ypred vector. Because ypred gets baked into the yloss,
    // which gets baked into the loss, which is the root of the static expression
    // graph, the Value nodes corresponding to ypred continue to stay alive
    // and get updated, and can be read in every training epoch.
    auto ypred_datav = ypred | std::views::transform([](const auto& v) { return v.data();} );

    // Calculate the square loss (this is still a lazy view, note!)
    // Also note, this fails to compile unless you have an auto& on the
    // yout, bc copies have been disabled on the Value class.
    auto yloss = std::views::zip_transform(
        [](const auto ygt, const auto& yout){ return misc::sqr(yout-ygt); },
        ys,
        ypred);

    // Finally collapse yloss into the total scalar loss. This "loss" Value
    // object carries the entire history of the forward pass.
    Value loss = std::ranges::fold_left(yloss, Value{0.0}, std::plus<>{});
    std::println("init_loss = {}", loss);

    double last_loss = std::numeric_limits<double>::max();
    double loss_eps  = 1e-4;

    Value& static_graph = loss; // Alias for semantics
    static_graph.compile();

    for (auto epoch : py::range(2000)) {

        // 1. Forward pass:
        // ----------------
        static_graph.forward();

        // 2. Reset grads
        // --------------
        // When we build the graph anew every time, only the params stay constant,
        // and all intermediate nodes are created anew. When we retain the static graph,
        // grads for _all_ nodes must be zeroed!
        static_graph.zero_grad();

        // 3. Backward pass:
        // ------------------
        loss.backward();

        // 4. Param update:
        // ----------------
        for (const auto& p : mlp.parameters()) {
            p.data() += -0.1*p.grad();
        }

        // Note the double colon format spec! Reqd because you need to use a colon
        // for the range, then a colon for the elements
        std::println("ST Epoch {:3}: loss = {:.3f}, ypred: {::.3f}", epoch, loss.data(), ypred_datav);

        // 5. Convergence + stall checks
        // -----------------------------
        if (loss.data() < loss_eps) {
            std::println("Converged at epoch {}!", epoch);
            break;
        }

        if (std::abs(last_loss - loss.data()) < 1e-7) {
            std::println("Loss stalled. Stopping.");
            break;
        }

        last_loss = loss.data();
    }

}


// dynamic_demo: Builds a new graph in each fwd pass. Only param nodes are
// retained.
// Note: Here we don't need to bake static Value nodes into the graph, so
// we just take the xs vec<vec> by const lref.
void dynamic_demo(const MLP& mlp, std::span<double> ys, const auto& xs)
{
    std::vector<Value> ypred;
    Value loss;
    double last_loss = std::numeric_limits<double>::max();
    double loss_eps  = 1e-3;

    for (auto epoch : py::range(2000)) {

        // 1. Forward pass:
        // ----------------
        // MLP output for each input is a vector<Value> (one Value corresponding
        // to each output Neuron). For a set of inputs, the output is a
        // vector<vector<Value>>. We are picking out the 0th element for our
        // single-output MLP(4,4,1), so ypred is a vector<Value>.
        ypred = xs
                | std::views::transform([&](auto& x){ return std::move(mlp(x)[0]); })
                | std::ranges::to<std::vector>();

        // Data view of ypred vector. Because ypred gets baked into the yloss,
        // which gets baked into the loss, which is the root of the static expression
        // graph, the Value nodes corresponding to ypred continue to stay alive
        // and get updated, and can be read in every training epoch.
        auto ypred_datav = ypred | std::views::transform([](const auto& v) { return v.data();} );

        // Calculate the square loss (this is still a lazy view, note!)
        // Also note, this fails to compile unless you have an auto& on the
        // yout, bc copies have been disabled on the Value class.
        auto yloss = std::views::zip_transform(
            [](const auto ygt, const auto& yout){ return misc::sqr(yout-ygt); },
            ys,
            ypred);

        // Finally collapse yloss into the total scalar loss. This "loss" Value
        // object carries the entire history of the forward pass.
        loss = std::ranges::fold_left(yloss, Value{0.0}, std::plus<>{});

        // 2. Reset grads
        // --------------
        // When we build the graph anew every time, only the params stay constant,
        // and all intermediate nodes are created anew (with grads zeroed at
        // construction). So we just need to zero out the param grads before doing
        // the backward pass.
        for (const auto& p : mlp.parameters()) {
            p.grad() = 0.0;
        }

        // 3. Backward pass:
        // ------------------
        loss.backward();

        // 4. Param update:
        // ----------------
        for (const auto& p : mlp.parameters()) {
            p.data() += -0.1*p.grad();
        }

        // Note the double colon format spec! Reqd because you need to use a colon
        // for the range, then a colon for the elements
        std::println("DY Epoch {:3}: loss = {:.3f}, ypred: {::.3f}", epoch, loss.data(), ypred_datav);

        // 5. Convergence + stall checks
        // -----------------------------
        if (loss.data() < loss_eps) {
            std::println("Converged at epoch {}!", epoch);
            break;
        }

        if (std::abs(last_loss - loss.data()) < 1e-7) {
            std::println("Loss stalled. Stopping.");
            break;
        }

        last_loss = loss.data();
    }

}
