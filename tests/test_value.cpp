// tests/test_value.cpp
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_session.hpp>

#include <random>

#include <torch/torch.h>
#include "../value.h"
#include "../nn.h"
#include "ptnn.h"

using Catch::Matchers::WithinAbs;
constexpr double ABS_TOLERANCE = 1e-6;

// === FORCE ALL LIBTORCH TENSORS TO DOUBLE ===
[[maybe_unused]] static const auto _force_double = []{
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kDouble));
    std::cout << "Forcing PyTorch default datatype to double\n";
    return 0;
}();

inline double rand_uniform_m1_1() {
    // One RNG per thread
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> urd(-1.0, 1.0);
    return urd(gen);
}

TEST_CASE("Value addition matches libtorch", "[basic]") {
    Value a(2.0), b(3.0);
    Value c = a + 0.1*b;
    c.backward();

    //auto ta = torch::tensor(2.0, torch::requires_grad().dtype(torch::kDouble));
    auto ta = torch::tensor(2.0, torch::requires_grad());
    auto tb = torch::tensor(3.0, torch::requires_grad());
    auto tc = ta + 0.1*tb;
    tc.backward();

    REQUIRE_THAT(c.data(), WithinAbs(tc.data().item<double>(), ABS_TOLERANCE));
    REQUIRE_THAT(a.grad(), WithinAbs(ta.grad().item<double>(), ABS_TOLERANCE));
    REQUIRE_THAT(b.grad(), WithinAbs(tb.grad().item<double>(), ABS_TOLERANCE));
}

TEST_CASE("ReLU matches libtorch", "[relu]") {
    Value x(-2.5);
    auto y = x.relu();
    y.backward();

    auto tx = torch::tensor(-2.5, torch::requires_grad().dtype(torch::kDouble));
    auto ty = torch::relu(tx);
    ty.backward();

    REQUIRE_THAT(y.data(), WithinAbs(ty.data().item<double>(), ABS_TOLERANCE));
    REQUIRE_THAT(x.grad(), WithinAbs(tx.grad().item<double>(), ABS_TOLERANCE));
}

TEST_CASE("Karpathy sanity check", "[karpathy-sanity]") {
    Value x{-4.0};
    auto z = 2*x + 2 + x;
    auto q = z.relu() + z * x;
    auto h = (z * z).relu();
    auto y = h + q + q * x;
    y.backward();

    auto tx = torch::tensor(-4.0, torch::requires_grad().dtype(torch::kDouble));
    auto tz = 2*tx + 2 + tx;
    auto tq = tz.relu() + tz * tx;
    auto th = (tz * tz).relu();
    auto ty = th + tq + tq * tx;
    ty.backward();

    REQUIRE_THAT(y.data(), WithinAbs(ty.data().item<double>(), ABS_TOLERANCE));
    REQUIRE_THAT(x.grad(), WithinAbs(tx.grad().item<double>(), ABS_TOLERANCE));
}

TEST_CASE("Karpathy more ops", "[karpathy-more]") {

    Value a{-4.0};
    Value b{2.0};
    auto c = a + b;
    auto d = a * b + b.pow(3);
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + (b + a).relu();
    d = d + 3 * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f / 2.0;
    g = g + 10.0/f;
    g.backward();

    auto ta = torch::tensor(-4.0, torch::requires_grad());
    auto tb = torch::tensor(2.0, torch::requires_grad());
    auto tc = ta + tb;
    auto td = ta * tb + tb.pow(3);
    tc = tc + tc + 1;
    tc = tc + 1 + tc + (-ta);
    td = td + td * 2 + (tb + ta).relu();
    td = td + 3 * td + (tb - ta).relu();
    auto te = tc - td;
    auto tf = te.pow(2);
    auto tg = tf / 2.0;
    tg = tg + 10.0/tf;
    tg.backward();

    // Fwd pass
    REQUIRE_THAT(g.data(), WithinAbs(tg.data().item<double>(), ABS_TOLERANCE));
    // Backward pass
    CHECK_THAT(a.grad(), WithinAbs(ta.grad().item<double>(), ABS_TOLERANCE));
    CHECK_THAT(b.grad(), WithinAbs(tb.grad().item<double>(), ABS_TOLERANCE));

}

TEST_CASE("Single neuron matches libtorch", "[neuron]") {

    std::array<double, 4> w_b;
    std::ranges::generate(w_b, rand_uniform_m1_1);  // Populates the array by calling rand_uniform repeatedly

    Neuron n{w_b};
    auto out = n({1.0, 2.0, 3.0});
    //std::cout << n;
    out.backward();

    TorchNeuron tn(3);
    // Ugh manual spec of elements
    //tn.linear->weight = torch::tensor({{w_b[0], w_b[1], w_b[2]}});
    tn.linear->weight.set_data(torch::from_blob(w_b.data(), {1, 3}));   // Courtesy Grok
    // Why don't these work??
    //tn.linear->weight = torch::tensor(w_b | std::views::take(3) | std::ranges::to<std::vector>());
    //tn.linear->weight = torch::tensor(w_b | std::views::take(3) | std::ranges::to<std::array>());
    //tn.linear->weight = torch::tensor(std::to_array(w_b | std::views::take(3)));

    // set_data ensures that the initial requires_grad persists
    tn.linear->bias.set_data(torch::tensor({w_b[3]}));
    auto tout = tn(torch::tensor({1.0, 2.0, 3.0}));
    tout.backward();
    std::cout << tn;

    std::array<double, 4> torch_grads = {
        tn.linear->weight.grad()[0][0].item<double>(),
        tn.linear->weight.grad()[0][1].item<double>(),
        tn.linear->weight.grad()[0][2].item<double>(),
        tn.linear->bias.grad().item<double>()
    };

    auto mg_params = n.parameters();

    REQUIRE_THAT(out.data(), WithinAbs(tout.data().item<double>(), ABS_TOLERANCE));
    for (int i = 0; i < 4; ++i) {
        INFO("Parameter index: " << i
                                 << " | micrograd grad: " << mg_params[i].grad()
                                 << " | torch grad: "     << torch_grads[i]);

        REQUIRE_THAT(mg_params[i].grad(), WithinAbs(torch_grads[i], ABS_TOLERANCE));
    }
}

TEST_CASE("MLP matches libtorch", "[mlp]") {

    std::array<double, 41> w_b;
    std::ranges::generate(w_b, rand_uniform_m1_1);  // Populates the array by calling rand_uniform repeatedly

    MLP mlp{3, {4,4,1}};
    std::println("# of params = {}", std::ranges::distance(mlp.parameters()));

    auto out = mlp({2.0, 3.0, -1.0});
    std::cout << "Random init output: " << out << "\n";

    // Note, cannot do the following:
    // std::ranges::copy(w_b, std::ranges::begin(mlp.parameters()));
    // This is bc mlp.parameters() is a join_view<transform_view<...>>. It is not a mutable
    // container that you can write into with std::ranges::copy.

    // Note neat method of initializing each element of params with the
    // corresponding element of w_b:
    for (auto&& [v, init] : std::views::zip(mlp.parameters(), w_b)) {
        v.data() = init;
    }

    out = mlp({2.0, 3.0, -1.0});
    std::cout << "With init vals: " << out << "\n";

    TorchMLP tmlp{3, {4,4,1}};
    auto tout = tmlp(torch::tensor({{2.0, 3.0, -1.0}}));
    std::println("TorchMLP output = {:.3f}", tout.data().item<double>());

    // Cannot do a similar zip-loop for TMLP! This is because tmlp.parameters()
    // returns a std::vector<torch::Tensor> that contains all parameters in the
    // order they were registered:
    //      fc0.weight   → shape [4, 3]   → 12 elements
    //      fc0.bias     → shape [4]      →  4 elements
    //      fc1.weight   → shape [4, 4]   → 16 elements
    //          . . .
    // In the zip loop you are pairing the i-th parameter tensor with the i-th double.
    // That means:
    // first tensor (12 elements) → gets only 1 double → "expected 12, got 1"
    // second tensor (4 elements) → gets the next single double → "expected 4, got 1"
    // You cannot zip 1-to-1 with the flat array — the tensors have different sizes.
    //
    // for (auto&& [p, init] : std::views::zip(tmlp.parameters(), w_b)) {
    //     p.set_data(torch::tensor({init}));  // braces around {init} needed to make param 1D
    // }

    // https://gemini.google.com/app/f1ce4c7f611085d1
    auto it = w_b.begin();
    for (auto& param : tmlp.parameters()) {
        int64_t n = param.numel();

        // 1. Create the tensor from the blob
        auto data = torch::from_blob(&*it, {n}, torch::kFloat64).clone();

        // 2. IMPORTANT: Reshape the 1D blob to the parameter's required dimensions
        // This turns a flat vector back into (out_features, in_features) for weights
        data = data.view(param.sizes());

        // 3. Update the parameter data
        // Use NoGradGuard to ensure this doesn't track history
        // copy_ vs set_data: While set_data works, using param.copy_(data) is often
        // preferred because it preserves the metadata and location of the original
        // parameter tensor while just updating the values.
        torch::NoGradGuard no_grad;
        param.copy_(data);

        std::advance(it, n);
    }

    tout = tmlp(torch::tensor({2.0, 3.0, -1.0}));
    std::println("TorchMLP output with init vals = {:.3f}", tout.data().item<double>());

}
