// tests/test_value.cpp
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_session.hpp>

#include <random>
#include <numeric>

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

// Swizzles PyTorch-format Layer-major input data into a flat neuron-major
// order to inject into class MLP
auto make_neuron_major(const MLP& mlp, std::span<const double> w_b)
{
    // Output swizzled vec
    std::vector<double> wb_swizzled;
    wb_swizzled.reserve(mlp.num_params());

    // Keep shrinking view of init data on every iteration
    auto remaining_data = w_b;

    for (auto&& [i,l] : mlp.layers | std::views::enumerate) {

        const auto num_lwts     = l.nout() * l.nin();
        const auto num_lbiases  = l.nout();
        const auto num_lparams  = num_lwts + num_lbiases;

        // 1. Slice out this layer's data from the front
        auto lwts       = remaining_data | std::views::take(num_lwts);
        auto lbiases    = remaining_data | std::views::drop(num_lwts) | std::views::take(num_lbiases);

        // 2. Swizzle: Zip chunks of weights with their corresponding bias
        // Using zip allows us to avoid manual indexing into lbiases
        auto neuron_data = std::views::zip(
                lwts | std::views::chunk(l.nin()),
                lbiases
            );

        for (auto&& [wts_chunk, bias] : neuron_data) {
            wb_swizzled.append_range(wts_chunk);
            wb_swizzled.push_back(bias);
        }

        // 3. Advance the "window" to the next layer
        remaining_data = remaining_data | std::views::drop(num_lparams);
    }

    return wb_swizzled;
}

// Returns MLP param gradients in layer-major order for direct comparison
// with PyTorch
auto make_layer_major(const MLP& mlp)
{
    // Output swizzled vec
    std::vector<double> lm_params;
    lm_params.reserve(mlp.num_params());

    // Linear neuron-major view of param data. Gemini suggested
    // keeping this as a view (no materialization) but that results in the
    // same strange segfault seen previously. HAVE to materialize to
    // work around.
    auto mg_data_vec = mlp.parameters()
                        | std::views::transform([](const Value& v) { return v.grad(); })
                        | std::ranges::to<std::vector>();

    // We use a subrange to "consume" the view as we iterate through layers
    // This MUST be a subrange, note! views::all returns a ref_view which
    // cannot be assigned to itself after a take.
    // XXX     auto remaining_data = std::views::all(mg_data_vec);
    auto remaining_data = std::ranges::subrange(mg_data_vec);

    for (auto&& [i,l] : mlp.layers | std::views::enumerate) {

        const auto num_lwts       = l.nout() * l.nin();
        const auto num_lbiases    = l.nout();
        const auto num_lparams    = num_lwts + num_lbiases;

        // Slice out exactly what this layer needs
        auto lparams = remaining_data | std::views::take(num_lparams);

        // 2. Chunk by (nin + 1) to isolate each neuron's [weights..., bias]
        auto neurons = lparams | std::views::chunk(l.nin() + 1);

        // 3. PyTorch Layout: All weights for the layer, THEN all biases
        // First pass: Append all weight chunks
        for (auto neuron : neurons) {
            lm_params.append_range(neuron | std::views::take(l.nin()));
        }

        // Second pass: Append all bias elements
        for (auto neuron : neurons) {
            lm_params.append_range(neuron | std::views::drop(l.nin()));
        }

        // 4. Advance the "window" for the next layer
        remaining_data = remaining_data | std::views::drop(num_lparams);
    }

    return lm_params;
}

TEST_CASE("MLP matches libtorch", "[mlp]") {

    std::array<double, 41> w_b;
    std::ranges::generate(w_b, rand_uniform_m1_1);  // Populates the array by calling rand_uniform repeatedly

    MLP mlp{3, {4,4,1}};
    auto num_params = mlp.num_params();
    std::println("# of params = {}", num_params);

    auto out = mlp({2.0, 3.0, -1.0});
    std::cout << "Random init output: " << out << "\n";

    // For testing: populate array with linear range from 0
    // Needs header <numeric>, note
    //std::ranges::iota(w_b, 0);  // Comment out to load real values!

    // Inject params (swizzled to neuron-major order) into MLP
    auto wb_swizzled = make_neuron_major(mlp, w_b);

    // Note, cannot do the following:
    // std::ranges::copy(w_b, std::ranges::begin(mlp.parameters()));
    // This is bc mlp.parameters() is a join_view<transform_view<...>>. It is not a mutable
    // container that you can write into with std::ranges::copy.

    // Note neat method of initializing each element of params with the
    // corresponding element of wb_swizzled:
    for (auto&& [v, init] : std::views::zip(mlp.parameters(), wb_swizzled)) {
        v.data() = init;
    }

    // Run fwd pass on a test input
    out = mlp({2.0, 3.0, -1.0});

    // Run backward pass; collect gradients in layer-major order
    out[0].backward();
    auto mg_grads = make_layer_major(mlp);

    TorchMLP tmlp{3, {4,4,1}};

    // Cannot do a similar zip-loop for TMLP to inject params:
    //      for (auto&& [p, init] : std::views::zip(tmlp.parameters(), w_b)) {
    //          p.set_data(torch::tensor({init}));
    //      }
    // This is because tmlp.parameters()
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

    // Inject params into PT MLP
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

    // Forward pass
    auto tout = tmlp(torch::tensor({2.0, 3.0, -1.0}));
    // Backward pass
    tout.backward();

    // Collect gradients
    std::vector<double> torch_grads;
    // PyTorch provides tmlp.parameters() which returns all weights/biases
    // in the exact order they were registered in the constructor.
    // Order: [fc0.weight, fc0.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias]
    for (const auto& param : tmlp.parameters()) {
        if (param.grad().defined()) {
            // 1. Ensure the gradient is on CPU and is a double
            // (Your model is kFloat64, but we call .to() just to be safe)
            torch::Tensor grad_cpu = param.grad().detach().to(torch::kCPU).to(torch::kFloat64);

            // 2. Get a pointer to the raw data
            const double* data_ptr = grad_cpu.data_ptr<double>();

            // 3. Append the entire tensor's worth of data to our vector
            torch_grads.insert(torch_grads.end(), data_ptr, data_ptr + grad_cpu.numel());
        } else {
            // If a gradient is undefined (e.g. backward wasn't called),
            // fill with zeros to maintain vector size alignment.
            std::vector<double> zeros(param.numel(), 0.0);
            torch_grads.insert(torch_grads.end(), zeros.begin(), zeros.end());
        }
    }

    // Compare data output:
    REQUIRE_THAT(out[0].data(), WithinAbs(tout.data().item<double>(), ABS_TOLERANCE));

    // Compare gradients:
    for (int i = 0; i < num_params; ++i) {
        INFO("Parameter index: " << i
                                 << " | micrograd grad: " << mg_grads[i]
                                 << " | torch grad: "     << torch_grads[i]);

        REQUIRE_THAT(mg_grads[i], WithinAbs(torch_grads[i], ABS_TOLERANCE));
    }


}

