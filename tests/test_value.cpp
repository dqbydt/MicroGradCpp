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

// --- Helper function to define the network architecture ---
// Since the swizzling needs the layer sizes, we need this structure to
// correctly map the dimensions.
struct LayerInfo {
    int64_t in_features;
    int64_t out_features;
    int64_t total_params;
};

// --- Function to perform the swizzling ---
std::vector<double> create_pytorch_flat_params(
    std::span<const double> w_b_custom_flat, // Your neuron-major flat array
    std::initializer_list<size_t> layer_sizes // {4, 4, 1}
    ) {
    // Determine the full architecture for iteration: 3 -> 4 -> 4 -> 1
    std::vector<LayerInfo> architecture;
    int64_t current_in = 3; // nin = 3, based on your tmlp{3, {4,4,1}}

    // Build the architecture info
    for (size_t out_size : layer_sizes) {
        int64_t current_out = static_cast<int64_t>(out_size);
        int64_t total = current_out * (current_in + 1);
        architecture.push_back({current_in, current_out, total});
        current_in = current_out;
    }

    // Verify the total size
    int64_t expected_total_size = 0;
    for (const auto& info : architecture) {
        expected_total_size += info.total_params;
    }
    if (expected_total_size != w_b_custom_flat.size()) {
        std::println("Error: Architecture size {} does not match input array size {}.",
                     expected_total_size, w_b_custom_flat.size());
        return {}; // Return empty vector on mismatch
    }

    // --- Swizzling Logic ---
    std::vector<double> w_b_pt; // The final PyTorch-ordered flat array
    auto custom_it = w_b_custom_flat.begin();

    for (const auto& layer : architecture) {
        std::vector<double> pt_flat_weights;
        std::vector<double> pt_flat_biases;

        std::println("Swizzling Layer: {} -> {}", layer.in_features, layer.out_features);

        // 1. Unpack from Custom (Neuron-major) Order
        // Custom order: [ (W_N1), B_N1, (W_N2), B_N2, ... ]
        for (int64_t n = 0; n < layer.out_features; ++n) {

            // A. Weights for Neuron N (in_features values)
            auto weights_range = std::ranges::subrange(custom_it, custom_it + layer.in_features);
            std::ranges::copy(weights_range, std::back_inserter(pt_flat_weights));
            custom_it += layer.in_features;

            // B. Bias for Neuron N (1 value)
            pt_flat_biases.push_back(*custom_it);
            custom_it++;
        }

        // 2. Repack into PyTorch (Layer-major) Order
        // PyTorch order for a layer: [ Weight Matrix (flattened) ], [ Bias Vector ]

        // Append all weights first (Weight Matrix: [out, in] flattened row-major)
        std::ranges::copy(pt_flat_weights, std::back_inserter(w_b_pt));

        // Append all biases next (Bias Vector: [out] )
        std::ranges::copy(pt_flat_biases, std::back_inserter(w_b_pt));
    }

    return w_b_pt;
}


// Function to read gradients from PyTorch parameters and flatten them
// into your custom neuron-major order, using a stack-allocated MLP.
// Function to read gradients from PyTorch parameters and flatten them
// into your custom neuron-major order, using a stack-allocated MLP.
std::vector<double> read_gradients_in_custom_order(
    TorchMLP& tmlp // Stack-allocated reference
    ) {
    // 1. Define the network architecture (same as before)
    std::vector<LayerInfo> architecture;
    int64_t current_in = 3;
    std::vector<size_t> layer_sizes = {4, 4, 1};

    for (size_t out_size : layer_sizes) {
        architecture.push_back({current_in, static_cast<int64_t>(out_size)});
        current_in = static_cast<int64_t>(out_size);
    }

    std::vector<double> grad_b_custom_flat;
    size_t layer_architecture_index = 0; // Tracks the index in the architecture vector

    // 2. Iterate over the named children of the Sequential container (net)
    // The keys will be "0", "1", "2", ...
    for (const auto& module_pair : tmlp.net->named_children()) {
        const std::string& index_str = module_pair.key();
        int index = std::stoi(index_str); // Convert key ("0", "1", etc.) to integer

        // --- Filter for Linear Layers ---
        // Linear layers are always at even indices (0, 2, 4, ...) in your Sequential
        if (index % 2 != 0) {
            continue; // Skip Tanh layers
        }

        // Ensure we haven't run out of architecture info (safety check)
        if (layer_architecture_index >= architecture.size()) {
            std::println("Error: Mismatch between Sequential size and architecture vector.");
            break;
        }

        const LayerInfo& layer = architecture[layer_architecture_index];
        int64_t out_features = layer.out_features;
        int64_t in_features  = layer.in_features;

        // Cast to LinearImpl to access weight and bias members
        // The value() is a shared_ptr<Module>, so we use .get()->as<T>()
        auto linear_ptr = module_pair.value().get()->as<torch::nn::LinearImpl>();

        if (!linear_ptr) {
            // This shouldn't happen for an even index, but check anyway
            std::println("Error: Module at index {} is not a Linear layer.", index);
            layer_architecture_index++;
            continue;
        }

        // Get the gradient Tensors (PyTorch Order: Weight, then Bias)
        torch::Tensor grad_weight = linear_ptr->weight.grad().to(torch::kCPU);
        torch::Tensor grad_bias   = linear_ptr->bias.grad().to(torch::kCPU);

        // Check if gradients exist before accessing data_ptr
        if (!grad_weight.defined() || !grad_bias.defined()) {
            std::println("Warning: Gradients not defined for layer {}. Did you call loss.backward()?", index);
            layer_architecture_index++;
            continue;
        }

        const double* w_grad_ptr = grad_weight.data_ptr<double>();
        const double* b_grad_ptr = grad_bias.data_ptr<double>();

        std::println("Reading Gradients from Sequential Index {} (Layer {}): {} -> {}",
                     index, layer_architecture_index + 1, in_features, out_features);

        // 3. Swizzle: Map PyTorch Order to Custom (Neuron-major) Order (same logic as before)
        for (int64_t n = 0; n < out_features; ++n) {

            // A. Read Weights Gradient for Neuron N
            size_t start_index = static_cast<size_t>(n * in_features);
            auto w_grad_span = std::span(w_grad_ptr + start_index, in_features);
            std::ranges::copy(w_grad_span, std::back_inserter(grad_b_custom_flat));

            // B. Read Bias Gradient for Neuron N
            grad_b_custom_flat.push_back(b_grad_ptr[n]);
        }

        // Advance to the next layer in the architecture vector
        layer_architecture_index++;
    }

    return grad_b_custom_flat;
}

TEST_CASE("MLP matches libtorch", "[mlp]") {

    std::array<double, 41> w_b = {
        0.4567586743459362,    -0.7891836110392254,   -0.44887830862390277, -0.8186330725257585,
        -0.22593813623021575,  -0.16432779500893546,   0.5937431020406805,   0.21789175846576891,
        -0.8215172707414129,    0.4945993482649922,   -0.10895162052883633,  0.2742519517612745,
        0.43529260458437236,    0.070438875589103,    -0.11732098445982286,  0.46756788860100507,
        0.776804984571531,      0.48655569212333094,  -0.8340129869412527,  -0.9612742574115734,
        0.2393245969601474,    -0.8521972230899275,   -0.6128703131175552,  -0.529124885030632,
        -0.8480265962031999,   -0.7814084981684273,   -0.8805687514921439,   0.32062710266821215,
        -0.7723148731011704,    0.7130170047171043,   -0.36211426771666577, -0.48555634644226164,
        0.10057022512856806,    0.7204828842048876,   -0.25174010184200957,  0.8549771745895092,
        0.6281135149040777,    -0.6472284682766034,    0.6355787586092168,  -0.7609336029822684,
        -0.3109609348451259,};
    //std::ranges::generate(w_b, rand_uniform_m1_1);  // Populates the array by calling rand_uniform repeatedly

    MLP mlp{3, {4,4,1}};
    auto num_params = std::ranges::distance(mlp.parameters());
    std::println("# of params = {}", num_params);

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

    std::println("Params:");
    std::ranges::for_each(mlp.layers[0].neurons[0].parameters(), [](const auto& v){ std::println("{}", v.data()); });
    std::println();

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

    // 2. Swizzle the custom parameters into the PyTorch format
    std::vector<double> w_b_pt = create_pytorch_flat_params(
        std::span<const double>(w_b),
        {4, 4, 1} // Must match the layers used in TorchMLP
    );

    // Inject swizzled params into PT MLP
    // https://gemini.google.com/app/f1ce4c7f611085d1
    auto it = w_b_pt.begin();
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

    std::println("\n--- PyTorch Model Parameters ---");
    for (const auto& pair : tmlp.named_parameters()) {
        const std::string& name = pair.key();
        const torch::Tensor& param = pair.value();

        // Convert the Tensor to a std::vector<double> for easy printing
        // Note: data_ptr<double>() gives a raw pointer to the underlying data
        const double* data_ptr = param.data().to(torch::kCPU).data_ptr<double>();

        // Create a view (or copy) of the data using C++23 ranges
        auto data_view = std::span(data_ptr, param.numel());

        // Format the sizes for printing
        std::stringstream sizes_ss;
        sizes_ss << param.sizes();

        std::println("{}: {}", name, sizes_ss.str());

        // Print up to the first 10 elements, then a summary
        size_t count = std::min((size_t)10, data_view.size());
        std::print("  [");
        for (size_t i = 0; i < count; ++i) {
            std::print("{}{}", data_view[i], (i < count - 1 ? ", " : ""));
        }
        if (data_view.size() > count) {
            std::print(", ... ({} more values)", data_view.size() - count);
        }
        std::println("]");
    }
    std::println("------------------------\n");


    // Input tensor (requires .unsqueeze(0) to become a batch of size 1)
    torch::Tensor input = torch::tensor({2.0, 3.0, -1.0}, torch::kFloat64).unsqueeze(0);
    std::cout << "Input Tensor (x):\n" << input << "\n";

    torch::Tensor current_output = input;
    size_t layer_idx = 0;

    std::println("\n--- Layer-wise Outputs ---");
    for (auto& module : *tmlp.net) {

        // Forward pass for the current module
        current_output = module.forward(current_output);

        std::string module_type;
        if (layer_idx % 2 == 0) {
            module_type = "Linear (pre-activation)";
            std::println("\n--- Layer {} ---", layer_idx / 2 + 1);
        } else {
            module_type = "Tanh (post-activation)";
        }

        // Convert the Tensor to a std::span<const double> for printing
        const double* data_ptr = current_output.to(torch::kCPU).data_ptr<double>();
        auto data_view = std::span(data_ptr, current_output.numel());

        std::println("{}: (size {})", module_type, current_output.numel());
        std::print("  [");
        for (size_t i = 0; i < data_view.size(); ++i) {
            std::print("{}{}", data_view[i], (i < data_view.size() - 1 ? ", " : ""));
        }
        std::println("]");

        layer_idx++;
    }
    std::println("--------------------------");

    //tout = current_output; // Final output
    tout.backward();
    auto torch_grads = read_gradients_in_custom_order(tmlp);
    std::println("{} Pytorch gradients:", torch_grads.size());
    for (auto& g : torch_grads) {
        std::print("{:.3f} ", g);
    }
    std::println();

    out[0].backward();
    for (auto&& v : mlp.parameters()) {
        std::print("{:.3f} ", v.grad());
    }
    std::println();

    REQUIRE_THAT(out[0].data(), WithinAbs(tout.data().item<double>(), ABS_TOLERANCE));

    auto mg_params_stable = mlp.parameters()
                            | std::views::transform([](const Value& v) { return &v; })
                            | std::ranges::to<std::vector<const Value*>>();

    //auto itp = mlp.parameters().begin();

    for (int i = 0; i < num_params; ++i) {
        //const auto& mg_param = *itp;
        const Value& mg_param = *mg_params_stable[i];
        double mg_grad = mg_param.grad();
        INFO("Parameter index: " << i
                                 << " | micrograd grad: " << mg_grad
                                 << " | torch grad: "     << torch_grads[i]);

        //std::print("{} ", mg_grad);
        //REQUIRE_THAT(mg_param.grad(), WithinAbs(torch_grads[i], ABS_TOLERANCE));
        REQUIRE_THAT(mg_grad, WithinAbs(torch_grads[i], ABS_TOLERANCE));
    }


}
