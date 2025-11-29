// tests/test_value.cpp
#define CATCH_CONFIG_COLOUR_ANSI
#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_session.hpp>

#include <torch/torch.h>
#include "../value.h"

using Catch::Matchers::WithinAbs;
constexpr double ABS_TOLERANCE = 1e-6;

// === FORCE ALL LIBTORCH TENSORS TO DOUBLE (match your Value class) ===
[[maybe_unused]] static const auto _force_double = []{
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kDouble));
    std::cout << "Forcing PyTorch default datatype to double\n";
    return 0;
}();

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
    g = g + f/10.0;
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
    tg = tg + tf/10.0;
    tg.backward();

    // Fwd pass
    REQUIRE_THAT(g.data(), WithinAbs(tg.data().item<double>(), ABS_TOLERANCE));
    // Backward pass
    CHECK_THAT(a.grad(), WithinAbs(ta.grad().item<double>(), ABS_TOLERANCE));
    CHECK_THAT(b.grad(), WithinAbs(tb.grad().item<double>(), ABS_TOLERANCE));

}
