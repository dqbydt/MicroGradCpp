#include <iostream>
#include <format>

#include "value.h"

int main()
{
    Value a{2.0, "a1"};
    Value b{-3.0, "b"};
    Value c{10.0, "c"};

    Value e = a*b; e.label() = "e1";
    Value d = e+c; d.label() = "d1";
    Value f{-2.0, "f"};
    Value L = d*f; L.label() = 'L';  // this is the final output of the graph
    auto L1 = L.data();

    double h = 0.001;   // nudge

    a = Value{2.0 + h, "a2"}; // uncomment to test ∂L/∂a
    // Re-evaluate all nodes
    e = a*b; e.label() = "e2";
    d = e+c; d.label() = "d2";  // d.data() += h; // uncomment to test ∂L/∂d
    // f.data() += h;   // uncomment to test ∂L/∂f
    L = d*f; L.label() = "L2";
    auto L2 = L.data();

    auto grad = (L2-L1)/h;

    std::cout << std::format("L1 = {:.6f}, L2 = {:.6f}, grad = {:.6f}\n", L1, L2, grad);
    std::cout.flush();
    return 0;
}
