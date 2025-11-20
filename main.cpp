#include <iostream>

#include "value.h"

int main()
{
    Value a{2.0, "a"};
    Value b{-3.0, "b"};
    Value c{10.0, "c"};

    Value e = a*b; e.label = 'e';
    Value d = e+c; d.label = 'd';
    Value f{-2.0, "f"};
    Value L = d*f; L.label = 'L';  // this is the final output of the graph

    std::cout << L << std::endl;
    L._prev();
    return 0;
}
