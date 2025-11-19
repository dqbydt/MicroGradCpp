#include <iostream>

#include "value.h"

int main()
{
    Value a{2.0};
    Value b{-3.0};
    Value c{10.0};

    Value d = a*b+c;

    std::cout << d << std::endl;
    d._prev();
    return 0;
}
