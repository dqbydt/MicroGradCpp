#ifndef MISC_H
#define MISC_H

namespace misc {

    template <class T>
    inline T sqr(const T& x) { return x*x; }

    // Insert space before +ve vals for pretty-printing
    constexpr const char* sgnspc(double d) { return (d >= 0)? " ":""; };

}


#endif // MISC_H
