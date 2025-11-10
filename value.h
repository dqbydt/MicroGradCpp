#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <format>

struct Value {

    double data;

    Value(double d) : data{d} {}

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        // Curly braces are a ph for a value that will be inserted into
        // the string. Colon introduces format specifiers.
        // v.data is the arg whose value will be inserted into the ph in
        // the format string
        os << std::format("Value(data={:.3f})", v.data);
        return os;
    }

    Value operator+(const Value& other) {
        return Value{data + other.data};
    }

    Value operator*(const Value& other) {
        return Value{data * other.data};
    }


};

#endif // VALUE_H
