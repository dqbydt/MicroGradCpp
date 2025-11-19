#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <format>
#include <set>
#include <memory>
#include <tuple>

struct _Value;
using _vtsp = std::tuple<std::shared_ptr<_Value>, std::shared_ptr<_Value>>;

// Shadow version of Value that backs automatic stack vars on the heap
struct _Value {

    double data = 0.0;
    std::string op = "";

    // Set of parents maintained as a set of shared_ptrs. This is
    // because in an expr like d = a*b + c, the temporary a*b gets
    // destroyed after that statement executes, but we do want the
    // backing shadow value to persist on the heap as a parent of
    // d.
    std::set<std::shared_ptr<_Value>> _prev;

    _Value(double d) : data{d} {}   // _prev default-init'd to empty set

    _Value(double d, const _vtsp& parents, std::string&& op) : data{d}, op{std::move(op)} {
        auto const& [p1, p2] = parents;
        _prev = {p1, p2}; // Copies SPs, increases ref count of both
    }

    ~_Value() {
        std::cout << std::format("_Value({:.3f}, \"{}\") dtor\n", data, op);
    }

};

class Value {
    // Member vars are initialized in the order they are declared.
    // The shadow ptr must be declared before the data ref so that
    // the _Value ctor runs before we try to init data.
private:
    std::shared_ptr<_Value> _spv;

public:
    // Ref members MUST be initialized in initializer list! And therefore
    // the _Val ptr must also be init'd in the init list, before these
    // ref members!
    double& data;
    std::string& op;

    // For Values constructed by themselves (e.g. Value a{2.0}),
    // the _spv->_prev must be an empty set.
    Value(double d) : _spv {std::make_shared<_Value>(d)}, data {_spv->data}, op {_spv->op} {}

    // For Values constructed in operations, the _Value tuple will
    // always be passed as an rvalue, from a std::make_tuple in an operator
    Value(double d, _vtsp&& parents, std::string&& op) :
        _spv {std::make_shared<_Value>(d, parents, std::move(op))},
        data {_spv->data},
        op {_spv->op}
    { }

    ~Value() {
        std::cout << std::format("Value({:.3f}) dtor\n", data);
    }

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        // Curly braces are a ph for a value that will be inserted into
        // the string. Colon introduces format specifiers.
        // v.data is the arg whose value will be inserted into the ph in
        // the format string
        os << std::format("Value(data={:.3f})", v.data);
        return os;
    }

    // SP ref counts:
    // --------------
    // 1. the std::make_tuple makes a copy of the SPs, incrementing their ref cnts
    // 2. The temp tuple is received as an rval ref in the Value ctor, no copies
    // 3. Value ctor invokes _Value ctor that takes the tuple by const ref. No copies.
    // 4. In the _Value ctor the tuple is unpacked into const refs. No copies.
    // 5. Finally the const refs are added to the set, which copies the SPs and
    //    increments ref counts.
    // 6. Temp tuple from #1 is destroyed, decrementing ref counts.
    // Net effect is that the ref counts on the parent SPs are incremented by 1.
    Value operator+(const Value& other) {
        return Value{data + other.data, std::make_tuple(_spv, other._spv), "+"};
    }

    Value operator*(const Value& other) {
        return Value{data * other.data, std::make_tuple(_spv, other._spv), "*"};
    }

    // 0. Missing ref in range-for with non trivial type warning?
    // 1. Why doesn't the std::format work?
    // 2. Rewrite as a view that yields the next elt from the set?
    void _prev() const {
        // If we iterate like so: for (auto p : _spv->_prev) then
        // p gets a copy of each element in the set -- which means we are
        // creating a new SP and incrementing the refcount needlessly.
        // Instead iterate with a const ref to the SP.
        //for (auto p : _spv->_prev) {
        for (const auto& p : _spv->_prev) {
            //std::cout << std::format("Value(data={.3f})", p->data);
            std::cout << p->data << std::endl;
        }
    }


};

#endif // VALUE_H
