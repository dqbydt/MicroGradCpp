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
    std::string label = "";

    // Set of parents maintained as a set of shared_ptrs. This is
    // because in an expr like d = a*b + c, the temporary a*b gets
    // destroyed after that statement executes, but we do want the
    // backing shadow value to persist on the heap as a parent of
    // d.
    std::set<std::shared_ptr<_Value>> _prev;

    _Value(double d) : data{d} {}   // _prev default-init'd to empty set

    _Value(double d, std::string&& label) : data{d}, label{std::move(label)} {}

    _Value(double d, _vtsp&& parents, std::string&& op) : data{d}, op{std::move(op)} {
        auto [p1, p2] = std::move(parents); // p1 and p2 move-ctor'd from the SPs in the tuple
        _prev = {std::move(p1), std::move(p2)}; // shared_ptrs moved all the way from the temp
    }

    ~_Value() {
        std::cout << std::format("_Value({:.3f}, \"{}\") dtor\n", data, label);
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
    std::string& label;

    // For Values constructed by themselves (e.g. Value a{2.0}),
    // the _spv->_prev must be an empty set.
    Value(double d) :
        _spv {std::make_shared<_Value>(d)},
        data {_spv->data},
        op {_spv->op},
        label {_spv->label}
    {}

    Value(double d, std::string label) :
        _spv {std::make_shared<_Value>(d, std::move(label))},
        data {_spv->data},
        op {_spv->op},
        label {_spv->label}
    {}

    // For Values constructed in operations, the _Value tuple will
    // always be passed as an rvalue, from a std::make_tuple in an operator
    Value(double d, _vtsp&& parents, std::string&& op) :
        _spv {std::make_shared<_Value>(d, std::move(parents), std::move(op))},
        data {_spv->data},
        op {_spv->op},
        label {_spv->label}
    { }

    ~Value() {
        std::cout << std::format("Value({:.3f}, \"{}\") dtor\n", data, label);
    }

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        // Curly braces are a ph for a value that will be inserted into
        // the string. Colon introduces format specifiers.
        // v.data is the arg whose value will be inserted into the ph in
        // the format string
        os << std::format("Value(data={:.3f}, label=\"{}\")", v.data, v.label);
        return os;
    }

    // SP ref counts:
    // --------------
    // 1. the std::make_tuple makes a copy of the SPs, incrementing their ref cnts
    // 2. The temp tuple is received as an rval ref in the Value ctor, no copies
    // 3. Value ctor invokes _Value ctor that moves the sp's out from the tuple.
    //    Ref cnt stays the same. SPs in the tuple are now nullptrs.
    // 4. In the _Value ctor the tuple is unpacked into rval refs. No copies..
    // 5. Finally the rval refs are moved to the set. The only ref cnt increment
    //    is the one from #1 above.
    // 6. Temp tuple from #1 is destroyed. Moved-from SPs getting destroyed does
    //    nothing.
    // Net effect is that the ref counts on the parent SPs are incremented by 1.
    Value operator+(const Value& other) {
        return Value{data + other.data, std::make_tuple(_spv, other._spv), "+"};
    }

    Value operator*(const Value& other) {
        return Value{data * other.data, std::make_tuple(_spv, other._spv), "*"};
    }

    // Print parents
    void _prev() const {
        // If we iterate like so: for (auto p : _spv->_prev) then
        // p gets a copy of each element in the set -- which means we are
        // creating a new SP and incrementing the refcount needlessly.
        // Instead iterate with a const ref to the SP.
        for (const auto& p : _spv->_prev) {
            // The following doesn't work bc std::format is consteval, can be called
            // only from non-const fns. Will be fixed in C++26.
            //std::cout << std::format("Parent: Value(data={.3f})", p->data);
            // Workaround for now: (needs arg index 0:, note!)
            std::cout << std::vformat("Parent: Value(data={0:.3f})\n", std::make_format_args(p->data));
        }
    }


};

#endif // VALUE_H
