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

    double data         = 0.0;
    double grad         = 0.0;
    std::string op      = "";
    std::string label   = "";

    // Set of parents of this node
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

    // Private ctor for common init of data ref members.
    // This takes just the SP to the _Value obj and
    // inits the ref data members
    explicit Value(std::shared_ptr<_Value> sp)
        : _spv(std::move(sp)),
        data(_spv->data),
        grad(_spv->grad),
        op(_spv->op),
        label(_spv->label)
    {}

public:
    // Ref members MUST be initialized in initializer list! And therefore
    // the _Val ptr must also be init'd in the init list, before these
    // ref members!
    double&         data;
    double&         grad;
    std::string&    op;
    std::string&    label;

    // For Values constructed by themselves (e.g. Value a{2.0}),
    // the _spv->_prev must be an empty set.
    Value(double d) : Value(std::make_shared<_Value>(d)) {}

    Value(double d, std::string label) :
        Value (std::make_shared<_Value>(d, std::move(label)))
    {}

    // For Values constructed in operations, the _Value tuple will
    // always be passed as an rvalue, from a std::make_tuple in an operator
    Value(double d, _vtsp&& parents, std::string&& op) :
        Value (std::make_shared<_Value>(d, std::move(parents), std::move(op)))
    {}

    ~Value() {
        std::cout << std::format("Value({:.3f}, \"{}\") dtor\n", data, label);
    }

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << std::format("Value(data={:.3f}, grad={:.3f}, label=\"{}\")", v.data, v.grad, v.label);
        return os;
    }

    Value operator+(const Value& other) {
        return Value{data + other.data, std::make_tuple(_spv, other._spv), "+"};
    }

    Value operator*(const Value& other) {
        return Value{data * other.data, std::make_tuple(_spv, other._spv), "*"};
    }

    // Print parents
    void _prev() const {
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
