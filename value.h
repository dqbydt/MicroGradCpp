#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <format>
#include <mutex>
#include <set>
#include <vector>
#include <memory>
#include <tuple>
#include <cmath>
#include <functional>

struct _Value;
using _vtsp = std::tuple<std::shared_ptr<_Value>, std::shared_ptr<_Value>>;

// Shadow version of Value that backs automatic stack vars on the heap
struct _Value {

    double data         = 0.0;
    double grad         = 0.0;
    std::string op      = "";
    std::string label   = "";

    bool visited = false;

    // Empty lambda by default (e.g. for a leaf node, there is nothing
    // to backprop. But in Value::add() for e.g. we are adding this and
    // other. We want to propagate the incoming gradient from "out" to
    // "this" and "other".
    std::function<void()> _backward = [](){};

    // Set of parents of this node
    std::set<std::shared_ptr<_Value>> _prev;

    _Value(double d) : data{d} {}   // _prev default-init'd to empty set

    _Value(double d, std::string&& label) : data{d}, label{std::move(label)} {}

    _Value(double d, _vtsp&& parents, std::string&& op) : data{d}, op{std::move(op)} {
        auto [p1, p2] = std::move(parents); // p1 and p2 move-ctor'd from the SPs in the tuple
        _prev = {std::move(p1), std::move(p2)}; // shared_ptrs moved all the way from the temp
    }

    ~_Value() {
        //std::cout << std::format("_Value({:.3f}, \"{}\") dtor\n", data, label);
    }

};

class Value {
private:
    std::shared_ptr<_Value> _spv;

    // Only used internally to set the backward lambda during expression buildup
    // NOTE: return type must be auto&, not auto! Otherwise it returns a copy,
    // not the actual stored lambda! So setting it to a specifc fn does nothing;
    // the _spv._backward member isn't changed.
    auto& _backward()   const { return _spv->_backward; }   // Lambda to backprop grads

    // We strictly just want to access the _Value nodes without
    // impacting the ownership (and incurring the atomic refcount inc/dec).
    // So we use raw pointers here.
    static inline std::once_flag btof{}; // for build_topo
    static inline std::vector<_Value*> topo;

    void build_topo(_Value* _pv) const {
        if (_pv->visited) return;
        _pv->visited = true;
        for (const auto& spp : _pv->_prev) {
            build_topo(spp.get());
        }
        Value::topo.push_back(_pv);
    }

public:
    // These were previously ref members initialized to the
    // corresponding members of the backing object. Have had
    // to abandon that approach since it breaks in move-assignment:
    // the underlying _spv is changed, but refs continue to refer
    // to old, deallocated _spv!
    double&         data()  const   { return _spv->data;    }
    double&         grad()  const   { return _spv->grad;    }
    std::string&    op()    const   { return _spv->op;      }
    std::string&    label() const   { return _spv->label;   }

    // For Values constructed by themselves (e.g. Value a{2.0}),
    // the _spv->_prev must be an empty set.
    Value(double d) : _spv{std::make_shared<_Value>(d)} {}

    Value(double d, std::string label) :
        _spv {std::make_shared<_Value>(d, std::move(label))}
    {}

    // For Values constructed in operations, the _Value tuple will
    // always be passed as an rvalue, from a std::make_tuple in an operator
    Value(double d, _vtsp&& parents, std::string&& op) :
        _spv {std::make_shared<_Value>(d, std::move(parents), std::move(op))}
    {}

    // Prevent copy - otherwise would corrupt expr graph
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    // Moves allowed, to enable reassigment of a node to a new value
    Value(Value&&) noexcept = default;
    Value& operator=(Value&& other) noexcept = default;

    // Getters valid only if dying object has not been moved from! Else
    // the _spv is a nullptr!
    ~Value() {
        if (_spv) {
            //std::cout << std::format("Value({:.3f}, \"{}\") dtor\n", data(), label());
        } else {
            std::cout << std::format("Moved-from-Value dtor\n");
        }
        std::cout.flush();
    }


    // Actually calls the _backward lambda that was set up below during the
    // expression build.
    void backward() {

        // Call once per graph, only on the output node
        std::call_once(Value::btof, [this](){ Value::topo.clear(); build_topo(_spv.get());} );

        std::cout << "Topo sorted graph:\n";
        for (auto& _pv : Value::topo) {
            std::cout << std::format("_Value(data={:.3f}, grad={:.3f}, label=\"{}\")\n",
                                     _pv->data, _pv->grad, _pv->label);
        }

        _spv->_backward();
    }

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << std::format("Value(data={:.3f}, grad={:.3f}, label=\"{}\")\n", v.data(), v.grad(), v.label());
        return os;
    }

    Value operator+(const Value& other) {
        auto out = Value{data() + other.data(), std::make_tuple(_spv, other._spv), "+"};

        // Note: must capture SPs to backing _Value objects that live on the heap!
        // Anything else would result in dangling ptrs/refs!
        out._backward() = [_p1 = _spv,
                          _p2 = other._spv,
                          _out = out._spv](){
            _p1->grad = 1.0 * _out->grad;
            _p2->grad = 1.0 * _out->grad;
        };
        return out;
    }

    Value operator*(const Value& other) {
        auto out = Value{data() * other.data(), std::make_tuple(_spv, other._spv), "*"};
        out._backward() = [_p1 = _spv,
                          _p2 = other._spv,
                          _out = out._spv](){
            // grad on one side = other side data * incoming grad
            _p1->grad = _p2->data * _out->grad;
            _p2->grad = _p1->data * _out->grad;
        };

        return out;
    }

    // tanh squashing fn for output of neuron
    Value tanh() {
        auto x = data();
        auto th = (std::exp(2*x) - 1)/(std::exp(2*x) + 1);
        // Need to repeat parent twice because Value ctor needs _vtsp which
        // is a tuple of two _Value SPs. Because the SPs are inserted into a set,
        // the repetition is benign.
        auto out = Value{th, std::make_tuple(_spv, _spv), "tanh"};
        // If L = tanh(n), then ∂L/∂n = (1 - tanh(n)**2)
        out._backward() = [_p1 = _spv, th,
                          _out = out._spv](){
            _p1->grad = (1 - th*th) * _out->grad;
        };
        return out;
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
