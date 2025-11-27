#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <format>
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
    // Caches the topo sort order the first time backward() is run on this node.
    // Subsequent backward() passes don't need to run the same sort over and over.
    // Note, we use raw ptrs since we strictly just want to access the _Value
    // nodes without impacting the ownership (and incurring the atomic refcount
    // inc/dec).
    std::vector<const _Value*> topo_cache;

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

    // Insert space before +ve vals for pretty-printing
    static constexpr const char* sgnspc(double d) { return (d >= 0)? " ":""; };

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

        // Alias names for semantic convenience
        auto* root = _spv.get();
        auto& topo_cache = root->topo_cache;

        // Build topo order if cache is empty
        if (topo_cache.empty()) {
            std::vector<_Value*> visited;   // Store visited nodes to reset flags later

            // Note: auto return type does not work for recursive calls; need to
            // fully specify the type of the lambda.
            // auto build_topo = [&](_Value* _pv) -> void {
            std::function<void(_Value*)> build_topo = [&](_Value* _pv) {
                if (_pv->visited) return;

                _pv->visited = true;
                visited.push_back(_pv);

                for (const auto& spp : _pv->_prev) {
                    build_topo(spp.get());
                }
                topo_cache.push_back(_pv);
            };

            // Build topo graph starting at this node
            build_topo(root);

            std::cout << "Topo sorted graph:\n";
            for (auto& _pv : topo_cache) {
                std::cout << std::format("_Value(data={}{:.3f}, grad={}{:.3f}, label=\"{}\")\n",
                                         sgnspc(_pv->data), _pv->data,
                                         sgnspc(_pv->grad), _pv->grad, _pv->label);
            }
            std::cout << "------------------\n";

            // Reset visited nodes - this enables re-computation of topo sort on a diff node.
            // Note special case auto* "deduce-as-pointer" syntax in range-for loop!
            for (auto* _pv : visited) _pv->visited = false;
        }

        // Do the backward pass on the topo-sorted list of nodes
        grad() = 1.0;
        for (auto it = topo_cache.rbegin(); it != topo_cache.rend(); it++) {
            (*it)->_backward();
        }

    }

    // operator<< overload for printing
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        auto d_sgnspc = v.sgnspc(v.data());   auto g_sgnspc = v.sgnspc(v.grad());
        os << std::format("Value(data={}{:.3f}, grad={}{:.3f}, label=\"{}\")\n",
                          d_sgnspc, v.data(), g_sgnspc, v.grad(), v.label());
        return os;
    }

    Value operator+(const Value& other) const {
        auto out = Value{data() + other.data(), std::make_tuple(_spv, other._spv), "+"};

        // Note: must capture SPs to backing _Value objects that live on the heap!
        // Anything else would result in dangling ptrs/refs!
        out._backward() = [ _p1  = _spv,
                            _p2  = other._spv,
                            _out = out._spv](){
            // Gradients must accumulate! For cases like b=a+a
            _p1->grad += _out->grad;
            _p2->grad += _out->grad;
        };
        return out;
    }

    // For Value + constant
    Value operator+(double other) const { return *this + Value{other}; }

    Value operator*(const Value& other) const {
        auto out = Value{data() * other.data(), std::make_tuple(_spv, other._spv), "*"};
        out._backward() = [ _p1  = _spv,
                            _p2  = other._spv,
                            _out = out._spv](){
            // Grad on one side = other side data * incoming grad
            // Also, gradients must accumulate! In order to backprop from
            // d = a*b; e = a+b
            _p1->grad += _p2->data * _out->grad;
            _p2->grad += _p1->data * _out->grad;
        };
        return out;
    }

    // For Value * constant
    Value operator*(double d) const { return *this * Value{d}; }

    // Unary -
    Value operator-() const { return *this * -1.0; }
    // Subtraction
    Value operator-(const Value& other) const { return *this + (-other); }
    // Value - constant
    Value operator-(double d) const { return *this + (-d); }

    // this^d
    Value pow(double d) const {
        auto x = data();
        auto trd = std::pow(x, d);
        auto out = Value{trd, std::make_tuple(_spv, _spv), std::format("**{:.3f}", d)};
        // If L = a.pow(d), then ∂L/∂a = d*a.pow(d-1.0)
        out._backward() = [_p1  = _spv, x, d,
                           _out = out._spv](){
            _p1->grad += d * std::pow(x, d-1.0) * _out->grad;
        };
        return out;
    }

    // Division
    Value operator/(const Value& other) const { return *this * other.pow(-1); }

    // exp(this)
    Value exp() {
        auto x = data();
        auto e_x = std::exp(x);
        auto out = Value{e_x, std::make_tuple(_spv, _spv), "exp"};
        // If L = exp(a), then ∂L/∂a = exp(a)
        out._backward() = [_p1  = _spv, e_x,
                           _out = out._spv](){
            _p1->grad += e_x * _out->grad;
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
        out._backward() = [_p1  = _spv, th,
                           _out = out._spv](){
            _p1->grad += (1 - th*th) * _out->grad;
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

// For constant +/* Value
Value operator*(double d, const Value& v) { return v*d; }
Value operator+(double d, const Value& v) { return v+d; }


#endif // VALUE_H
