#ifndef _GENERAL_GRAPH_OPTIMIZOR_JET_H_
#define _GENERAL_GRAPH_OPTIMIZOR_JET_H_

#include "basic_type.h"
#include "cmath"
#include "iostream"

namespace slam_solver {

/**
 * @brief Jet class for Automatic Differentiation (Forward Mode AD).
 * This class implements the dual number (Jet) data structure, which simultaneously
 * stores a function's value and its partial derivatives (infinitesimal part).
 * Jet<T, N> encapsulates a scalar value of type T and an N-dimensional derivative vector.
 * It is the core component for computing analytical derivatives automatically in optimization problems.
 */
template <typename T, int N>
struct Jet {
    T a;                   // Primitive value of the function (function output)
    Eigen::Matrix<T, N, 1> v; // Infinitesimal part: partial derivatives w.r.t. N variables

    // Default constructor: initialize value to 0 and derivatives to zero vector
    Jet() : a(0), v(Eigen::Matrix<T, N, 1>::Zero()) {}
    // Constructor from scalar value: value initialized, derivatives set to zero
    explicit Jet(T value) : a(value), v(Eigen::Matrix<T, N, 1>::Zero()) {}
    /**
     * @brief Constructor for independent variables (seed for automatic differentiation)
     * @param value The value of the independent variable
     * @param index The index of this variable in the parameter vector (derivative = 1 at this index)
     */
    Jet(T value, int index) : a(value), v(Eigen::Matrix<T, N, 1>::Zero()) {
        if (index >= 0 && index < N) {
            v(index) = T(1);
        }
    }
    // Full constructor: directly set value and derivative vector
    Jet(T value, const Eigen::Matrix<T, N, 1>& derivatives) : a(value), v(derivatives) {}

    // Compound assignment operators with derivative chain rules applied
    Jet<T, N>& operator+=(const Jet<T, N>& f) {
        a += f.a;
        v += f.v;
        return *this;
    }
    Jet<T, N>& operator-=(const Jet<T, N>& f) {
        a -= f.a;
        v -= f.v;
        return *this;
    }
    Jet<T, N>& operator*=(const Jet<T, N>& f) {
        // Product rule: d(ab) = a*db + b*da
        v = a * f.v + f.a * v;
        a *= f.a;
        return *this;
    }
    Jet<T, N>& operator/=(const Jet<T, N>& f) {
        // Quotient rule: d(a/b) = (b*da - a*db) / b²
        T inv_fa = T(1.0) / f.a;
        v = (v * f.a - a * f.v) * inv_fa * inv_fa;
        a *= inv_fa;
        return *this;
    }

    // Unary arithmetic operators
    Jet<T, N> operator-() const { return Jet<T, N>(-a, -v); }
    Jet<T, N> operator+() const { return *this; }

    // Implicit conversion to primitive scalar type (use the value component)
    operator T() const { return a; }
};

// Binary arithmetic operators with forward-mode AD derivative rules
template <typename T, int N>
inline Jet<T, N> operator+(const Jet<T, N>& f, const Jet<T, N>& g) {
    return Jet<T, N>(f.a + g.a, f.v + g.v);
}
template <typename T, int N>
inline Jet<T, N> operator+(T f, const Jet<T, N>& g) {
    return Jet<T, N>(f + g.a, g.v);
}
template <typename T, int N>
inline Jet<T, N> operator+(const Jet<T, N>& f, T g) {
    return Jet<T, N>(f.a + g, f.v);
}

template <typename T, int N>
inline Jet<T, N> operator-(const Jet<T, N>& f, const Jet<T, N>& g) {
    return Jet<T, N>(f.a - g.a, f.v - g.v);
}
template <typename T, int N>
inline Jet<T, N> operator-(T f, const Jet<T, N>& g) {
    return Jet<T, N>(f - g.a, -g.v);
}
template <typename T, int N>
inline Jet<T, N> operator-(const Jet<T, N>& f, T g) {
    return Jet<T, N>(f.a - g, f.v);
}

template <typename T, int N>
inline Jet<T, N> operator*(const Jet<T, N>& f, const Jet<T, N>& g) {
    return Jet<T, N>(f.a * g.a, f.a * g.v + g.a * f.v);
}
template <typename T, int N>
inline Jet<T, N> operator*(T f, const Jet<T, N>& g) {
    return Jet<T, N>(f * g.a, f * g.v);
}
template <typename T, int N>
inline Jet<T, N> operator*(const Jet<T, N>& f, T g) {
    return Jet<T, N>(f.a * g, f.v * g);
}

template <typename T, int N>
inline Jet<T, N> operator/(const Jet<T, N>& f, const Jet<T, N>& g) {
    T inv_ga = T(1.0) / g.a;
    return Jet<T, N>(f.a * inv_ga, (f.v * g.a - f.a * g.v) * inv_ga * inv_ga);
}
template <typename T, int N>
inline Jet<T, N> operator/(T f, const Jet<T, N>& g) {
    T inv_ga = T(1.0) / g.a;
    return Jet<T, N>(f * inv_ga, -f * g.v * inv_ga * inv_ga);
}
template <typename T, int N>
inline Jet<T, N> operator/(const Jet<T, N>& f, T g) {
    T inv_g = T(1.0) / g;
    return Jet<T, N>(f.a * inv_g, f.v * inv_g);
}

// Comparison operators: only compare the primitive value (derivatives ignored)
template <typename T, int N>
inline bool operator==(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a == g.a; }
template <typename T, int N>
inline bool operator!=(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a != g.a; }
template <typename T, int N>
inline bool operator<(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a < g.a; }
template <typename T, int N>
inline bool operator<=(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a <= g.a; }
template <typename T, int N>
inline bool operator>(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a > g.a; }
template <typename T, int N>
inline bool operator>=(const Jet<T, N>& f, const Jet<T, N>& g) { return f.a >= g.a; }

} // namespace slam_solver

// Eigen library type traits specialization for Jet type
namespace Eigen {
template <typename T, int N>
struct NumTraits<slam_solver::Jet<T, N>> : GenericNumTraits<T> {
    typedef slam_solver::Jet<T, N> Real;
    typedef slam_solver::Jet<T, N> NonInteger;
    typedef slam_solver::Jet<T, N> Nested;
    enum {
        IsComplex = 0,        // Not a complex number
        IsInteger = 0,        // Not an integer type
        IsSigned = 1,         // Supports signed values
        RequireInitialization = 1, // Requires explicit initialization
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
};
} // namespace Eigen

namespace slam_solver {

/**
 * @brief Overloaded mathematical functions for Jet type (automatic differentiation support)
 * Each function computes both the output value and its analytical derivative using chain rule
 */
template <typename T, int N>
inline Jet<T, N> sin(const Jet<T, N>& f) {
    return Jet<T, N>(std::sin(f.a), std::cos(f.a) * f.v);
}
template <typename T, int N>
inline Jet<T, N> cos(const Jet<T, N>& f) {
    return Jet<T, N>(std::cos(f.a), -std::sin(f.a) * f.v);
}
template <typename T, int N>
inline Jet<T, N> tan(const Jet<T, N>& f) {
    T sec = T(1.0) / std::cos(f.a);
    return Jet<T, N>(std::tan(f.a), sec * sec * f.v);
}
template <typename T, int N>
inline Jet<T, N> exp(const Jet<T, N>& f) {
    T val = std::exp(f.a);
    return Jet<T, N>(val, val * f.v);
}
template <typename T, int N>
inline Jet<T, N> log(const Jet<T, N>& f) {
    return Jet<T, N>(std::log(f.a), f.v / f.a);
}
template <typename T, int N>
inline Jet<T, N> sqrt(const Jet<T, N>& f) {
    T val = std::sqrt(f.a);
    return Jet<T, N>(val, f.v / (T(2.0) * val));
}
template <typename T, int N>
inline Jet<T, N> pow(const Jet<T, N>& f, T g) {
    T val = std::pow(f.a, g);
    return Jet<T, N>(val, g * std::pow(f.a, g - T(1.0)) * f.v);
}
template <typename T, int N>
inline Jet<T, N> abs(const Jet<T, N>& f) {
    return f.a >= 0 ? f : -f;
}
template <typename T, int N>
inline Jet<T, N> atan2(const Jet<T, N>& g, const Jet<T, N>& f) {
    T inv_mag2 = T(1.0) / (f.a * f.a + g.a * g.a);
    return Jet<T, N>(std::atan2(g.a, f.a), (f.a * g.v - g.a * f.v) * inv_mag2);
}
template <typename T, int N>
inline Jet<T, N> asin(const Jet<T, N>& f) {
    return Jet<T, N>(std::asin(f.a), f.v / std::sqrt(T(1.0) - f.a * f.a));
}
template <typename T, int N>
inline Jet<T, N> acos(const Jet<T, N>& f) {
    return Jet<T, N>(std::acos(f.a), -f.v / std::sqrt(T(1.0) - f.a * f.a));
}
template <typename T, int N>
inline Jet<T, N> atan(const Jet<T, N>& f) {
    return Jet<T, N>(std::atan(f.a), f.v / (T(1.0) + f.a * f.a));
}

} // namespace slam_solver

// Forward std math function calls to slam_solver's Jet-specialized implementations
namespace std {
template <typename T, int N>
inline slam_solver::Jet<T, N> sin(const slam_solver::Jet<T, N>& f) { return slam_solver::sin(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> cos(const slam_solver::Jet<T, N>& f) { return slam_solver::cos(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> tan(const slam_solver::Jet<T, N>& f) { return slam_solver::tan(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> exp(const slam_solver::Jet<T, N>& f) { return slam_solver::exp(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> log(const slam_solver::Jet<T, N>& f) { return slam_solver::log(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> sqrt(const slam_solver::Jet<T, N>& f) { return slam_solver::sqrt(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> abs(const slam_solver::Jet<T, N>& f) { return slam_solver::abs(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> atan2(const slam_solver::Jet<T, N>& g, const slam_solver::Jet<T, N>& f) { return slam_solver::atan2(g, f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> asin(const slam_solver::Jet<T, N>& f) { return slam_solver::asin(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> acos(const slam_solver::Jet<T, N>& f) { return slam_solver::acos(f); }
template <typename T, int N>
inline slam_solver::Jet<T, N> atan(const slam_solver::Jet<T, N>& f) { return slam_solver::atan(f); }
}

#endif // _GENERAL_GRAPH_OPTIMIZOR_JET_H_
