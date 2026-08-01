#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Norm computations collected as static members. The scaled variants
// avoid overflow when entries are near the top of the exponent range.
template <typename T>
class Norm {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Kind {
        enum class Vector { One, Two, Infinity, P };
        enum class Matrix { One, Two, Infinity, Frobenius, Max, Nuclear };
    };

    static Real vectorOne(const Vector<T>& x);
    static Real vectorTwo(const Vector<T>& x);
    static Real vectorInfinity(const Vector<T>& x);
    static Real vectorP(const Vector<T>& x, Real p);       // p >= 1
    static Real vectorTwoScaled(const Vector<T>& x);       // overflow-safe two-pass or running rescale

    // matrixOne/matrixInfinity are the max absolute column/row sums.
    // matrixTwo (largest singular value) and matrixNuclear (sum of
    // singular values) require an SVD and are O(n^3).
    static Real matrixOne(const Matrix<T>& a);
    static Real matrixInfinity(const Matrix<T>& a);
    static Real matrixFrobenius(const Matrix<T>& a);
    static Real matrixMax(const Matrix<T>& a);             // largest |a(i,j)|; not submultiplicative
    static Real matrixTwo(const Matrix<T>& a);
    static Real matrixNuclear(const Matrix<T>& a);

    // Diagnostics used throughout the test suite.
    static Real distance(const Vector<T>& x, const Vector<T>& y);          // ||x - y||_2
    static Real relativeError(const Vector<T>& approx, const Vector<T>& exact); // ||approx - exact|| / ||exact||
    static Real residualNorm(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b); // ||b - A x||_2
    // Normwise backward error ||b - A x|| / (||A|| ||x|| + ||b||): ~epsilon
    // means x is the exact solution of a nearby problem, regardless of
    // conditioning.
    static Real backwardError(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b);
    static Real orthogonalityDefect(const Matrix<T>& q);   // ||Q^H Q - I||_F
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Norm<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
