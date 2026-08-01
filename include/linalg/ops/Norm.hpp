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

    /// @brief Vector 1-norm, the sum of element magnitudes.
    /// @param x The vector.
    /// @return The 1-norm.
    static Real vectorOne(const Vector<T>& x);
    /// @brief Vector 2-norm (Euclidean norm).
    /// @param x The vector.
    /// @return The 2-norm.
    static Real vectorTwo(const Vector<T>& x);
    /// @brief Vector infinity-norm, the largest element magnitude.
    /// @param x The vector.
    /// @return The infinity-norm.
    static Real vectorInfinity(const Vector<T>& x);
    /// @brief Vector p-norm, (sum |x_i|^p)^(1/p).
    /// @param x The vector.
    /// @param p The norm order (p >= 1).
    /// @return The p-norm.
    static Real vectorP(const Vector<T>& x, Real p);
    /// @brief Vector 2-norm with overflow-safe scaling.
    /// @param x The vector.
    /// @return The 2-norm, computed without intermediate overflow.
    static Real vectorTwoScaled(const Vector<T>& x);

    // matrixOne/matrixInfinity are the max absolute column/row sums.
    // matrixTwo (largest singular value) and matrixNuclear (sum of
    // singular values) require an SVD and are O(n^3).
    /// @brief Matrix 1-norm, the maximum absolute column sum.
    /// @param a The matrix.
    /// @return The 1-norm.
    static Real matrixOne(const Matrix<T>& a);
    /// @brief Matrix infinity-norm, the maximum absolute row sum.
    /// @param a The matrix.
    /// @return The infinity-norm.
    static Real matrixInfinity(const Matrix<T>& a);
    /// @brief Frobenius norm, the square root of the sum of squared magnitudes.
    /// @param a The matrix.
    /// @return The Frobenius norm.
    static Real matrixFrobenius(const Matrix<T>& a);
    /// @brief Max-norm, the largest element magnitude (not submultiplicative).
    /// @param a The matrix.
    /// @return The maximum |a(i,j)|.
    static Real matrixMax(const Matrix<T>& a);
    /// @brief Matrix 2-norm, the largest singular value (via SVD; O(n^3)).
    /// @param a The matrix.
    /// @return The spectral norm.
    static Real matrixTwo(const Matrix<T>& a);
    /// @brief Nuclear norm, the sum of the singular values (via SVD; O(n^3)).
    /// @param a The matrix.
    /// @return The nuclear norm.
    static Real matrixNuclear(const Matrix<T>& a);

    // Diagnostics used throughout the test suite.
    /// @brief Euclidean distance between two vectors, ||x - y||_2.
    /// @param x First vector.
    /// @param y Second vector.
    /// @return The distance.
    static Real distance(const Vector<T>& x, const Vector<T>& y);
    /// @brief Relative error, ||approx - exact|| / ||exact||.
    /// @param approx The approximate vector.
    /// @param exact The reference vector.
    /// @return The relative error.
    static Real relativeError(const Vector<T>& approx, const Vector<T>& exact);
    /// @brief Residual norm of a linear system, ||b - A x||_2.
    /// @param a The system matrix.
    /// @param x The candidate solution.
    /// @param b The right-hand side.
    /// @return The residual norm.
    static Real residualNorm(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b);
    // Normwise backward error ||b - A x|| / (||A|| ||x|| + ||b||): ~epsilon
    // means x is the exact solution of a nearby problem, regardless of
    // conditioning.
    /// @brief Normwise backward error, ||b - A x|| / (||A|| ||x|| + ||b||).
    /// @param a The system matrix.
    /// @param x The candidate solution.
    /// @param b The right-hand side.
    /// @return The backward error.
    static Real backwardError(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b);
    /// @brief Orthogonality defect of a matrix, ||Q^H Q - I||_F.
    /// @param q The (approximately orthonormal) matrix.
    /// @return The orthogonality defect.
    static Real orthogonalityDefect(const Matrix<T>& q);
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Norm<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
