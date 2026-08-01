#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Forward and back substitution. Every direct solver bottoms out here,
// so it is worth having blocked and unit-diagonal variants.
template <typename T>
class TriangularSolver {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Options {
        Triangle::Kind      uplo;          // which triangle of `a` holds the data; the other is never read
        Diagonal::Kind      diagonal;      // Unit: diagonal assumed all ones, not read
        Transposition::Kind transposition; // solves op(A) x = b
        bool                useBlocked;
        Index               blockSize;
    };

    TriangularSolver();
    explicit TriangularSolver(const Options& options);

    // Forward substitution for Lower, back substitution for Upper.
    // Throws SingularMatrix on a zero diagonal entry (NonUnit only). The
    // matrix overloads treat each column of b as a right-hand side.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) const;
    Matrix<T> solve(const Matrix<T>& a, const Matrix<T>& b) const;
    void      solveInPlace(const Matrix<T>& a, Vector<T>& b) const;
    void      solveInPlace(const Matrix<T>& a, Matrix<T>& b) const;

    Matrix<T> inverse(const Matrix<T>& a) const;  // triangular inverse, same structure
    Real      reciprocalConditionEstimate(const Matrix<T>& a) const;
    bool      isNonsingular(const Matrix<T>& a, Real tolerance) const;  // all |diagonal| > tolerance

private:
    Options options_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class TriangularSolver<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
