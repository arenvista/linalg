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

    /// @brief Constructs a solver with default options.
    TriangularSolver();
    /// @brief Constructs a solver with explicit options.
    /// @param options Triangle, diagonal, transposition, and blocking settings.
    explicit TriangularSolver(const Options& options);

    // Forward substitution for Lower, back substitution for Upper.
    // Throws SingularMatrix on a zero diagonal entry (NonUnit only). The
    // matrix overloads treat each column of b as a right-hand side.
    /// @brief Solves op(A) x = b for a triangular A.
    /// @param a Triangular coefficient matrix.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    /// @throws SingularMatrix on a zero diagonal entry (NonUnit only).
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) const;
    /// @brief Solves op(A) X = B for a triangular A (one column per system).
    /// @param a Triangular coefficient matrix.
    /// @param b Right-hand side matrix.
    /// @return The solution matrix X.
    /// @throws SingularMatrix on a zero diagonal entry (NonUnit only).
    Matrix<T> solve(const Matrix<T>& a, const Matrix<T>& b) const;
    /// @brief Solves op(A) x = b in place, overwriting b with the solution.
    /// @param a Triangular coefficient matrix.
    /// @param b Right-hand side, overwritten with the solution.
    /// @throws SingularMatrix on a zero diagonal entry (NonUnit only).
    void      solveInPlace(const Matrix<T>& a, Vector<T>& b) const;
    /// @brief Solves op(A) X = B in place, overwriting B with the solution.
    /// @param a Triangular coefficient matrix.
    /// @param b Right-hand sides, overwritten with the solutions.
    /// @throws SingularMatrix on a zero diagonal entry (NonUnit only).
    void      solveInPlace(const Matrix<T>& a, Matrix<T>& b) const;

    /// @brief Inverts a triangular matrix, preserving its structure.
    /// @param a Triangular matrix to invert.
    /// @return The triangular inverse.
    Matrix<T> inverse(const Matrix<T>& a) const;
    /// @brief Cheap reciprocal condition-number estimate for a triangular matrix.
    /// @param a Triangular matrix.
    /// @return An estimate of 1 / cond(a) in [0, 1].
    Real      reciprocalConditionEstimate(const Matrix<T>& a) const;
    /// @brief Tests nonsingularity (all |diagonal| above a tolerance).
    /// @param a Triangular matrix.
    /// @param tolerance Cutoff for the diagonal magnitudes.
    /// @return True if every diagonal entry exceeds tolerance.
    bool      isNonsingular(const Matrix<T>& a, Real tolerance) const;

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
