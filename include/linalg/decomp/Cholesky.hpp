#pragma once

#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// A = L L^H for Hermitian positive definite A.
template <typename T>
class Cholesky {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Options {
        Triangle::Kind readFrom;          // which triangle of the input holds A; the other is never read
        bool           useBlocked;
        Index          blockSize;
        bool           throwOnIndefinite; // throw NotPositiveDefinite vs record failure
    };

    Cholesky();
    explicit Cholesky(const Matrix<T>& a);
    Cholesky(const Matrix<T>& a, const Options& options);

    void compute(const Matrix<T>& a);
    bool isComputed() const;
    bool isPositiveDefinite() const;  // false when a pivot came out non-positive (failedPivot_)

    Matrix<T> lower() const;  // L, with A = L L^H
    Matrix<T> upper() const;  // L^H

    Vector<T> solve(const Vector<T>& b) const;
    Matrix<T> solve(const Matrix<T>& b) const;
    Matrix<T> inverse() const;

    T    determinant() const;      // product of squared diagonal entries of L
    Real logDeterminant() const;   // 2 * sum log L(i,i); safe from overflow
    Real reciprocalConditionEstimate() const;

    // Rank-one modifications without refactorizing (O(n^2) vs O(n^3)):
    // update refactors A + x x^H, downdate refactors A - x x^H. A downdate
    // can destroy positive definiteness, in which case it throws
    // NotPositiveDefinite and leaves the factor unchanged.
    void update(const Vector<T>& x);
    void downdate(const Vector<T>& x);

private:
    void unblockedFactorize();
    void blockedFactorize(Index blockSize);

    Matrix<T> factor_;
    Options   options_;
    bool      computed_;
    bool      positiveDefinite_;
    Index     failedPivot_;
};

// L D L^H: no square roots, tolerates semidefinite and some indefinite
// matrices where plain Cholesky breaks down.
template <typename T>
class LDLT {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    LDLT();
    explicit LDLT(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    // Factorization is P^T A P = L D L^H with L unit lower triangular and
    // D strictly diagonal (no 2x2 blocks — diagonal pivoting only, so
    // strongly indefinite matrices may still be inaccurate).
    Matrix<T> lower() const;
    Vector<T> diagonal() const;           // D's diagonal; signs reveal inertia
    Matrix<T> permutationMatrix() const;

    Vector<T> solve(const Vector<T>& b) const;
    bool      isPositiveSemiDefinite(Real tolerance) const; // all D(i) >= -tolerance
    bool      isNegativeSemiDefinite(Real tolerance) const;
    Index     rank(Real tolerance) const;                   // count of |D(i)| > tolerance

private:
    Matrix<T>          factor_;
    Vector<T>          diagonal_;
    std::vector<Index> pivots_;
    bool               computed_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Cholesky<SCALAR>; \
    extern template class LDLT<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
