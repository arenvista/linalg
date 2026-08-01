#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Householder.hpp"

namespace linalg {

// A = Q H Q^H with H upper Hessenberg. The mandatory first step of
// the general eigenvalue problem: it makes each QR sweep O(n^2).
template <typename T>
class Hessenberg {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    Hessenberg();
    explicit Hessenberg(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Matrix<T> matrixH() const;  // upper Hessenberg: zero below the first subdiagonal
    Matrix<T> matrixQ() const;
    const HouseholderSequence<T>& reflectors() const;

    // A Hessenberg matrix is unreduced when no subdiagonal entry is
    // negligible; a negligible one splits the eigenproblem into two
    // independent blocks (that is the QR iteration's deflation test).
    bool  isUnreduced(Real tolerance) const;
    Index firstNegligibleSubdiagonal(Real tolerance) const;

private:
    Matrix<T>              factors_;
    HouseholderSequence<T> reflectors_;
    bool                   computed_;
};

// A = Q T Q^H with T symmetric tridiagonal, for Hermitian input.
template <typename T>
class Tridiagonal {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    Tridiagonal();
    explicit Tridiagonal(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    // The tridiagonal factor is real even for complex Hermitian input:
    // the reflectors are chosen to rotate the off-diagonal phases away.
    Vector<Real> diagonal() const;
    Vector<Real> subdiagonal() const;
    Matrix<T>    matrixT() const;
    Matrix<T>    matrixQ() const;

    // Sturm sequence count: eigenvalues strictly below a shift.
    // Bisection on this gives eigenvalues in a range without iterating.
    Index eigenvalueCountBelow(Real shift) const;

private:
    Matrix<T>              factors_;
    Vector<Real>           diagonal_;
    Vector<Real>           subdiagonal_;
    HouseholderSequence<T> reflectors_;
    bool                   computed_;
};

// A = U B V^H with B upper bidiagonal: the reduction step feeding
// the Golub-Kahan SVD iteration.
template <typename T>
class Bidiagonal {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    Bidiagonal();
    explicit Bidiagonal(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Vector<Real> diagonal() const;       // real, as with Tridiagonal
    Vector<Real> superdiagonal() const;
    Matrix<T>    matrixB() const;
    Matrix<T>    matrixU() const;
    Matrix<T>    matrixV() const;

    bool usedRPreprocessing() const;  // true if m >> n triggered QR-first, then bidiagonalizing R

private:
    Matrix<T>              factors_;
    Vector<Real>           diagonal_;
    Vector<Real>           superdiagonal_;
    HouseholderSequence<T> leftReflectors_;
    HouseholderSequence<T> rightReflectors_;
    bool                   rPreprocessed_;
    bool                   computed_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Hessenberg<SCALAR>; \
    extern template class Tridiagonal<SCALAR>; \
    extern template class Bidiagonal<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
