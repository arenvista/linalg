#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Abstract base: a preconditioner is anything that can approximate
// M^-1 * r cheaply. Krylov solvers hold a reference to one of these.
template <typename T>
class Preconditioner {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    virtual ~Preconditioner();

    virtual void      setup(const Matrix<T>& a) = 0;               // factor/extract whatever apply() needs
    virtual Vector<T> apply(const Vector<T>& r) const = 0;         // ~ M^-1 r; must be cheap (O(n) or O(n^2))
    virtual Vector<T> applyTranspose(const Vector<T>& r) const = 0; // ~ M^-H r, for transpose/adjoint methods
    virtual bool      isSymmetric() const = 0;                     // true is required for use with CG
    virtual Index     dimension() const = 0;
};

template <typename T>
class IdentityPreconditioner : public Preconditioner<T> {
public:
    using Index = std::size_t;

    IdentityPreconditioner();

    void      setup(const Matrix<T>& a) override;
    Vector<T> apply(const Vector<T>& r) const override;
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    bool      isSymmetric() const override;
    Index     dimension() const override;

private:
    Index dimension_;
};

// M = diag(A): free to set up, one divide per entry to apply.
template <typename T>
class JacobiPreconditioner : public Preconditioner<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    JacobiPreconditioner();

    void      setup(const Matrix<T>& a) override;
    Vector<T> apply(const Vector<T>& r) const override;
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    bool      isSymmetric() const override;
    Index     dimension() const override;

private:
    Vector<T> inverseDiagonal_;
};

// Symmetric SOR: one forward and one backward Gauss-Seidel-like sweep
// per apply. relaxation must lie in (0, 2); 1 gives symmetric
// Gauss-Seidel. Symmetric when A is, so usable with CG.
template <typename T>
class SSORPreconditioner : public Preconditioner<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    SSORPreconditioner();
    explicit SSORPreconditioner(Real relaxation);

    void      setup(const Matrix<T>& a) override;
    Vector<T> apply(const Vector<T>& r) const override;
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    bool      isSymmetric() const override;
    Index     dimension() const override;

    Real optimalRelaxationEstimate() const;

private:
    Matrix<T> operator_;
    Real      relaxation_;
};

// Incomplete Cholesky with zero fill-in: the Cholesky loop, but entries
// that are zero in A stay zero in the factor. Can break down (negative
// pivot) even for SPD input; `shift` scales the diagonal up as a
// compensation, and breakdownOccurred() reports whether it still failed.
template <typename T>
class IncompleteCholeskyPreconditioner : public Preconditioner<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    IncompleteCholeskyPreconditioner();
    explicit IncompleteCholeskyPreconditioner(Real shift);

    void      setup(const Matrix<T>& a) override;
    Vector<T> apply(const Vector<T>& r) const override;
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    bool      isSymmetric() const override;
    Index     dimension() const override;

    bool breakdownOccurred() const;

private:
    Matrix<T> factor_;
    Real      shift_;
    bool      breakdown_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Preconditioner<SCALAR>; \
    extern template class IdentityPreconditioner<SCALAR>; \
    extern template class JacobiPreconditioner<SCALAR>; \
    extern template class SSORPreconditioner<SCALAR>; \
    extern template class IncompleteCholeskyPreconditioner<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
