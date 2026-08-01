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

    /// @brief Virtual destructor for the abstract preconditioner interface.
    virtual ~Preconditioner();

    /// @brief Prepares whatever apply() needs from the system matrix.
    /// @param a The system matrix A.
    virtual void      setup(const Matrix<T>& a) = 0;
    /// @brief Applies the preconditioner, approximating M^-1 * r cheaply.
    /// @param r The residual vector.
    /// @return An approximation of M^-1 * r.
    virtual Vector<T> apply(const Vector<T>& r) const = 0;
    /// @brief Applies the adjoint preconditioner, approximating M^-H * r.
    /// @param r The residual vector.
    /// @return An approximation of M^-H * r.
    virtual Vector<T> applyTranspose(const Vector<T>& r) const = 0;
    /// @brief Reports whether the preconditioner is symmetric (required for CG).
    /// @return True if symmetric.
    virtual bool      isSymmetric() const = 0;
    /// @brief The dimension of the operator.
    /// @return The problem size n.
    virtual Index     dimension() const = 0;
};

/// @brief Trivial preconditioner M = I; apply() returns its input unchanged.
template <typename T>
class IdentityPreconditioner : public Preconditioner<T> {
public:
    using Index = std::size_t;

    /// @brief Constructs an identity preconditioner.
    IdentityPreconditioner();

    /// @brief Records the problem dimension from A.
    /// @param a The system matrix A.
    void      setup(const Matrix<T>& a) override;
    /// @brief Returns r unchanged (M^-1 = I).
    /// @param r The residual vector.
    /// @return A copy of r.
    Vector<T> apply(const Vector<T>& r) const override;
    /// @brief Returns r unchanged (M^-H = I).
    /// @param r The residual vector.
    /// @return A copy of r.
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    /// @brief The identity is symmetric.
    /// @return Always true.
    bool      isSymmetric() const override;
    /// @brief The recorded problem dimension.
    /// @return The problem size n.
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

    /// @brief Constructs a Jacobi (diagonal) preconditioner.
    JacobiPreconditioner();

    /// @brief Extracts and inverts the diagonal of A.
    /// @param a The system matrix A.
    void      setup(const Matrix<T>& a) override;
    /// @brief Applies M^-1 = diag(A)^-1 by scaling each entry.
    /// @param r The residual vector.
    /// @return diag(A)^-1 * r.
    Vector<T> apply(const Vector<T>& r) const override;
    /// @brief Applies the adjoint diagonal preconditioner.
    /// @param r The residual vector.
    /// @return conj(diag(A))^-1 * r.
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    /// @brief Reports whether the diagonal is real (symmetric preconditioner).
    /// @return True if symmetric.
    bool      isSymmetric() const override;
    /// @brief The problem dimension.
    /// @return The problem size n.
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

    /// @brief Constructs an SSOR preconditioner with default relaxation.
    SSORPreconditioner();
    /// @brief Constructs an SSOR preconditioner with a relaxation factor.
    /// @param relaxation Relaxation parameter in (0, 2); 1 is symmetric Gauss-Seidel.
    explicit SSORPreconditioner(Real relaxation);

    /// @brief Stores A for the forward/backward sweeps.
    /// @param a The system matrix A.
    void      setup(const Matrix<T>& a) override;
    /// @brief Applies one forward and one backward SSOR sweep.
    /// @param r The residual vector.
    /// @return The SSOR-preconditioned vector.
    Vector<T> apply(const Vector<T>& r) const override;
    /// @brief Applies the adjoint SSOR sweeps.
    /// @param r The residual vector.
    /// @return The adjoint-preconditioned vector.
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    /// @brief Reports whether the preconditioner is symmetric.
    /// @return True when A is symmetric.
    bool      isSymmetric() const override;
    /// @brief The problem dimension.
    /// @return The problem size n.
    Index     dimension() const override;

    /// @brief Heuristic estimate of the optimal relaxation parameter.
    /// @return The estimated optimal relaxation in (0, 2).
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

    /// @brief Constructs an incomplete-Cholesky preconditioner with no shift.
    IncompleteCholeskyPreconditioner();
    /// @brief Constructs an incomplete-Cholesky preconditioner with a diagonal shift.
    /// @param shift Diagonal shift added to compensate for breakdown.
    explicit IncompleteCholeskyPreconditioner(Real shift);

    /// @brief Computes the zero-fill incomplete Cholesky factor of A.
    /// @param a The system matrix A.
    void      setup(const Matrix<T>& a) override;
    /// @brief Applies M^-1 via forward and backward triangular solves.
    /// @param r The residual vector.
    /// @return The preconditioned vector.
    Vector<T> apply(const Vector<T>& r) const override;
    /// @brief Applies the adjoint preconditioner.
    /// @param r The residual vector.
    /// @return The adjoint-preconditioned vector.
    Vector<T> applyTranspose(const Vector<T>& r) const override;
    /// @brief Reports whether the preconditioner is symmetric.
    /// @return True (the incomplete Cholesky factor is symmetric).
    bool      isSymmetric() const override;
    /// @brief The problem dimension.
    /// @return The problem size n.
    Index     dimension() const override;

    /// @brief Reports whether factorization broke down (a non-positive pivot).
    /// @return True if breakdown occurred during setup().
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
