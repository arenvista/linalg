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

    /// @brief Constructs an empty reduction; call compute() before use.
    Hessenberg();
    /// @brief Constructs and immediately reduces a matrix to Hessenberg form.
    /// @param a Matrix to reduce.
    explicit Hessenberg(const Matrix<T>& a);

    /// @brief Computes the A = Q H Q^H reduction to upper Hessenberg form.
    /// @param a Matrix to reduce.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a reduction is available.
    /// @return True if compute() has run.
    bool isComputed() const;

    /// @brief The upper Hessenberg factor H (zero below the first subdiagonal).
    /// @return The H factor.
    Matrix<T> matrixH() const;
    /// @brief The orthogonal/unitary factor Q.
    /// @return The Q factor.
    Matrix<T> matrixQ() const;
    /// @brief The implicit Householder reflector sequence representing Q.
    /// @return Reference to the reflector sequence.
    const HouseholderSequence<T>& reflectors() const;

    // A Hessenberg matrix is unreduced when no subdiagonal entry is
    // negligible; a negligible one splits the eigenproblem into two
    // independent blocks (that is the QR iteration's deflation test).
    /// @brief Reports whether every subdiagonal entry is non-negligible.
    /// @param tolerance Threshold below which a subdiagonal counts as zero.
    /// @return True if the matrix is unreduced.
    bool  isUnreduced(Real tolerance) const;
    /// @brief Finds the first negligible subdiagonal entry (a deflation point).
    /// @param tolerance Threshold below which a subdiagonal counts as zero.
    /// @return The index of the first negligible subdiagonal.
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

    /// @brief Constructs an empty reduction; call compute() before use.
    Tridiagonal();
    /// @brief Constructs and immediately reduces a Hermitian matrix.
    /// @param a Hermitian matrix to reduce.
    explicit Tridiagonal(const Matrix<T>& a);

    /// @brief Computes the A = Q T Q^H reduction to tridiagonal form.
    /// @param a Hermitian matrix to reduce.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a reduction is available.
    /// @return True if compute() has run.
    bool isComputed() const;

    // The tridiagonal factor is real even for complex Hermitian input:
    // the reflectors are chosen to rotate the off-diagonal phases away.
    /// @brief The (real) main diagonal of T.
    /// @return A vector of the diagonal entries.
    Vector<Real> diagonal() const;
    /// @brief The (real) subdiagonal of T.
    /// @return A vector of the subdiagonal entries.
    Vector<Real> subdiagonal() const;
    /// @brief The symmetric tridiagonal factor T.
    /// @return The T factor.
    Matrix<T>    matrixT() const;
    /// @brief The orthogonal/unitary factor Q.
    /// @return The Q factor.
    Matrix<T>    matrixQ() const;

    // Sturm sequence count: eigenvalues strictly below a shift.
    // Bisection on this gives eigenvalues in a range without iterating.
    /// @brief Sturm-sequence count of eigenvalues strictly below a shift.
    /// @param shift The shift value.
    /// @return The number of eigenvalues less than `shift`.
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

    /// @brief Constructs an empty reduction; call compute() before use.
    Bidiagonal();
    /// @brief Constructs and immediately reduces a matrix to bidiagonal form.
    /// @param a Matrix to reduce.
    explicit Bidiagonal(const Matrix<T>& a);

    /// @brief Computes the A = U B V^H reduction to upper bidiagonal form.
    /// @param a Matrix to reduce.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a reduction is available.
    /// @return True if compute() has run.
    bool isComputed() const;

    /// @brief The (real) main diagonal of B.
    /// @return A vector of the diagonal entries.
    Vector<Real> diagonal() const;
    /// @brief The (real) superdiagonal of B.
    /// @return A vector of the superdiagonal entries.
    Vector<Real> superdiagonal() const;
    /// @brief The upper bidiagonal factor B.
    /// @return The B factor.
    Matrix<T>    matrixB() const;
    /// @brief The left orthogonal/unitary factor U.
    /// @return The U factor.
    Matrix<T>    matrixU() const;
    /// @brief The right orthogonal/unitary factor V.
    /// @return The V factor.
    Matrix<T>    matrixV() const;

    /// @brief Reports whether the m >> n QR-first preprocessing path was used.
    /// @return True if R-preprocessing was applied.
    bool usedRPreprocessing() const;

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
