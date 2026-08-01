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

    /// @brief Constructs an empty factorization; call compute() before use.
    Cholesky();
    /// @brief Constructs and immediately factorizes a matrix with defaults.
    /// @param a Hermitian positive-definite matrix to factorize.
    explicit Cholesky(const Matrix<T>& a);
    /// @brief Constructs and immediately factorizes a matrix with options.
    /// @param a Hermitian positive-definite matrix to factorize.
    /// @param options Factorization options (triangle, blocking, error policy).
    Cholesky(const Matrix<T>& a, const Options& options);

    /// @brief Computes the A = L L^H factorization.
    /// @param a Hermitian positive-definite matrix to factorize.
    /// @throws NotPositiveDefinite if indefinite and options.throwOnIndefinite is set.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the factorization found the matrix positive definite.
    /// @return False if a non-positive pivot was encountered.
    bool isPositiveDefinite() const;

    /// @brief The lower-triangular factor L, with A = L L^H.
    /// @return The L factor.
    Matrix<T> lower() const;
    /// @brief The upper-triangular factor L^H.
    /// @return The L^H factor.
    Matrix<T> upper() const;

    /// @brief Solves A x = b using the factorization.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Solves A X = B for multiple right-hand sides.
    /// @param b Right-hand side matrix (one column per system).
    /// @return The solution matrix X.
    Matrix<T> solve(const Matrix<T>& b) const;
    /// @brief Matrix inverse computed from the factor.
    /// @return The inverse matrix.
    Matrix<T> inverse() const;

    /// @brief Determinant, the product of the squared diagonal entries of L.
    /// @return The determinant.
    T    determinant() const;
    /// @brief Log determinant, 2 * sum log L(i,i); safe from overflow.
    /// @return log(det(A)).
    Real logDeterminant() const;
    /// @brief Cheap reciprocal condition-number estimate.
    /// @return An estimate of 1 / cond(A) in [0, 1].
    Real reciprocalConditionEstimate() const;

    // Rank-one modifications without refactorizing (O(n^2) vs O(n^3)):
    // update refactors A + x x^H, downdate refactors A - x x^H. A downdate
    // can destroy positive definiteness, in which case it throws
    // NotPositiveDefinite and leaves the factor unchanged.
    /// @brief Rank-one update, re-factoring A + x x^H in O(n^2).
    /// @param x The update vector.
    void update(const Vector<T>& x);
    /// @brief Rank-one downdate, re-factoring A - x x^H in O(n^2).
    /// @param x The downdate vector.
    /// @throws NotPositiveDefinite if the downdate loses positive definiteness
    /// (the factor is left unchanged).
    void downdate(const Vector<T>& x);

private:
    /// @brief Performs the unblocked (level-2) factorization.
    void unblockedFactorize();
    /// @brief Performs the blocked factorization.
    /// @param blockSize Panel width for the blocked algorithm.
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

    /// @brief Constructs an empty factorization; call compute() before use.
    LDLT();
    /// @brief Constructs and immediately factorizes a matrix.
    /// @param a Hermitian matrix to factorize.
    explicit LDLT(const Matrix<T>& a);

    /// @brief Computes the P^T A P = L D L^H factorization.
    /// @param a Hermitian matrix to factorize.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has run.
    bool isComputed() const;

    // Factorization is P^T A P = L D L^H with L unit lower triangular and
    // D strictly diagonal (no 2x2 blocks — diagonal pivoting only, so
    // strongly indefinite matrices may still be inaccurate).
    /// @brief The unit lower-triangular factor L.
    /// @return The L factor.
    Matrix<T> lower() const;
    /// @brief The diagonal factor D; the signs reveal the inertia.
    /// @return A vector of D's diagonal entries.
    Vector<T> diagonal() const;
    /// @brief Builds the symmetric permutation matrix P.
    /// @return The permutation matrix.
    Matrix<T> permutationMatrix() const;

    /// @brief Solves A x = b using the factorization.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Tests positive semidefiniteness (all D(i) >= -tolerance).
    /// @param tolerance Absolute tolerance on the diagonal signs.
    /// @return True if positive semidefinite within tolerance.
    bool      isPositiveSemiDefinite(Real tolerance) const;
    /// @brief Tests negative semidefiniteness (all D(i) <= tolerance).
    /// @param tolerance Absolute tolerance on the diagonal signs.
    /// @return True if negative semidefinite within tolerance.
    bool      isNegativeSemiDefinite(Real tolerance) const;
    /// @brief Rank, the count of |D(i)| above tolerance.
    /// @param tolerance Cutoff below which a diagonal entry counts as zero.
    /// @return The numerical rank.
    Index     rank(Real tolerance) const;

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
