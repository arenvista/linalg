#pragma once

#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Givens.hpp"
#include "linalg/ops/Householder.hpp"

namespace linalg {

// A = QR by Householder reflections. Q is kept implicit as a reflector
// sequence; materialize only when asked.
template <typename T>
class QR {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // Thin: matrixQ() is m x min(m, n); Full: m x m. R is
    // min(m, n) x n either way.
    struct Mode {
        enum class Kind { Thin, Full };
    };

    struct Options {
        typename Mode::Kind mode;
        bool                useBlocked;
        Index               blockSize;
        bool                accumulateQ;  // materialize Q during compute() instead of on first request
    };

    /// @brief Constructs an empty factorization; call compute() before use.
    QR();
    /// @brief Constructs and immediately factorizes a matrix with defaults.
    /// @param a Matrix to factorize.
    explicit QR(const Matrix<T>& a);
    /// @brief Constructs and immediately factorizes a matrix with options.
    /// @param a Matrix to factorize.
    /// @param options Factorization options (mode, blocking, Q accumulation).
    QR(const Matrix<T>& a, const Options& options);

    /// @brief Computes the A = QR factorization by Householder reflections.
    /// @param a Matrix to factorize.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has succeeded.
    bool isComputed() const;

    /// @brief Materializes the orthogonal factor Q.
    /// @return Q, of shape m x min(m,n) (thin) or m x m (full).
    Matrix<T>                       matrixQ() const;
    /// @brief Materializes the upper-triangular factor R.
    /// @return R, of shape min(m,n) x n.
    Matrix<T>                       matrixR() const;
    /// @brief The implicit Householder reflector sequence representing Q.
    /// @return Reference to the reflector sequence.
    const HouseholderSequence<T>&   reflectors() const;

    /// @brief Multiplies a target matrix by Q on the left, in place (Q * target).
    /// @param target Matrix overwritten with the product.
    void applyQOnLeft(Matrix<T>& target) const;
    /// @brief Multiplies a target by Q^H on the left, in place (Q^H * target).
    /// @param target Matrix overwritten with the product.
    void applyQTransposeOnLeft(Matrix<T>& target) const;

    // solve expects square nonsingular A; solveLeastSquares minimizes
    // ||Ax - b|| for m >= n. Both are Q^H * b then a back-substitution.
    /// @brief Solves A x = b for square nonsingular A.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Solves A X = B for square nonsingular A.
    /// @param b Right-hand side matrix (one column per system).
    /// @return The solution matrix X.
    Matrix<T> solve(const Matrix<T>& b) const;
    /// @brief Least-squares solve minimizing ||A x - b|| for m >= n.
    /// @param b Right-hand side vector.
    /// @return The least-squares solution x.
    Vector<T> solveLeastSquares(const Vector<T>& b) const;

    /// @brief Absolute determinant, the product of |R(i,i)|.
    /// @return |det(A)|.
    T     absDeterminant() const;
    /// @brief Logarithm of the absolute determinant.
    /// @return log|det(A)|.
    Real  logAbsDeterminant() const;
    /// @brief Rank estimate from |R(i,i)|; unpivoted QR is not rank-revealing.
    /// @param tolerance Cutoff below which a diagonal entry counts as zero.
    /// @return The estimated rank.
    Index rank(Real tolerance) const;

    // Updating and downdating without recomputing from scratch: O(n^2)
    // Givens repair work per modification instead of an O(n^3) refactor.
    /// @brief Updates the factorization for an appended row (O(n^2)).
    /// @param row The row to add.
    void appendRow(const Vector<T>& row);
    /// @brief Updates the factorization for an appended column (O(n^2)).
    /// @param column The column to add.
    void appendColumn(const Vector<T>& column);
    /// @brief Updates the factorization for a removed row (O(n^2)).
    /// @param index Index of the row to remove.
    void removeRow(Index index);
    /// @brief Rank-one update, re-factoring A + u v^H (O(n^2)).
    /// @param u Left update vector.
    /// @param v Right update vector.
    void rankOneUpdate(const Vector<T>& u, const Vector<T>& v);

private:
    /// @brief Performs the unblocked (level-2) factorization.
    void unblockedFactorize();
    /// @brief Performs the blocked factorization.
    /// @param blockSize Panel width for the blocked algorithm.
    void blockedFactorize(Index blockSize);

    Matrix<T>              factors_;
    HouseholderSequence<T> reflectors_;
    Options                options_;
    bool                   computed_;
};

// A P = Q R with column pivoting: rank-revealing, and the basis for
// least squares on rank-deficient systems.
template <typename T>
class ColPivQR {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    /// @brief Constructs an empty factorization; call compute() before use.
    ColPivQR();
    /// @brief Constructs and immediately factorizes a matrix.
    /// @param a Matrix to factorize.
    explicit ColPivQR(const Matrix<T>& a);

    /// @brief Computes the A P = Q R factorization with column pivoting.
    /// @param a Matrix to factorize.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has succeeded.
    bool isComputed() const;

    /// @brief Materializes the orthogonal factor Q.
    /// @return The Q factor.
    Matrix<T>                 matrixQ() const;
    /// @brief Materializes the upper-triangular factor R.
    /// @return The R factor.
    Matrix<T>                 matrixR() const;
    /// @brief Builds the column permutation matrix P.
    /// @return The permutation matrix.
    Matrix<T>                 permutationMatrix() const;
    /// @brief The column pivot order.
    /// @return Reference to the column-pivot vector.
    const std::vector<Index>& columnPivots() const;

    // solve is basic least squares (zeroes the free variables on rank
    // deficiency); solveMinimumNorm returns the least-squares solution of
    // smallest ||x||. rank counts |R(i,i)| above tolerance * |R(0,0)|.
    /// @brief Basic least-squares solve (free variables zeroed on rank deficiency).
    /// @param b Right-hand side vector.
    /// @return A least-squares solution x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Minimum-norm least-squares solve (smallest ||x||).
    /// @param b Right-hand side vector.
    /// @return The minimum-norm least-squares solution.
    Vector<T> solveMinimumNorm(const Vector<T>& b) const;
    /// @brief Numerical rank from |R(i,i)| above tolerance * |R(0,0)|.
    /// @param tolerance Relative cutoff.
    /// @return The numerical rank.
    Index     rank(Real tolerance) const;
    /// @brief Basis for the numerical null space of A.
    /// @param tolerance Relative rank cutoff.
    /// @return A matrix whose columns span the kernel.
    Matrix<T> kernel(Real tolerance) const;
    /// @brief Cheap reciprocal condition-number estimate.
    /// @return An estimate of 1 / cond(A) in [0, 1].
    Real      reciprocalConditionEstimate() const;

private:
    Matrix<T>              factors_;
    HouseholderSequence<T> reflectors_;
    std::vector<Index>     colPivots_;
    Vector<T>              columnNorms_;
    bool                   computed_;
};

// Gram-Schmidt, kept for teaching and for the modified/re-orthogonalized
// variants that Krylov methods actually need.
template <typename T>
class GramSchmidt {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // Classical: one pass, loses orthogonality as cond(A) grows.
    // Modified: orthogonalize against each q_j in sequence; much better.
    // ClassicalTwice: classical with reorthogonalization ("twice is
    // enough"), the strongest of the three.
    struct Variant {
        enum class Kind { Classical, Modified, ClassicalTwice };
    };

    /// @brief Constructs an empty factorization; call compute() before use.
    GramSchmidt();
    /// @brief Constructs and immediately orthogonalizes a matrix.
    /// @param a Matrix whose columns are orthogonalized.
    /// @param variant Which Gram-Schmidt variant to use.
    GramSchmidt(const Matrix<T>& a, typename Variant::Kind variant);

    /// @brief Orthogonalizes the columns of a matrix into Q and R.
    /// @param a Matrix to factorize.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has succeeded.
    bool isComputed() const;

    /// @brief The orthonormal factor Q.
    /// @return The Q factor.
    Matrix<T> matrixQ() const;
    /// @brief The upper-triangular factor R.
    /// @return The R factor.
    Matrix<T> matrixR() const;
    /// @brief Loss of orthogonality, ||Q^H Q - I||.
    /// @return The orthogonality-loss metric.
    Real      orthogonalityLoss() const;

private:
    Matrix<T>              q_;
    Matrix<T>              r_;
    typename Variant::Kind variant_;
    bool                   computed_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class QR<SCALAR>; \
    extern template class ColPivQR<SCALAR>; \
    extern template class GramSchmidt<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
