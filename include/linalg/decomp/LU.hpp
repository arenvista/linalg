#pragma once

#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// PA = LU with partial pivoting. Factors are packed into one matrix;
// L is unit lower, U is upper.
template <typename T>
class LU {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Options {
        bool  useBlocked;      // right-looking blocked factorization (perf only)
        Index blockSize;
        Real  pivotTolerance;  // pivots with |p| <= this count as zero
        bool  throwOnSingular; // throw SingularMatrix vs record and carry on
    };

    /// @brief Constructs an empty factorization; call compute() before use.
    LU();
    /// @brief Constructs and immediately factorizes a matrix with defaults.
    /// @param a Matrix to factorize.
    explicit LU(const Matrix<T>& a);
    /// @brief Constructs and immediately factorizes a matrix with options.
    /// @param a Matrix to factorize.
    /// @param options Factorization options (blocking, pivot tolerance, ...).
    LU(const Matrix<T>& a, const Options& options);

    /// @brief Computes the PA = LU factorization of a matrix.
    /// @param a Matrix to factorize.
    /// @throws SingularMatrix if singular and options.throwOnSingular is set.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has succeeded.
    bool isComputed() const;

    // packedFactors is L and U in one matrix (L's unit diagonal implicit);
    // lower/upper unpack copies. pivots() uses the LAPACK ipiv convention:
    // at step k, row k was swapped with row pivots()[k].
    /// @brief The L and U factors packed into a single matrix (L's unit
    /// diagonal implicit).
    /// @return The packed factor matrix.
    Matrix<T>                 packedFactors() const;
    /// @brief Unpacks the unit lower-triangular factor L.
    /// @return The L factor as a full matrix.
    Matrix<T>                 lower() const;
    /// @brief Unpacks the upper-triangular factor U.
    /// @return The U factor as a full matrix.
    Matrix<T>                 upper() const;
    /// @brief Builds the row permutation matrix P from the pivots.
    /// @return The permutation matrix.
    Matrix<T>                 permutationMatrix() const;
    /// @brief The pivot sequence (LAPACK ipiv convention).
    /// @return Reference to the pivot vector.
    const std::vector<Index>& pivots() const;
    /// @brief Parity of the row swaps; det(P) = (-1)^permutationSign().
    /// @return The row-swap count modulo 2.
    Index                     permutationSign() const;

    // Solves reuse the factorization: permute b, then two triangular
    // solves. solveTranspose solves A^T x = b from the same factors.
    // refineSolution runs iterative refinement, x += solve(b - A x),
    // which needs the original A alongside the factors.
    /// @brief Solves A x = b for a single right-hand side.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Solves A X = B for multiple right-hand sides.
    /// @param b Right-hand side matrix (one column per system).
    /// @return The solution matrix X.
    Matrix<T> solve(const Matrix<T>& b) const;
    /// @brief Solves A^T x = b using the existing factors.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solveTranspose(const Vector<T>& b) const;
    /// @brief Improves a solution via iterative refinement, x += solve(b - A x).
    /// @param a The original (unfactorized) matrix A.
    /// @param b The original right-hand side.
    /// @param x The approximate solution to refine.
    /// @param steps Number of refinement iterations.
    /// @return The refined solution.
    Vector<T> refineSolution(const Matrix<T>& a, const Vector<T>& b,
                             const Vector<T>& x, Index steps) const;

    /// @brief Determinant, the product of U's diagonal times det(P).
    /// @return The determinant.
    T     determinant() const;
    /// @brief Logarithm of the absolute determinant; immune to over/underflow.
    /// @return log|det(A)|.
    T     logAbsDeterminant() const;
    /// @brief Matrix inverse computed from the factors.
    /// @return The inverse matrix.
    Matrix<T> inverse() const;
    /// @brief Rank estimate from the magnitudes of U's diagonal.
    /// @param tolerance Cutoff below which a diagonal entry counts as zero.
    /// @return The estimated rank.
    Index rank(Real tolerance) const;
    /// @brief Reports whether the matrix is (numerically) singular.
    /// @param tolerance Pivot magnitude cutoff for singularity.
    /// @return True if a pivot is at or below tolerance.
    bool  isSingular(Real tolerance) const;
    /// @brief Cheap 1-norm reciprocal condition estimate (Hager/Higham).
    /// @return An estimate of 1 / cond_1(A) in [0, 1].
    Real  reciprocalConditionEstimate() const;

private:
    /// @brief Performs the unblocked (level-2) factorization.
    void unblockedFactorize();
    /// @brief Performs the right-looking blocked factorization.
    /// @param blockSize Panel width for the blocked algorithm.
    void blockedFactorize(Index blockSize);
    /// @brief Applies the recorded row pivots to a right-hand side in place.
    /// @param b Right-hand side to permute.
    void applyPivotsTo(Matrix<T>& b) const;

    Matrix<T>          factors_;
    std::vector<Index> pivots_;
    Options            options_;
    Index              sign_;
    bool               computed_;
    bool               singular_;
};

// Full pivoting variant: rank-revealing, slower, used when the matrix
// is suspected to be rank deficient.
template <typename T>
class FullPivLU {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    /// @brief Constructs an empty factorization; call compute() before use.
    FullPivLU();
    /// @brief Constructs and immediately factorizes a matrix.
    /// @param a Matrix to factorize.
    explicit FullPivLU(const Matrix<T>& a);

    /// @brief Computes the full-pivoting LU factorization of a matrix.
    /// @param a Matrix to factorize.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has succeeded.
    bool isComputed() const;

    /// @brief Unpacks the unit lower-triangular factor L.
    /// @return The L factor as a full matrix.
    Matrix<T> lower() const;
    /// @brief Unpacks the upper-triangular factor U.
    /// @return The U factor as a full matrix.
    Matrix<T> upper() const;
    /// @brief Builds the row permutation matrix.
    /// @return The row permutation matrix.
    Matrix<T> rowPermutation() const;
    /// @brief Builds the column permutation matrix.
    /// @return The column permutation matrix.
    Matrix<T> columnPermutation() const;

    /// @brief Solves A x = b using the factorization.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Basis for the null space of A.
    /// @return A matrix whose columns span the kernel.
    Matrix<T> kernel() const;
    /// @brief Basis for the column space of A.
    /// @return A matrix whose columns span the image.
    Matrix<T> image() const;
    /// @brief Rank revealed by the full-pivoting factorization.
    /// @param tolerance Cutoff below which a pivot counts as zero.
    /// @return The estimated rank.
    Index     rank(Real tolerance) const;

private:
    Matrix<T>          factors_;
    std::vector<Index> rowPivots_;
    std::vector<Index> colPivots_;
    bool               computed_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class LU<SCALAR>; \
    extern template class FullPivLU<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
