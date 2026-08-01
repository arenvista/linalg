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

    LU();
    explicit LU(const Matrix<T>& a);
    LU(const Matrix<T>& a, const Options& options);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    // packedFactors is L and U in one matrix (L's unit diagonal implicit);
    // lower/upper unpack copies. pivots() uses the LAPACK ipiv convention:
    // at step k, row k was swapped with row pivots()[k].
    Matrix<T>                 packedFactors() const;
    Matrix<T>                 lower() const;
    Matrix<T>                 upper() const;
    Matrix<T>                 permutationMatrix() const;
    const std::vector<Index>& pivots() const;
    Index                     permutationSign() const;  // row swaps mod 2: det(P) = (-1)^sign

    // Solves reuse the factorization: permute b, then two triangular
    // solves. solveTranspose solves A^T x = b from the same factors.
    // refineSolution runs iterative refinement, x += solve(b - A x),
    // which needs the original A alongside the factors.
    Vector<T> solve(const Vector<T>& b) const;
    Matrix<T> solve(const Matrix<T>& b) const;
    Vector<T> solveTranspose(const Vector<T>& b) const;
    Vector<T> refineSolution(const Matrix<T>& a, const Vector<T>& b,
                             const Vector<T>& x, Index steps) const;

    T     determinant() const;         // product of U's diagonal times det(P)
    T     logAbsDeterminant() const;   // log|det|, immune to over/underflow
    Matrix<T> inverse() const;
    Index rank(Real tolerance) const;  // from |U(i,i)|; only trustworthy-ish with pivoting
    bool  isSingular(Real tolerance) const;
    Real  reciprocalConditionEstimate() const; // Hager/Higham 1-norm estimate; cheap after compute()

private:
    void unblockedFactorize();
    void blockedFactorize(Index blockSize);
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

    FullPivLU();
    explicit FullPivLU(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Matrix<T> lower() const;
    Matrix<T> upper() const;
    Matrix<T> rowPermutation() const;
    Matrix<T> columnPermutation() const;

    Vector<T> solve(const Vector<T>& b) const;
    Matrix<T> kernel() const;  // columns span the null space
    Matrix<T> image() const;   // columns span the column space
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
