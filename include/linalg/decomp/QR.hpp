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

    QR();
    explicit QR(const Matrix<T>& a);
    QR(const Matrix<T>& a, const Options& options);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Matrix<T>                       matrixQ() const;
    Matrix<T>                       matrixR() const;
    const HouseholderSequence<T>&   reflectors() const;

    void applyQOnLeft(Matrix<T>& target) const;
    void applyQTransposeOnLeft(Matrix<T>& target) const;

    // solve expects square nonsingular A; solveLeastSquares minimizes
    // ||Ax - b|| for m >= n. Both are Q^H * b then a back-substitution.
    Vector<T> solve(const Vector<T>& b) const;
    Matrix<T> solve(const Matrix<T>& b) const;
    Vector<T> solveLeastSquares(const Vector<T>& b) const;

    T     absDeterminant() const;      // product of |R(i,i)|; Q contributes only a unit factor
    Real  logAbsDeterminant() const;
    Index rank(Real tolerance) const;  // from |R(i,i)|; unpivoted QR is NOT rank-revealing — prefer ColPivQR

    // Updating and downdating without recomputing from scratch: O(n^2)
    // Givens repair work per modification instead of an O(n^3) refactor.
    void appendRow(const Vector<T>& row);
    void appendColumn(const Vector<T>& column);
    void removeRow(Index index);
    void rankOneUpdate(const Vector<T>& u, const Vector<T>& v);  // factors A + u v^H

private:
    void unblockedFactorize();
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

    ColPivQR();
    explicit ColPivQR(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Matrix<T>                 matrixQ() const;
    Matrix<T>                 matrixR() const;
    Matrix<T>                 permutationMatrix() const;
    const std::vector<Index>& columnPivots() const;

    // solve is basic least squares (zeroes the free variables on rank
    // deficiency); solveMinimumNorm returns the least-squares solution of
    // smallest ||x||. rank counts |R(i,i)| above tolerance * |R(0,0)|.
    Vector<T> solve(const Vector<T>& b) const;
    Vector<T> solveMinimumNorm(const Vector<T>& b) const;
    Index     rank(Real tolerance) const;
    Matrix<T> kernel(Real tolerance) const;  // columns span the (numerical) null space
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

    GramSchmidt();
    GramSchmidt(const Matrix<T>& a, typename Variant::Kind variant);

    void compute(const Matrix<T>& a);
    bool isComputed() const;

    Matrix<T> matrixQ() const;
    Matrix<T> matrixR() const;
    Real      orthogonalityLoss() const;  // ||Q^H Q - I||: the metric the variants differ on

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
