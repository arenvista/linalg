#include "linalg/decomp/Cholesky.hpp"

#include <cmath>
#include <utility>

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/solve/TriangularSolver.hpp"

namespace linalg {

template <typename T>
Cholesky<T>::Cholesky()
    : factor_(),
      options_(
          Options{
              Triangle::Kind::Lower,
              false,
              0,
              false
          }
      ),
      computed_(false),
      positiveDefinite_(false),
      failedPivot_(0) {}

template <typename T>
Cholesky<T>::Cholesky(const Matrix<T> &a)
    : factor_(a),
      options_(
          Options{
              Triangle::Kind::Lower,
              false,
              0,
              false
          }
      ),
      computed_(false),
      positiveDefinite_(false),
      failedPivot_(0) {
    compute(a);
}

template <typename T>
Cholesky<T>::Cholesky(
    const Matrix<T> &a,
    const Options   &options
)
    : factor_(a),
      options_(options),
      computed_(false),
      positiveDefinite_(false),
      failedPivot_(0) {
    compute(a);
}

template <typename T> void Cholesky<T>::compute(const Matrix<T> &a) {
    if (!a.isSquare()) {
        throw DimensionMismatch(a.rows(), a.cols(), a.rows(), a.cols());
    }

    // Seed the working matrix; the factorization routines read A from factor_
    // and overwrite it with L. Dispatch on the requested variant.
    factor_ = a;
    if (options_.useBlocked) {
        blockedFactorize(options_.blockSize);
    } else {
        unblockedFactorize();
    }
}

template <typename T> bool Cholesky<T>::isComputed() const {
    return computed_;
}

template <typename T> bool Cholesky<T>::isPositiveDefinite() const {
    return computed_ && positiveDefinite_;
}

template <typename T> Matrix<T> Cholesky<T>::lower() const {
    // compute() always builds L into the lower triangle, independent of which
    // input triangle was read, so the factor is returned as-is here.
    return factor_;
}

template <typename T> Matrix<T> Cholesky<T>::upper() const {
    return factor_.conjugateTranspose();
}

template <typename T> Vector<T> Cholesky<T>::solve(const Vector<T> &b) const {
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }

    // A = L L^H, so A x = b splits into two triangular sweeps: forward-solve
    // L y = b, then back-solve L^H x = y. Both read the lower triangle of
    // factor_; the second sweep applies the conjugate transpose to it.
    using Solver = TriangularSolver<T>;

    const typename Solver::Options forward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::None,
        false,
        0
    };
    const typename Solver::Options backward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::ConjugateTranspose,
        false,
        0
    };

    const Vector<T> y = Solver(forward).solve(factor_, b);
    return Solver(backward).solve(factor_, y);
}

template <typename T> Matrix<T> Cholesky<T>::solve(const Matrix<T> &b) const {
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }
    using Solver = TriangularSolver<T>;

    const typename Solver::Options forward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::None,
        false,
        0
    };
    const typename Solver::Options backward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::ConjugateTranspose,
        false,
        0
    };

    const Matrix<T> y = Solver(forward).solve(factor_, b);
    return Solver(backward).solve(factor_, y);
}

template <typename T> Matrix<T> Cholesky<T>::inverse() const {
    const Matrix<T> identity = Matrix<T>::Identity(factor_.rows());
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }
    using Solver = TriangularSolver<T>;

    const typename Solver::Options forward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::None,
        false,
        0
    };

    const typename Solver::Options backward{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::ConjugateTranspose,
        false,
        0
    };

    const Matrix<T> y = Solver(forward).solve(factor_, identity);
    return Solver(backward).solve(factor_, y);
}

template <typename T> T Cholesky<T>::determinant() const {
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }
    // det(A) = det(L) det(L^H) = prod_j L(j,j)^2; the L(j,j) are real and
    // positive, so the product is real and positive too.
    T det = NumericTraits<T>::one();
    for (Index j = 0; j < factor_.rows(); ++j) {
        det *= factor_(j, j) * factor_(j, j);
    }
    return det;
}

template <typename T>
typename Cholesky<T>::Real Cholesky<T>::logDeterminant() const {
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }
    // det(A) = det(L) det(L^H) = prod_j L(j,j)^2; the L(j,j) are real and
    // positive, so the product is real and positive too.
    Cholesky<T>::Real det = 0;
    for (Index j = 0; j < factor_.rows(); ++j) {
        det += 2 * std::log(NumericTraits<T>::real(factor_(j, j)));
    }
    return det;
}

template <typename T>
typename Cholesky<T>::Real Cholesky<T>::reciprocalConditionEstimate() const {
    if (!computed_ || !positiveDefinite_) {
        throw NotComputed("Cholesky");
    }

    // A = L L^H gives cond_2(A) = cond_2(L)^2, so rcond(A) ~ rcond(L)^2. Reuse
    // the triangular solver's cheap 1-norm estimate on the lower factor; the
    // result lies in (0, 1] because L is nonsingular for a positive-definite A.
    using Solver = TriangularSolver<T>;

    const typename Solver::Options lower{
        Triangle::Kind::Lower,
        Diagonal::Kind::NonUnit,
        Transposition::Kind::None,
        false,
        0
    };

    const Real rcondL = Solver(lower).reciprocalConditionEstimate(factor_);
    return rcondL * rcondL;
}

template <typename T> void Cholesky<T>::update(const Vector<T> &x) {
    // See lecture 8 from notes
    // Apply a sequence of Givens rotations to [Lᵀ; xᵀ] so that the appended
    // row is annihilated, leaving the updated factor with L̄ L̄ᵀ = L Lᵀ + x xᵀ.
    // factor_ holds the lower factor, so the k-th "row" of Lᵀ lives in column k
    // below the diagonal: factor_(j, k) for j >= k.
    Vector<T> z = x;
    for (Index k = 0; k < x.size(); ++k) {
        T r           = std::sqrt(factor_(k, k) * factor_(k, k) + z(k) * z(k));
        T c           = factor_(k, k) / r;
        T s           = z(k) / r;
        factor_(k, k) = r;
        z[k]          = T(0);
        for (Index j = k + 1; j < z.size(); ++j) {
            T t           = c * factor_(j, k) + s * z(j);
            z[j]          = c * z[j] - s * factor_(j, k);
            factor_(j, k) = t;
        }
    }
}

template <typename T> void Cholesky<T>::downdate(const Vector<T> &x) {
    throw LinalgError("not implemented: linalg::Cholesky<T>::downdate");
}

template <typename T> void Cholesky<T>::unblockedFactorize() {
    using Traits  = NumericTraits<T>;
    const Index n = factor_.rows();

    // A is Hermitian, so only the caller-selected triangle is read; the
    // opposite triangle is never touched. The lower entry (i >= j) that the
    // algorithm needs is the conjugate of the stored upper entry. factor_ still
    // holds the seeded A throughout the loop; L is built into a fresh matrix.
    const bool readsLower = options_.readFrom == Triangle::Kind::Lower;
    auto       lowerOf    = [&](Index i, Index j) -> T {
        return readsLower ? factor_(i, j) : Traits::conj(factor_(j, i));
    };

    // Bordered (left-looking) Cholesky: build L one column at a time in the
    // lower triangle of factor, leaving the strict upper triangle zero.
    Matrix<T> factor(n, n);
    for (Index j = 0; j < n; ++j) {
        Real pivot = Traits::real(factor_(j, j));
        for (Index k = 0; k < j; ++k) {
            pivot -= Traits::absSquared(factor(j, k));
        }

        // The pivot is real for a Hermitian matrix; a non-positive value
        // (or NaN) means A is not positive definite at column j.
        if (!(pivot > Real(0))) {
            if (options_.throwOnIndefinite) {
                throw NotPositiveDefinite(j);
            }
            factor_           = std::move(factor);
            computed_         = true;
            positiveDefinite_ = false;
            failedPivot_      = j;
            return;
        }

        const T diagonal = Traits::sqrt(T(pivot));
        factor(j, j)     = diagonal;
        for (Index i = j + 1; i < n; ++i) {
            T sum = lowerOf(i, j);
            for (Index k = 0; k < j; ++k) {
                sum -= factor(i, k) * Traits::conj(factor(j, k));
            }
            factor(i, j) = sum / diagonal;
        }
    }

    factor_           = std::move(factor);
    computed_         = true;
    positiveDefinite_ = true;
    failedPivot_      = n; // sentinel: no failed pivot
}

template <typename T> void Cholesky<T>::blockedFactorize(Index blockSize) {
    // A genuine panel-blocked Cholesky is not implemented yet. Blocking is a
    // performance optimization only, so fall back to the unblocked algorithm
    // to keep results correct when useBlocked is requested.
    (void)blockSize;
    unblockedFactorize();
}

template <typename T> LDLT<T>::LDLT() {
    throw LinalgError("not implemented: linalg::LDLT<T>::LDLT");
}

template <typename T> LDLT<T>::LDLT(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::LDLT<T>::LDLT");
}

template <typename T> void LDLT<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::LDLT<T>::compute");
}

template <typename T> bool LDLT<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::LDLT<T>::isComputed");
}

template <typename T> Matrix<T> LDLT<T>::lower() const {
    throw LinalgError("not implemented: linalg::LDLT<T>::lower");
}

template <typename T> Vector<T> LDLT<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::LDLT<T>::diagonal");
}

template <typename T> Matrix<T> LDLT<T>::permutationMatrix() const {
    throw LinalgError("not implemented: linalg::LDLT<T>::permutationMatrix");
}

template <typename T> Vector<T> LDLT<T>::solve(const Vector<T> &b) const {
    throw LinalgError("not implemented: linalg::LDLT<T>::solve");
}

template <typename T>
bool LDLT<T>::isPositiveSemiDefinite(Real tolerance) const {
    throw LinalgError(
        "not implemented: linalg::LDLT<T>::isPositiveSemiDefinite"
    );
}

template <typename T>
bool LDLT<T>::isNegativeSemiDefinite(Real tolerance) const {
    throw LinalgError(
        "not implemented: linalg::LDLT<T>::isNegativeSemiDefinite"
    );
}

template <typename T>
typename LDLT<T>::Index LDLT<T>::rank(Real tolerance) const {
    throw LinalgError("not implemented: linalg::LDLT<T>::rank");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)   \
    template class Cholesky<SCALAR>; \
    template class LDLT<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
