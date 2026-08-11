#include "linalg/decomp/Cholesky.hpp"

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
              true
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
              true
          }
      ),
      computed_(false),
      positiveDefinite_(false),
      failedPivot_(0) {}

template <typename T>
Cholesky<T>::Cholesky(
    const Matrix<T> &a,
    const Options   &options
)
    : factor_(a),
      options_(options),
      computed_(false),
      positiveDefinite_(false),
      failedPivot_(0) {}

template <typename T> void Cholesky<T>::compute(const Matrix<T> &a) {
    if (!a.isSquare()) {
        throw DimensionMismatch(a.rows(), a.cols(), a.rows(), a.cols());
    }

    using Traits  = NumericTraits<T>;
    const Index n = a.rows();

    // A is Hermitian, so only the caller-selected triangle is read; the
    // opposite triangle is never touched. The lower entry (i >= j) that the
    // algorithm needs is the conjugate of the stored upper entry.
    const bool readsLower = options_.readFrom == Triangle::Kind::Lower;
    auto       lowerOf    = [&](Index i, Index j) -> T {
        return readsLower ? a(i, j) : Traits::conj(a(j, i));
    };

    // Bordered (left-looking) Cholesky: build L one column at a time in the
    // lower triangle of factor_, leaving the strict upper triangle zero.
    Matrix<T> factor(n, n);
    for (Index j = 0; j < n; ++j) {
        Real pivot = Traits::real(a(j, j));
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

template <typename T> bool Cholesky<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Cholesky<T>::isComputed");
}

template <typename T> bool Cholesky<T>::isPositiveDefinite() const {
    throw LinalgError(
        "not implemented: linalg::Cholesky<T>::isPositiveDefinite"
    );
}

template <typename T> Matrix<T> Cholesky<T>::lower() const {
    throw LinalgError("not implemented: linalg::Cholesky<T>::lower");
}

template <typename T> Matrix<T> Cholesky<T>::upper() const {
    throw LinalgError("not implemented: linalg::Cholesky<T>::upper");
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
    throw LinalgError("not implemented: linalg::Cholesky<T>::logDeterminant");
}

template <typename T>
typename Cholesky<T>::Real Cholesky<T>::reciprocalConditionEstimate() const {
    throw LinalgError(
        "not implemented: linalg::Cholesky<T>::reciprocalConditionEstimate"
    );
}

template <typename T> void Cholesky<T>::update(const Vector<T> &x) {
    throw LinalgError("not implemented: linalg::Cholesky<T>::update");
}

template <typename T> void Cholesky<T>::downdate(const Vector<T> &x) {
    throw LinalgError("not implemented: linalg::Cholesky<T>::downdate");
}

template <typename T> void Cholesky<T>::unblockedFactorize() {
    throw LinalgError(
        "not implemented: linalg::Cholesky<T>::unblockedFactorize"
    );
}

template <typename T> void Cholesky<T>::blockedFactorize(Index blockSize) {
    throw LinalgError("not implemented: linalg::Cholesky<T>::blockedFactorize");
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
