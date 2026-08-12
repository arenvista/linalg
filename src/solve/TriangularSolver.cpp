#include "linalg/solve/TriangularSolver.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

namespace {

// The effective operator op(A) is triangular; which triangle depends on both
// the stored triangle and whether op transposes. A^T (or A^H) of a lower
// triangle is upper and vice versa.
bool isEffectiveLower(
    Triangle::Kind      uplo,
    Transposition::Kind trans
) {
    const bool transposed = trans != Transposition::Kind::None;
    return (uplo == Triangle::Kind::Lower) != transposed;
}

// Solves op(A) x = b in place, x holding b on entry and the solution on exit.
// op(A)(i, j) is read lazily from the stored triangle so the opposite triangle
// is never touched. Forward substitution for an effectively-lower operator,
// back substitution for an effectively-upper one.
template <typename T>
void substitute(
    const Matrix<T>    &a,
    Vector<T>          &x,
    Triangle::Kind      uplo,
    Diagonal::Kind      diag,
    Transposition::Kind trans
) {
    using Traits = NumericTraits<T>;
    using Real   = typename Traits::Real;
    using Index  = std::size_t;

    if (a.rows() != a.cols() || a.cols() != x.size()) {
        throw DimensionMismatch(a.rows(), a.cols(), x.size(), 1);
    }

    const Index n          = a.rows();
    const bool  transposed = trans != Transposition::Kind::None;
    const bool  conjugate  = trans == Transposition::Kind::ConjugateTranspose;
    const bool  unit       = diag == Diagonal::Kind::Unit;

    // Element (i, j) of op(A).
    auto op = [&](Index i, Index j) -> T {
        const T &v = transposed ? a(j, i) : a(i, j);
        return conjugate ? Traits::conj(v) : v;
    };

    if (isEffectiveLower(uplo, trans)) {
        for (Index i = 0; i < n; ++i) {
            T s = x(i);
            for (Index j = 0; j < i; ++j) {
                s -= op(i, j) * x(j);
            }
            if (!unit) {
                const T d = op(i, i);
                if (Traits::isApproxZero(d, Real(0))) {
                    throw SingularMatrix(i);
                }
                s /= d;
            }
            x(i) = s;
        }
    } else {
        for (Index i = n; i-- > 0;) {
            T s = x(i);
            for (Index j = i + 1; j < n; ++j) {
                s -= op(i, j) * x(j);
            }
            if (!unit) {
                const T d = op(i, i);
                if (Traits::isApproxZero(d, Real(0))) {
                    throw SingularMatrix(i);
                }
                s /= d;
            }
            x(i) = s;
        }
    }
}

// Materializes op(A) as a dense triangular matrix with the untouched triangle
// zeroed and the diagonal filled with ones under a unit-diagonal assumption.
// Used for the one-norm in the condition estimate.
template <typename T>
Matrix<T> effectiveOperator(
    const Matrix<T>    &a,
    Triangle::Kind      uplo,
    Diagonal::Kind      diag,
    Transposition::Kind trans
) {
    using Traits = NumericTraits<T>;
    using Index  = std::size_t;

    const Index n          = a.rows();
    const bool  transposed = trans != Transposition::Kind::None;
    const bool  conjugate  = trans == Transposition::Kind::ConjugateTranspose;
    const bool  lower      = isEffectiveLower(uplo, trans);

    auto op = [&](Index i, Index j) -> T {
        const T &v = transposed ? a(j, i) : a(i, j);
        return conjugate ? Traits::conj(v) : v;
    };

    Matrix<T> m(n, n);
    for (Index i = 0; i < n; ++i) {
        const Index begin = lower ? 0 : i;
        const Index end   = lower ? i + 1 : n;
        for (Index j = begin; j < end; ++j) {
            m(i, j) = op(i, j);
        }
        if (diag == Diagonal::Kind::Unit) {
            m(i, i) = Traits::one();
        }
    }
    return m;
}

} // namespace

template <typename T>
TriangularSolver<T>::TriangularSolver()
    : options_(Options()) {}

template <typename T>
TriangularSolver<T>::TriangularSolver(const Options &options)
    : options_(options) {}

template <typename T>
Vector<T> TriangularSolver<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) const {
    Vector<T> x = b;
    solveInPlace(a, x);
    return x;
}

template <typename T>
Matrix<T> TriangularSolver<T>::solve(
    const Matrix<T> &a,
    const Matrix<T> &b
) const {
    Matrix<T> x = b;
    solveInPlace(a, x);
    return x;
}

template <typename T>
void TriangularSolver<T>::solveInPlace(
    const Matrix<T> &a,
    Vector<T>       &b
) const {
    substitute(a, b, options_.uplo, options_.diagonal, options_.transposition);
}

template <typename T>
void TriangularSolver<T>::solveInPlace(
    const Matrix<T> &a,
    Matrix<T>       &b
) const {
    if (a.rows() != a.cols() || a.cols() != b.rows()) {
        throw DimensionMismatch(a.rows(), a.cols(), b.rows(), b.cols());
    }
    // Each column of b is an independent right-hand side.
    for (Index j = 0; j < b.cols(); ++j) {
        Vector<T> column = b.col(j);
        substitute(
            a,
            column,
            options_.uplo,
            options_.diagonal,
            options_.transposition
        );
        b.setCol(j, column);
    }
}

template <typename T>
Matrix<T> TriangularSolver<T>::inverse(const Matrix<T> &a) const {
    if (a.rows() != a.cols()) {
        throw DimensionMismatch(a.rows(), a.cols(), a.rows(), a.cols());
    }
    // op(A) * X = I; substitution preserves the triangular structure, so X
    // comes out triangular of the same kind.
    Matrix<T> x = Matrix<T>::Identity(a.rows());
    solveInPlace(a, x);
    return x;
}

template <typename T>
typename TriangularSolver<T>::Real TriangularSolver<
    T>::reciprocalConditionEstimate(const Matrix<T> &a) const {
    // rcond = 1 / (||op(A)||_1 * ||op(A)^-1||_1), clamped to 0 when
    // singular.
    Matrix<T> m = effectiveOperator(
        a,
        options_.uplo,
        options_.diagonal,
        options_.transposition
    );
    const Real normA = m.oneNorm();
    if (normA == Real(0)) {
        return Real(0);
    }
    Real normInv;
    try {
        normInv = inverse(a).oneNorm();
    } catch (const SingularMatrix &) {
        return Real(0);
    }
    if (normInv == Real(0)) {
        return Real(0);
    }
    return Real(1) / (normA * normInv);
}

template <typename T>
bool TriangularSolver<T>::isNonsingular(
    const Matrix<T> &a,
    Real             tolerance
) const {
    // A unit diagonal is an implied all-ones diagonal, always nonsingular.
    if (options_.diagonal == Diagonal::Kind::Unit) {
        return true;
    }
    const Index n = a.rows() < a.cols() ? a.rows() : a.cols();
    for (Index i = 0; i < n; ++i) {
        if (NumericTraits<T>::abs(a(i, i)) <= tolerance) {
            return false;
        }
    }
    return true;
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class TriangularSolver<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
