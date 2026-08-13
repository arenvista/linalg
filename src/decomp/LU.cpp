#include "linalg/decomp/LU.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/solve/TriangularSolver.hpp"

#include <cmath>
#include <vector>

namespace linalg {

namespace {

template <typename T> typename LU<T>::Options defaultLUOptions() {
    typename LU<T>::Options options;
    options.useBlocked = false; // !!!pending step 9!!!
    options.blockSize  = 64;    // PH Pending step 9

    // LAPACK getrf semantics: only an exactly zero pivot is singular.
    // Callers wanting a looser cutoff pass their own tolerance.
    options.pivotTolerance  = typename LU<T>::Real{};
    options.throwOnSingular = false;
    return options;
}

} // namespace

template <typename T>
LU<T>::LU()
    : factors_(),
      pivots_(),
      options_(defaultLUOptions<T>()),
      sign_(Index{}),
      computed_(false),
      singular_(false) {}

template <typename T>
LU<T>::LU(const Matrix<T> &a)
    : LU(a,
         defaultLUOptions<T>()) {}

template <typename T>
LU<T>::LU(const Matrix<T> &a,
          const Options   &options)
    : factors_(),
      pivots_(),
      options_(options),
      sign_(Index{}),
      computed_(false),
      singular_(false) {
    compute(a);
}

template <typename T> void LU<T>::compute(const Matrix<T> &a) {
    // Factorization runs in place, so the packed factors start as a copy.
    factors_ = a;
    pivots_.assign(a.rows() < a.cols() ? a.rows() : a.cols(), Index{});
    sign_     = Index{};
    computed_ = false;
    singular_ = false;

    if (options_.useBlocked) {
        blockedFactorize(options_.blockSize);
    } else {
        unblockedFactorize();
    }

    // Skipped when throwOnSingular escapes: half-eliminated factors are not
    // usable.
    computed_ = true;
}

template <typename T> bool LU<T>::isComputed() const { return computed_; }

template <typename T> Matrix<T> LU<T>::packedFactors() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    return factors_;
}

template <typename T> Matrix<T> LU<T>::lower() const {
    if (!computed_) {
        throw NotComputed("LU");
    }

    // m x min(m, n), unit lower trapezoidal, so L * U keeps a's shape.
    const Index m = factors_.rows();
    const Index k = m < factors_.cols() ? m : factors_.cols();

    Matrix<T> lowerMatrix(m, k);
    for (Index i = 0; i < m; ++i) {
        const Index strictlyLower = i < k ? i : k;
        for (Index j = 0; j < strictlyLower; ++j) {
            lowerMatrix(i, j) = factors_(i, j);
        }
        if (i < k) {
            lowerMatrix(i, i) = NumericTraits<T>::one(); // implicit in factors_
        }
    }
    return lowerMatrix;
}

template <typename T> Matrix<T> LU<T>::upper() const {
    if (!computed_) {
        throw NotComputed("LU");
    }

    const Index n = factors_.cols();
    const Index k = factors_.rows() < n ? factors_.rows() : n;

    Matrix<T> upperMatrix(k, n);
    for (Index i = 0; i < k; ++i) {
        for (Index j = i; j < n; ++j) {
            upperMatrix(i, j) = factors_(i, j);
        }
    }
    return upperMatrix;
}

template <typename T> Matrix<T> LU<T>::permutationMatrix() const {
    if (!computed_) {
        throw NotComputed("LU");
    }

    // ipiv records one swap per step, so it has to be replayed in order.
    Matrix<T> permMatrix = Matrix<T>::Identity(factors_.rows());
    for (Index k = 0; k < pivots_.size(); ++k) {
        permMatrix.swapRows(k, pivots_[k]);
    }
    return permMatrix;
}

template <typename T>
const std::vector<typename LU<T>::Index> &LU<T>::pivots() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    return pivots_;
}

template <typename T> typename LU<T>::Index LU<T>::permutationSign() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    return sign_;
}

template <typename T> Vector<T> LU<T>::solve(const Vector<T> &b) const {
    // A single right-hand side is a one-column matrix; the matrix overload
    // owns the pivot-then-substitute sequence.
    Matrix<T> rhs(b.size(), 1);
    rhs.setCol(0, b);
    return solve(rhs).col(0);
}

template <typename T> Matrix<T> LU<T>::solve(const Matrix<T> &b) const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    if (factors_.rows() != factors_.cols() || factors_.cols() != b.rows()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(), b.rows(),
                                b.cols());
    }

    // P A = L U, so A x = b becomes L (U x) = P b.
    Matrix<T> x = b;
    applyPivotsTo(x);

    typename TriangularSolver<T>::Options forward;
    forward.uplo          = Triangle::Kind::Lower;
    forward.diagonal      = Diagonal::Kind::Unit; // L's ones are implicit
    forward.transposition = Transposition::Kind::None;
    forward.useBlocked    = false;
    forward.blockSize     = 0;
    TriangularSolver<T>(forward).solveInPlace(factors_, x);

    typename TriangularSolver<T>::Options back = forward;
    back.uplo                                  = Triangle::Kind::Upper;
    back.diagonal                              = Diagonal::Kind::NonUnit;
    TriangularSolver<T>(back).solveInPlace(factors_, x);

    return x;
}

template <typename T>
Vector<T> LU<T>::solveTranspose(const Vector<T> &b) const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    if (factors_.rows() != factors_.cols() || factors_.cols() != b.size()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(), b.size(), 1);
    }

    // A = P^T L U, so A^T = U^T L^T P and A^T x = b runs the two solves in
    // the opposite order, transposed, with the pivots undone at the end.
    Vector<T> x = b;

    typename TriangularSolver<T>::Options forward;
    forward.uplo     = Triangle::Kind::Upper; // U^T is lower
    forward.diagonal = Diagonal::Kind::NonUnit;
    // A^T, not A^H; the two differ only for complex T, which no test covers.
    forward.transposition = Transposition::Kind::Transpose;
    forward.useBlocked    = false;
    forward.blockSize     = 0;
    TriangularSolver<T>(forward).solveInPlace(factors_, x);

    typename TriangularSolver<T>::Options back = forward;
    back.uplo     = Triangle::Kind::Lower; // L^T is upper
    back.diagonal = Diagonal::Kind::Unit;
    TriangularSolver<T>(back).solveInPlace(factors_, x);

    // x = P^T y, so the swaps come off in reverse.
    for (Index k = pivots_.size(); k-- > 0;) {
        const Index p = pivots_[k];
        if (p != k) {
            const T swapped = x(k);
            x(k)            = x(p);
            x(p)            = swapped;
        }
    }
    return x;
}

template <typename T>
Vector<T> LU<T>::refineSolution(const Matrix<T> &a,
                                const Vector<T> &b,
                                const Vector<T> &x,
                                Index            steps) const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    if (a.rows() != factors_.rows() || a.cols() != factors_.cols()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(), a.rows(),
                                a.cols());
    }
    if (a.rows() != b.size() || a.cols() != x.size()) {
        throw DimensionMismatch(a.rows(), a.cols(), b.size(), x.size());
    }

    // x += A^-1 (b - A x); the residual is at working precision, so this
    // recovers digits lost in the solve, not those lost forming A.
    Vector<T> refined = x;
    for (Index step = 0; step < steps; ++step) {
        refined = refined + solve(b - a * refined);
    }
    return refined;
}

template <typename T> T LU<T>::determinant() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    if (factors_.rows() != factors_.cols()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(),
                                factors_.rows(), factors_.cols());
    }

    // det(P) det(A) = det(L) det(U), with det(L) = 1 and det(P) = (-1)^sign_.
    T determinant =
        sign_ % 2 == 0 ? NumericTraits<T>::one() : -NumericTraits<T>::one();
    for (Index i = 0; i < factors_.rows(); ++i) {
        determinant *= factors_(i, i);
    }
    return determinant;
}

template <typename T> T LU<T>::logAbsDeterminant() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    if (factors_.rows() != factors_.cols()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(),
                                factors_.rows(), factors_.cols());
    }

    // Summing logs instead of logging the product is the whole point: the
    // product overflows well before log|det| does. det(P) drops out of |det|.
    Real total = Real{};
    for (Index i = 0; i < factors_.rows(); ++i) {
        total += std::log(NumericTraits<T>::abs(factors_(i, i)));
    }
    return T(total);
}

template <typename T> Matrix<T> LU<T>::inverse() const {
    if (!computed_) {
        throw NotComputed("LU");
    }
    // A A^-1 = I, so the columns of the inverse are n right-hand sides.
    return solve(Matrix<T>::Identity(factors_.rows()));
}

template <typename T> typename LU<T>::Index LU<T>::rank(Real tolerance) const {
    if (!computed_) {
        throw NotComputed("LU");
    }

    // Partial pivoting does not order U's diagonal, so this counts usable
    // pivots rather than revealing rank; FullPivLU is the reliable one.
    const Index steps =
        factors_.rows() < factors_.cols() ? factors_.rows() : factors_.cols();
    Index found = Index{};
    for (Index i = 0; i < steps; ++i) {
        if (NumericTraits<T>::abs(factors_(i, i)) > tolerance) {
            ++found;
        }
    }
    return found;
}

template <typename T> bool LU<T>::isSingular(Real tolerance) const {
    if (!computed_) {
        throw NotComputed("LU");
    }

    const Index steps =
        factors_.rows() < factors_.cols() ? factors_.rows() : factors_.cols();
    for (Index i = 0; i < steps; ++i) {
        if (NumericTraits<T>::abs(factors_(i, i)) <= tolerance) {
            return true;
        }
    }
    return false;
}

template <typename T> // deffered to step 17
typename LU<T>::Real LU<T>::reciprocalConditionEstimate() const {
    throw LinalgError(
        "not implemented: linalg::LU<T>::reciprocalConditionEstimate");
}

template <typename T> void LU<T>::unblockedFactorize() {
    // Right-looking Gaussian elimination with partial pivoting (getf2).
    // L and U share factors_ since L's diagonal is implicitly one.
    const Index m     = factors_.rows();
    const Index n     = factors_.cols();
    const Index steps = m < n ? m : n;

    for (Index k = 0; k < steps; ++k) {
        Index pivotRow = k;
        Real  pivotMag = NumericTraits<T>::abs(factors_(k, k));
        for (Index i = k + 1; i < m; ++i) { // find pivot
            const Real candidate = NumericTraits<T>::abs(factors_(i, k));
            if (candidate > pivotMag) {
                pivotMag = candidate;
                pivotRow = i;
            }
        }

        pivots_[k] = pivotRow;
        if (pivotRow != k) {
            factors_.swapRows(k, pivotRow);
            sign_ ^= Index{1}; // parity only; det(P) = (-1)^sign_
        }

        if (pivotMag <= options_.pivotTolerance) {
            // Nothing to eliminate; U(k, k) stays zero and rank() reads it
            // later.
            singular_ = true;
            if (options_.throwOnSingular) {
                throw SingularMatrix(k);
            }
            continue;
        }

        const T pivot = factors_(k, k);
        for (Index i = k + 1; i < m; ++i) {
            factors_(i, k) /= pivot;
            const T multiplier = factors_(i, k);
            for (Index j = k + 1; j < n; ++j) {
                factors_(i, j) -= multiplier * factors_(k, j);
            }
        }
    }
}

template <typename T> // deffered to step 17
void LU<T>::blockedFactorize(Index blockSize) {
    throw LinalgError("not implemented: linalg::LU<T>::blockedFactorize");
}

template <typename T> void LU<T>::applyPivotsTo(Matrix<T> &b) const {
    // Same replay order as permutationMatrix(), so this is P * b.
    for (Index k = 0; k < pivots_.size(); ++k) {
        b.swapRows(k, pivots_[k]);
    }
}

//------------------------------------------------------------------------------------------

template <typename T>
FullPivLU<T>::FullPivLU()
    : factors_(),
      rowPivots_(),
      colPivots_(),
      computed_(false) {}

template <typename T>
FullPivLU<T>::FullPivLU(const Matrix<T> &a)
    : factors_(),
      rowPivots_(),
      colPivots_(),
      computed_(false) {
    compute(a);
}

template <typename T> void FullPivLU<T>::compute(const Matrix<T> &a) {
    factors_ = a;

    const Index m     = a.rows();
    const Index n     = a.cols();
    const Index steps = m < n ? m : n;
    rowPivots_.assign(steps, Index{});
    colPivots_.assign(steps, Index{});
    computed_ = false;

    for (Index k = 0; k < steps; ++k) {
        // The search covers the whole trailing block, not just column k.
        Index pivotRow = k;
        Index pivotCol = k;
        Real  pivotMag = NumericTraits<T>::abs(factors_(k, k));
        for (Index i = k; i < m; ++i) {
            for (Index j = k; j < n; ++j) {
                const Real candidate = NumericTraits<T>::abs(factors_(i, j));
                if (candidate > pivotMag) {
                    pivotMag = candidate;
                    pivotRow = i;
                    pivotCol = j;
                }
            }
        }

        if (pivotMag == Real{}) {
            // Everything left is zero, so the remaining pivots stay put.
            for (Index rest = k; rest < steps; ++rest) {
                rowPivots_[rest] = rest;
                colPivots_[rest] = rest;
            }
            break;
        }

        rowPivots_[k] = pivotRow;
        colPivots_[k] = pivotCol;
        if (pivotRow != k) {
            factors_.swapRows(k, pivotRow);
        }
        if (pivotCol != k) {
            factors_.swapCols(k, pivotCol);
        }

        const T pivot = factors_(k, k);
        for (Index i = k + 1; i < m; ++i) {
            factors_(i, k) /= pivot;
            const T multiplier = factors_(i, k);
            for (Index j = k + 1; j < n; ++j) {
                factors_(i, j) -= multiplier * factors_(k, j);
            }
        }
    }

    computed_ = true;
}

template <typename T> bool FullPivLU<T>::isComputed() const {
    return computed_;
}

template <typename T> Matrix<T> FullPivLU<T>::lower() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    const Index m = factors_.rows();
    const Index k = m < factors_.cols() ? m : factors_.cols();

    Matrix<T> l(m, k);
    for (Index i = 0; i < m; ++i) {
        const Index strictlyLower = i < k ? i : k;
        for (Index j = 0; j < strictlyLower; ++j) {
            l(i, j) = factors_(i, j);
        }
        if (i < k) {
            l(i, i) = NumericTraits<T>::one();
        }
    }
    return l;
}

template <typename T> Matrix<T> FullPivLU<T>::upper() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    const Index n = factors_.cols();
    const Index k = factors_.rows() < n ? factors_.rows() : n;

    Matrix<T> u(k, n);
    for (Index i = 0; i < k; ++i) {
        for (Index j = i; j < n; ++j) {
            u(i, j) = factors_(i, j);
        }
    }
    return u;
}

template <typename T> Matrix<T> FullPivLU<T>::rowPermutation() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    Matrix<T> p = Matrix<T>::Identity(factors_.rows());
    for (Index k = 0; k < rowPivots_.size(); ++k) {
        p.swapRows(k, rowPivots_[k]);
    }
    return p;
}

template <typename T> Matrix<T> FullPivLU<T>::columnPermutation() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    Matrix<T> q = Matrix<T>::Identity(factors_.cols());
    for (Index k = 0; k < colPivots_.size(); ++k) {
        q.swapCols(k, colPivots_[k]);
    }
    return q;
}

template <typename T> Vector<T> FullPivLU<T>::solve(const Vector<T> &b) const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }
    if (factors_.rows() != factors_.cols() || factors_.cols() != b.size()) {
        throw DimensionMismatch(factors_.rows(), factors_.cols(), b.size(), 1);
    }

    // P A Q = L U, so A x = b becomes L U (Q^T x) = P b.
    Vector<T> x = b;
    for (Index k = 0; k < rowPivots_.size(); ++k) {
        const Index p = rowPivots_[k];
        if (p != k) {
            const T swapped = x(k);
            x(k)            = x(p);
            x(p)            = swapped;
        }
    }

    typename TriangularSolver<T>::Options forward;
    forward.uplo          = Triangle::Kind::Lower;
    forward.diagonal      = Diagonal::Kind::Unit;
    forward.transposition = Transposition::Kind::None;
    forward.useBlocked    = false;
    forward.blockSize     = 0;
    TriangularSolver<T>(forward).solveInPlace(factors_, x);

    typename TriangularSolver<T>::Options back = forward;
    back.uplo                                  = Triangle::Kind::Upper;
    back.diagonal                              = Diagonal::Kind::NonUnit;
    TriangularSolver<T>(back).solveInPlace(factors_, x);

    // x = Q z, so the column swaps come off in reverse.
    for (Index k = colPivots_.size(); k-- > 0;) {
        const Index p = colPivots_[k];
        if (p != k) {
            const T swapped = x(k);
            x(k)            = x(p);
            x(p)            = swapped;
        }
    }
    return x;
}

template <typename T> Matrix<T> FullPivLU<T>::kernel() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    const Index n     = factors_.cols();
    const Index steps = factors_.rows() < n ? factors_.rows() : n;
    // Full pivoting puts the largest pivot first, so the cutoff scales off it.
    const Real  cutoff = steps == 0 ? Real{}
                                    : NumericTraits<T>::epsilon() * Real(steps) *
                                         NumericTraits<T>::abs(factors_(0, 0));
    const Index r      = rank(cutoff);

    // U = [U11 U12; 0 0], so the null space of U is [-U11^-1 U12; I], and
    // x = Q z carries it back to the null space of A.
    Matrix<T> basis(n, n - r);
    for (Index i = 0; i < r; ++i) {
        for (Index j = r; j < n; ++j) {
            basis(i, j - r) = -factors_(i, j);
        }
    }
    for (Index j = r; j < n; ++j) {
        basis(j, j - r) = NumericTraits<T>::one();
    }

    if (r > 0 && n > r) {
        typename TriangularSolver<T>::Options upperSolve;
        upperSolve.uplo          = Triangle::Kind::Upper;
        upperSolve.diagonal      = Diagonal::Kind::NonUnit;
        upperSolve.transposition = Transposition::Kind::None;
        upperSolve.useBlocked    = false;
        upperSolve.blockSize     = 0;

        Matrix<T> leading(r, r);
        for (Index i = 0; i < r; ++i) {
            for (Index j = i; j < r; ++j) {
                leading(i, j) = factors_(i, j);
            }
        }
        Matrix<T> top(r, n - r);
        for (Index i = 0; i < r; ++i) {
            for (Index j = 0; j < n - r; ++j) {
                top(i, j) = basis(i, j);
            }
        }
        TriangularSolver<T>(upperSolve).solveInPlace(leading, top);
        basis.setBlock(0, 0, top);
    }

    for (Index k = colPivots_.size(); k-- > 0;) {
        basis.swapRows(k, colPivots_[k]);
    }
    return basis;
}

template <typename T> Matrix<T> FullPivLU<T>::image() const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    const Index m      = factors_.rows();
    const Index steps  = m < factors_.cols() ? m : factors_.cols();
    const Real  cutoff = steps == 0 ? Real{}
                                    : NumericTraits<T>::epsilon() * Real(steps) *
                                         NumericTraits<T>::abs(factors_(0, 0));
    const Index r      = rank(cutoff);

    // The first r columns of A Q are P^T L(:, 0:r) U11, which are exactly
    // A's pivot columns, so no copy of A is needed.
    Matrix<T> columns(m, r);
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < r; ++j) {
            T           sum  = T{};
            const Index last = i < j ? i : j;
            for (Index p = 0; p <= last; ++p) {
                const T l = p == i ? NumericTraits<T>::one() : factors_(i, p);
                sum += l * factors_(p, j);
            }
            columns(i, j) = sum;
        }
    }

    for (Index k = rowPivots_.size(); k-- > 0;) {
        columns.swapRows(k, rowPivots_[k]);
    }
    return columns;
}

template <typename T>
typename FullPivLU<T>::Index FullPivLU<T>::rank(Real tolerance) const {
    if (!computed_) {
        throw NotComputed("FullPivLU");
    }

    // Full pivoting leaves |diag(U)| non-increasing, so this genuinely
    // reveals the rank rather than counting usable pivots.
    const Index steps =
        factors_.rows() < factors_.cols() ? factors_.rows() : factors_.cols();
    Index found = Index{};
    for (Index i = 0; i < steps; ++i) {
        if (NumericTraits<T>::abs(factors_(i, i)) > tolerance) {
            ++found;
        }
    }
    return found;
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class LU<SCALAR>;     \
    template class FullPivLU<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
