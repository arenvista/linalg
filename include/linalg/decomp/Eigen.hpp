#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Hermitian eigenproblem: real eigenvalues, orthonormal eigenvectors.
// Tridiagonalize, then implicit-QL/QR with Wilkinson shifts.
// compute() assumes the input is Hermitian and does not verify it; only
// the lower triangle is read.
template <typename T>
class SymmetricEigen {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Algorithm {
        enum class Kind { ImplicitQR, Jacobi, DivideAndConquer, Bisection };
    };

    struct Order {
        enum class Kind { Ascending, Descending, ByMagnitude };
    };

    struct Options {
        typename Algorithm::Kind algorithm;
        typename Order::Kind     order;
        bool                     computeVectors;
        Index                    maxIterations;
        Real                     tolerance;
    };

    SymmetricEigen();
    explicit SymmetricEigen(const Matrix<T>& a);
    SymmetricEigen(const Matrix<T>& a, const Options& options);

    void compute(const Matrix<T>& a);
    bool isComputed() const;
    bool converged() const;

    const Vector<Real>& eigenvalues() const;   // sorted per Options::order
    const Matrix<T>&    eigenvectors() const;  // eigenvector k is column k, matching eigenvalues()[k]
    Vector<T>           eigenvector(Index k) const;
    Real                eigenvalue(Index k) const;

    // Partial spectra, without forming the whole decomposition.
    Vector<Real> eigenvaluesInRange(Real lower, Real upper) const;
    Vector<Real> smallestEigenvalues(Index count) const;
    Vector<Real> largestEigenvalues(Index count) const;

    Real      spectralRadius() const;   // max |lambda|
    Real      conditionNumber() const;  // |lambda|_max / |lambda|_min
    Real      residual(Index k, const Matrix<T>& a) const;  // ||A q_k - lambda_k q_k||
    // f(A) = Q diag(f(lambda)) Q^H: caller supplies f applied entrywise to
    // eigenvalues(); squareRoot/exponential are the common cases canned
    // (square root requires positive semidefinite input).
    Matrix<T> functionOfMatrix(const Vector<Real>& transformedEigenvalues) const;
    Matrix<T> matrixSquareRoot() const;
    Matrix<T> matrixExponential() const;

private:
    Vector<Real> eigenvalues_;
    Matrix<T>    eigenvectors_;
    Options      options_;
    bool         computed_;
    bool         converged_;
};

// A = Q T Q^H, T quasi-upper-triangular over the reals.
template <typename T>
class Schur {
public:
    using Scalar  = T;
    using Real    = typename NumericTraits<T>::Real;
    using Complex = std::complex<Real>;
    using Index   = std::size_t;

    Schur();
    explicit Schur(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;
    bool converged() const;

    // For real T, matrixT is quasi-triangular: 1x1 blocks for real
    // eigenvalues, 2x2 blocks for complex conjugate pairs. For complex T
    // it is strictly upper triangular with the eigenvalues on the diagonal.
    Matrix<T>           matrixT() const;
    Matrix<T>           matrixQ() const;
    Vector<Complex>     eigenvalues() const;
    Index               iterationsUsed() const;

    // Moves selected eigenvalues to the top-left of T (swapping adjacent
    // diagonal blocks), e.g. to extract an invariant subspace.
    void reorder(const std::vector<Index>& desiredOrder);

private:
    // The double-shift trick keeps arithmetic real for real T; the complex
    // instantiations should use a single Wilkinson shift instead.
    void   francisDoubleShiftSweep(Index lo, Index hi);
    Index  detectDeflation(Real tolerance) const;

    Matrix<T> t_;
    Matrix<T> q_;
    Index     iterations_;
    bool      computed_;
    bool      converged_;
};

// General (non-Hermitian) eigenproblem, built on the Schur form.
template <typename T>
class GeneralEigen {
public:
    using Scalar  = T;
    using Real    = typename NumericTraits<T>::Real;
    using Complex = std::complex<Real>;
    using Index   = std::size_t;

    GeneralEigen();
    explicit GeneralEigen(const Matrix<T>& a);

    void compute(const Matrix<T>& a);
    bool isComputed() const;
    bool converged() const;

    // Results are complex even for real T, since real matrices have
    // complex eigenpairs. Right vectors satisfy A x = lambda x (one per
    // column), left vectors y^H A = lambda y^H; columns are unit-norm.
    const Vector<Complex>&    eigenvalues() const;
    const Matrix<Complex>&    rightEigenvectors() const;
    const Matrix<Complex>&    leftEigenvectors() const;

    Vector<Real> conditionNumbers() const;   // per-eigenvalue sensitivity: 1 / |y^H x|
    bool         isDiagonalizable(Real tolerance) const;
    Real         spectralRadius() const;     // max |lambda|
    Real         spectralAbscissa() const;   // max Re(lambda); < 0 means a stable system

private:
    // balance(): diagonal similarity scaling before Schur, undone on the
    // vectors afterward — improves eigenvalue accuracy for badly scaled A.
    void balance();
    void solveTriangularEigenvectors();

    Schur<T>        schur_;
    Vector<Complex> eigenvalues_;
    Matrix<Complex> rightVectors_;
    Matrix<Complex> leftVectors_;
    Vector<Real>    balanceScaling_;
    bool            computed_;
    bool            converged_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class SymmetricEigen<SCALAR>; \
    extern template class Schur<SCALAR>; \
    extern template class GeneralEigen<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
