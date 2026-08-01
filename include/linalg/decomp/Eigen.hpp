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

    /// @brief Constructs an empty decomposition; call compute() before use.
    SymmetricEigen();
    /// @brief Constructs and immediately solves the eigenproblem with defaults.
    /// @param a Hermitian matrix (only the lower triangle is read).
    explicit SymmetricEigen(const Matrix<T>& a);
    /// @brief Constructs and immediately solves the eigenproblem with options.
    /// @param a Hermitian matrix (only the lower triangle is read).
    /// @param options Solver options (algorithm, ordering, vectors, limits).
    SymmetricEigen(const Matrix<T>& a, const Options& options);

    /// @brief Solves the Hermitian eigenproblem for a matrix.
    /// @param a Hermitian matrix (assumed Hermitian; lower triangle read).
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a decomposition is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the iteration converged.
    /// @return True on convergence.
    bool converged() const;

    /// @brief The eigenvalues, sorted per Options::order.
    /// @return Reference to the eigenvalue vector.
    const Vector<Real>& eigenvalues() const;
    /// @brief The eigenvectors; column k matches eigenvalues()[k].
    /// @return Reference to the eigenvector matrix.
    const Matrix<T>&    eigenvectors() const;
    /// @brief A single eigenvector.
    /// @param k Eigenpair index (0-based).
    /// @return Eigenvector k.
    Vector<T>           eigenvector(Index k) const;
    /// @brief A single eigenvalue.
    /// @param k Eigenpair index (0-based).
    /// @return Eigenvalue k.
    Real                eigenvalue(Index k) const;

    // Partial spectra, without forming the whole decomposition.
    /// @brief Eigenvalues lying within an interval.
    /// @param lower Lower bound of the interval.
    /// @param upper Upper bound of the interval.
    /// @return The eigenvalues in [lower, upper].
    Vector<Real> eigenvaluesInRange(Real lower, Real upper) const;
    /// @brief The smallest eigenvalues.
    /// @param count How many to return.
    /// @return The `count` smallest eigenvalues.
    Vector<Real> smallestEigenvalues(Index count) const;
    /// @brief The largest eigenvalues.
    /// @param count How many to return.
    /// @return The `count` largest eigenvalues.
    Vector<Real> largestEigenvalues(Index count) const;

    /// @brief Spectral radius, max |lambda|.
    /// @return The largest eigenvalue magnitude.
    Real      spectralRadius() const;
    /// @brief Condition number, |lambda|_max / |lambda|_min.
    /// @return The eigenvalue condition number.
    Real      conditionNumber() const;
    /// @brief Residual of an eigenpair, ||A q_k - lambda_k q_k||.
    /// @param k Eigenpair index (0-based).
    /// @param a The original matrix.
    /// @return The residual norm.
    Real      residual(Index k, const Matrix<T>& a) const;
    // f(A) = Q diag(f(lambda)) Q^H: caller supplies f applied entrywise to
    // eigenvalues(); squareRoot/exponential are the common cases canned
    // (square root requires positive semidefinite input).
    /// @brief Matrix function f(A) = Q diag(f(lambda)) Q^H.
    /// @param transformedEigenvalues f applied entrywise to the eigenvalues.
    /// @return The matrix function value.
    Matrix<T> functionOfMatrix(const Vector<Real>& transformedEigenvalues) const;
    /// @brief Matrix square root (requires positive semidefinite input).
    /// @return The principal square root of the matrix.
    Matrix<T> matrixSquareRoot() const;
    /// @brief Matrix exponential exp(A).
    /// @return The matrix exponential.
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

    /// @brief Constructs an empty decomposition; call compute() before use.
    Schur();
    /// @brief Constructs and immediately computes the Schur form.
    /// @param a Matrix to decompose.
    explicit Schur(const Matrix<T>& a);

    /// @brief Computes the Schur decomposition A = Q T Q^H.
    /// @param a Matrix to decompose.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a decomposition is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the iteration converged.
    /// @return True on convergence.
    bool converged() const;

    // For real T, matrixT is quasi-triangular: 1x1 blocks for real
    // eigenvalues, 2x2 blocks for complex conjugate pairs. For complex T
    // it is strictly upper triangular with the eigenvalues on the diagonal.
    /// @brief The (quasi-)triangular Schur factor T.
    /// @return The T factor.
    Matrix<T>           matrixT() const;
    /// @brief The orthogonal/unitary Schur factor Q.
    /// @return The Q factor.
    Matrix<T>           matrixQ() const;
    /// @brief The eigenvalues read from the Schur form (always complex).
    /// @return The eigenvalue vector.
    Vector<Complex>     eigenvalues() const;
    /// @brief Number of QR sweeps performed.
    /// @return The iteration count.
    Index               iterationsUsed() const;

    // Moves selected eigenvalues to the top-left of T (swapping adjacent
    // diagonal blocks), e.g. to extract an invariant subspace.
    /// @brief Reorders diagonal blocks so selected eigenvalues come first.
    /// @param desiredOrder Target order of the diagonal blocks.
    void reorder(const std::vector<Index>& desiredOrder);

private:
    // The double-shift trick keeps arithmetic real for real T; the complex
    // instantiations should use a single Wilkinson shift instead.
    /// @brief Performs one Francis double-shift QR sweep on a subrange.
    /// @param lo First index of the active block.
    /// @param hi Last index of the active block.
    void   francisDoubleShiftSweep(Index lo, Index hi);
    /// @brief Finds a deflation point where a subdiagonal entry is negligible.
    /// @param tolerance Deflation threshold.
    /// @return The index at which the problem deflates.
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

    /// @brief Constructs an empty decomposition; call compute() before use.
    GeneralEigen();
    /// @brief Constructs and immediately solves the eigenproblem.
    /// @param a Matrix to decompose.
    explicit GeneralEigen(const Matrix<T>& a);

    /// @brief Solves the general (non-Hermitian) eigenproblem.
    /// @param a Matrix to decompose.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a decomposition is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the iteration converged.
    /// @return True on convergence.
    bool converged() const;

    // Results are complex even for real T, since real matrices have
    // complex eigenpairs. Right vectors satisfy A x = lambda x (one per
    // column), left vectors y^H A = lambda y^H; columns are unit-norm.
    /// @brief The eigenvalues (complex even for real T).
    /// @return Reference to the eigenvalue vector.
    const Vector<Complex>&    eigenvalues() const;
    /// @brief The right eigenvectors, one per column (A x = lambda x).
    /// @return Reference to the right-eigenvector matrix.
    const Matrix<Complex>&    rightEigenvectors() const;
    /// @brief The left eigenvectors, one per column (y^H A = lambda y^H).
    /// @return Reference to the left-eigenvector matrix.
    const Matrix<Complex>&    leftEigenvectors() const;

    /// @brief Per-eigenvalue sensitivities, 1 / |y^H x|.
    /// @return A vector of eigenvalue condition numbers.
    Vector<Real> conditionNumbers() const;
    /// @brief Tests whether the matrix is diagonalizable within a tolerance.
    /// @param tolerance Tolerance on eigenvector independence.
    /// @return True if diagonalizable within tolerance.
    bool         isDiagonalizable(Real tolerance) const;
    /// @brief Spectral radius, max |lambda|.
    /// @return The largest eigenvalue magnitude.
    Real         spectralRadius() const;
    /// @brief Spectral abscissa, max Re(lambda); negative implies stability.
    /// @return The largest real part among the eigenvalues.
    Real         spectralAbscissa() const;

private:
    // balance(): diagonal similarity scaling before Schur, undone on the
    // vectors afterward — improves eigenvalue accuracy for badly scaled A.
    /// @brief Applies diagonal similarity scaling to improve conditioning.
    void balance();
    /// @brief Back-solves the triangular Schur form for the eigenvectors.
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
