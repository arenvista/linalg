#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// A = U S V^H. Golub-Kahan by default: bidiagonalize, then chase
// with implicitly shifted QR sweeps on the bidiagonal form.
template <typename T>
class SVD {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Algorithm {
        enum class Kind { GolubKahan, Jacobi, DivideAndConquer };
    };

    struct Mode {
        enum class Kind { None, Thin, Full };
    };

    struct Options {
        typename Algorithm::Kind algorithm;
        typename Mode::Kind      leftVectors;   // None/Thin/Full for U
        typename Mode::Kind      rightVectors;  // None/Thin/Full for V
        Index                    maxIterations; // budget for the bidiagonal QR sweeps
        Real                     tolerance;     // deflation threshold, relative to neighboring entries
    };

    /// @brief Constructs an empty decomposition; call compute() before use.
    SVD();
    /// @brief Constructs and immediately decomposes a matrix with defaults.
    /// @param a Matrix to decompose.
    explicit SVD(const Matrix<T>& a);
    /// @brief Constructs and immediately decomposes a matrix with options.
    /// @param a Matrix to decompose.
    /// @param options Decomposition options (algorithm, vector modes, limits).
    SVD(const Matrix<T>& a, const Options& options);

    /// @brief Computes the A = U S V^H decomposition.
    /// @param a Matrix to decompose.
    void compute(const Matrix<T>& a);
    /// @brief Reports whether a decomposition is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the iteration converged within its budget.
    /// @return True if the sweeps converged.
    bool converged() const;
    /// @brief Number of QR sweeps actually performed.
    /// @return The iteration count.
    Index iterationsUsed() const;

    /// @brief The singular values (real, non-negative, sorted descending).
    /// @return Reference to the singular-value vector.
    const Vector<Real>& singularValues() const;
    /// @brief The left singular vectors U.
    /// @return Reference to the U matrix.
    const Matrix<T>&    matrixU() const;
    /// @brief The right singular vectors V (V itself, not V^H; A = U S V^H).
    /// @return Reference to the V matrix.
    const Matrix<T>&    matrixV() const;

    /// @brief The largest singular value.
    /// @return sigma_max.
    Real  largestSingularValue() const;
    /// @brief The smallest singular value.
    /// @return sigma_min.
    Real  smallestSingularValue() const;
    /// @brief Condition number, sigma_max / sigma_min.
    /// @return The 2-norm condition number.
    Real  conditionNumber() const;
    /// @brief Numerical rank, the count of singular values above a relative cutoff.
    /// @param tolerance Relative cutoff (times sigma_max).
    /// @return The numerical rank.
    Index rank(Real tolerance) const;
    /// @brief Spectral norm, equal to the largest singular value.
    /// @return The spectral norm.
    Real  spectralNorm() const;
    /// @brief Nuclear norm, the sum of the singular values.
    /// @return The nuclear norm.
    Real  nuclearNorm() const;

    // Regularized solves for ill-posed problems: solve is minimum-norm
    // least squares (pseudoinverse applied to b); solveTruncated keeps only
    // the `keep` largest singular triplets (TSVD); solveTikhonov damps with
    // filter factors s_i / (s_i^2 + lambda^2). `tolerance` arguments are
    // relative to the largest singular value.
    /// @brief Minimum-norm least-squares solve (pseudoinverse applied to b).
    /// @param b Right-hand side vector.
    /// @return The minimum-norm least-squares solution.
    Vector<T> solve(const Vector<T>& b) const;
    /// @brief Truncated-SVD solve keeping the `keep` largest singular triplets.
    /// @param b Right-hand side vector.
    /// @param keep Number of leading singular triplets to retain.
    /// @return The regularized solution.
    Vector<T> solveTruncated(const Vector<T>& b, Index keep) const;
    /// @brief Tikhonov-damped solve with filter factors s/(s^2 + lambda^2).
    /// @param b Right-hand side vector.
    /// @param lambda Regularization (damping) parameter.
    /// @return The regularized solution.
    Vector<T> solveTikhonov(const Vector<T>& b, Real lambda) const;
    /// @brief Moore-Penrose pseudoinverse.
    /// @param tolerance Relative singular-value cutoff.
    /// @return The pseudoinverse.
    Matrix<T> pseudoInverse(Real tolerance) const;
    /// @brief Best rank-`rank` approximation (Eckart-Young).
    /// @param rank Target rank.
    /// @return The nearest matrix of the given rank.
    Matrix<T> lowRankApproximation(Index rank) const;
    /// @brief Orthonormal basis for the null space.
    /// @param tolerance Relative singular-value cutoff.
    /// @return A matrix whose columns span the null space.
    Matrix<T> nullSpace(Real tolerance) const;
    /// @brief Orthonormal basis for the column space.
    /// @param tolerance Relative singular-value cutoff.
    /// @return A matrix whose columns span the column space.
    Matrix<T> columnSpace(Real tolerance) const;
    // Polar decomposition A = W P from the SVD: W = U V^H is the nearest
    // unitary matrix to A, P = V S V^H is Hermitian positive semidefinite.
    /// @brief Orthogonal polar factor W = U V^H (nearest unitary matrix to A).
    /// @return The orthogonal/unitary polar factor.
    Matrix<T> orthogonalPolarFactor() const;
    /// @brief Hermitian polar factor P = V S V^H (positive semidefinite).
    /// @return The Hermitian polar factor.
    Matrix<T> hermitianPolarFactor() const;

private:
    /// @brief Runs the Golub-Kahan bidiagonalization and QR sweeps.
    void computeGolubKahan();
    /// @brief Runs the one-sided Jacobi algorithm.
    void computeJacobi();
    /// @brief Sorts the singular values (and vectors) in descending order.
    void sortSingularValues();

    Vector<Real> singularValues_;
    Matrix<T>    u_;
    Matrix<T>    v_;
    Options      options_;
    Index        iterations_;
    bool         computed_;
    bool         converged_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class SVD<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
