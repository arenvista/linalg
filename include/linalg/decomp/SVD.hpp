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

    SVD();
    explicit SVD(const Matrix<T>& a);
    SVD(const Matrix<T>& a, const Options& options);

    void compute(const Matrix<T>& a);
    bool isComputed() const;
    bool converged() const;
    Index iterationsUsed() const;

    const Vector<Real>& singularValues() const;  // always real, non-negative, sorted descending
    const Matrix<T>&    matrixU() const;
    const Matrix<T>&    matrixV() const;         // V itself, not V^H: A == U * S * V^H

    Real  largestSingularValue() const;
    Real  smallestSingularValue() const;
    Real  conditionNumber() const;
    Index rank(Real tolerance) const;
    Real  spectralNorm() const;
    Real  nuclearNorm() const;

    // Regularized solves for ill-posed problems: solve is minimum-norm
    // least squares (pseudoinverse applied to b); solveTruncated keeps only
    // the `keep` largest singular triplets (TSVD); solveTikhonov damps with
    // filter factors s_i / (s_i^2 + lambda^2). `tolerance` arguments are
    // relative to the largest singular value.
    Vector<T> solve(const Vector<T>& b) const;
    Vector<T> solveTruncated(const Vector<T>& b, Index keep) const;
    Vector<T> solveTikhonov(const Vector<T>& b, Real lambda) const;
    Matrix<T> pseudoInverse(Real tolerance) const;
    Matrix<T> lowRankApproximation(Index rank) const;  // Eckart-Young: nearest rank-`rank` matrix
    Matrix<T> nullSpace(Real tolerance) const;         // orthonormal basis columns
    Matrix<T> columnSpace(Real tolerance) const;       // orthonormal basis columns
    // Polar decomposition A = W P from the SVD: W = U V^H is the nearest
    // unitary matrix to A, P = V S V^H is Hermitian positive semidefinite.
    Matrix<T> orthogonalPolarFactor() const;
    Matrix<T> hermitianPolarFactor() const;

private:
    void computeGolubKahan();
    void computeJacobi();
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
