#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/iterative/Preconditioner.hpp"

namespace linalg {

// A few dominant or interior eigenpairs without a full decomposition.
template <typename T>
class PowerIteration {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    struct Options {
        Index maxIterations;
        Real  tolerance;
        bool  useRayleighQuotient;  // Rayleigh-quotient estimate: squares the convergence rate for Hermitian A
    };

    PowerIteration();
    explicit PowerIteration(const Options& options);

    // Converges to the eigenvalue of largest magnitude, at a rate set by
    // |lambda_2 / lambda_1|. shiftedInverseEigenvalue finds the eigenvalue
    // nearest `shift` by iterating with (A - shift I)^-1 (one LU, reused
    // every iteration).
    Real      dominantEigenvalue(const Matrix<T>& a);
    Vector<T> dominantEigenvector(const Matrix<T>& a);
    Real      shiftedInverseEigenvalue(const Matrix<T>& a, Real shift);
    Real      rayleighQuotient(const Matrix<T>& a, const Vector<T>& x) const;  // x^H A x / x^H x

    Index iterationsUsed() const;
    bool  converged() const;

private:
    Options options_;
    Index   iterations_;
    bool    converged_;
};

// Lanczos for Hermitian problems: builds a tridiagonal projection.
template <typename T>
class Lanczos {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    // In exact arithmetic the three-term recurrence keeps the basis
    // orthogonal; in floating point it drifts and duplicate ("ghost")
    // Ritz values appear. None: fastest, accepts ghosts. Full:
    // re-orthogonalize every vector against the whole basis. Selective/
    // Partial: re-orthogonalize only when a drift estimate crosses
    // sqrt(epsilon).
    struct Reorthogonalization {
        enum class Kind { None, Full, Selective, Partial };
    };

    Lanczos();
    Lanczos(Index krylovDimension, typename Reorthogonalization::Kind strategy);

    void compute(const Matrix<T>& a, const Vector<T>& startVector);
    bool isComputed() const;

    // Ritz values are the eigenvalues of the projected tridiagonal matrix
    // (extreme ones converge first); ritzResidualBounds gives the cheap
    // per-pair error bound |beta_m| * |last component of the projected
    // eigenvector| — no residual against A needed.
    Vector<Real>  diagonal() const;
    Vector<Real>  subdiagonal() const;
    Matrix<T>     basis() const;
    Vector<Real>  ritzValues() const;
    Matrix<T>     ritzVectors() const;
    Vector<Real>  ritzResidualBounds() const;
    Real          orthogonalityLoss() const;  // ||V^H V - I|| of the basis actually built

private:
    Matrix<T>                          basis_;
    Vector<Real>                       diagonal_;
    Vector<Real>                       subdiagonal_;
    Index                              krylovDimension_;
    typename Reorthogonalization::Kind strategy_;
    bool                               computed_;
};

// Arnoldi with implicit restarts, for nonsymmetric problems.
template <typename T>
class ImplicitlyRestartedArnoldi {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    struct Target {
        enum class Kind { LargestMagnitude, SmallestMagnitude,
                          LargestReal, SmallestReal };
    };

    struct Options {
        Index                 wantedCount;      // eigenpairs to converge (k)
        Index                 krylovDimension;  // working subspace size m > k, typically ~2k
        Index                 maxRestarts;
        Real                  tolerance;
        typename Target::Kind target;           // which end of the spectrum to chase
    };

    ImplicitlyRestartedArnoldi();
    explicit ImplicitlyRestartedArnoldi(const Options& options);

    void compute(const Matrix<T>& a, const Vector<T>& startVector);
    bool isComputed() const;
    bool converged() const;

    Vector<T> ritzValues() const;
    Matrix<T> ritzVectors() const;
    Index     restartsUsed() const;

private:
    void expandKrylovSpace(const Matrix<T>& a, Index from, Index to);
    // The implicit restart: QR steps on the projected Hessenberg with the
    // unwanted Ritz values as shifts, compressing the subspace back to
    // wantedCount vectors while keeping it a valid Arnoldi factorization.
    void applyShiftedQRRestart(const Vector<T>& shifts);

    Matrix<T> basis_;
    Matrix<T> hessenberg_;
    Options   options_;
    Index     restarts_;
    bool      computed_;
    bool      converged_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class PowerIteration<SCALAR>; \
    extern template class Lanczos<SCALAR>; \
    extern template class ImplicitlyRestartedArnoldi<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
