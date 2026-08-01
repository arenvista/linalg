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

    /// @brief Constructs a power-iteration solver with default options.
    PowerIteration();
    /// @brief Constructs a power-iteration solver with explicit options.
    /// @param options Iteration limit, tolerance, and Rayleigh-quotient flag.
    explicit PowerIteration(const Options& options);

    // Converges to the eigenvalue of largest magnitude, at a rate set by
    // |lambda_2 / lambda_1|. shiftedInverseEigenvalue finds the eigenvalue
    // nearest `shift` by iterating with (A - shift I)^-1 (one LU, reused
    // every iteration).
    /// @brief Computes the eigenvalue of largest magnitude.
    /// @param a The matrix.
    /// @return The dominant eigenvalue.
    Real      dominantEigenvalue(const Matrix<T>& a);
    /// @brief Computes the eigenvector for the eigenvalue of largest magnitude.
    /// @param a The matrix.
    /// @return The dominant eigenvector (unit norm).
    Vector<T> dominantEigenvector(const Matrix<T>& a);
    /// @brief Finds the eigenvalue nearest a shift via inverse iteration.
    /// @param a The matrix.
    /// @param shift The target value to iterate toward.
    /// @return The eigenvalue closest to `shift`.
    Real      shiftedInverseEigenvalue(const Matrix<T>& a, Real shift);
    /// @brief Rayleigh quotient x^H A x / x^H x.
    /// @param a The matrix.
    /// @param x The vector.
    /// @return The Rayleigh quotient.
    Real      rayleighQuotient(const Matrix<T>& a, const Vector<T>& x) const;

    /// @brief Number of iterations performed by the last call.
    /// @return The iteration count.
    Index iterationsUsed() const;
    /// @brief Reports whether the last call converged.
    /// @return True on convergence.
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

    /// @brief Constructs an empty Lanczos solver; call compute() before use.
    Lanczos();
    /// @brief Constructs a Lanczos solver with a subspace size and strategy.
    /// @param krylovDimension Maximum Krylov subspace dimension.
    /// @param strategy Reorthogonalization strategy.
    Lanczos(Index krylovDimension, typename Reorthogonalization::Kind strategy);

    /// @brief Builds the Lanczos tridiagonalization from a start vector.
    /// @param a Hermitian matrix.
    /// @param startVector Initial Krylov vector.
    void compute(const Matrix<T>& a, const Vector<T>& startVector);
    /// @brief Reports whether a factorization is available.
    /// @return True if compute() has run.
    bool isComputed() const;

    // Ritz values are the eigenvalues of the projected tridiagonal matrix
    // (extreme ones converge first); ritzResidualBounds gives the cheap
    // per-pair error bound |beta_m| * |last component of the projected
    // eigenvector| — no residual against A needed.
    /// @brief Main diagonal of the projected tridiagonal matrix.
    /// @return The diagonal entries.
    Vector<Real>  diagonal() const;
    /// @brief Subdiagonal of the projected tridiagonal matrix.
    /// @return The subdiagonal entries.
    Vector<Real>  subdiagonal() const;
    /// @brief The Lanczos basis actually built.
    /// @return The basis matrix.
    Matrix<T>     basis() const;
    /// @brief Ritz values (eigenvalues of the projected tridiagonal).
    /// @return The Ritz values.
    Vector<Real>  ritzValues() const;
    /// @brief Ritz vectors (approximate eigenvectors).
    /// @return The Ritz vectors.
    Matrix<T>     ritzVectors() const;
    /// @brief Cheap per-pair Ritz residual bounds.
    /// @return The residual bound for each Ritz pair.
    Vector<Real>  ritzResidualBounds() const;
    /// @brief Loss of orthogonality of the basis, ||V^H V - I||.
    /// @return The orthogonality-loss metric.
    Real          orthogonalityLoss() const;

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

    /// @brief Constructs an empty solver with default options.
    ImplicitlyRestartedArnoldi();
    /// @brief Constructs a solver with explicit options.
    /// @param options Wanted count, subspace size, restart limit, and target.
    explicit ImplicitlyRestartedArnoldi(const Options& options);

    /// @brief Runs the implicitly restarted Arnoldi iteration.
    /// @param a The matrix.
    /// @param startVector Initial Krylov vector.
    void compute(const Matrix<T>& a, const Vector<T>& startVector);
    /// @brief Reports whether a result is available.
    /// @return True if compute() has run.
    bool isComputed() const;
    /// @brief Reports whether the wanted eigenpairs converged.
    /// @return True on convergence.
    bool converged() const;

    /// @brief The converged Ritz values (approximate eigenvalues).
    /// @return The Ritz values.
    Vector<T> ritzValues() const;
    /// @brief The converged Ritz vectors (approximate eigenvectors).
    /// @return The Ritz vectors.
    Matrix<T> ritzVectors() const;
    /// @brief Number of restarts performed.
    /// @return The restart count.
    Index     restartsUsed() const;

private:
    /// @brief Extends the Krylov/Arnoldi factorization over a range of steps.
    /// @param a The matrix.
    /// @param from First step index to fill.
    /// @param to One past the last step index to fill.
    void expandKrylovSpace(const Matrix<T>& a, Index from, Index to);
    // The implicit restart: QR steps on the projected Hessenberg with the
    // unwanted Ritz values as shifts, compressing the subspace back to
    // wantedCount vectors while keeping it a valid Arnoldi factorization.
    /// @brief Applies a shifted-QR implicit restart to compress the subspace.
    /// @param shifts The unwanted Ritz values used as shifts.
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
