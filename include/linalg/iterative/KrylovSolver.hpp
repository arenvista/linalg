#pragma once

#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/iterative/Preconditioner.hpp"

namespace linalg {

// Shared configuration and reporting for iterative methods.
template <typename T>
class KrylovSolver {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Options {
        Index maxIterations;
        Real  relativeTolerance;  // converged when ||r|| <= relative * ||b|| ...
        Real  absoluteTolerance;  // ... or ||r|| <= absolute, whichever is met first
        bool  recordHistory;      // fill Report::residualHistory per iteration
        bool  useInitialGuess;    // otherwise x0 = 0 and the 2-arg solve is used
    };

    struct Report {
        Index             iterations;
        Real              finalResidual;     // ||b - A x||
        Real              relativeResidual;  // finalResidual / ||b||
        bool              converged;
        bool              breakdown;         // a recurrence divisor vanished; result unreliable
        std::vector<Real> residualHistory;   // empty unless recordHistory
    };

    /// @brief Constructs a solver with default options.
    KrylovSolver();
    /// @brief Constructs a solver with explicit options.
    /// @param options Iteration limits, tolerances, and reporting settings.
    explicit KrylovSolver(const Options& options);
    /// @brief Virtual destructor for the abstract solver interface.
    virtual ~KrylovSolver();

    /// @brief Solves A x = b starting from a zero initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @return The computed solution x.
    virtual Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) = 0;
    /// @brief Solves A x = b starting from a supplied initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @param initialGuess The starting iterate x0.
    /// @return The computed solution x.
    virtual Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                            const Vector<T>& initialGuess) = 0;

    // Non-owning; the preconditioner must outlive the solver and already
    // be set up. nullptr means unpreconditioned.
    /// @brief Sets the (non-owning) preconditioner; nullptr disables it.
    /// @param preconditioner A set-up preconditioner that outlives the solver.
    void setPreconditioner(const Preconditioner<T>* preconditioner);
    /// @brief Replaces the solver options.
    /// @param options The new options.
    void setOptions(const Options& options);

    /// @brief The current solver options.
    /// @return Reference to the options.
    const Options& options() const;
    /// @brief Diagnostics from the most recent solve.
    /// @return Reference to the last report.
    const Report&  lastReport() const;

protected:
    /// @brief Tests the convergence criterion against a reference residual.
    /// @param residual The current residual norm.
    /// @param referenceResidual The reference norm (typically ||b||).
    /// @return True if converged.
    bool hasConverged(Real residual, Real referenceResidual) const;
    /// @brief Records a residual into the report (and history if enabled).
    /// @param residual The residual norm for the current iteration.
    void recordResidual(Real residual);

    Options                  options_;
    Report                   report_;
    const Preconditioner<T>* preconditioner_;
};

// Preconditioned conjugate gradients: symmetric positive definite only.
template <typename T>
class ConjugateGradient : public KrylovSolver<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    /// @brief Constructs a CG solver with default options.
    ConjugateGradient();
    /// @brief Constructs a CG solver with explicit options.
    /// @param options Iteration limits, tolerances, and reporting settings.
    explicit ConjugateGradient(const typename KrylovSolver<T>::Options& options);

    /// @brief Solves the SPD system A x = b from a zero initial guess.
    /// @param a Symmetric positive-definite matrix.
    /// @param b The right-hand side.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    /// @brief Solves the SPD system A x = b from an initial guess.
    /// @param a Symmetric positive-definite matrix.
    /// @param b The right-hand side.
    /// @param initialGuess The starting iterate x0.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    // The CG coefficients build a tridiagonal matrix whose eigenvalues
    // approximate the spectrum of A; useful for diagnosing convergence.
    /// @brief Main diagonal of the Lanczos tridiagonal built from CG coefficients.
    /// @return The diagonal entries.
    Vector<Real> lanczosDiagonal() const;
    /// @brief Subdiagonal of the Lanczos tridiagonal built from CG coefficients.
    /// @return The subdiagonal entries.
    Vector<Real> lanczosSubdiagonal() const;
    /// @brief Condition-number estimate from the Lanczos spectrum.
    /// @return An estimate of cond(A).
    Real         conditionNumberEstimate() const;
    /// @brief Per-iteration energy-norm error history.
    /// @return The energy-norm history.
    Vector<Real> energyNormHistory() const;

private:
    Vector<Real> alphas_;
    Vector<Real> betas_;
};

// CG on the normal equations, for least squares without forming A^T A.
// Accepts rectangular A; with lambda set it solves the damped problem
// min ||Ax - b||^2 + lambda^2 ||x||^2.
template <typename T>
class LSQR : public KrylovSolver<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    /// @brief Constructs an LSQR solver with default options.
    LSQR();
    /// @brief Constructs an LSQR solver with explicit options.
    /// @param options Iteration limits, tolerances, and reporting settings.
    explicit LSQR(const typename KrylovSolver<T>::Options& options);

    /// @brief Solves the least-squares problem min ||A x - b|| from x0 = 0.
    /// @param a The (possibly rectangular) system matrix.
    /// @param b The right-hand side.
    /// @return The computed least-squares solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    /// @brief Solves the least-squares problem from an initial guess.
    /// @param a The (possibly rectangular) system matrix.
    /// @param b The right-hand side.
    /// @param initialGuess The starting iterate x0.
    /// @return The computed least-squares solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    /// @brief Sets the Tikhonov damping lambda for the regularized problem.
    /// @param lambda Regularization parameter (0 disables damping).
    void setRegularization(Real lambda);
    /// @brief Estimate of ||A|| accumulated during the iteration.
    /// @return The matrix-norm estimate.
    Real matrixNormEstimate() const;
    /// @brief Estimate of cond(A) accumulated during the iteration.
    /// @return The condition-number estimate.
    Real conditionNumberEstimate() const;

private:
    Real lambda_;
    Real normEstimate_;
    Real conditionEstimate_;
};

// GMRES with Arnoldi, Givens-rotation least squares, and restarting.
template <typename T>
class GMRES : public KrylovSolver<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    // How each new Arnoldi vector is orthogonalized against the basis;
    // the tradeoff mirrors GramSchmidt::Variant (speed vs orthogonality).
    struct Orthogonalization {
        enum class Kind { ModifiedGramSchmidt, Householder, ClassicalTwice };
    };

    /// @brief Constructs a GMRES solver with default options.
    GMRES();
    /// @brief Constructs a GMRES solver with explicit options.
    /// @param options Iteration limits, tolerances, and reporting settings.
    explicit GMRES(const typename KrylovSolver<T>::Options& options);

    /// @brief Solves A x = b from a zero initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    /// @brief Solves A x = b from an initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @param initialGuess The starting iterate x0.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    /// @brief Sets the restart length m of GMRES(m); 0 means never restart.
    /// @param restart Basis size before restarting.
    void  setRestart(Index restart);
    /// @brief Selects how Arnoldi vectors are orthogonalized.
    /// @param kind The orthogonalization scheme.
    void  setOrthogonalization(typename Orthogonalization::Kind kind);
    /// @brief The current restart length.
    /// @return The restart parameter m.
    Index restart() const;

    /// @brief The Arnoldi basis from the last cycle.
    /// @return Reference to the basis matrix.
    const Matrix<T>& arnoldiBasis() const;
    /// @brief The upper Hessenberg matrix from the last cycle.
    /// @return Reference to the Hessenberg matrix.
    const Matrix<T>& hessenbergMatrix() const;
    /// @brief Number of restarts performed in the last solve.
    /// @return The restart count.
    Index            restartCount() const;

private:
    /// @brief Performs one Arnoldi step, extending the basis.
    /// @param a The system matrix.
    /// @param k The step index.
    void arnoldiStep(const Matrix<T>& a, Index k);
    /// @brief Applies the stored Givens rotations to the new Hessenberg column.
    /// @param k The current column index.
    void applyStoredRotations(Index k);
    /// @brief Solves the projected least-squares problem for the iterate.
    /// @param k The current subspace dimension.
    void solveProjectedProblem(Index k);

    Matrix<T>                       basis_;
    Matrix<T>                       hessenberg_;
    Index                           restart_;
    Index                           restartCount_;
    typename Orthogonalization::Kind orthogonalization_;
};

// Short-recurrence method for nonsymmetric systems.
template <typename T>
class BiCGSTAB : public KrylovSolver<T> {
public:
    using Real  = typename NumericTraits<T>::Real;
    using Index = std::size_t;

    /// @brief Constructs a BiCGSTAB solver with default options.
    BiCGSTAB();
    /// @brief Constructs a BiCGSTAB solver with explicit options.
    /// @param options Iteration limits, tolerances, and reporting settings.
    explicit BiCGSTAB(const typename KrylovSolver<T>::Options& options);

    /// @brief Solves the nonsymmetric system A x = b from a zero initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    /// @brief Solves the nonsymmetric system A x = b from an initial guess.
    /// @param a The system matrix (or operator).
    /// @param b The right-hand side.
    /// @param initialGuess The starting iterate x0.
    /// @return The computed solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    /// @brief Reports whether a breakdown was detected during the run.
    /// @return True if a recurrence divisor vanished.
    bool breakdownDetected() const;
    /// @brief The smallest |omega| encountered; near zero warns of stagnation.
    /// @return The minimum omega magnitude over the run.
    Real smallestOmega() const;

private:
    bool breakdown_;
    Real smallestOmega_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class KrylovSolver<SCALAR>; \
    extern template class ConjugateGradient<SCALAR>; \
    extern template class LSQR<SCALAR>; \
    extern template class GMRES<SCALAR>; \
    extern template class BiCGSTAB<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
