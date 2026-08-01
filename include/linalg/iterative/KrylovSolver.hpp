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

    KrylovSolver();
    explicit KrylovSolver(const Options& options);
    virtual ~KrylovSolver();

    virtual Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) = 0;
    virtual Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                            const Vector<T>& initialGuess) = 0;

    // Non-owning; the preconditioner must outlive the solver and already
    // be set up. nullptr means unpreconditioned.
    void setPreconditioner(const Preconditioner<T>* preconditioner);
    void setOptions(const Options& options);

    const Options& options() const;
    const Report&  lastReport() const;

protected:
    bool hasConverged(Real residual, Real referenceResidual) const;
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

    ConjugateGradient();
    explicit ConjugateGradient(const typename KrylovSolver<T>::Options& options);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    // The CG coefficients build a tridiagonal matrix whose eigenvalues
    // approximate the spectrum of A; useful for diagnosing convergence.
    Vector<Real> lanczosDiagonal() const;
    Vector<Real> lanczosSubdiagonal() const;
    Real         conditionNumberEstimate() const;
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

    LSQR();
    explicit LSQR(const typename KrylovSolver<T>::Options& options);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    void setRegularization(Real lambda);
    Real matrixNormEstimate() const;
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

    GMRES();
    explicit GMRES(const typename KrylovSolver<T>::Options& options);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    void  setRestart(Index restart);  // the m of GMRES(m): basis size before restarting; 0 = never restart
    void  setOrthogonalization(typename Orthogonalization::Kind kind);
    Index restart() const;

    const Matrix<T>& arnoldiBasis() const;
    const Matrix<T>& hessenbergMatrix() const;
    Index            restartCount() const;

private:
    void arnoldiStep(const Matrix<T>& a, Index k);
    void applyStoredRotations(Index k);
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

    BiCGSTAB();
    explicit BiCGSTAB(const typename KrylovSolver<T>::Options& options);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b) override;
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b,
                    const Vector<T>& initialGuess) override;

    bool breakdownDetected() const;
    Real smallestOmega() const;  // min |omega| over the run; near zero warns of stagnation

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
