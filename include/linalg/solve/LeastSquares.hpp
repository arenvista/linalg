#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// min ||Ax - b||. Normal equations are exposed only as a comparison
// point: they square the condition number and should not be the default.
template <typename T>
class LeastSquares {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    struct Method {
        enum class Kind { HouseholderQR, ColPivQR, SVD, NormalEquations };
    };

    struct Report {
        Real  residualNorm;
        Real  solutionNorm;
        Index effectiveRank;
        Real  conditionEstimate;
        bool  rankDeficient;
    };

    /// @brief Constructs a solver with the default method.
    LeastSquares();
    /// @brief Constructs a solver with an explicit method.
    /// @param method Which factorization to use.
    explicit LeastSquares(typename Method::Kind method);

    /// @brief Solves the least-squares problem min ||A x - b||.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @return The least-squares solution x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b);
    // Minimum-norm: among all least-squares minimizers (rank-deficient
    // case), the x with smallest ||x||.
    /// @brief Solves for the minimum-norm least-squares solution.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @return The minimizer of smallest ||x||.
    Vector<T> solveMinimumNorm(const Matrix<T>& a, const Vector<T>& b);
    // Weighted: minimizes ||W^(1/2) (A x - b)|| for diagonal W = weights.
    /// @brief Solves the weighted least-squares problem min ||W^(1/2)(A x - b)||.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @param weights Diagonal weights W (one per row).
    /// @return The weighted least-squares solution.
    Vector<T> solveWeighted(const Matrix<T>& a, const Vector<T>& b,
                            const Vector<T>& weights);
    // Ridge/Tikhonov: minimizes ||A x - b||^2 + lambda^2 ||x||^2.
    /// @brief Solves the ridge/Tikhonov problem min ||A x - b||^2 + lambda^2 ||x||^2.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @param lambda Regularization parameter.
    /// @return The regularized solution.
    Vector<T> solveRegularized(const Matrix<T>& a, const Vector<T>& b,
                               Real lambda);
    // Equality-constrained: minimizes ||A x - b|| subject to C x = d
    // (nullspace method: QR of C^H, then unconstrained solve in the
    // remaining directions).
    /// @brief Solves min ||A x - b|| subject to the constraint C x = d.
    /// @param a Coefficient matrix of the objective.
    /// @param b Right-hand side of the objective.
    /// @param c Constraint matrix.
    /// @param d Constraint right-hand side.
    /// @return The constrained least-squares solution.
    Vector<T> solveConstrained(const Matrix<T>& a, const Vector<T>& b,
                               const Matrix<T>& c, const Vector<T>& d);

    // Regression statistics, in the Gauss-Markov sense.
    /// @brief Residual of a fit, b - A x.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @param x The solution.
    /// @return The residual vector.
    Vector<T> residual(const Matrix<T>& a, const Vector<T>& b,
                       const Vector<T>& x) const;
    /// @brief Parameter covariance matrix, variance * (A^H A)^-1.
    /// @param a Coefficient matrix.
    /// @param variance The noise variance estimate.
    /// @return The covariance matrix.
    Matrix<T> covarianceMatrix(const Matrix<T>& a, Real variance) const;
    /// @brief Standard errors of the parameters, sqrt(diag(covariance)).
    /// @param a Coefficient matrix.
    /// @param residual The fit residual.
    /// @return A vector of per-parameter standard errors.
    Vector<T> standardErrors(const Matrix<T>& a, const Vector<T>& residual) const;
    /// @brief Hat (projection) matrix A (A^H A)^-1 A^H onto range(A).
    /// @param a Coefficient matrix.
    /// @return The hat matrix.
    Matrix<T> hatMatrix(const Matrix<T>& a) const;

    /// @brief Diagnostics from the most recent solve.
    /// @return Reference to the last report.
    const Report& lastReport() const;

private:
    typename Method::Kind method_;
    Report                report_;
};

// Nonlinear least squares by Gauss-Newton / Levenberg-Marquardt,
// where each step is a linear least squares solve on the Jacobian.
template <typename T>
class GaussNewton {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // Users derive from this instead of passing a std::function,
    // keeping the interface entirely class-based.
    class Model {
    public:
        /// @brief Virtual destructor for the abstract model interface.
        virtual ~Model() = default;
        /// @brief Residual vector r(parameters) to be minimized.
        /// @param parameters Current parameter estimate.
        /// @return The residual vector.
        virtual Vector<T> residual(const Vector<T>& parameters) const = 0;
        /// @brief Jacobian of the residual with respect to the parameters.
        /// @param parameters Current parameter estimate.
        /// @return The Jacobian matrix.
        virtual Matrix<T> jacobian(const Vector<T>& parameters) const = 0;
        /// @brief Number of residual components.
        /// @return The residual count.
        virtual Index     residualCount() const = 0;
        /// @brief Number of parameters.
        /// @return The parameter count.
        virtual Index     parameterCount() const = 0;
    };

    struct Options {
        Index maxIterations;
        Real  gradientTolerance;      // stop when ||J^H r|| falls below this
        Real  stepTolerance;          // stop when the step is negligible relative to x
        Real  initialDamping;         // LM lambda_0
        Real  dampingIncrease;        // multiplier after a rejected (cost-increasing) step
        Real  dampingDecrease;        // multiplier after an accepted step
        bool  useLevenbergMarquardt;  // false: undamped Gauss-Newton steps
    };

    struct Report {
        Index iterations;
        Real  finalCost;
        Real  gradientNorm;
        Real  finalDamping;
        bool  converged;
    };

    /// @brief Constructs an optimizer with default options.
    GaussNewton();
    /// @brief Constructs an optimizer with explicit options.
    /// @param options Iteration limits, tolerances, and damping settings.
    explicit GaussNewton(const Options& options);

    /// @brief Minimizes the model's residual sum of squares.
    /// @param model The residual/Jacobian model to fit.
    /// @param initialGuess Starting parameter estimate.
    /// @return The optimized parameters.
    Vector<T> minimize(const Model& model, const Vector<T>& initialGuess);

    /// @brief Diagnostics from the most recent minimize().
    /// @return Reference to the last report.
    const Report& lastReport() const;

private:
    /// @brief Computes one (possibly damped) Gauss-Newton/LM step.
    /// @param jacobian Jacobian at the current estimate.
    /// @param residual Residual at the current estimate.
    /// @param damping The Levenberg-Marquardt damping lambda.
    /// @return The parameter step.
    Vector<T> computeStep(const Matrix<T>& jacobian, const Vector<T>& residual,
                          Real damping) const;

    Options options_;
    Report  report_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class LeastSquares<SCALAR>; \
    extern template class GaussNewton<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
