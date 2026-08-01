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

    LeastSquares();
    explicit LeastSquares(typename Method::Kind method);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b);
    // Minimum-norm: among all least-squares minimizers (rank-deficient
    // case), the x with smallest ||x||.
    Vector<T> solveMinimumNorm(const Matrix<T>& a, const Vector<T>& b);
    // Weighted: minimizes ||W^(1/2) (A x - b)|| for diagonal W = weights.
    Vector<T> solveWeighted(const Matrix<T>& a, const Vector<T>& b,
                            const Vector<T>& weights);
    // Ridge/Tikhonov: minimizes ||A x - b||^2 + lambda^2 ||x||^2.
    Vector<T> solveRegularized(const Matrix<T>& a, const Vector<T>& b,
                               Real lambda);
    // Equality-constrained: minimizes ||A x - b|| subject to C x = d
    // (nullspace method: QR of C^H, then unconstrained solve in the
    // remaining directions).
    Vector<T> solveConstrained(const Matrix<T>& a, const Vector<T>& b,
                               const Matrix<T>& c, const Vector<T>& d);

    // Regression statistics, in the Gauss-Markov sense.
    Vector<T> residual(const Matrix<T>& a, const Vector<T>& b,
                       const Vector<T>& x) const;                          // b - A x
    Matrix<T> covarianceMatrix(const Matrix<T>& a, Real variance) const;   // variance * (A^H A)^-1
    Vector<T> standardErrors(const Matrix<T>& a, const Vector<T>& residual) const; // sqrt(diag(covariance))
    Matrix<T> hatMatrix(const Matrix<T>& a) const;   // A (A^H A)^-1 A^H: projector onto range(A)

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
        virtual ~Model() = default;
        virtual Vector<T> residual(const Vector<T>& parameters) const = 0;
        virtual Matrix<T> jacobian(const Vector<T>& parameters) const = 0;
        virtual Index     residualCount() const = 0;
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

    GaussNewton();
    explicit GaussNewton(const Options& options);

    Vector<T> minimize(const Model& model, const Vector<T>& initialGuess);

    const Report& lastReport() const;

private:
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
