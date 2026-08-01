#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Front door for Ax = b. Inspects A and dispatches to Cholesky,
// LDLT, LU, or QR rather than making the caller choose.
template <typename T>
class LinearSolver {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // Automatic dispatch: Cholesky for Hermitian matrices that factor
    // (falling back to LDLT when indefinite), LU for general square,
    // ColPivQR/SVD for anything rank-suspect or non-square.
    struct Method {
        enum class Kind { Automatic, LU, FullPivLU, Cholesky, LDLT, QR, ColPivQR, SVD };
    };

    struct Report {
        typename Method::Kind chosenMethod;       // what Automatic actually picked
        Real                  residualNorm;       // ||b - A x||
        Real                  backwardError;      // ||b - A x|| / (||A|| ||x|| + ||b||)
        Real                  conditionEstimate;
        Index                 refinementSteps;    // refinement iterations actually taken
        bool                  succeeded;
    };

    struct Options {
        typename Method::Kind method;
        bool                  iterativeRefinement; // polish x with residual corrections
        Index                 maxRefinementSteps;
        Real                  tolerance;           // singularity / rank threshold passed down
        bool                  equilibrate;         // row/column scale A first (helps badly scaled systems)
    };

    /// @brief Constructs a solver with default options.
    LinearSolver();
    /// @brief Constructs a solver with explicit options.
    /// @param options Method choice, refinement, tolerance, and scaling settings.
    explicit LinearSolver(const Options& options);

    /// @brief Solves A x = b, dispatching to a suitable factorization.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side vector.
    /// @return The solution vector x.
    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b);
    /// @brief Solves A X = B for multiple right-hand sides.
    /// @param a Coefficient matrix.
    /// @param b Right-hand side matrix (one column per system).
    /// @return The solution matrix X.
    Matrix<T> solve(const Matrix<T>& a, const Matrix<T>& b);

    /// @brief Diagnostics from the most recent solve.
    /// @return Reference to the last report.
    const Report& lastReport() const;

    /// @brief The Automatic dispatch policy, exposed for tests.
    /// @param a Coefficient matrix to inspect.
    /// @return The method Automatic would pick for a.
    typename Method::Kind selectMethod(const Matrix<T>& a) const;
    /// @brief Polishes a solution with iterative refinement.
    /// @param a The original coefficient matrix.
    /// @param b The original right-hand side.
    /// @param x The approximate solution to refine.
    /// @param steps Number of refinement iterations.
    /// @return The refined solution.
    Vector<T>             refine(const Matrix<T>& a, const Vector<T>& b,
                                 const Vector<T>& x, Index steps) const;
    // Scales A in place to D_r A D_c with rows/columns of comparable norm;
    // returns the scalings so the solution can be unscaled afterward.
    /// @brief Equilibrates A in place to D_r A D_c with balanced row/column norms.
    /// @param a Matrix scaled in place.
    /// @param rowScale Output row scaling D_r.
    /// @param colScale Output column scaling D_c.
    void                  equilibrate(Matrix<T>& a, Vector<T>& rowScale,
                                      Vector<T>& colScale) const;

private:
    Options options_;
    Report  report_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class LinearSolver<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
