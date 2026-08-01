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

    LinearSolver();
    explicit LinearSolver(const Options& options);

    Vector<T> solve(const Matrix<T>& a, const Vector<T>& b);
    Matrix<T> solve(const Matrix<T>& a, const Matrix<T>& b);

    const Report& lastReport() const;

    typename Method::Kind selectMethod(const Matrix<T>& a) const;  // the Automatic policy, exposed for tests
    Vector<T>             refine(const Matrix<T>& a, const Vector<T>& b,
                                 const Vector<T>& x, Index steps) const;
    // Scales A in place to D_r A D_c with rows/columns of comparable norm;
    // returns the scalings so the solution can be unscaled afterward.
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
