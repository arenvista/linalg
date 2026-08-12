#include "linalg/solve/LeastSquares.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> LeastSquares<T>::LeastSquares() {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::LeastSquares");
}

template <typename T>
LeastSquares<T>::LeastSquares(typename Method::Kind method) {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::LeastSquares");
}

template <typename T>
Vector<T> LeastSquares<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::solve");
}

template <typename T>
Vector<T> LeastSquares<T>::solveMinimumNorm(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::solveMinimumNorm"
    );
}

template <typename T>
Vector<T> LeastSquares<T>::solveWeighted(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &weights
) {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::solveWeighted"
    );
}

template <typename T>
Vector<T> LeastSquares<T>::solveRegularized(
    const Matrix<T> &a,
    const Vector<T> &b,
    Real             lambda
) {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::solveRegularized"
    );
}

template <typename T>
Vector<T> LeastSquares<T>::solveConstrained(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Matrix<T> &c,
    const Vector<T> &d
) {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::solveConstrained"
    );
}

template <typename T>
Vector<T> LeastSquares<T>::residual(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &x
) const {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::residual");
}

template <typename T>
Matrix<T> LeastSquares<T>::covarianceMatrix(
    const Matrix<T> &a,
    Real             variance
) const {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::covarianceMatrix"
    );
}

template <typename T>
Vector<T> LeastSquares<T>::standardErrors(
    const Matrix<T> &a,
    const Vector<T> &residual
) const {
    throw LinalgError(
        "not implemented: linalg::LeastSquares<T>::standardErrors"
    );
}

template <typename T>
Matrix<T> LeastSquares<T>::hatMatrix(const Matrix<T> &a) const {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::hatMatrix");
}

template <typename T>
const typename LeastSquares<T>::Report &LeastSquares<T>::lastReport() const {
    throw LinalgError("not implemented: linalg::LeastSquares<T>::lastReport");
}

template <typename T> GaussNewton<T>::GaussNewton() {
    throw LinalgError("not implemented: linalg::GaussNewton<T>::GaussNewton");
}

template <typename T> GaussNewton<T>::GaussNewton(const Options &options) {
    throw LinalgError("not implemented: linalg::GaussNewton<T>::GaussNewton");
}

template <typename T>
Vector<T> GaussNewton<T>::minimize(
    const Model     &model,
    const Vector<T> &initialGuess
) {
    throw LinalgError("not implemented: linalg::GaussNewton<T>::minimize");
}

template <typename T>
const typename GaussNewton<T>::Report &GaussNewton<T>::lastReport() const {
    throw LinalgError("not implemented: linalg::GaussNewton<T>::lastReport");
}

template <typename T>
Vector<T> GaussNewton<T>::computeStep(
    const Matrix<T> &jacobian,
    const Vector<T> &residual,
    Real             damping
) const {
    throw LinalgError("not implemented: linalg::GaussNewton<T>::computeStep");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)       \
    template class LeastSquares<SCALAR>; \
    template class GaussNewton<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
