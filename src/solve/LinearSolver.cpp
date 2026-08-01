#include "linalg/solve/LinearSolver.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
LinearSolver<T>::LinearSolver()
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::LinearSolver");
}

template <typename T>
LinearSolver<T>::LinearSolver(const Options& options)
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::LinearSolver");
}

template <typename T>
Vector<T> LinearSolver<T>::solve(const Matrix<T>& a, const Vector<T>& b)
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::solve");
}

template <typename T>
Matrix<T> LinearSolver<T>::solve(const Matrix<T>& a, const Matrix<T>& b)
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::solve");
}

template <typename T>
const typename LinearSolver<T>::Report& LinearSolver<T>::lastReport() const
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::lastReport");
}

template <typename T>
typename LinearSolver<T>::Method::Kind LinearSolver<T>::selectMethod(const Matrix<T>& a) const
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::selectMethod");
}

template <typename T>
Vector<T> LinearSolver<T>::refine(const Matrix<T>& a, const Vector<T>& b, const Vector<T>& x, Index steps) const
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::refine");
}

template <typename T>
void LinearSolver<T>::equilibrate(Matrix<T>& a, Vector<T>& rowScale, Vector<T>& colScale) const
{
    throw LinalgError("not implemented: linalg::LinearSolver<T>::equilibrate");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class LinearSolver<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
