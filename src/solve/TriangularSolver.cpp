#include "linalg/solve/TriangularSolver.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
TriangularSolver<T>::TriangularSolver()
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::TriangularSolver");
}

template <typename T>
TriangularSolver<T>::TriangularSolver(const Options& options)
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::TriangularSolver");
}

template <typename T>
Vector<T> TriangularSolver<T>::solve(const Matrix<T>& a, const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::solve");
}

template <typename T>
Matrix<T> TriangularSolver<T>::solve(const Matrix<T>& a, const Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::solve");
}

template <typename T>
void TriangularSolver<T>::solveInPlace(const Matrix<T>& a, Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::solveInPlace");
}

template <typename T>
void TriangularSolver<T>::solveInPlace(const Matrix<T>& a, Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::solveInPlace");
}

template <typename T>
Matrix<T> TriangularSolver<T>::inverse(const Matrix<T>& a) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::inverse");
}

template <typename T>
typename TriangularSolver<T>::Real TriangularSolver<T>::reciprocalConditionEstimate(const Matrix<T>& a) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::reciprocalConditionEstimate");
}

template <typename T>
bool TriangularSolver<T>::isNonsingular(const Matrix<T>& a, Real tolerance) const
{
    throw LinalgError("not implemented: linalg::TriangularSolver<T>::isNonsingular");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class TriangularSolver<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
