#include "linalg/ops/Norm.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
typename Norm<T>::Real Norm<T>::vectorOne(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Norm<T>::vectorOne");
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorTwo(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Norm<T>::vectorTwo");
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorInfinity(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Norm<T>::vectorInfinity");
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorP(const Vector<T>& x, Real p)
{
    throw LinalgError("not implemented: linalg::Norm<T>::vectorP");
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorTwoScaled(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Norm<T>::vectorTwoScaled");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixOne(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixOne");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixInfinity(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixInfinity");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixFrobenius(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixFrobenius");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixMax(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixMax");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixTwo(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixTwo");
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixNuclear(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Norm<T>::matrixNuclear");
}

template <typename T>
typename Norm<T>::Real Norm<T>::distance(const Vector<T>& x, const Vector<T>& y)
{
    throw LinalgError("not implemented: linalg::Norm<T>::distance");
}

template <typename T>
typename Norm<T>::Real Norm<T>::relativeError(const Vector<T>& approx, const Vector<T>& exact)
{
    throw LinalgError("not implemented: linalg::Norm<T>::relativeError");
}

template <typename T>
typename Norm<T>::Real Norm<T>::residualNorm(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b)
{
    throw LinalgError("not implemented: linalg::Norm<T>::residualNorm");
}

template <typename T>
typename Norm<T>::Real Norm<T>::backwardError(const Matrix<T>& a, const Vector<T>& x, const Vector<T>& b)
{
    throw LinalgError("not implemented: linalg::Norm<T>::backwardError");
}

template <typename T>
typename Norm<T>::Real Norm<T>::orthogonalityDefect(const Matrix<T>& q)
{
    throw LinalgError("not implemented: linalg::Norm<T>::orthogonalityDefect");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Norm<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
