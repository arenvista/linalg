#include "linalg/decomp/SVD.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
SVD<T>::SVD()
{
    throw LinalgError("not implemented: linalg::SVD<T>::SVD");
}

template <typename T>
SVD<T>::SVD(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::SVD<T>::SVD");
}

template <typename T>
SVD<T>::SVD(const Matrix<T>& a, const Options& options)
{
    throw LinalgError("not implemented: linalg::SVD<T>::SVD");
}

template <typename T>
void SVD<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::SVD<T>::compute");
}

template <typename T>
bool SVD<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::isComputed");
}

template <typename T>
bool SVD<T>::converged() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::converged");
}

template <typename T>
typename SVD<T>::Index SVD<T>::iterationsUsed() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::iterationsUsed");
}

template <typename T>
const Vector<typename SVD<T>::Real>& SVD<T>::singularValues() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::singularValues");
}

template <typename T>
const Matrix<T>& SVD<T>::matrixU() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::matrixU");
}

template <typename T>
const Matrix<T>& SVD<T>::matrixV() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::matrixV");
}

template <typename T>
typename SVD<T>::Real SVD<T>::largestSingularValue() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::largestSingularValue");
}

template <typename T>
typename SVD<T>::Real SVD<T>::smallestSingularValue() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::smallestSingularValue");
}

template <typename T>
typename SVD<T>::Real SVD<T>::conditionNumber() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::conditionNumber");
}

template <typename T>
typename SVD<T>::Index SVD<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::rank");
}

template <typename T>
typename SVD<T>::Real SVD<T>::spectralNorm() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::spectralNorm");
}

template <typename T>
typename SVD<T>::Real SVD<T>::nuclearNorm() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::nuclearNorm");
}

template <typename T>
Vector<T> SVD<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::solve");
}

template <typename T>
Vector<T> SVD<T>::solveTruncated(const Vector<T>& b, Index keep) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::solveTruncated");
}

template <typename T>
Vector<T> SVD<T>::solveTikhonov(const Vector<T>& b, Real lambda) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::solveTikhonov");
}

template <typename T>
Matrix<T> SVD<T>::pseudoInverse(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::pseudoInverse");
}

template <typename T>
Matrix<T> SVD<T>::lowRankApproximation(Index rank) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::lowRankApproximation");
}

template <typename T>
Matrix<T> SVD<T>::nullSpace(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::nullSpace");
}

template <typename T>
Matrix<T> SVD<T>::columnSpace(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::SVD<T>::columnSpace");
}

template <typename T>
Matrix<T> SVD<T>::orthogonalPolarFactor() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::orthogonalPolarFactor");
}

template <typename T>
Matrix<T> SVD<T>::hermitianPolarFactor() const
{
    throw LinalgError("not implemented: linalg::SVD<T>::hermitianPolarFactor");
}

template <typename T>
void SVD<T>::computeGolubKahan()
{
    throw LinalgError("not implemented: linalg::SVD<T>::computeGolubKahan");
}

template <typename T>
void SVD<T>::computeJacobi()
{
    throw LinalgError("not implemented: linalg::SVD<T>::computeJacobi");
}

template <typename T>
void SVD<T>::sortSingularValues()
{
    throw LinalgError("not implemented: linalg::SVD<T>::sortSingularValues");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class SVD<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
