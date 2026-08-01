#include "linalg/decomp/QR.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
QR<T>::QR()
{
    throw LinalgError("not implemented: linalg::QR<T>::QR");
}

template <typename T>
QR<T>::QR(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::QR<T>::QR");
}

template <typename T>
QR<T>::QR(const Matrix<T>& a, const Options& options)
{
    throw LinalgError("not implemented: linalg::QR<T>::QR");
}

template <typename T>
void QR<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::QR<T>::compute");
}

template <typename T>
bool QR<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::QR<T>::isComputed");
}

template <typename T>
Matrix<T> QR<T>::matrixQ() const
{
    throw LinalgError("not implemented: linalg::QR<T>::matrixQ");
}

template <typename T>
Matrix<T> QR<T>::matrixR() const
{
    throw LinalgError("not implemented: linalg::QR<T>::matrixR");
}

template <typename T>
const HouseholderSequence<T>& QR<T>::reflectors() const
{
    throw LinalgError("not implemented: linalg::QR<T>::reflectors");
}

template <typename T>
void QR<T>::applyQOnLeft(Matrix<T>& target) const
{
    throw LinalgError("not implemented: linalg::QR<T>::applyQOnLeft");
}

template <typename T>
void QR<T>::applyQTransposeOnLeft(Matrix<T>& target) const
{
    throw LinalgError("not implemented: linalg::QR<T>::applyQTransposeOnLeft");
}

template <typename T>
Vector<T> QR<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::QR<T>::solve");
}

template <typename T>
Matrix<T> QR<T>::solve(const Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::QR<T>::solve");
}

template <typename T>
Vector<T> QR<T>::solveLeastSquares(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::QR<T>::solveLeastSquares");
}

template <typename T>
T QR<T>::absDeterminant() const
{
    throw LinalgError("not implemented: linalg::QR<T>::absDeterminant");
}

template <typename T>
typename QR<T>::Real QR<T>::logAbsDeterminant() const
{
    throw LinalgError("not implemented: linalg::QR<T>::logAbsDeterminant");
}

template <typename T>
typename QR<T>::Index QR<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::QR<T>::rank");
}

template <typename T>
void QR<T>::appendRow(const Vector<T>& row)
{
    throw LinalgError("not implemented: linalg::QR<T>::appendRow");
}

template <typename T>
void QR<T>::appendColumn(const Vector<T>& column)
{
    throw LinalgError("not implemented: linalg::QR<T>::appendColumn");
}

template <typename T>
void QR<T>::removeRow(Index index)
{
    throw LinalgError("not implemented: linalg::QR<T>::removeRow");
}

template <typename T>
void QR<T>::rankOneUpdate(const Vector<T>& u, const Vector<T>& v)
{
    throw LinalgError("not implemented: linalg::QR<T>::rankOneUpdate");
}

template <typename T>
void QR<T>::unblockedFactorize()
{
    throw LinalgError("not implemented: linalg::QR<T>::unblockedFactorize");
}

template <typename T>
void QR<T>::blockedFactorize(Index blockSize)
{
    throw LinalgError("not implemented: linalg::QR<T>::blockedFactorize");
}

template <typename T>
ColPivQR<T>::ColPivQR()
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::ColPivQR");
}

template <typename T>
ColPivQR<T>::ColPivQR(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::ColPivQR");
}

template <typename T>
void ColPivQR<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::compute");
}

template <typename T>
bool ColPivQR<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::isComputed");
}

template <typename T>
Matrix<T> ColPivQR<T>::matrixQ() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::matrixQ");
}

template <typename T>
Matrix<T> ColPivQR<T>::matrixR() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::matrixR");
}

template <typename T>
Matrix<T> ColPivQR<T>::permutationMatrix() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::permutationMatrix");
}

template <typename T>
const std::vector<typename ColPivQR<T>::Index>& ColPivQR<T>::columnPivots() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::columnPivots");
}

template <typename T>
Vector<T> ColPivQR<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::solve");
}

template <typename T>
Vector<T> ColPivQR<T>::solveMinimumNorm(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::solveMinimumNorm");
}

template <typename T>
typename ColPivQR<T>::Index ColPivQR<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::rank");
}

template <typename T>
Matrix<T> ColPivQR<T>::kernel(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::kernel");
}

template <typename T>
typename ColPivQR<T>::Real ColPivQR<T>::reciprocalConditionEstimate() const
{
    throw LinalgError("not implemented: linalg::ColPivQR<T>::reciprocalConditionEstimate");
}

template <typename T>
GramSchmidt<T>::GramSchmidt()
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::GramSchmidt");
}

template <typename T>
GramSchmidt<T>::GramSchmidt(const Matrix<T>& a, typename Variant::Kind variant)
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::GramSchmidt");
}

template <typename T>
void GramSchmidt<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::compute");
}

template <typename T>
bool GramSchmidt<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::isComputed");
}

template <typename T>
Matrix<T> GramSchmidt<T>::matrixQ() const
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::matrixQ");
}

template <typename T>
Matrix<T> GramSchmidt<T>::matrixR() const
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::matrixR");
}

template <typename T>
typename GramSchmidt<T>::Real GramSchmidt<T>::orthogonalityLoss() const
{
    throw LinalgError("not implemented: linalg::GramSchmidt<T>::orthogonalityLoss");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class QR<SCALAR>; \
    template class ColPivQR<SCALAR>; \
    template class GramSchmidt<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
