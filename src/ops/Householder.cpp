#include "linalg/ops/Householder.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
Householder<T>::Householder()
{
    throw LinalgError("not implemented: linalg::Householder<T>::Householder");
}

template <typename T>
Householder<T>::Householder(const Vector<T>& essential, const T& beta)
{
    throw LinalgError("not implemented: linalg::Householder<T>::Householder");
}

template <typename T>
Householder<T> Householder<T>::FromVector(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Householder<T>::FromVector");
}

template <typename T>
Householder<T> Householder<T>::FromColumn(const ConstMatrixView<T>& a, Index col, Index startRow)
{
    throw LinalgError("not implemented: linalg::Householder<T>::FromColumn");
}

template <typename T>
const Vector<T>& Householder<T>::essential() const
{
    throw LinalgError("not implemented: linalg::Householder<T>::essential");
}

template <typename T>
const T& Householder<T>::beta() const
{
    throw LinalgError("not implemented: linalg::Householder<T>::beta");
}

template <typename T>
typename Householder<T>::Index Householder<T>::size() const
{
    throw LinalgError("not implemented: linalg::Householder<T>::size");
}

template <typename T>
bool Householder<T>::isIdentity() const
{
    throw LinalgError("not implemented: linalg::Householder<T>::isIdentity");
}

template <typename T>
void Householder<T>::applyLeft(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::Householder<T>::applyLeft");
}

template <typename T>
void Householder<T>::applyRight(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::Householder<T>::applyRight");
}

template <typename T>
void Householder<T>::applyLeftConjugate(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::Householder<T>::applyLeftConjugate");
}

template <typename T>
void Householder<T>::apply(Vector<T>& x) const
{
    throw LinalgError("not implemented: linalg::Householder<T>::apply");
}

template <typename T>
Matrix<T> Householder<T>::toMatrix(Index dimension) const
{
    throw LinalgError("not implemented: linalg::Householder<T>::toMatrix");
}

template <typename T>
HouseholderSequence<T>::HouseholderSequence()
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::HouseholderSequence");
}

template <typename T>
HouseholderSequence<T>::HouseholderSequence(const Matrix<T>& reflectors, const Vector<T>& betas)
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::HouseholderSequence");
}

template <typename T>
void HouseholderSequence<T>::append(const Householder<T>& reflector)
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::append");
}

template <typename T>
typename HouseholderSequence<T>::Index HouseholderSequence<T>::count() const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::count");
}

template <typename T>
void HouseholderSequence<T>::applyLeft(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::applyLeft");
}

template <typename T>
void HouseholderSequence<T>::applyRight(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::applyRight");
}

template <typename T>
void HouseholderSequence<T>::applyLeftTranspose(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::applyLeftTranspose");
}

template <typename T>
Matrix<T> HouseholderSequence<T>::toMatrix(Index dimension) const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::toMatrix");
}

template <typename T>
Matrix<T> HouseholderSequence<T>::firstColumns(Index dimension, Index count) const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::firstColumns");
}

template <typename T>
Matrix<T> HouseholderSequence<T>::blockV() const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::blockV");
}

template <typename T>
Matrix<T> HouseholderSequence<T>::blockT() const
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::blockT");
}

template <typename T>
void HouseholderSequence<T>::buildBlockRepresentation(Index blockSize)
{
    throw LinalgError("not implemented: linalg::HouseholderSequence<T>::buildBlockRepresentation");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Householder<SCALAR>; \
    template class HouseholderSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
