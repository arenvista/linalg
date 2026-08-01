#include "linalg/ops/Givens.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
Givens<T>::Givens()
{
    throw LinalgError("not implemented: linalg::Givens<T>::Givens");
}

template <typename T>
Givens<T>::Givens(const T& cosine, const T& sine, Index p, Index q)
{
    throw LinalgError("not implemented: linalg::Givens<T>::Givens");
}

template <typename T>
Givens<T> Givens<T>::FromPair(const T& a, const T& b, Index p, Index q)
{
    throw LinalgError("not implemented: linalg::Givens<T>::FromPair");
}

template <typename T>
Givens<T> Givens<T>::Identity(Index p, Index q)
{
    throw LinalgError("not implemented: linalg::Givens<T>::Identity");
}

template <typename T>
const T& Givens<T>::cosine() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::cosine");
}

template <typename T>
const T& Givens<T>::sine() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::sine");
}

template <typename T>
typename Givens<T>::Index Givens<T>::firstIndex() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::firstIndex");
}

template <typename T>
typename Givens<T>::Index Givens<T>::secondIndex() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::secondIndex");
}

template <typename T>
T Givens<T>::radius() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::radius");
}

template <typename T>
Givens<T> Givens<T>::transposed() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::transposed");
}

template <typename T>
Givens<T> Givens<T>::inverse() const
{
    throw LinalgError("not implemented: linalg::Givens<T>::inverse");
}

template <typename T>
void Givens<T>::applyLeft(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::Givens<T>::applyLeft");
}

template <typename T>
void Givens<T>::applyRight(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::Givens<T>::applyRight");
}

template <typename T>
void Givens<T>::apply(Vector<T>& x) const
{
    throw LinalgError("not implemented: linalg::Givens<T>::apply");
}

template <typename T>
Matrix<T> Givens<T>::toMatrix(Index dimension) const
{
    throw LinalgError("not implemented: linalg::Givens<T>::toMatrix");
}

template <typename T>
GivensSequence<T>::GivensSequence()
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::GivensSequence");
}

template <typename T>
void GivensSequence<T>::append(const Givens<T>& rotation)
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::append");
}

template <typename T>
void GivensSequence<T>::clear()
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::clear");
}

template <typename T>
typename GivensSequence<T>::Index GivensSequence<T>::count() const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::count");
}

template <typename T>
const Givens<T>& GivensSequence<T>::operator[](Index k) const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::operator[]");
}

template <typename T>
void GivensSequence<T>::applyLeft(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::applyLeft");
}

template <typename T>
void GivensSequence<T>::applyRight(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::applyRight");
}

template <typename T>
void GivensSequence<T>::applyLeftReversed(MatrixView<T> target) const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::applyLeftReversed");
}

template <typename T>
GivensSequence<T> GivensSequence<T>::reversed() const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::reversed");
}

template <typename T>
Matrix<T> GivensSequence<T>::toMatrix(Index dimension) const
{
    throw LinalgError("not implemented: linalg::GivensSequence<T>::toMatrix");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Givens<SCALAR>; \
    template class GivensSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
