#include "linalg/iterative/Preconditioner.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
Preconditioner<T>::~Preconditioner()
{
}

template <typename T>
IdentityPreconditioner<T>::IdentityPreconditioner()
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::IdentityPreconditioner");
}

template <typename T>
void IdentityPreconditioner<T>::setup(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::setup");
}

template <typename T>
Vector<T> IdentityPreconditioner<T>::apply(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::apply");
}

template <typename T>
Vector<T> IdentityPreconditioner<T>::applyTranspose(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::applyTranspose");
}

template <typename T>
bool IdentityPreconditioner<T>::isSymmetric() const
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::isSymmetric");
}

template <typename T>
typename IdentityPreconditioner<T>::Index IdentityPreconditioner<T>::dimension() const
{
    throw LinalgError("not implemented: linalg::IdentityPreconditioner<T>::dimension");
}

template <typename T>
JacobiPreconditioner<T>::JacobiPreconditioner()
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::JacobiPreconditioner");
}

template <typename T>
void JacobiPreconditioner<T>::setup(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::setup");
}

template <typename T>
Vector<T> JacobiPreconditioner<T>::apply(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::apply");
}

template <typename T>
Vector<T> JacobiPreconditioner<T>::applyTranspose(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::applyTranspose");
}

template <typename T>
bool JacobiPreconditioner<T>::isSymmetric() const
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::isSymmetric");
}

template <typename T>
typename JacobiPreconditioner<T>::Index JacobiPreconditioner<T>::dimension() const
{
    throw LinalgError("not implemented: linalg::JacobiPreconditioner<T>::dimension");
}

template <typename T>
SSORPreconditioner<T>::SSORPreconditioner()
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::SSORPreconditioner");
}

template <typename T>
SSORPreconditioner<T>::SSORPreconditioner(Real relaxation)
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::SSORPreconditioner");
}

template <typename T>
void SSORPreconditioner<T>::setup(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::setup");
}

template <typename T>
Vector<T> SSORPreconditioner<T>::apply(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::apply");
}

template <typename T>
Vector<T> SSORPreconditioner<T>::applyTranspose(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::applyTranspose");
}

template <typename T>
bool SSORPreconditioner<T>::isSymmetric() const
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::isSymmetric");
}

template <typename T>
typename SSORPreconditioner<T>::Index SSORPreconditioner<T>::dimension() const
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::dimension");
}

template <typename T>
typename SSORPreconditioner<T>::Real SSORPreconditioner<T>::optimalRelaxationEstimate() const
{
    throw LinalgError("not implemented: linalg::SSORPreconditioner<T>::optimalRelaxationEstimate");
}

template <typename T>
IncompleteCholeskyPreconditioner<T>::IncompleteCholeskyPreconditioner()
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::IncompleteCholeskyPreconditioner");
}

template <typename T>
IncompleteCholeskyPreconditioner<T>::IncompleteCholeskyPreconditioner(Real shift)
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::IncompleteCholeskyPreconditioner");
}

template <typename T>
void IncompleteCholeskyPreconditioner<T>::setup(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::setup");
}

template <typename T>
Vector<T> IncompleteCholeskyPreconditioner<T>::apply(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::apply");
}

template <typename T>
Vector<T> IncompleteCholeskyPreconditioner<T>::applyTranspose(const Vector<T>& r) const
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::applyTranspose");
}

template <typename T>
bool IncompleteCholeskyPreconditioner<T>::isSymmetric() const
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::isSymmetric");
}

template <typename T>
typename IncompleteCholeskyPreconditioner<T>::Index IncompleteCholeskyPreconditioner<T>::dimension() const
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::dimension");
}

template <typename T>
bool IncompleteCholeskyPreconditioner<T>::breakdownOccurred() const
{
    throw LinalgError("not implemented: linalg::IncompleteCholeskyPreconditioner<T>::breakdownOccurred");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Preconditioner<SCALAR>; \
    template class IdentityPreconditioner<SCALAR>; \
    template class JacobiPreconditioner<SCALAR>; \
    template class SSORPreconditioner<SCALAR>; \
    template class IncompleteCholeskyPreconditioner<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
