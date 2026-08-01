#include "linalg/decomp/Cholesky.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
Cholesky<T>::Cholesky()
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::Cholesky");
}

template <typename T>
Cholesky<T>::Cholesky(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::Cholesky");
}

template <typename T>
Cholesky<T>::Cholesky(const Matrix<T>& a, const Options& options)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::Cholesky");
}

template <typename T>
void Cholesky<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::compute");
}

template <typename T>
bool Cholesky<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::isComputed");
}

template <typename T>
bool Cholesky<T>::isPositiveDefinite() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::isPositiveDefinite");
}

template <typename T>
Matrix<T> Cholesky<T>::lower() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::lower");
}

template <typename T>
Matrix<T> Cholesky<T>::upper() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::upper");
}

template <typename T>
Vector<T> Cholesky<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::solve");
}

template <typename T>
Matrix<T> Cholesky<T>::solve(const Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::solve");
}

template <typename T>
Matrix<T> Cholesky<T>::inverse() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::inverse");
}

template <typename T>
T Cholesky<T>::determinant() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::determinant");
}

template <typename T>
typename Cholesky<T>::Real Cholesky<T>::logDeterminant() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::logDeterminant");
}

template <typename T>
typename Cholesky<T>::Real Cholesky<T>::reciprocalConditionEstimate() const
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::reciprocalConditionEstimate");
}

template <typename T>
void Cholesky<T>::update(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::update");
}

template <typename T>
void Cholesky<T>::downdate(const Vector<T>& x)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::downdate");
}

template <typename T>
void Cholesky<T>::unblockedFactorize()
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::unblockedFactorize");
}

template <typename T>
void Cholesky<T>::blockedFactorize(Index blockSize)
{
    throw LinalgError("not implemented: linalg::Cholesky<T>::blockedFactorize");
}

template <typename T>
LDLT<T>::LDLT()
{
    throw LinalgError("not implemented: linalg::LDLT<T>::LDLT");
}

template <typename T>
LDLT<T>::LDLT(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::LDLT<T>::LDLT");
}

template <typename T>
void LDLT<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::LDLT<T>::compute");
}

template <typename T>
bool LDLT<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::isComputed");
}

template <typename T>
Matrix<T> LDLT<T>::lower() const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::lower");
}

template <typename T>
Vector<T> LDLT<T>::diagonal() const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::diagonal");
}

template <typename T>
Matrix<T> LDLT<T>::permutationMatrix() const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::permutationMatrix");
}

template <typename T>
Vector<T> LDLT<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::solve");
}

template <typename T>
bool LDLT<T>::isPositiveSemiDefinite(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::isPositiveSemiDefinite");
}

template <typename T>
bool LDLT<T>::isNegativeSemiDefinite(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::isNegativeSemiDefinite");
}

template <typename T>
typename LDLT<T>::Index LDLT<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::LDLT<T>::rank");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Cholesky<SCALAR>; \
    template class LDLT<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
