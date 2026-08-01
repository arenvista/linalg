#include "linalg/decomp/LU.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
LU<T>::LU()
{
    throw LinalgError("not implemented: linalg::LU<T>::LU");
}

template <typename T>
LU<T>::LU(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::LU<T>::LU");
}

template <typename T>
LU<T>::LU(const Matrix<T>& a, const Options& options)
{
    throw LinalgError("not implemented: linalg::LU<T>::LU");
}

template <typename T>
void LU<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::LU<T>::compute");
}

template <typename T>
bool LU<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::LU<T>::isComputed");
}

template <typename T>
Matrix<T> LU<T>::packedFactors() const
{
    throw LinalgError("not implemented: linalg::LU<T>::packedFactors");
}

template <typename T>
Matrix<T> LU<T>::lower() const
{
    throw LinalgError("not implemented: linalg::LU<T>::lower");
}

template <typename T>
Matrix<T> LU<T>::upper() const
{
    throw LinalgError("not implemented: linalg::LU<T>::upper");
}

template <typename T>
Matrix<T> LU<T>::permutationMatrix() const
{
    throw LinalgError("not implemented: linalg::LU<T>::permutationMatrix");
}

template <typename T>
const std::vector<typename LU<T>::Index>& LU<T>::pivots() const
{
    throw LinalgError("not implemented: linalg::LU<T>::pivots");
}

template <typename T>
typename LU<T>::Index LU<T>::permutationSign() const
{
    throw LinalgError("not implemented: linalg::LU<T>::permutationSign");
}

template <typename T>
Vector<T> LU<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::LU<T>::solve");
}

template <typename T>
Matrix<T> LU<T>::solve(const Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::LU<T>::solve");
}

template <typename T>
Vector<T> LU<T>::solveTranspose(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::LU<T>::solveTranspose");
}

template <typename T>
Vector<T> LU<T>::refineSolution(const Matrix<T>& a, const Vector<T>& b, const Vector<T>& x, Index steps) const
{
    throw LinalgError("not implemented: linalg::LU<T>::refineSolution");
}

template <typename T>
T LU<T>::determinant() const
{
    throw LinalgError("not implemented: linalg::LU<T>::determinant");
}

template <typename T>
T LU<T>::logAbsDeterminant() const
{
    throw LinalgError("not implemented: linalg::LU<T>::logAbsDeterminant");
}

template <typename T>
Matrix<T> LU<T>::inverse() const
{
    throw LinalgError("not implemented: linalg::LU<T>::inverse");
}

template <typename T>
typename LU<T>::Index LU<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::LU<T>::rank");
}

template <typename T>
bool LU<T>::isSingular(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::LU<T>::isSingular");
}

template <typename T>
typename LU<T>::Real LU<T>::reciprocalConditionEstimate() const
{
    throw LinalgError("not implemented: linalg::LU<T>::reciprocalConditionEstimate");
}

template <typename T>
void LU<T>::unblockedFactorize()
{
    throw LinalgError("not implemented: linalg::LU<T>::unblockedFactorize");
}

template <typename T>
void LU<T>::blockedFactorize(Index blockSize)
{
    throw LinalgError("not implemented: linalg::LU<T>::blockedFactorize");
}

template <typename T>
void LU<T>::applyPivotsTo(Matrix<T>& b) const
{
    throw LinalgError("not implemented: linalg::LU<T>::applyPivotsTo");
}

template <typename T>
FullPivLU<T>::FullPivLU()
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::FullPivLU");
}

template <typename T>
FullPivLU<T>::FullPivLU(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::FullPivLU");
}

template <typename T>
void FullPivLU<T>::compute(const Matrix<T>& a)
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::compute");
}

template <typename T>
bool FullPivLU<T>::isComputed() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::isComputed");
}

template <typename T>
Matrix<T> FullPivLU<T>::lower() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::lower");
}

template <typename T>
Matrix<T> FullPivLU<T>::upper() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::upper");
}

template <typename T>
Matrix<T> FullPivLU<T>::rowPermutation() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::rowPermutation");
}

template <typename T>
Matrix<T> FullPivLU<T>::columnPermutation() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::columnPermutation");
}

template <typename T>
Vector<T> FullPivLU<T>::solve(const Vector<T>& b) const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::solve");
}

template <typename T>
Matrix<T> FullPivLU<T>::kernel() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::kernel");
}

template <typename T>
Matrix<T> FullPivLU<T>::image() const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::image");
}

template <typename T>
typename FullPivLU<T>::Index FullPivLU<T>::rank(Real tolerance) const
{
    throw LinalgError("not implemented: linalg::FullPivLU<T>::rank");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class LU<SCALAR>; \
    template class FullPivLU<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
