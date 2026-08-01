#include "linalg/ops/Kernels.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
void Kernels<T>::scal(Index n, const T& alpha, T* x, Index incx)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::scal");
}

template <typename T>
void Kernels<T>::axpy(Index n, const T& alpha, const T* x, Index incx, T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::axpy");
}

template <typename T>
T Kernels<T>::dot(Index n, const T* x, Index incx, const T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::dot");
}

template <typename T>
T Kernels<T>::dotc(Index n, const T* x, Index incx, const T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::dotc");
}

template <typename T>
typename Kernels<T>::Real Kernels<T>::nrm2(Index n, const T* x, Index incx)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::nrm2");
}

template <typename T>
typename Kernels<T>::Real Kernels<T>::asum(Index n, const T* x, Index incx)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::asum");
}

template <typename T>
typename Kernels<T>::Index Kernels<T>::iamax(Index n, const T* x, Index incx)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::iamax");
}

template <typename T>
void Kernels<T>::swap(Index n, T* x, Index incx, T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::swap");
}

template <typename T>
void Kernels<T>::copy(Index n, const T* x, Index incx, T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::copy");
}

template <typename T>
void Kernels<T>::gemv(Transposition::Kind trans, const T& alpha, const ConstMatrixView<T>& a, const T* x, Index incx, const T& beta, T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::gemv");
}

template <typename T>
void Kernels<T>::ger(const T& alpha, const T* x, Index incx, const T* y, Index incy, MatrixView<T> a)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::ger");
}

template <typename T>
void Kernels<T>::trsv(Triangle::Kind uplo, Transposition::Kind trans, Diagonal::Kind diag, const ConstMatrixView<T>& a, T* x, Index incx)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::trsv");
}

template <typename T>
void Kernels<T>::symv(Triangle::Kind uplo, const T& alpha, const ConstMatrixView<T>& a, const T* x, Index incx, const T& beta, T* y, Index incy)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::symv");
}

template <typename T>
void Kernels<T>::gemm(Transposition::Kind transA, Transposition::Kind transB, const T& alpha, const ConstMatrixView<T>& a, const ConstMatrixView<T>& b, const T& beta, MatrixView<T> c)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::gemm");
}

template <typename T>
void Kernels<T>::syrk(Triangle::Kind uplo, Transposition::Kind trans, const T& alpha, const ConstMatrixView<T>& a, const T& beta, MatrixView<T> c)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::syrk");
}

template <typename T>
void Kernels<T>::trsm(Triangle::Kind uplo, Transposition::Kind trans, Diagonal::Kind diag, const T& alpha, const ConstMatrixView<T>& a, MatrixView<T> b)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::trsm");
}

template <typename T>
typename Kernels<T>::BlockSizes Kernels<T>::tunedBlockSizes()
{
    throw LinalgError("not implemented: linalg::Kernels<T>::tunedBlockSizes");
}

template <typename T>
void Kernels<T>::packPanelA(const ConstMatrixView<T>& a, T* buffer)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::packPanelA");
}

template <typename T>
void Kernels<T>::packPanelB(const ConstMatrixView<T>& b, T* buffer)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::packPanelB");
}

template <typename T>
void Kernels<T>::microKernel(Index kc, const T* packedA, const T* packedB, MatrixView<T> c)
{
    throw LinalgError("not implemented: linalg::Kernels<T>::microKernel");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Kernels<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
