#include "linalg/ops/Kernels.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

#include <algorithm>

namespace linalg {

template <typename T>
void Kernels<T>::scal(Index    n,
                      const T &alpha,
                      T       *x,
                      Index    incx) {
    for (Index i = 0; i < n; ++i) {
        x[i * incx] *= alpha;
    }
}

template <typename T>
void Kernels<T>::axpy(Index    n,
                      const T &alpha,
                      const T *x,
                      Index    incx,
                      T       *y,
                      Index    incy) {
    for (Index i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

template <typename T>
T Kernels<T>::dot(Index    n,
                  const T *x,
                  Index    incx,
                  const T *y,
                  Index    incy) {
    T result = T{};
    for (Index i = 0; i < n; ++i) {
        result += y[i * incy] * x[i * incx];
    }
    return result;
}

template <typename T>
T Kernels<T>::dotc(Index    n,
                   const T *x,
                   Index    incx,
                   const T *y,
                   Index    incy) {
    T result = T{};
    for (Index i = 0; i < n; ++i) {
        result += NumericTraits<T>::conj(x[i * incx]) * y[i * incy];
    }
    return result;
}

template <typename T>
typename Kernels<T>::Real Kernels<T>::nrm2(Index    n,
                                           const T *x,
                                           Index    incx) {
    // Overflow-safe scaling (BLAS nrm2): factor out the running maximum
    // magnitude so |x_i|^2 is never formed directly.
    Real scale = Real{};
    Real ssq   = Real(1);
    for (Index i = 0; i < n; ++i) {
        const Real ax = NumericTraits<T>::abs(x[i * incx]);
        if (ax != Real{}) {
            if (scale < ax) {
                const Real r = scale / ax;
                ssq          = Real(1) + ssq * r * r;
                scale        = ax;
            } else {
                const Real r = ax / scale;
                ssq += r * r;
            }
        }
    }
    return scale * NumericTraits<Real>::sqrt(ssq);
}

template <typename T>
typename Kernels<T>::Real Kernels<T>::asum(Index    n,
                                           const T *x,
                                           Index    incx) {
    Real sum = Real{};
    for (Index i = 0; i < n; ++i) {
        sum += NumericTraits<T>::abs(x[i * incx]);
    }
    return sum;
}

template <typename T>
typename Kernels<T>::Index Kernels<T>::iamax(Index    n,
                                             const T *x,
                                             Index    incx) {
    if (n == 0) {
        return 0; // BLAS convention for n < 1; also avoids reading x[0]
    }
    Index bestI = 0;
    Real  best  = NumericTraits<T>::abs(x[0]);
    for (Index i = 1; i < n; ++i) {
        const Real m = NumericTraits<T>::abs(x[i * incx]);
        if (m > best) { // strict > keeps the FIRST max on ties
            best  = m;
            bestI = i;
        }
    }
    return bestI;
}

template <typename T>
void Kernels<T>::swap(Index n,
                      T    *x,
                      Index incx,
                      T    *y,
                      Index incy) {
    for (int i = 0; i < n; ++i) {
        std::swap(x[i * incx], y[i * incy]);
    }
}

template <typename T>
void Kernels<T>::copy(Index    n,
                      const T *x,
                      Index    incx,
                      T       *y,
                      Index    incy) {
    for (int i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
}

template <typename T>
void Kernels<T>::gemv(Transposition::Kind       trans,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const T                  *x,
                      Index                     incx,
                      const T                  &beta,
                      T                        *y,
                      Index                     incy) {
    const bool transposed = (trans != Transposition::Kind::None);
    const bool conjugate  = (trans == Transposition::Kind::ConjugateTranspose);

    // op(a) is outLen x inLen; y has length outLen, x has length inLen.
    const Index outLen = transposed ? a.cols() : a.rows();
    const Index inLen  = transposed ? a.rows() : a.cols();

    for (Index p = 0; p < outLen; ++p) {
        T acc = T{};
        for (Index q = 0; q < inLen; ++q) {
            T aval = transposed ? a(q, p) : a(p, q);
            if (conjugate) {
                aval = NumericTraits<T>::conj(aval);
            }
            acc += aval * x[q * incx];
        }
        // beta == 0 means y is write-only: do not read it (may be
        // uninitialized).
        if (beta == T{}) {
            y[p * incy] = alpha * acc;
        } else {
            y[p * incy] = alpha * acc + beta * y[p * incy];
        }
    }
}

template <typename T>
void Kernels<T>::ger(const T      &alpha,
                     const T      *x,
                     Index         incx,
                     const T      *y,
                     Index         incy,
                     MatrixView<T> a) {
    // Unconjugated rank-one update: a(i,j) += alpha * x[i] * y[j].
    const Index rows = a.rows();
    const Index cols = a.cols();
    for (Index i = 0; i < rows; ++i) {
        const T ax = alpha * x[i * incx]; // hoist the row-constant factor
        for (Index j = 0; j < cols; ++j) {
            a(i, j) += ax * y[j * incy];
        }
    }
}

template <typename T>
void Kernels<T>::trsv(Triangle::Kind            uplo,
                      Transposition::Kind       trans,
                      Diagonal::Kind            diag,
                      const ConstMatrixView<T> &a,
                      T                        *x,
                      Index                     incx) {
    throw LinalgError("not implemented: linalg::Kernels<T>::trsv");
}

template <typename T>
void Kernels<T>::symv(Triangle::Kind            uplo,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const T                  *x,
                      Index                     incx,
                      const T                  &beta,
                      T                        *y,
                      Index                     incy) {
    throw LinalgError("not implemented: linalg::Kernels<T>::symv");
}

template <typename T>
void Kernels<T>::gemm(Transposition::Kind       transA,
                      Transposition::Kind       transB,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const ConstMatrixView<T> &b,
                      const T                  &beta,
                      MatrixView<T>             c) {
    throw LinalgError("not implemented: linalg::Kernels<T>::gemm");
}

template <typename T>
void Kernels<T>::syrk(Triangle::Kind            uplo,
                      Transposition::Kind       trans,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const T                  &beta,
                      MatrixView<T>             c) {
    throw LinalgError("not implemented: linalg::Kernels<T>::syrk");
}

template <typename T>
void Kernels<T>::trsm(Triangle::Kind            uplo,
                      Transposition::Kind       trans,
                      Diagonal::Kind            diag,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      MatrixView<T>             b) {
    throw LinalgError("not implemented: linalg::Kernels<T>::trsm");
}

template <typename T>
typename Kernels<T>::BlockSizes Kernels<T>::tunedBlockSizes() {
    throw LinalgError("not implemented: linalg::Kernels<T>::tunedBlockSizes");
}

template <typename T>
void Kernels<T>::packPanelA(const ConstMatrixView<T> &a,
                            T                        *buffer) {
    throw LinalgError("not implemented: linalg::Kernels<T>::packPanelA");
}

template <typename T>
void Kernels<T>::packPanelB(const ConstMatrixView<T> &b,
                            T                        *buffer) {
    throw LinalgError("not implemented: linalg::Kernels<T>::packPanelB");
}

template <typename T>
void Kernels<T>::microKernel(Index         kc,
                             const T      *packedA,
                             const T      *packedB,
                             MatrixView<T> c) {
    throw LinalgError("not implemented: linalg::Kernels<T>::microKernel");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Kernels<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
