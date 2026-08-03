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
    for (Index i = 0; i < n; ++i) {
        std::swap(x[i * incx], y[i * incy]);
    }
}

template <typename T>
void Kernels<T>::copy(Index    n,
                      const T *x,
                      Index    incx,
                      T       *y,
                      Index    incy) {
    for (Index i = 0; i < n; ++i) {
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
    const bool  transposed = (trans != Transposition::Kind::None);
    const bool  conjugate  = (trans == Transposition::Kind::ConjugateTranspose);
    const bool  unit       = (diag == Diagonal::Kind::Unit);
    const Index n          = a.rows();

    // Element of op(a) at logical (i, j): a itself, or a^T / a^H.
    const auto opA = [&](Index i, Index j) -> T {
        if (!transposed) {
            return a(i, j);
        }
        const T v = a(j, i);
        return conjugate ? NumericTraits<T>::conj(v) : v;
    };

    // Transposing flips which triangle op(a) occupies, so the substitution
    // direction follows the *effective* triangle, not the stored one.
    bool opLower = (uplo == Triangle::Kind::Lower);
    if (transposed) {
        opLower = !opLower;
    }

    if (opLower) { // forward substitution
        for (Index i = 0; i < n; ++i) {
            T s = x[i * incx];
            for (Index j = 0; j < i; ++j) {
                s -= opA(i, j) * x[j * incx];
            }
            if (!unit) {
                s /= opA(i, i);
            }
            x[i * incx] = s;
        }
    } else { // back substitution
        for (Index i = n; i-- > 0;) {
            T s = x[i * incx];
            for (Index j = i + 1; j < n; ++j) {
                s -= opA(i, j) * x[j * incx];
            }
            if (!unit) {
                s /= opA(i, i);
            }
            x[i * incx] = s;
        }
    }
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
    const Index n = a.rows();

    // Symmetric (not Hermitian): the untouched triangle mirrors the stored
    // one with no conjugation, so a(i,j) == a(j,i).
    const auto sym = [&](Index i, Index j) -> T {
        if (uplo == Triangle::Kind::Upper) {
            return (j >= i) ? a(i, j) : a(j, i);
        }
        return (i >= j) ? a(i, j) : a(j, i);
    };

    for (Index i = 0; i < n; ++i) {
        T acc = T{};
        for (Index j = 0; j < n; ++j) {
            acc += sym(i, j) * x[j * incx];
        }
        // beta == 0 means y is write-only: do not read it.
        if (beta == T{}) {
            y[i * incy] = alpha * acc;
        } else {
            y[i * incy] = alpha * acc + beta * y[i * incy];
        }
    }
}

template <typename T>
void Kernels<T>::gemm(Transposition::Kind       transA,
                      Transposition::Kind       transB,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const ConstMatrixView<T> &b,
                      const T                  &beta,
                      MatrixView<T>             c) {
    // op(m) element at logical (i, j): m, m^T, or m^H.
    const auto op = [](const ConstMatrixView<T> &m, Transposition::Kind t,
                       Index i, Index j) -> T {
        if (t == Transposition::Kind::None) {
            return m(i, j);
        }
        const T v = m(j, i);
        return (t == Transposition::Kind::ConjugateTranspose)
                   ? NumericTraits<T>::conj(v)
                   : v;
    };

    const Index m = c.rows();
    const Index n = c.cols();
    // Shared inner dimension: cols of op(a) == rows of op(b).
    const Index k = (transA == Transposition::Kind::None) ? a.cols() : a.rows();

    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < n; ++j) {
            T acc = T{};
            for (Index l = 0; l < k; ++l) {
                acc += op(a, transA, i, l) * op(b, transB, l, j);
            }
            // beta == 0 means c is write-only: do not read it.
            if (beta == T{}) {
                c(i, j) = alpha * acc;
            } else {
                c(i, j) = alpha * acc + beta * c(i, j);
            }
        }
    }
}

template <typename T>
void Kernels<T>::syrk(Triangle::Kind            uplo,
                      Transposition::Kind       trans,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      const T                  &beta,
                      MatrixView<T>             c) {
    // op(a) is a (None) or a^H (ConjugateTranspose); result c = alpha *
    // op(a) * op(a)^H + beta * c, writing only the `uplo` triangle.
    const bool  conjugate = (trans != Transposition::Kind::None);
    const Index n         = conjugate ? a.cols() : a.rows();
    const Index k         = conjugate ? a.rows() : a.cols();

    const auto opA = [&](Index i, Index l) -> T { // op(a)(i, l)
        return conjugate ? NumericTraits<T>::conj(a(l, i)) : a(i, l);
    };

    for (Index i = 0; i < n; ++i) {
        const Index jlo = (uplo == Triangle::Kind::Upper) ? i : 0;
        const Index jhi = (uplo == Triangle::Kind::Upper) ? n : i + 1;
        for (Index j = jlo; j < jhi; ++j) {
            T acc = T{};
            for (Index l = 0; l < k; ++l) {
                // (op(a) op(a)^H)(i,j) = sum_l op(a)(i,l) * conj(op(a)(j,l)).
                acc += opA(i, l) * NumericTraits<T>::conj(opA(j, l));
            }
            if (beta == T{}) {
                c(i, j) = alpha * acc;
            } else {
                c(i, j) = alpha * acc + beta * c(i, j);
            }
        }
    }
}

template <typename T>
void Kernels<T>::trsm(Triangle::Kind            uplo,
                      Transposition::Kind       trans,
                      Diagonal::Kind            diag,
                      const T                  &alpha,
                      const ConstMatrixView<T> &a,
                      MatrixView<T>             b) {
    // b <- alpha * op(a)^-1 * b: solve op(a) X = alpha * b for each column,
    // reusing the trsv substitution structure.
    const bool  transposed = (trans != Transposition::Kind::None);
    const bool  conjugate  = (trans == Transposition::Kind::ConjugateTranspose);
    const bool  unit       = (diag == Diagonal::Kind::Unit);
    const Index n          = a.rows();
    const Index nrhs       = b.cols();

    const auto opA = [&](Index i, Index j) -> T {
        if (!transposed) {
            return a(i, j);
        }
        const T v = a(j, i);
        return conjugate ? NumericTraits<T>::conj(v) : v;
    };

    bool opLower = (uplo == Triangle::Kind::Lower);
    if (transposed) {
        opLower = !opLower;
    }

    for (Index col = 0; col < nrhs; ++col) {
        if (opLower) { // forward substitution
            for (Index i = 0; i < n; ++i) {
                T s = alpha * b(i, col);
                for (Index j = 0; j < i; ++j) {
                    s -= opA(i, j) * b(j, col);
                }
                if (!unit) {
                    s /= opA(i, i);
                }
                b(i, col) = s;
            }
        } else { // back substitution
            for (Index i = n; i-- > 0;) {
                T s = alpha * b(i, col);
                for (Index j = i + 1; j < n; ++j) {
                    s -= opA(i, j) * b(j, col);
                }
                if (!unit) {
                    s /= opA(i, i);
                }
                b(i, col) = s;
            }
        }
    }
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
