#pragma once

#include <cstddef>

#include "linalg/core/MatrixView.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"

namespace linalg {

// BLAS-shaped kernels, grouped as static members so no free functions
// are introduced. Level 3 routines are the tuning surface: blocking,
// packing, and SIMD live behind these declarations.
//
// Conventions follow BLAS: raw-pointer vectors carry an explicit stride
// (incx/incy), matrix operands arrive as views (their strides play the
// role of the leading dimension), and beta == 0 means the output operand
// is write-only — it must not be read, so it may hold NaNs or garbage.
template <typename T>
class Kernels {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // Cache-blocking parameters for the packed gemm: an mc x kc panel of A
    // sized for L2, a kc x nc panel of B sized for L1/L3, and an mr x nr
    // register tile computed by the micro-kernel.
    struct BlockSizes {
        Index mc;
        Index nc;
        Index kc;
        Index mr;
        Index nr;
    };

    // Level 1
    static void   scal(Index n, const T& alpha, T* x, Index incx);  // x <- alpha x
    static void   axpy(Index n, const T& alpha, const T* x, Index incx, T* y, Index incy); // y <- alpha x + y
    static T      dot(Index n, const T* x, Index incx, const T* y, Index incy);  // sum x_i y_i, unconjugated
    static T      dotc(Index n, const T* x, Index incx, const T* y, Index incy); // sum conj(x_i) y_i
    static Real   nrm2(Index n, const T* x, Index incx);   // ||x||_2, overflow-safe scaling
    static Real   asum(Index n, const T* x, Index incx);   // sum |x_i|
    static Index  iamax(Index n, const T* x, Index incx);  // index of first largest |x_i|
    static void   swap(Index n, T* x, Index incx, T* y, Index incy);
    static void   copy(Index n, const T* x, Index incx, T* y, Index incy);

    // Level 2
    // y <- alpha op(a) x + beta y
    static void gemv(Transposition::Kind trans, const T& alpha,
                     const ConstMatrixView<T>& a, const T* x, Index incx,
                     const T& beta, T* y, Index incy);
    // a <- alpha x y^T + a (rank-one update, unconjugated)
    static void ger(const T& alpha, const T* x, Index incx,
                    const T* y, Index incy, MatrixView<T> a);
    // x <- op(a)^-1 x for triangular a; only the `uplo` triangle is read,
    // and the diagonal is assumed 1 when diag is Unit.
    static void trsv(Triangle::Kind uplo, Transposition::Kind trans,
                     Diagonal::Kind diag, const ConstMatrixView<T>& a,
                     T* x, Index incx);
    // y <- alpha a x + beta y for symmetric a stored in one triangle.
    static void symv(Triangle::Kind uplo, const T& alpha,
                     const ConstMatrixView<T>& a, const T* x, Index incx,
                     const T& beta, T* y, Index incy);

    // Level 3
    // c <- alpha op(a) op(b) + beta c; c must not alias a or b.
    static void gemm(Transposition::Kind transA, Transposition::Kind transB,
                     const T& alpha, const ConstMatrixView<T>& a,
                     const ConstMatrixView<T>& b, const T& beta,
                     MatrixView<T> c);
    // c <- alpha op(a) op(a)^H + beta c; only the `uplo` triangle of c
    // is referenced or written.
    static void syrk(Triangle::Kind uplo, Transposition::Kind trans,
                     const T& alpha, const ConstMatrixView<T>& a,
                     const T& beta, MatrixView<T> c);
    // b <- alpha op(a)^-1 b for triangular a (multiple right-hand sides).
    static void trsm(Triangle::Kind uplo, Transposition::Kind trans,
                     Diagonal::Kind diag, const T& alpha,
                     const ConstMatrixView<T>& a, MatrixView<T> b);

    // Internals of the blocked gemm, exposed for benchmarking: pack copies
    // a panel into a contiguous, kernel-friendly layout; microKernel does
    // the mr x nr rank-kc update from two packed panels.
    static BlockSizes tunedBlockSizes();
    static void       packPanelA(const ConstMatrixView<T>& a, T* buffer);
    static void       packPanelB(const ConstMatrixView<T>& b, T* buffer);
    static void       microKernel(Index kc, const T* packedA, const T* packedB,
                                  MatrixView<T> c);
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Kernels<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
