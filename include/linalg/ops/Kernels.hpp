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
    /// @brief Scales a vector in place: x <- alpha * x.
    /// @param n Number of elements.
    /// @param alpha Scalar multiplier.
    /// @param x Vector data (modified in place).
    /// @param incx Stride between consecutive elements of x.
    static void   scal(Index n, const T& alpha, T* x, Index incx);
    /// @brief Fused scale-and-add: y <- alpha * x + y.
    /// @param n Number of elements.
    /// @param alpha Scalar multiplier applied to x.
    /// @param x Source vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param y Destination vector (modified in place).
    /// @param incy Stride between consecutive elements of y.
    static void   axpy(Index n, const T& alpha, const T* x, Index incx, T* y, Index incy);
    /// @brief Unconjugated dot product, sum x_i * y_i.
    /// @param n Number of elements.
    /// @param x First vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param y Second vector.
    /// @param incy Stride between consecutive elements of y.
    /// @return The dot product.
    static T      dot(Index n, const T* x, Index incx, const T* y, Index incy);
    /// @brief Conjugated dot product, sum conj(x_i) * y_i.
    /// @param n Number of elements.
    /// @param x First vector (conjugated).
    /// @param incx Stride between consecutive elements of x.
    /// @param y Second vector.
    /// @param incy Stride between consecutive elements of y.
    /// @return The conjugated dot product.
    static T      dotc(Index n, const T* x, Index incx, const T* y, Index incy);
    /// @brief Euclidean norm ||x||_2 with overflow-safe scaling.
    /// @param n Number of elements.
    /// @param x The vector.
    /// @param incx Stride between consecutive elements of x.
    /// @return The 2-norm of x.
    static Real   nrm2(Index n, const T* x, Index incx);
    /// @brief Sum of element magnitudes, sum |x_i|.
    /// @param n Number of elements.
    /// @param x The vector.
    /// @param incx Stride between consecutive elements of x.
    /// @return The 1-norm of x.
    static Real   asum(Index n, const T* x, Index incx);
    /// @brief Index of the first element with the largest magnitude.
    /// @param n Number of elements.
    /// @param x The vector.
    /// @param incx Stride between consecutive elements of x.
    /// @return The index of the maximum-magnitude element.
    static Index  iamax(Index n, const T* x, Index incx);
    /// @brief Swaps the contents of two vectors.
    /// @param n Number of elements.
    /// @param x First vector (modified in place).
    /// @param incx Stride between consecutive elements of x.
    /// @param y Second vector (modified in place).
    /// @param incy Stride between consecutive elements of y.
    static void   swap(Index n, T* x, Index incx, T* y, Index incy);
    /// @brief Copies one vector into another: y <- x.
    /// @param n Number of elements.
    /// @param x Source vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param y Destination vector.
    /// @param incy Stride between consecutive elements of y.
    static void   copy(Index n, const T* x, Index incx, T* y, Index incy);

    // Level 2
    /// @brief General matrix-vector product: y <- alpha * op(a) * x + beta * y.
    /// @param trans Whether to use a, a^T, or a^H as op(a).
    /// @param alpha Scalar multiplier for the product.
    /// @param a Matrix operand.
    /// @param x Input vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param beta Scalar multiplier for y (beta == 0 means y is write-only).
    /// @param y Input/output vector (modified in place).
    /// @param incy Stride between consecutive elements of y.
    static void gemv(Transposition::Kind trans, const T& alpha,
                     const ConstMatrixView<T>& a, const T* x, Index incx,
                     const T& beta, T* y, Index incy);
    /// @brief Unconjugated rank-one update: a <- alpha * x * y^T + a.
    /// @param alpha Scalar multiplier for the outer product.
    /// @param x Left vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param y Right vector.
    /// @param incy Stride between consecutive elements of y.
    /// @param a Matrix updated in place.
    static void ger(const T& alpha, const T* x, Index incx,
                    const T* y, Index incy, MatrixView<T> a);
    /// @brief Triangular solve: x <- op(a)^-1 * x.
    /// @param uplo Which triangle of a holds the data.
    /// @param trans Whether to use a, a^T, or a^H as op(a).
    /// @param diag Whether a's diagonal is unit (implied ones) or explicit.
    /// @param a Triangular matrix operand.
    /// @param x Right-hand side / solution vector (modified in place).
    /// @param incx Stride between consecutive elements of x.
    static void trsv(Triangle::Kind uplo, Transposition::Kind trans,
                     Diagonal::Kind diag, const ConstMatrixView<T>& a,
                     T* x, Index incx);
    /// @brief Symmetric matrix-vector product: y <- alpha * a * x + beta * y.
    /// @param uplo Which triangle of a holds the symmetric data.
    /// @param alpha Scalar multiplier for the product.
    /// @param a Symmetric matrix operand.
    /// @param x Input vector.
    /// @param incx Stride between consecutive elements of x.
    /// @param beta Scalar multiplier for y (beta == 0 means y is write-only).
    /// @param y Input/output vector (modified in place).
    /// @param incy Stride between consecutive elements of y.
    static void symv(Triangle::Kind uplo, const T& alpha,
                     const ConstMatrixView<T>& a, const T* x, Index incx,
                     const T& beta, T* y, Index incy);

    // Level 3
    /// @brief General matrix-matrix product: c <- alpha * op(a) * op(b) + beta * c.
    /// @param transA Whether to use a, a^T, or a^H as op(a).
    /// @param transB Whether to use b, b^T, or b^H as op(b).
    /// @param alpha Scalar multiplier for the product.
    /// @param a Left matrix operand.
    /// @param b Right matrix operand.
    /// @param beta Scalar multiplier for c (beta == 0 means c is write-only).
    /// @param c Output matrix (must not alias a or b).
    static void gemm(Transposition::Kind transA, Transposition::Kind transB,
                     const T& alpha, const ConstMatrixView<T>& a,
                     const ConstMatrixView<T>& b, const T& beta,
                     MatrixView<T> c);
    /// @brief Symmetric rank-k update: c <- alpha * op(a) * op(a)^H + beta * c.
    /// @param uplo Which triangle of c is referenced and written.
    /// @param trans Whether to use a or a^H as op(a).
    /// @param alpha Scalar multiplier for the product.
    /// @param a Matrix operand.
    /// @param beta Scalar multiplier for c (beta == 0 means c is write-only).
    /// @param c Output symmetric matrix (one triangle).
    static void syrk(Triangle::Kind uplo, Transposition::Kind trans,
                     const T& alpha, const ConstMatrixView<T>& a,
                     const T& beta, MatrixView<T> c);
    /// @brief Triangular solve with multiple right-hand sides:
    /// b <- alpha * op(a)^-1 * b.
    /// @param uplo Which triangle of a holds the data.
    /// @param trans Whether to use a, a^T, or a^H as op(a).
    /// @param diag Whether a's diagonal is unit (implied ones) or explicit.
    /// @param alpha Scalar multiplier applied to b.
    /// @param a Triangular matrix operand.
    /// @param b Right-hand sides / solutions (modified in place).
    static void trsm(Triangle::Kind uplo, Transposition::Kind trans,
                     Diagonal::Kind diag, const T& alpha,
                     const ConstMatrixView<T>& a, MatrixView<T> b);

    // Internals of the blocked gemm, exposed for benchmarking: pack copies
    // a panel into a contiguous, kernel-friendly layout; microKernel does
    // the mr x nr rank-kc update from two packed panels.
    /// @brief The cache-blocking parameters tuned for the current scalar type.
    /// @return The chosen block sizes.
    static BlockSizes tunedBlockSizes();
    /// @brief Packs a panel of A into a contiguous, kernel-friendly layout.
    /// @param a The panel of A to pack.
    /// @param buffer Destination buffer for the packed panel.
    static void       packPanelA(const ConstMatrixView<T>& a, T* buffer);
    /// @brief Packs a panel of B into a contiguous, kernel-friendly layout.
    /// @param b The panel of B to pack.
    /// @param buffer Destination buffer for the packed panel.
    static void       packPanelB(const ConstMatrixView<T>& b, T* buffer);
    /// @brief Computes the mr x nr rank-kc register-tile update.
    /// @param kc Depth of the packed panels.
    /// @param packedA Packed panel of A.
    /// @param packedB Packed panel of B.
    /// @param c Output tile updated in place.
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
