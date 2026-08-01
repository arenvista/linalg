#pragma once

#include <complex>
#include <cstddef>
#include "linalg/Instantiate.hpp"

namespace linalg {

// Scalar-type abstraction: everything downstream asks these questions
// instead of assuming `double`.
template <typename T>
struct NumericTraits {
    using Scalar = T;
    using Real   = T;            // magnitude type; equals Scalar only for real T
    using Index  = std::size_t;

    /// @brief Machine epsilon: the gap between 1 and the next representable Real.
    /// @return The machine epsilon for Real.
    static Real   epsilon();
    /// @brief Smallest positive value s such that 1/s is finite; the scaling
    /// threshold used to avoid overflow.
    /// @return The safe minimum for Real.
    static Real   safeMin();
    /// @brief Magnitude of a scalar.
    /// @param x The scalar.
    /// @return |x| as a Real.
    static Real   abs(const Scalar& x);
    /// @brief Squared magnitude of a scalar, computed without a square root.
    /// @param x The scalar.
    /// @return |x|^2 as a Real.
    static Real   absSquared(const Scalar& x);
    /// @brief Complex conjugate; the identity for real T.
    /// @param x The scalar.
    /// @return The conjugate of x.
    static Scalar conj(const Scalar& x);
    /// @brief Real part of a scalar.
    /// @param x The scalar.
    /// @return The real part.
    static Real   real(const Scalar& x);
    /// @brief Imaginary part of a scalar; zero for real T.
    /// @param x The scalar.
    /// @return The imaginary part.
    static Real   imag(const Scalar& x);
    /// @brief Principal square root of a scalar.
    /// @param x The scalar.
    /// @return The principal square root.
    static Scalar sqrt(const Scalar& x);
    /// @brief The additive identity for the scalar type.
    /// @return Zero.
    static Scalar zero();
    /// @brief The multiplicative identity for the scalar type.
    /// @return One.
    static Scalar one();
    /// @brief Tests whether a scalar is negligible against a tolerance.
    /// @param x The scalar.
    /// @param tol Absolute magnitude tolerance.
    /// @return True if |x| <= tol.
    static bool   isApproxZero(const Scalar& x, Real tol);
};

// Complex specialization: Real is the component type; abs/conj/sqrt take
// their usual complex meanings. Same contract as the primary template.
template <typename T>
struct NumericTraits<std::complex<T>> {
    using Scalar = std::complex<T>;
    using Real   = T;
    using Index  = std::size_t;

    /// @brief Machine epsilon of the component type.
    /// @return The machine epsilon for Real.
    static Real   epsilon();
    /// @brief Smallest positive value s such that 1/s is finite (component type).
    /// @return The safe minimum for Real.
    static Real   safeMin();
    /// @brief Complex modulus.
    /// @param x The complex scalar.
    /// @return |x| as a Real.
    static Real   abs(const Scalar& x);
    /// @brief Squared complex modulus, computed without a square root.
    /// @param x The complex scalar.
    /// @return |x|^2 as a Real.
    static Real   absSquared(const Scalar& x);
    /// @brief Complex conjugate.
    /// @param x The complex scalar.
    /// @return The conjugate of x.
    static Scalar conj(const Scalar& x);
    /// @brief Real part.
    /// @param x The complex scalar.
    /// @return The real component.
    static Real   real(const Scalar& x);
    /// @brief Imaginary part.
    /// @param x The complex scalar.
    /// @return The imaginary component.
    static Real   imag(const Scalar& x);
    /// @brief Principal complex square root.
    /// @param x The complex scalar.
    /// @return The principal square root.
    static Scalar sqrt(const Scalar& x);
    /// @brief The additive identity (0 + 0i).
    /// @return Zero.
    static Scalar zero();
    /// @brief The multiplicative identity (1 + 0i).
    /// @return One.
    static Scalar one();
    /// @brief Tests whether the modulus is negligible against a tolerance.
    /// @param x The complex scalar.
    /// @param tol Absolute magnitude tolerance.
    /// @return True if |x| <= tol.
    static bool   isApproxZero(const Scalar& x, Real tol);
};

/// @brief Trait reporting whether T is a std::complex type; false by default.
template <typename T>
struct IsComplex {
    static constexpr bool value = false;
};

/// @brief IsComplex specialization for std::complex, reporting true.
template <typename T>
struct IsComplex<std::complex<T>> {
    static constexpr bool value = true;
};

// Tag carriers. Enums are nested so that no bare enumeration leaks
// into namespace scope.
/// @brief Memory layout of a dense matrix (row-major or column-major).
struct StorageOrder {
    enum class Kind { RowMajor, ColumnMajor };
};

// Which triangle of a matrix an algorithm reads or writes; the opposite
// triangle is never touched.
/// @brief Selects the upper or lower triangle of a matrix.
struct Triangle {
    enum class Kind { Upper, Lower };
};

// Selects op(A) in kernel and solver signatures: A, A^T, or A^H.
// Transpose and ConjugateTranspose coincide for real scalars.
/// @brief Selects op(A): the matrix itself, its transpose, or its conjugate transpose.
struct Transposition {
    enum class Kind { None, Transpose, ConjugateTranspose };
};

// Unit means the diagonal is assumed to be all ones and is never read,
// as with the L factor of an LU decomposition stored packed.
/// @brief Selects whether a triangular matrix's diagonal is unit (implied ones) or explicit.
struct Diagonal {
    enum class Kind { Unit, NonUnit };
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template struct NumericTraits<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
