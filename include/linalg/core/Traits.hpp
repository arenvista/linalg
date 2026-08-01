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

    static Real   epsilon();     // machine epsilon: gap between 1 and the next Real
    static Real   safeMin();     // smallest positive s with 1/s finite; scaling threshold
    static Real   abs(const Scalar& x);        // |x|, always a Real
    static Real   absSquared(const Scalar& x); // |x|^2, computed without a sqrt
    static Scalar conj(const Scalar& x);       // identity for real T
    static Real   real(const Scalar& x);
    static Real   imag(const Scalar& x);       // zero for real T
    static Scalar sqrt(const Scalar& x);       // principal square root
    static Scalar zero();
    static Scalar one();
    static bool   isApproxZero(const Scalar& x, Real tol);  // |x| <= tol
};

// Complex specialization: Real is the component type; abs/conj/sqrt take
// their usual complex meanings. Same contract as the primary template.
template <typename T>
struct NumericTraits<std::complex<T>> {
    using Scalar = std::complex<T>;
    using Real   = T;
    using Index  = std::size_t;

    static Real   epsilon();
    static Real   safeMin();
    static Real   abs(const Scalar& x);
    static Real   absSquared(const Scalar& x);
    static Scalar conj(const Scalar& x);
    static Real   real(const Scalar& x);
    static Real   imag(const Scalar& x);
    static Scalar sqrt(const Scalar& x);
    static Scalar zero();
    static Scalar one();
    static bool   isApproxZero(const Scalar& x, Real tol);
};

template <typename T>
struct IsComplex {
    static constexpr bool value = false;
};

template <typename T>
struct IsComplex<std::complex<T>> {
    static constexpr bool value = true;
};

// Tag carriers. Enums are nested so that no bare enumeration leaks
// into namespace scope.
// Memory layout of a dense matrix.
struct StorageOrder {
    enum class Kind { RowMajor, ColumnMajor };
};

// Which triangle of a matrix an algorithm reads or writes; the opposite
// triangle is never touched.
struct Triangle {
    enum class Kind { Upper, Lower };
};

// Selects op(A) in kernel and solver signatures: A, A^T, or A^H.
// Transpose and ConjugateTranspose coincide for real scalars.
struct Transposition {
    enum class Kind { None, Transpose, ConjugateTranspose };
};

// Unit means the diagonal is assumed to be all ones and is never read,
// as with the L factor of an LU decomposition stored packed.
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
