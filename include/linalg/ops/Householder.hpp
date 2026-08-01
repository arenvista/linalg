#pragma once

#include <cstddef>

#include "linalg/core/Matrix.hpp"
#include "linalg/core/MatrixView.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// A single reflector H = I - beta * v * v^H, stored implicitly.
// Applying it is O(mn), never O(n^3): the dense form is only for tests.
//
// Storage convention: the full vector is v = [1; essential] — the leading
// 1 is implicit, only the tail is kept. beta is real-valued even for
// complex T (the reflector is Hermitian and unitary either way).
template <typename T>
class Householder {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    /// @brief Constructs an identity reflector (beta == 0).
    Householder();
    /// @brief Constructs a reflector from its essential part and beta.
    /// @param essential The tail of v (the leading 1 is implicit).
    /// @param beta The reflector coefficient.
    Householder(const Vector<T>& essential, const T& beta);

    // Build the reflector mapping x to +/- ||x|| e_1. The sign is chosen
    // opposite to x_1 (for complex T, opposite its phase) so the leading
    // subtraction cannot cancel. FromColumn reads a(startRow.., col) in
    // place, saving the copy in factorization inner loops.
    /// @brief Builds the reflector mapping x to +/- ||x|| e_1.
    /// @param x The vector to reflect onto the first axis.
    /// @return The constructed reflector.
    static Householder FromVector(const Vector<T>& x);
    /// @brief Builds a reflector from a matrix column, read in place.
    /// @param a Matrix view providing the column.
    /// @param col Column index to read.
    /// @param startRow First row of the column to include.
    /// @return The constructed reflector.
    static Householder FromColumn(const ConstMatrixView<T>& a, Index col, Index startRow);

    /// @brief The essential (stored tail) part of v.
    /// @return Reference to the essential vector.
    const Vector<T>& essential() const;
    /// @brief The reflector coefficient beta.
    /// @return Reference to beta.
    const T&         beta() const;
    /// @brief Length of the full reflector vector v (essential().size() + 1).
    /// @return The reflector dimension.
    Index            size() const;
    /// @brief Reports whether this is the identity reflector (beta == 0).
    /// @return True if applying the reflector is a no-op.
    bool             isIdentity() const;

    // Two-pass application: w = v^H * target, then target -= beta * v * w.
    // H is never formed.
    /// @brief Applies H on the left in place: target <- H * target.
    /// @param target Matrix view overwritten with the product.
    void applyLeft(MatrixView<T> target) const;
    /// @brief Applies H on the right in place: target <- target * H.
    /// @param target Matrix view overwritten with the product.
    void applyRight(MatrixView<T> target) const;
    /// @brief Applies H^H on the left in place: target <- H^H * target.
    /// @param target Matrix view overwritten with the product.
    void applyLeftConjugate(MatrixView<T> target) const;
    /// @brief Applies the reflector to a vector in place.
    /// @param x Vector overwritten with H * x.
    void apply(Vector<T>& x) const;

    /// @brief Materializes the dense reflector, embedded bottom-right in an
    /// identity (tests only).
    /// @param dimension Size of the surrounding identity matrix.
    /// @return The dense reflector matrix.
    Matrix<T> toMatrix(Index dimension) const;

private:
    Vector<T> essential_;
    T         beta_;
};

// Aggregated reflectors in WY form, so a sequence can be applied with
// level-3 kernels instead of a loop of rank-1 updates.
//
// Order convention: with reflectors H_0 .. H_{k-1} in append order, the
// sequence represents Q = H_0 H_1 ... H_{k-1} (the Q of Householder QR).
// applyLeft forms Q * target; applyLeftTranspose forms Q^H * target.
template <typename T>
class HouseholderSequence {
public:
    using Scalar = T;
    using Index  = std::size_t;

    /// @brief Constructs an empty sequence.
    HouseholderSequence();
    /// @brief Constructs a sequence from packed reflectors and their betas.
    /// @param reflectors Matrix whose columns hold the reflector vectors.
    /// @param betas The per-reflector coefficients.
    HouseholderSequence(const Matrix<T>& reflectors, const Vector<T>& betas);

    /// @brief Appends a reflector to the end of the sequence.
    /// @param reflector The reflector to add.
    void  append(const Householder<T>& reflector);
    /// @brief Number of reflectors in the sequence.
    /// @return The reflector count.
    Index count() const;

    /// @brief Applies Q on the left in place: target <- Q * target.
    /// @param target Matrix view overwritten with the product.
    void applyLeft(MatrixView<T> target) const;
    /// @brief Applies Q on the right in place: target <- target * Q.
    /// @param target Matrix view overwritten with the product.
    void applyRight(MatrixView<T> target) const;
    /// @brief Applies Q^H on the left in place: target <- Q^H * target.
    /// @param target Matrix view overwritten with the product.
    void applyLeftTranspose(MatrixView<T> target) const;

    /// @brief Materializes the full orthogonal factor Q.
    /// @param dimension Size of the identity Q acts within.
    /// @return The dense Q matrix.
    Matrix<T> toMatrix(Index dimension) const;
    /// @brief Materializes the leading columns of Q (a thin Q).
    /// @param dimension Size of the identity Q acts within.
    /// @param count Number of leading columns to form.
    /// @return The leading `count` columns of Q.
    Matrix<T> firstColumns(Index dimension, Index count) const;

    // Compact WY representation: Q = I - V * Tmat * V^H, with V the unit
    // lower-trapezoidal reflector matrix and Tmat k x k upper triangular.
    // Must be rebuilt (blockValid_) after append().
    /// @brief The V factor of the compact WY representation.
    /// @return The unit lower-trapezoidal reflector matrix V.
    Matrix<T> blockV() const;
    /// @brief The T factor of the compact WY representation.
    /// @return The upper-triangular matrix T.
    Matrix<T> blockT() const;
    /// @brief Builds the compact WY block representation.
    /// @param blockSize Panel width for the block representation.
    void      buildBlockRepresentation(Index blockSize);

private:
    Matrix<T> reflectors_;
    Vector<T> betas_;
    Matrix<T> blockV_;
    Matrix<T> blockT_;
    bool      blockValid_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Householder<SCALAR>; \
    extern template class HouseholderSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
