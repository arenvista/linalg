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

    Householder();
    Householder(const Vector<T>& essential, const T& beta);

    // Build the reflector mapping x to +/- ||x|| e_1. The sign is chosen
    // opposite to x_1 (for complex T, opposite its phase) so the leading
    // subtraction cannot cancel. FromColumn reads a(startRow.., col) in
    // place, saving the copy in factorization inner loops.
    static Householder FromVector(const Vector<T>& x);
    static Householder FromColumn(const ConstMatrixView<T>& a, Index col, Index startRow);

    const Vector<T>& essential() const;
    const T&         beta() const;
    Index            size() const;      // length of the full v, essential().size() + 1
    bool             isIdentity() const; // beta == 0: apply* are no-ops

    // Two-pass application: w = v^H * target, then target -= beta * v * w.
    // H is never formed.
    void applyLeft(MatrixView<T> target) const;          // target <- H * target
    void applyRight(MatrixView<T> target) const;         // target <- target * H
    void applyLeftConjugate(MatrixView<T> target) const; // target <- H^H * target
    void apply(Vector<T>& x) const;

    Matrix<T> toMatrix(Index dimension) const;  // dense H embedded bottom-right in I_dimension; tests only

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

    HouseholderSequence();
    HouseholderSequence(const Matrix<T>& reflectors, const Vector<T>& betas);

    void  append(const Householder<T>& reflector);
    Index count() const;

    void applyLeft(MatrixView<T> target) const;
    void applyRight(MatrixView<T> target) const;
    void applyLeftTranspose(MatrixView<T> target) const;

    Matrix<T> toMatrix(Index dimension) const;
    Matrix<T> firstColumns(Index dimension, Index count) const; // thin Q: leading `count` columns only

    // Compact WY representation: Q = I - V * Tmat * V^H, with V the unit
    // lower-trapezoidal reflector matrix and Tmat k x k upper triangular.
    // Must be rebuilt (blockValid_) after append().
    Matrix<T> blockV() const;
    Matrix<T> blockT() const;
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
