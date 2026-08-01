#pragma once

#include <cstddef>
#include <vector>

#include "linalg/core/Matrix.hpp"
#include "linalg/core/MatrixView.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// Plane rotation zeroing a single entry. Preferred over Householder
// where the pattern is sparse: bidiagonal sweeps, QR updating, Hessenberg
// reduction inside GMRES.
//
// Convention (LAPACK lartg): acting on rows p and q,
//   row_p <-        c  * row_p + s * row_q
//   row_q <- -conj(s) * row_p + c * row_q
// with c always real, so the rotation is unitary for complex T too.
template <typename T>
class Givens {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    Givens();
    Givens(const T& cosine, const T& sine, Index p, Index q);

    // Choose (c, s) so that the rotation maps (a, b) to (r, 0). Must be
    // computed without squaring a or b (dlartg-style scaling), so it stays
    // exact near the overflow and underflow thresholds.
    static Givens FromPair(const T& a, const T& b, Index p, Index q);
    static Givens Identity(Index p, Index q);  // c = 1, s = 0

    const T& cosine() const;
    const T& sine() const;
    Index    firstIndex() const;
    Index    secondIndex() const;
    T        radius() const;   // the r produced by FromPair: what (a, b) rotates onto

    Givens transposed() const;
    Givens inverse() const;    // same as transposed(): rotations are unitary

    // O(cols) / O(rows): only the two indexed rows/columns are touched.
    void applyLeft(MatrixView<T> target) const;   // rows p, q
    void applyRight(MatrixView<T> target) const;  // cols p, q
    void apply(Vector<T>& x) const;

    Matrix<T> toMatrix(Index dimension) const;    // dense embedding; tests only

private:
    T     cosine_;
    T     sine_;
    T     radius_;
    Index p_;
    Index q_;
};

// Ordered product of rotations, e.g. the accumulated sweep of a
// Jacobi eigenvalue pass or a bidiagonal chase.
template <typename T>
class GivensSequence {
public:
    using Index = std::size_t;

    GivensSequence();

    void  append(const Givens<T>& rotation);
    void  clear();
    Index count() const;

    const Givens<T>& operator[](Index k) const;

    // applyLeft applies rotations in append order:
    // target <- G_{k-1} ... G_1 G_0 * target; applyLeftReversed is the
    // opposite order (the inverse sequence uses reversed + transposed).
    void applyLeft(MatrixView<T> target) const;
    void applyRight(MatrixView<T> target) const;
    void applyLeftReversed(MatrixView<T> target) const;

    GivensSequence  reversed() const;
    Matrix<T>       toMatrix(Index dimension) const;

private:
    std::vector<Givens<T>> rotations_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Givens<SCALAR>; \
    extern template class GivensSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
