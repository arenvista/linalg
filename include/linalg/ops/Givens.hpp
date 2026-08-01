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

    /// @brief Constructs an uninitialized rotation.
    Givens();
    /// @brief Constructs a rotation from its coefficients and plane indices.
    /// @param cosine The cosine coefficient c.
    /// @param sine The sine coefficient s.
    /// @param p First plane index.
    /// @param q Second plane index.
    Givens(const T& cosine, const T& sine, Index p, Index q);

    // Choose (c, s) so that the rotation maps (a, b) to (r, 0). Must be
    // computed without squaring a or b (dlartg-style scaling), so it stays
    // exact near the overflow and underflow thresholds.
    /// @brief Builds the rotation mapping the pair (a, b) to (r, 0).
    /// @param a First component to rotate.
    /// @param b Second component (the one zeroed).
    /// @param p First plane index.
    /// @param q Second plane index.
    /// @return The constructed rotation.
    static Givens FromPair(const T& a, const T& b, Index p, Index q);
    /// @brief Builds the identity rotation (c = 1, s = 0).
    /// @param p First plane index.
    /// @param q Second plane index.
    /// @return The identity rotation on the (p, q) plane.
    static Givens Identity(Index p, Index q);

    /// @brief The cosine coefficient c.
    /// @return Reference to c.
    const T& cosine() const;
    /// @brief The sine coefficient s.
    /// @return Reference to s.
    const T& sine() const;
    /// @brief The first (lower) plane index.
    /// @return Index p.
    Index    firstIndex() const;
    /// @brief The second (upper) plane index.
    /// @return Index q.
    Index    secondIndex() const;
    /// @brief The radius r produced by FromPair (what (a, b) rotates onto).
    /// @return The radius r.
    T        radius() const;

    /// @brief The transpose of this rotation.
    /// @return The transposed rotation.
    Givens transposed() const;
    /// @brief The inverse of this rotation (equals the transpose; unitary).
    /// @return The inverse rotation.
    Givens inverse() const;

    // O(cols) / O(rows): only the two indexed rows/columns are touched.
    /// @brief Applies the rotation on the left in place (rows p and q).
    /// @param target Matrix view overwritten with the rotated rows.
    void applyLeft(MatrixView<T> target) const;
    /// @brief Applies the rotation on the right in place (columns p and q).
    /// @param target Matrix view overwritten with the rotated columns.
    void applyRight(MatrixView<T> target) const;
    /// @brief Applies the rotation to a vector in place (entries p and q).
    /// @param x Vector overwritten with the rotated entries.
    void apply(Vector<T>& x) const;

    /// @brief Materializes the dense rotation embedded in an identity (tests only).
    /// @param dimension Size of the surrounding identity matrix.
    /// @return The dense rotation matrix.
    Matrix<T> toMatrix(Index dimension) const;

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

    /// @brief Constructs an empty sequence.
    GivensSequence();

    /// @brief Appends a rotation to the end of the sequence.
    /// @param rotation The rotation to add.
    void  append(const Givens<T>& rotation);
    /// @brief Removes all rotations from the sequence.
    void  clear();
    /// @brief Number of rotations in the sequence.
    /// @return The rotation count.
    Index count() const;

    /// @brief Read-only access to a rotation by position.
    /// @param k Position in the sequence (0-based).
    /// @return Reference to rotation k.
    const Givens<T>& operator[](Index k) const;

    // applyLeft applies rotations in append order:
    // target <- G_{k-1} ... G_1 G_0 * target; applyLeftReversed is the
    // opposite order (the inverse sequence uses reversed + transposed).
    /// @brief Applies all rotations on the left in append order.
    /// @param target Matrix view overwritten with the result.
    void applyLeft(MatrixView<T> target) const;
    /// @brief Applies all rotations on the right in append order.
    /// @param target Matrix view overwritten with the result.
    void applyRight(MatrixView<T> target) const;
    /// @brief Applies all rotations on the left in reverse append order.
    /// @param target Matrix view overwritten with the result.
    void applyLeftReversed(MatrixView<T> target) const;

    /// @brief Returns the sequence in reverse order.
    /// @return The reversed sequence.
    GivensSequence  reversed() const;
    /// @brief Materializes the accumulated rotation as a dense matrix.
    /// @param dimension Size of the identity the rotations act within.
    /// @return The dense product matrix.
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
