#include "linalg/ops/Givens.hpp"

#include <cmath>

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T>
Givens<T>::Givens()
    : cosine_(T{}),
      sine_(T{}),
      radius_(T{}),
      p_(Index{}),
      q_(Index{}) {}

template <typename T>
Givens<T>::Givens(const T &cosine,
                  const T &sine,
                  Index    p,
                  Index    q)
    : cosine_(cosine),
      sine_(sine),
      radius_(NumericTraits<T>::zero()),
      p_(p),
      q_(q) {}

template <typename T>
Givens<T> Givens<T>::FromPair(const T &a,
                              const T &b,
                              Index    p,
                              Index    q) {
    // Referenced:
    // https://www.netlib.org/lapack/explore-html/da/dd3/group__lartg_ga86f8f877eaea0386cdc2c3c175d9ea88.html
    //
    // Convention (header): [c, s; -conj(s), c] * (a, b) = (r, 0) with c real.
    // Solving gives, for d = sqrt(|a|^2 + |b|^2):
    //   c = |a|/d,   s = (a/|a|) * conj(b)/d,   r = (a/|a|) * d.
    // The magnitudes are formed with a scale factor so neither |a|^2 nor
    // |b|^2 is ever materialized (dlartg-style: exact near over/underflow).
    const Real magA     = NumericTraits<T>::abs(a);
    const Real magB     = NumericTraits<T>::abs(b);
    T          cosCoeff = {};
    T          sinCoeff = {};
    T          radius   = {};
    if (magB == Real{}) {
        // Nothing to zero; identity (radius carries a's sign/phase).
        cosCoeff = NumericTraits<T>::one();
        sinCoeff = NumericTraits<T>::zero();
        radius   = a;
    } else if (magA == Real{}) {
        // Pure swap; the phase of a is undefined so radius is the real
        // magnitude.
        cosCoeff = NumericTraits<T>::zero();
        sinCoeff = NumericTraits<T>::conj(b) / magB;
        radius   = T(magB);
    } else {
        const Real scale      = magA > magB ? magA : magB;
        const Real scaledMagA = magA / scale;
        const Real scaledMagB = magB / scale;
        const Real scaledNorm =
            std::sqrt(scaledMagA * scaledMagA + scaledMagB * scaledMagB);
        const Real norm  = scale * scaledNorm; // = sqrt(|a|^2 + |b|^2)
        const T    phase = a / magA;           // unit-modulus direction of a

        cosCoeff = T(scaledMagA / scaledNorm); // = |a|/norm
        sinCoeff = phase * (NumericTraits<T>::conj(b) / norm);
        radius   = phase * norm;
    }

    Givens rotation(cosCoeff, sinCoeff, p, q);
    rotation.radius_ = radius;
    return rotation;
}

template <typename T>
Givens<T> Givens<T>::Identity(Index p,
                              Index q) {
    // c = 1, s = 0: the rotation is a no-op on the (p, q) plane.
    Givens rotation(NumericTraits<T>::one(), NumericTraits<T>::zero(), p, q);
    rotation.radius_ = NumericTraits<T>::zero();
    return rotation;
}

template <typename T> const T &Givens<T>::cosine() const { return cosine_; }

template <typename T> const T &Givens<T>::sine() const { return sine_; }

template <typename T> typename Givens<T>::Index Givens<T>::firstIndex() const {
    return p_;
}

template <typename T> typename Givens<T>::Index Givens<T>::secondIndex() const {
    return q_;
}

template <typename T> T Givens<T>::radius() const { return radius_; }

template <typename T> Givens<T> Givens<T>::transposed() const {
    // Transposing [c, s; -conj(s), c] gives [c, -conj(s); s, c], the same
    // rotation form with c unchanged and s -> -conj(s).
    Givens rotation(cosine_, -NumericTraits<T>::conj(sine_), p_, q_);
    rotation.radius_ = radius_;
    return rotation;
}

template <typename T> Givens<T> Givens<T>::inverse() const {
    // The rotation is unitary, so the inverse is the conjugate transpose G^H.
    // Conjugate-transposing [c, s; -conj(s), c] (c real) gives
    // See notes on:
    // http://ec2-54-146-246-77.compute-1.amazonaws.com/notes/Numerical-Linear-Algebra/L15-Structured-Decomposition
    // [c, -s; conj(s), c]: the same rotation form with c unchanged, s -> -s.
    // For real T this coincides with transposed(); for complex T it does not.
    Givens rotation(cosine_, -sine_, p_, q_);
    rotation.radius_ = radius_;
    return rotation;
}

template <typename T> void Givens<T>::applyLeft(MatrixView<T> target) const {
    // Rotate rows p and q in place; every other row is untouched.
    //   row_p <-        c  * row_p + s * row_q
    //   row_q <- -conj(s) * row_p + c * row_q
    const T     negConjSine = -NumericTraits<T>::conj(sine_);
    const Index cols        = target.cols();
    for (Index j = 0; j < cols; ++j) {
        const T rowP  = target(p_, j);
        const T rowQ  = target(q_, j);
        target(p_, j) = cosine_ * rowP + sine_ * rowQ;
        target(q_, j) = negConjSine * rowP + cosine_ * rowQ;
    }
}

template <typename T> void Givens<T>::applyRight(MatrixView<T> target) const {
    // Rotate columns p and q in place; every other column is untouched.
    // With M' = M * [c, s; -conj(s), c] on the (p, q) plane:
    //   col_p <- c * col_p - conj(s) * col_q
    //   col_q <- s * col_p +       c * col_q
    const T     negConjSine = -NumericTraits<T>::conj(sine_);
    const Index rows        = target.rows();
    for (Index i = 0; i < rows; ++i) {
        const T colP  = target(i, p_);
        const T colQ  = target(i, q_);
        target(i, p_) = cosine_ * colP + negConjSine * colQ;
        target(i, q_) = sine_ * colP + cosine_ * colQ;
    }
}

template <typename T> void Givens<T>::apply(Vector<T> &x) const {
    throw LinalgError("not implemented: linalg::Givens<T>::apply");
}

template <typename T> Matrix<T> Givens<T>::toMatrix(Index dimension) const {
    throw LinalgError("not implemented: linalg::Givens<T>::toMatrix");
}

template <typename T> GivensSequence<T>::GivensSequence() {
    throw LinalgError(
        "not implemented: linalg::GivensSequence<T>::GivensSequence");
}

template <typename T>
void GivensSequence<T>::append(const Givens<T> &rotation) {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::append");
}

template <typename T> void GivensSequence<T>::clear() {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::clear");
}

template <typename T>
typename GivensSequence<T>::Index GivensSequence<T>::count() const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::count");
}

template <typename T>
const Givens<T> &GivensSequence<T>::operator[](Index k) const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::operator[]");
}

template <typename T>
void GivensSequence<T>::applyLeft(MatrixView<T> target) const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::applyLeft");
}

template <typename T>
void GivensSequence<T>::applyRight(MatrixView<T> target) const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::applyRight");
}

template <typename T>
void GivensSequence<T>::applyLeftReversed(MatrixView<T> target) const {
    throw LinalgError(
        "not implemented: linalg::GivensSequence<T>::applyLeftReversed");
}

template <typename T> GivensSequence<T> GivensSequence<T>::reversed() const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::reversed");
}

template <typename T>
Matrix<T> GivensSequence<T>::toMatrix(Index dimension) const {
    throw LinalgError("not implemented: linalg::GivensSequence<T>::toMatrix");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Givens<SCALAR>; \
    template class GivensSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
