#include "linalg/ops/Householder.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

// This library uses the *real-beta* (Hermitian) reflector convention:
// H = I - beta * v * v^H with beta real, so H is both unitary and
// Hermitian (H == H^H, H*H == I). The reflector vector is normalized to
// a unit leading entry, v = [1; essential]; only `essential` is stored.
//
// For a target x with leading entry alpha and tail t:
//   r    = -phase(alpha) * ||x||     (phase(alpha) = alpha / |alpha|)
//   v    = (x - r e_1) / (alpha - r) so v_0 == 1
//   beta = 2 |alpha - r|^2 / (|alpha - r|^2 + ||t||^2) = 2 / (v^H v)
// The sign/phase of r is chosen opposite alpha so the subtraction
// alpha - r never cancels. H x then lands exactly on r e_1, and because
// conj(r) alpha is real, the tail is annihilated.

template <typename T>
Householder<T>::Householder()
    : essential_(0),
      beta_(0) {}

template <typename T>
Householder<T>::Householder(
    const Vector<T> &essential,
    const T         &beta
)
    : essential_(essential),
      beta_(beta) {}

template <typename T>
Householder<T> Householder<T>::FromVector(const Vector<T> &x) {
    const Index n = x.size();
    if (n == 0) {
        return Householder(Vector<T>(0), T{});
    }

    const T    alpha      = x(0);
    const Real absAlphaSq = NumericTraits<T>::absSquared(alpha);

    // Tail squared norm, accumulated directly to avoid the cancellation
    // of forming ||x||^2 - |alpha|^2.
    Real tailSq = Real{};
    for (Index i = 1; i < n; ++i) {
        tailSq += NumericTraits<T>::absSquared(x(i));
    }

    Vector<T> essential(n - 1);

    // Already a (real) multiple of e_1: no reflection needed. The
    // imaginary-part test keeps a complex alpha from slipping through,
    // since we still want to rotate its phase onto the real axis.
    if (tailSq == Real{} && NumericTraits<T>::imag(alpha) == Real{}) {
        for (Index i = 0; i + 1 < n; ++i) {
            essential(i) = T{};
        }
        return Householder(essential, T{});
    }

    const Real normx    = NumericTraits<Real>::sqrt(absAlphaSq + tailSq);
    const Real absAlpha = NumericTraits<Real>::sqrt(absAlphaSq);

    // phase(alpha) = alpha / |alpha|; when alpha == 0 any unit phase
    // works, so take 1 (giving a real, negative r).
    const T phase = (absAlpha == Real{}) ? NumericTraits<T>::one()
                                         : alpha * (Real(1) / absAlpha);
    const T r     = -(phase * normx);
    const T d     = alpha - r;

    for (Index i = 0; i + 1 < n; ++i) {
        essential(i) = x(i + 1) / d;
    }

    const Real dSq  = NumericTraits<T>::absSquared(d);
    const Real beta = Real(2) * dSq / (dSq + tailSq);
    return Householder(essential, T(beta));
}

template <typename T>
Householder<T> Householder<T>::FromColumn(
    const ConstMatrixView<T> &a,
    Index                     col,
    Index                     startRow
) {
    const Index n = a.rows() - startRow;
    Vector<T>   x(n);
    for (Index i = 0; i < n; ++i) {
        x(i) = a(startRow + i, col);
    }
    return FromVector(x);
}

template <typename T> const Vector<T> &Householder<T>::essential() const {
    return essential_;
}

template <typename T> const T &Householder<T>::beta() const {
    return beta_;
}

template <typename T>
typename Householder<T>::Index Householder<T>::size() const {
    return essential_.size() + 1;
}

template <typename T> bool Householder<T>::isIdentity() const {
    return beta_ == T{};
}

// --- application ------------------------------------------------------------
// Every apply is the two-pass H = I - beta v v^H update, never forming H.
// v_p is 1 for p == 0 and essential_(p-1) otherwise. When the reflector is
// shorter than the operand, it acts on the trailing size() rows/columns,
// which reproduces the bottom-right embedding used by toMatrix and the
// QR-style sequence.

template <typename T> void Householder<T>::apply(Vector<T> &x) const {
    if (isIdentity()) {
        return;
    }
    const Index k   = size();
    const Index off = x.size() - k;

    // w = v^H x
    T w = x(off);
    for (Index p = 1; p < k; ++p) {
        w += NumericTraits<T>::conj(essential_(p - 1)) * x(off + p);
    }
    // x -= beta * v * w
    x(off) -= beta_ * w;
    for (Index p = 1; p < k; ++p) {
        x(off + p) -= beta_ * essential_(p - 1) * w;
    }
}

template <typename T>
void Householder<T>::applyLeft(MatrixView<T> target) const {
    if (isIdentity()) {
        return;
    }
    const Index k    = size();
    const Index off  = target.rows() - k;
    const Index cols = target.cols();

    for (Index c = 0; c < cols; ++c) {
        // w = v^H (column c)
        T w = target(off, c);
        for (Index p = 1; p < k; ++p) {
            w += NumericTraits<T>::conj(essential_(p - 1)) * target(off + p, c);
        }
        target(off, c) -= beta_ * w;
        for (Index p = 1; p < k; ++p) {
            target(off + p, c) -= beta_ * essential_(p - 1) * w;
        }
    }
}

template <typename T>
void Householder<T>::applyRight(MatrixView<T> target) const {
    if (isIdentity()) {
        return;
    }
    const Index k    = size();
    const Index off  = target.cols() - k;
    const Index rows = target.rows();

    for (Index r = 0; r < rows; ++r) {
        // u = (row r) v
        T u = target(r, off);
        for (Index q = 1; q < k; ++q) {
            u += target(r, off + q) * essential_(q - 1);
        }
        // row r -= beta * u * v^H
        target(r, off) -= beta_ * u;
        for (Index q = 1; q < k; ++q) {
            target(r, off + q) -=
                beta_ * u * NumericTraits<T>::conj(essential_(q - 1));
        }
    }
}

template <typename T>
void Householder<T>::applyLeftConjugate(MatrixView<T> target) const {
    // beta is real and H is Hermitian, so H^H == H.
    applyLeft(target);
}

template <typename T>
Matrix<T> Householder<T>::toMatrix(Index dimension) const {
    // Materializes the dense reflector; needs a fully implemented core
    // Matrix (Phase 0). Left as a stub until Matrix lands.
    (void)dimension;
    throw LinalgError("not implemented: linalg::Householder<T>::toMatrix");
}

// --- HouseholderSequence ----------------------------------------------------
// Order convention (header): Q = H_0 H_1 ... H_{k-1} in append order, and
// every H_j is Hermitian, so Q^H = H_{k-1} ... H_0. Hence:
//   applyLeft  (Q  * T): apply H_{k-1} .. H_0   (reverse append order)
//   applyRight (T  * Q): apply H_0 .. H_{k-1}   (forward)
//   applyLeftTranspose (Q^H * T): apply H_0 .. H_{k-1} on the left

template <typename T> HouseholderSequence<T>::HouseholderSequence() {}

template <typename T>
HouseholderSequence<T>::HouseholderSequence(
    const Matrix<T> &reflectors,
    const Vector<T> &betas
) {
    // Unpack a packed panel (LAPACK convention): column j holds reflector
    // j with an implicit unit diagonal, so its essential part is the
    // sub-diagonal entries reflectors(j+1.., j) and it acts at offset j.
    const Index m = reflectors.rows();
    const Index k = betas.size();
    reflectors_.reserve(k);
    for (Index j = 0; j < k; ++j) {
        Vector<T> essential(m - j - 1);
        for (Index i = 0; i + j + 1 < m; ++i) {
            essential(i) = reflectors(j + 1 + i, j);
        }
        reflectors_.push_back(Householder<T>(essential, betas(j)));
    }
}

template <typename T>
void HouseholderSequence<T>::append(const Householder<T> &reflector) {
    reflectors_.push_back(reflector);
    blockValid_ = false;
}

template <typename T>
typename HouseholderSequence<T>::Index HouseholderSequence<T>::count() const {
    return reflectors_.size();
}

template <typename T>
void HouseholderSequence<T>::applyLeft(MatrixView<T> target) const {
    // Q * target = H_0 ( H_1 ( ... ( H_{k-1} target ) ) )
    for (Index j = reflectors_.size(); j-- > 0;) {
        reflectors_[j].applyLeft(target);
    }
}

template <typename T>
void HouseholderSequence<T>::applyRight(MatrixView<T> target) const {
    // target * Q = ( ( target H_0 ) H_1 ) ... H_{k-1}
    for (Index j = 0; j < reflectors_.size(); ++j) {
        reflectors_[j].applyRight(target);
    }
}

template <typename T>
void HouseholderSequence<T>::applyLeftTranspose(MatrixView<T> target) const {
    // Q^H * target = H_{k-1} ( ... ( H_0 target ) ), each H_j Hermitian.
    for (Index j = 0; j < reflectors_.size(); ++j) {
        reflectors_[j].applyLeft(target);
    }
}

template <typename T>
Matrix<T> HouseholderSequence<T>::toMatrix(Index dimension) const {
    // Needs core Matrix (Phase 0). Stub until Matrix lands.
    (void)dimension;
    throw LinalgError(
        "not implemented: linalg::HouseholderSequence<T>::toMatrix"
    );
}

template <typename T>
Matrix<T> HouseholderSequence<T>::firstColumns(
    Index dimension,
    Index count
) const {
    (void)dimension;
    (void)count;
    throw LinalgError(
        "not implemented: linalg::HouseholderSequence<T>::firstColumns"
    );
}

template <typename T> Matrix<T> HouseholderSequence<T>::blockV() const {
    throw LinalgError(
        "not implemented: linalg::HouseholderSequence<T>::blockV"
    );
}

template <typename T> Matrix<T> HouseholderSequence<T>::blockT() const {
    throw LinalgError(
        "not implemented: linalg::HouseholderSequence<T>::blockT"
    );
}

template <typename T>
void HouseholderSequence<T>::buildBlockRepresentation(Index blockSize) {
    // Compact WY form: Phase 6.
    (void)blockSize;
    throw LinalgError(
        "not implemented: "
        "linalg::HouseholderSequence<T>::buildBlockRepresentation"
    );
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)      \
    template class Householder<SCALAR>; \
    template class HouseholderSequence<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
