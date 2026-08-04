#include "linalg/ops/Norm.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T>
typename Norm<T>::Real Norm<T>::vectorOne(const Vector<T> &x) {
    return x.oneNorm();
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorTwo(const Vector<T> &x) {
    return x.norm();
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorInfinity(const Vector<T> &x) {
    return x.infinityNorm();
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorP(const Vector<T> &x,
                                        Real             p) {
    return x.pNorm(p);
}

template <typename T>
typename Norm<T>::Real Norm<T>::vectorTwoScaled(const Vector<T> &x) {
    return x.norm();
}

// TODO: Requires SVD
template <typename T>
typename Norm<T>::Real Norm<T>::matrixOne(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Norm<T>::matrixOne");
}

// Maximum absolute row sum. No SVD needed.
template <typename T>
typename Norm<T>::Real Norm<T>::matrixInfinity(const Matrix<T> &a) {
    Real maxSum = Real{};
    for (Index i = 0; i < a.rows(); ++i) {
        Real rowSum = Real{};
        for (Index j = 0; j < a.cols(); ++j) {
            rowSum += NumericTraits<T>::abs(a(i, j));
        }
        if (rowSum > maxSum) {
            maxSum = rowSum;
        }
    }
    return maxSum;
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixFrobenius(const Matrix<T> &a) {
    Real sumSq = Real{};
    for (Index i = 0; i < a.rows(); ++i) {
        for (Index j = 0; j < a.cols(); ++j) {
            sumSq += NumericTraits<T>::absSquared(a(i, j));
        }
    }
    return NumericTraits<Real>::sqrt(sumSq);
}

template <typename T>
typename Norm<T>::Real Norm<T>::matrixMax(const Matrix<T> &a) {
    Real maxMag = Real{};
    for (Index i = 0; i < a.rows(); ++i) {
        for (Index j = 0; j < a.cols(); ++j) {
            const Real mag = NumericTraits<T>::abs(a(i, j));
            if (mag > maxMag) {
                maxMag = mag;
            }
        }
    }
    return maxMag;
}

// TODO: Requires SVD
template <typename T>
typename Norm<T>::Real Norm<T>::matrixTwo(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Norm<T>::matrixTwo");
}

// TODO: Requires SVD
template <typename T>
typename Norm<T>::Real Norm<T>::matrixNuclear(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Norm<T>::matrixNuclear");
}

template <typename T>
typename Norm<T>::Real Norm<T>::distance(const Vector<T> &x,
                                         const Vector<T> &y) {
    return (x - y).norm();
}

template <typename T>
typename Norm<T>::Real Norm<T>::relativeError(const Vector<T> &approx,
                                              const Vector<T> &exact) {
    throw LinalgError("not implemented: linalg::Norm<T>::relativeError");
}

template <typename T>
typename Norm<T>::Real Norm<T>::residualNorm(const Matrix<T> &a,
                                             const Vector<T> &x,
                                             const Vector<T> &b) {
    Vector<T> Ax = a * x;
    return (b - Ax).norm();
}

template <typename T>
typename Norm<T>::Real Norm<T>::backwardError(const Matrix<T> &a,
                                              const Vector<T> &x,
                                              const Vector<T> &b) {
    throw LinalgError("not implemented: linalg::Norm<T>::backwardError");
}

template <typename T>
typename Norm<T>::Real Norm<T>::orthogonalityDefect(const Matrix<T> &q) {
    throw LinalgError("not implemented: linalg::Norm<T>::orthogonalityDefect");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Norm<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
