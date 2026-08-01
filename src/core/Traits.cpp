#include "linalg/core/Traits.hpp"

#include <cmath>
#include <limits>

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::epsilon() {
  return std::numeric_limits<Real>::epsilon();
}

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::safeMin() {
  // LAPACK dlamch('S'): smallest s with 1/s finite. On IEEE this reduces
  // to min(); the guard covers formats where 1/min() would overflow.
  const Real tiny = std::numeric_limits<Real>::min();
  const Real small = Real(1) / std::numeric_limits<Real>::max();
  if (small >= tiny) {
    return small * (Real(1) + std::numeric_limits<Real>::epsilon());
  }
  return tiny;
}

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::abs(const Scalar &x) {
  return std::abs(x); // NOTE: using cmath abs to avoid signed zero issues for
                      // floating point types
}

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::absSquared(const Scalar &x) {
  return std::pow(x, 2);
}

template <typename T>
typename NumericTraits<T>::Scalar NumericTraits<T>::conj(const Scalar &x) {
  return x;
}

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::real(const Scalar &x) {
  return x;
}

template <typename T>
typename NumericTraits<T>::Real NumericTraits<T>::imag(const Scalar &x) {
  throw LinalgError("not implemented: linalg::NumericTraits<T>::imag");
}

template <typename T>
typename NumericTraits<T>::Scalar NumericTraits<T>::sqrt(const Scalar &x) {
  return std::sqrt(x);
}

template <typename T>
typename NumericTraits<T>::Scalar NumericTraits<T>::zero() {
  return Scalar(0);
}

template <typename T>
typename NumericTraits<T>::Scalar NumericTraits<T>::one() {
  return Scalar(1);
}

template <typename T>
bool NumericTraits<T>::isApproxZero(const Scalar &x, Real tol) {
  return abs(x) <= tol;
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::epsilon() {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::epsilon");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::safeMin() {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::safeMin");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::abs(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::abs");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::absSquared(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::absSquared");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Scalar
NumericTraits<std::complex<T>>::conj(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::conj");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::real(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::real");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Real
NumericTraits<std::complex<T>>::imag(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::imag");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Scalar
NumericTraits<std::complex<T>>::sqrt(const Scalar &x) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::sqrt");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Scalar
NumericTraits<std::complex<T>>::zero() {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::zero");
}

template <typename T>
typename NumericTraits<std::complex<T>>::Scalar
NumericTraits<std::complex<T>>::one() {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::one");
}

template <typename T>
bool NumericTraits<std::complex<T>>::isApproxZero(const Scalar &x, Real tol) {
  throw LinalgError(
      "not implemented: linalg::NumericTraits<std::complex<T>>::isApproxZero");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template struct NumericTraits<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
