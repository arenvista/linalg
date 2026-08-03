#include "linalg/core/Vector.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace linalg {

template <typename T>
Vector<T>::Vector()
    : storage_() {}

template <typename T>
Vector<T>::Vector(Index size)
    : storage_(size) {}

template <typename T>
Vector<T>::Vector(Index    size,
                  const T &fill)
    : storage_(size,
               fill) {}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> values)
    : storage_(values) {}

template <typename T>
Vector<T>::Vector(const std::vector<T> &values)
    : storage_(values) {}

template <typename T>
Vector<T>::Vector(const Vector &other)
    : storage_(other.storage_) {}

template <typename T>
Vector<T>::Vector(Vector &&other) noexcept
    : storage_(std::move(other.storage_)) {}

template <typename T> Vector<T>::~Vector() {}

template <typename T> Vector<T> &Vector<T>::operator=(const Vector &other) {
    if (this != &other) {
        storage_ = other.storage_;
    }
    return *this;
}

template <typename T> Vector<T> &Vector<T>::operator=(Vector &&other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);
    }
    return *this;
}

template <typename T> Vector<T> Vector<T>::Zeros(Index size) {
    return Vector(size, T{});
}

template <typename T> Vector<T> Vector<T>::Ones(Index size) {
    return Vector(size, T(1));
}

template <typename T>
Vector<T> Vector<T>::Constant(Index    size,
                              const T &value) {
    return Vector(size, T(value));
}

template <typename T>
Vector<T> Vector<T>::Unit(Index size,
                          Index axis) {
    if (axis >= size) {
        throw IndexOutOfRange(axis, size);
    }
    Vector result(size, T{});
    result(axis) = T(1);
    return result;
}

template <typename T>
Vector<T> Vector<T>::Random(Index         size,
                            unsigned long seed) {
    Vector                               result(size);
    std::mt19937_64                      gen(seed);
    std::uniform_real_distribution<Real> dist(Real(-1), Real(1));
    for (Index i = 0; i < size; ++i) {
        if constexpr (IsComplex<T>::value) {
            result(i) = T(dist(gen), dist(gen));
        } else {
            result(i) = dist(gen);
        }
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::LinSpace(Index    size,
                              const T &begin,
                              const T &end) {
    Vector result(size);
    if (size == 0) {
        return result;
    }
    if (size == 1) {
        result(0) = begin;
        return result;
    }
    const T step = (end - begin) / T(static_cast<Real>(size - 1));
    for (Index i = 0; i < size; ++i) {
        result(i) = begin + step * T(static_cast<Real>(i));
    }
    result(size - 1) = end; // exact inclusive endpoint, free of drift
    return result;
}

template <typename T> T &Vector<T>::operator()(Index i) { return storage_[i]; }

template <typename T> const T &Vector<T>::operator()(Index i) const {
    return storage_[i];
}

template <typename T> T &Vector<T>::operator[](Index i) { return storage_[i]; }

template <typename T> const T &Vector<T>::operator[](Index i) const {
    return storage_[i];
}

template <typename T> T &Vector<T>::at(Index i) {
    checkBounds(i);
    return storage_[i];
}

template <typename T> const T &Vector<T>::at(Index i) const {
    checkBounds(i);
    return storage_[i];
}

template <typename T> T *Vector<T>::data() { return storage_.data(); }

template <typename T> const T *Vector<T>::data() const {
    return storage_.data();
}

template <typename T> typename Vector<T>::Index Vector<T>::size() const {
    return storage_.size();
}

template <typename T> bool Vector<T>::isEmpty() const {
    return storage_.empty();
}

template <typename T> Vector<T> Vector<T>::operator+(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i) + rhs(i);
    }
    return result;
}

template <typename T> Vector<T> Vector<T>::operator-(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i)-rhs(i);
    }
    return result;
}

template <typename T> Vector<T> Vector<T>::operator*(const T &scalar) const {
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i)*scalar;
    }
    return result;
}

template <typename T> Vector<T> Vector<T>::operator/(const T &scalar) const {
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i) / scalar;
    }
    return result;
}

template <typename T> Vector<T> Vector<T>::operator-() const {
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = -(*this)(i);
    }
    return result;
}

template <typename T> Vector<T> &Vector<T>::operator+=(const Vector &rhs) {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    std::transform(storage_.begin(), storage_.end(), rhs.storage_.begin(),
                   storage_.begin(), std::plus<T>());
    return *this;
}

template <typename T> Vector<T> &Vector<T>::operator-=(const Vector &rhs) {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    std::transform(storage_.begin(), storage_.end(), rhs.storage_.begin(),
                   storage_.begin(), std::minus<T>());
    return *this;
}

template <typename T> Vector<T> &Vector<T>::operator*=(const T &scalar) {
    std::transform(storage_.begin(), storage_.end(), storage_.begin(),
                   [scalar](const T &x) { return x * scalar; });
    return *this;
}

template <typename T> Vector<T> &Vector<T>::operator/=(const T &scalar) {
    std::transform(storage_.begin(), storage_.end(), storage_.begin(),
                   [scalar](const T &x) { return x / scalar; });
    return *this;
}

template <typename T> bool Vector<T>::operator==(const Vector &rhs) const {
    return (size() == rhs.size()) &&
           std::equal(storage_.begin(), storage_.end(), rhs.storage_.begin());
}

template <typename T> bool Vector<T>::operator!=(const Vector &rhs) const {
    return !(*this == rhs);
}

template <typename T> Vector<T> Vector<T>::scaledBy(const T &scalar) const {
    Vector result(size());
    std::transform(storage_.begin(), storage_.end(), result.storage_.begin(),
                   [scalar](const T &x) { return x * scalar; });
    return result;
}

template <typename T>
Vector<T> Vector<T>::elementwiseProduct(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i)*rhs(i);
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::elementwiseQuotient(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = (*this)(i) / rhs(i);
    }
    return result;
}

template <typename T> T Vector<T>::dot(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    T sum = T{};
    for (Index i = 0; i < size(); ++i) {
        sum += (*this)(i)*rhs(i);
    }
}

template <typename T> T Vector<T>::hermitianDot(const Vector &rhs) const {
    if (size() != rhs.size()) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    T sum = T{};
    for (Index i = 0; i < size(); ++i) {
        sum += NumericTraits<T>::conj((*this)(i)) * rhs(i);
    }
    return sum;
}

template <typename T> Matrix<T> Vector<T>::outer(const Vector &rhs) const {
    Matrix<T> result(size(), rhs.size());
    for (Index i = 0; i < size(); ++i) {
        for (Index j = 0; j < rhs.size(); ++j) {
            result(i, j) = (*this)(i)*NumericTraits<T>::conj(rhs(j));
        }
    }
    return result;
}

template <typename T> Vector<T> Vector<T>::cross(const Vector &rhs) const {
    if (size() != 3 || rhs.size() != 3) {
        throw DimensionMismatch(size(), 1, rhs.size(), 1);
    }
    Vector result(3);
    result(0) = (*this)(1) * rhs(2) - (*this)(2) * rhs(1);
    result(1) = (*this)(2) * rhs(0) - (*this)(0) * rhs(2);
    result(2) = (*this)(0) * rhs(1) - (*this)(1) * rhs(0);
    return result;
}

template <typename T>
Vector<T> Vector<T>::axpy(const T      &alpha,
                          const Vector &y) const {
    if (size() != y.size()) {
        throw DimensionMismatch(size(), 1, y.size(), 1);
    }
    Vector result(size());
    for (Index i = 0; i < size(); ++i) {
        result(i) = alpha * (*this)(i) + y(i);
    }
    return result;
}

template <typename T> typename Vector<T>::Real Vector<T>::norm() const {
    // Overflow-safe scaling: factor out the running maximum magnitude so
    // |x_i|^2 is never formed directly. Mirrors BLAS nrm2.
    Real scale = Real{};
    Real ssq   = Real(1);
    for (Index i = 0; i < size(); ++i) {
        const Real ax = NumericTraits<T>::abs((*this)(i));
        if (ax != Real{}) {
            if (scale < ax) {
                const Real r = scale / ax;
                ssq          = Real(1) + ssq * r * r;
                scale        = ax;
            } else {
                const Real r = ax / scale;
                ssq += r * r;
            }
        }
    }
    return scale * NumericTraits<Real>::sqrt(ssq);
}

template <typename T> typename Vector<T>::Real Vector<T>::squaredNorm() const {
    Real sumSq = Real{};
    for (Index i = 0; i < size(); ++i) {
        sumSq += NumericTraits<T>::absSquared((*this)(i));
    }
    return sumSq;
}

template <typename T> typename Vector<T>::Real Vector<T>::oneNorm() const {
    Real sum = Real{};
    for (Index i = 0; i < size(); ++i) {
        sum += NumericTraits<T>::abs((*this)(i));
    }
    return sum;
}

template <typename T> typename Vector<T>::Real Vector<T>::infinityNorm() const {
    Real maxMag = Real{};
    for (Index i = 0; i < size(); ++i) {
        const Real ax = NumericTraits<T>::abs((*this)(i));
        if (ax > maxMag) {
            maxMag = ax;
        }
    }
    return maxMag;
}

template <typename T> typename Vector<T>::Real Vector<T>::pNorm(Real p) const {
    Real sum = Real{};
    for (Index i = 0; i < size(); ++i) {
        sum += std::pow(NumericTraits<T>::abs((*this)(i)), p);
    }
    return std::pow(sum, Real(1) / p);
}

template <typename T> T Vector<T>::sum() const {
    return std::accumulate(storage_.begin(), storage_.end(), T{});
}

template <typename T> T Vector<T>::product() const {
    return std::accumulate(storage_.begin(), storage_.end(), T{1},
                           std::multiplies<T>());
}

template <typename T> Vector<T> Vector<T>::normalized() const {
    return (*this) / T(norm());
}

template <typename T> void Vector<T>::normalize() {
    const Real n = norm();
    if (n == Real{}) {
        throw LinalgError("cannot normalize zero vector");
    }
    (*this) /= T(n);
}

template <typename T>
Vector<T> Vector<T>::segment(Index start,
                             Index count) const {
    return Vector(std::vector<T>(storage_.begin() + start,
                                 storage_.begin() + start + count));
}

template <typename T> Vector<T> Vector<T>::head(Index count) const {
    return Vector(std::vector<T>(storage_.begin(), storage_.begin() + count));
}

template <typename T> Vector<T> Vector<T>::tail(Index count) const {
    return Vector(std::vector<T>(storage_.end() - count, storage_.end()));
}

template <typename T> Vector<T> Vector<T>::concat(const Vector &rhs) const {
    std::vector<T> result = std::vector<T>(storage_.begin(), storage_.end());
    result.reserve(size() + rhs.size());
    result.insert(result.end(), rhs.storage_.begin(), rhs.storage_.end());
    return Vector(result);
}

template <typename T> Vector<T> Vector<T>::reversed() const {
    return Vector(std::vector<T>(storage_.rbegin(), storage_.rend()));
}

template <typename T> Matrix<T> Vector<T>::asColumnMatrix() const {
    return Matrix<T>(size(), 1, storage_);
}

template <typename T> Matrix<T> Vector<T>::asRowMatrix() const {
    return Matrix<T>(1, size(), storage_);
}

template <typename T> Matrix<T> Vector<T>::asDiagonalMatrix() const {
    Matrix<T> result(size(), size(), T{});
    for (Index i = 0; i < size(); ++i) {
        result(i, i) = (*this)(i);
    }
    return result;
}

template <typename T> void Vector<T>::resize(Index size) {
    // Contract: discard all existing contents; zero-fill the new length.
    storage_.assign(size, T{});
}

template <typename T> void Vector<T>::conservativeResize(Index size) {
    throw LinalgError("not implemented: linalg::Vector<T>::conservativeResize");
}

template <typename T> void Vector<T>::fill(const T &value) {
    std::fill(storage_.begin(), storage_.end(), value);
}

template <typename T> void Vector<T>::setZero() {
    std::fill(storage_.begin(), storage_.end(), T{});
}

template <typename T> void Vector<T>::setUnit(Index axis) {
    setZero(), storage_[axis] = T(1);
}

template <typename T> void Vector<T>::swap(Vector &other) {
    storage_.swap(other.storage_);
}

template <typename T> typename Vector<T>::Index Vector<T>::maxAbsIndex() const {
    T           maxValue = storage_[0];
    std::size_t maxIndex = 0;
    for (std::size_t i = 1; i < storage_.size(); i++) {
        if (NumericTraits<T>::abs(storage_[i]) >
            NumericTraits<T>::abs(maxValue)) {
            maxValue = storage_[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

template <typename T> typename Vector<T>::Index Vector<T>::minAbsIndex() const {
    T           minValue = storage_[0];
    std::size_t minIndex = 0;
    for (std::size_t i = 1; i < storage_.size(); i++) {
        if (NumericTraits<T>::abs(storage_[i]) <
            NumericTraits<T>::abs(minValue)) {
            minValue = storage_[i];
            minIndex = i;
        }
    }
    return minIndex;
}

template <typename T> T Vector<T>::maxCoefficient() const {
    T           maxValue = storage_[0];
    std::size_t maxIndex = 0;
    for (std::size_t i = 1; i < storage_.size(); i++) {
        if (NumericTraits<T>::abs(storage_[i]) >
            NumericTraits<T>::abs(maxValue)) {
            maxValue = storage_[i];
            maxIndex = i;
        }
    }
    return maxValue;
}

template <typename T> T Vector<T>::minCoefficient() const {
    T           minValue = storage_[0];
    std::size_t minIndex = 0;
    for (std::size_t i = 1; i < storage_.size(); i++) {
        if (NumericTraits<T>::abs(storage_[i]) <
            NumericTraits<T>::abs(minValue)) {
            minValue = storage_[i];
            minIndex = i;
        }
    }
    return minValue;
}

template <typename T>
bool Vector<T>::isApprox(const Vector &other,
                         Real          tolerance) const {
    throw LinalgError("not implemented: linalg::Vector<T>::isApprox");
}

template <typename T> bool Vector<T>::hasNaN() const {
    for (const auto &value : storage_) {
        // Covers complex T too: imag() is 0 for real T, so one path serves
        // both.
        if (std::isnan(NumericTraits<T>::real(value)) ||
            std::isnan(NumericTraits<T>::imag(value))) {
            return true;
        }
    }
    return false;
}

template <typename T> std::string Vector<T>::toString(int precision) const {
    std::ostringstream oss;
    oss << std::setprecision(precision);
    oss << "[";
    for (Index i = 0; i < size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << (*this)(i); // std::complex has its own operator<<, so both work
    }
    oss << "]";
    return oss.str();
}

template <typename T> void Vector<T>::checkBounds(Index i) const {
    if (i >= storage_.size()) {
        throw IndexOutOfRange(i, storage_.size());
    }
}

template <typename T> void Vector<T>::checkSameSize(const Vector &other) const {
    if (size() != other.size()) {
        throw DimensionMismatch(size(), 1, other.size(), 1);
    }
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Vector<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
