#include "linalg/core/Vector.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <utility>

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
    throw LinalgError("not implemented: linalg::Vector<T>::outer");
}

template <typename T> Vector<T> Vector<T>::cross(const Vector &rhs) const {
    throw LinalgError("not implemented: linalg::Vector<T>::cross");
}

template <typename T>
Vector<T> Vector<T>::axpy(const T      &alpha,
                          const Vector &y) const {
    throw LinalgError("not implemented: linalg::Vector<T>::axpy");
}

template <typename T> typename Vector<T>::Real Vector<T>::norm() const {
    throw LinalgError("not implemented: linalg::Vector<T>::norm");
}

template <typename T> typename Vector<T>::Real Vector<T>::squaredNorm() const {
    throw LinalgError("not implemented: linalg::Vector<T>::squaredNorm");
}

template <typename T> typename Vector<T>::Real Vector<T>::oneNorm() const {
    throw LinalgError("not implemented: linalg::Vector<T>::oneNorm");
}

template <typename T> typename Vector<T>::Real Vector<T>::infinityNorm() const {
    throw LinalgError("not implemented: linalg::Vector<T>::infinityNorm");
}

template <typename T> typename Vector<T>::Real Vector<T>::pNorm(Real p) const {
    throw LinalgError("not implemented: linalg::Vector<T>::pNorm");
}

template <typename T> T Vector<T>::sum() const {
    throw LinalgError("not implemented: linalg::Vector<T>::sum");
}

template <typename T> T Vector<T>::product() const {
    throw LinalgError("not implemented: linalg::Vector<T>::product");
}

template <typename T> Vector<T> Vector<T>::normalized() const {
    throw LinalgError("not implemented: linalg::Vector<T>::normalized");
}

template <typename T> void Vector<T>::normalize() {
    throw LinalgError("not implemented: linalg::Vector<T>::normalize");
}

template <typename T>
Vector<T> Vector<T>::segment(Index start,
                             Index count) const {
    throw LinalgError("not implemented: linalg::Vector<T>::segment");
}

template <typename T> Vector<T> Vector<T>::head(Index count) const {
    throw LinalgError("not implemented: linalg::Vector<T>::head");
}

template <typename T> Vector<T> Vector<T>::tail(Index count) const {
    throw LinalgError("not implemented: linalg::Vector<T>::tail");
}

template <typename T> Vector<T> Vector<T>::concat(const Vector &rhs) const {
    throw LinalgError("not implemented: linalg::Vector<T>::concat");
}

template <typename T> Vector<T> Vector<T>::reversed() const {
    throw LinalgError("not implemented: linalg::Vector<T>::reversed");
}

template <typename T> Matrix<T> Vector<T>::asColumnMatrix() const {
    throw LinalgError("not implemented: linalg::Vector<T>::asColumnMatrix");
}

template <typename T> Matrix<T> Vector<T>::asRowMatrix() const {
    throw LinalgError("not implemented: linalg::Vector<T>::asRowMatrix");
}

template <typename T> Matrix<T> Vector<T>::asDiagonalMatrix() const {
    throw LinalgError("not implemented: linalg::Vector<T>::asDiagonalMatrix");
}

template <typename T> void Vector<T>::resize(Index size) {
    throw LinalgError("not implemented: linalg::Vector<T>::resize");
}

template <typename T> void Vector<T>::conservativeResize(Index size) {
    throw LinalgError("not implemented: linalg::Vector<T>::conservativeResize");
}

template <typename T> void Vector<T>::fill(const T &value) {
    throw LinalgError("not implemented: linalg::Vector<T>::fill");
}

template <typename T> void Vector<T>::setZero() {
    throw LinalgError("not implemented: linalg::Vector<T>::setZero");
}

template <typename T> void Vector<T>::setUnit(Index axis) {
    throw LinalgError("not implemented: linalg::Vector<T>::setUnit");
}

template <typename T> void Vector<T>::swap(Vector &other) {
    throw LinalgError("not implemented: linalg::Vector<T>::swap");
}

template <typename T> typename Vector<T>::Index Vector<T>::maxAbsIndex() const {
    throw LinalgError("not implemented: linalg::Vector<T>::maxAbsIndex");
}

template <typename T> typename Vector<T>::Index Vector<T>::minAbsIndex() const {
    throw LinalgError("not implemented: linalg::Vector<T>::minAbsIndex");
}

template <typename T> T Vector<T>::maxCoefficient() const {
    throw LinalgError("not implemented: linalg::Vector<T>::maxCoefficient");
}

template <typename T> T Vector<T>::minCoefficient() const {
    throw LinalgError("not implemented: linalg::Vector<T>::minCoefficient");
}

template <typename T>
bool Vector<T>::isApprox(const Vector &other,
                         Real          tolerance) const {
    throw LinalgError("not implemented: linalg::Vector<T>::isApprox");
}

template <typename T> bool Vector<T>::hasNaN() const {
    throw LinalgError("not implemented: linalg::Vector<T>::hasNaN");
}

template <typename T> std::string Vector<T>::toString(int precision) const {
    throw LinalgError("not implemented: linalg::Vector<T>::toString");
}

template <typename T> void Vector<T>::checkBounds(Index i) const {
    if (i >= storage_.size()) {
        throw IndexOutOfRange(i, storage_.size());
    }
}

template <typename T> void Vector<T>::checkSameSize(const Vector &other) const {
    throw LinalgError("not implemented: linalg::Vector<T>::checkSameSize");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Vector<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
