#include "linalg/core/Vector.hpp"

#include "linalg/core/Matrix.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"

namespace linalg {

template <typename T>
Vector<T>::Vector()
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(Index size)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(Index size, const T& fill)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> values)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(const std::vector<T>& values)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(const Vector& other)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Vector");
}

template <typename T>
Vector<T>::Vector(Vector&& other) noexcept
{
    // TODO: transfer ownership from `other`. Declared noexcept,
    // so this stub cannot throw the way the others do.
}

template <typename T>
Vector<T>::~Vector()
{
}

template <typename T>
Vector<T>& Vector<T>::operator=(const Vector& other)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator=");
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector&& other) noexcept
{
    // TODO: transfer ownership from `other`. Declared noexcept,
    // so this stub cannot throw the way the others do.
    return *this;
}

template <typename T>
Vector<T> Vector<T>::Zeros(Index size)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Zeros");
}

template <typename T>
Vector<T> Vector<T>::Ones(Index size)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Ones");
}

template <typename T>
Vector<T> Vector<T>::Constant(Index size, const T& value)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Constant");
}

template <typename T>
Vector<T> Vector<T>::Unit(Index size, Index axis)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Unit");
}

template <typename T>
Vector<T> Vector<T>::Random(Index size, unsigned long seed)
{
    throw LinalgError("not implemented: linalg::Vector<T>::Random");
}

template <typename T>
Vector<T> Vector<T>::LinSpace(Index size, const T& begin, const T& end)
{
    throw LinalgError("not implemented: linalg::Vector<T>::LinSpace");
}

template <typename T>
T& Vector<T>::operator()(Index i)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator()");
}

template <typename T>
const T& Vector<T>::operator()(Index i) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator()");
}

template <typename T>
T& Vector<T>::operator[](Index i)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator[]");
}

template <typename T>
const T& Vector<T>::operator[](Index i) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator[]");
}

template <typename T>
T& Vector<T>::at(Index i)
{
    throw LinalgError("not implemented: linalg::Vector<T>::at");
}

template <typename T>
const T& Vector<T>::at(Index i) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::at");
}

template <typename T>
T* Vector<T>::data()
{
    throw LinalgError("not implemented: linalg::Vector<T>::data");
}

template <typename T>
const T* Vector<T>::data() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::data");
}

template <typename T>
typename Vector<T>::Index Vector<T>::size() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::size");
}

template <typename T>
bool Vector<T>::isEmpty() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::isEmpty");
}

template <typename T>
Vector<T> Vector<T>::operator+(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator+");
}

template <typename T>
Vector<T> Vector<T>::operator-(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator-");
}

template <typename T>
Vector<T> Vector<T>::operator*(const T& scalar) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator*");
}

template <typename T>
Vector<T> Vector<T>::operator/(const T& scalar) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator/");
}

template <typename T>
Vector<T> Vector<T>::operator-() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator-");
}

template <typename T>
Vector<T>& Vector<T>::operator+=(const Vector& rhs)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator+=");
}

template <typename T>
Vector<T>& Vector<T>::operator-=(const Vector& rhs)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator-=");
}

template <typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator*=");
}

template <typename T>
Vector<T>& Vector<T>::operator/=(const T& scalar)
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator/=");
}

template <typename T>
bool Vector<T>::operator==(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator==");
}

template <typename T>
bool Vector<T>::operator!=(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::operator!=");
}

template <typename T>
Vector<T> Vector<T>::scaledBy(const T& scalar) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::scaledBy");
}

template <typename T>
Vector<T> Vector<T>::elementwiseProduct(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::elementwiseProduct");
}

template <typename T>
Vector<T> Vector<T>::elementwiseQuotient(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::elementwiseQuotient");
}

template <typename T>
T Vector<T>::dot(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::dot");
}

template <typename T>
T Vector<T>::hermitianDot(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::hermitianDot");
}

template <typename T>
Matrix<T> Vector<T>::outer(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::outer");
}

template <typename T>
Vector<T> Vector<T>::cross(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::cross");
}

template <typename T>
Vector<T> Vector<T>::axpy(const T& alpha, const Vector& y) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::axpy");
}

template <typename T>
typename Vector<T>::Real Vector<T>::norm() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::norm");
}

template <typename T>
typename Vector<T>::Real Vector<T>::squaredNorm() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::squaredNorm");
}

template <typename T>
typename Vector<T>::Real Vector<T>::oneNorm() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::oneNorm");
}

template <typename T>
typename Vector<T>::Real Vector<T>::infinityNorm() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::infinityNorm");
}

template <typename T>
typename Vector<T>::Real Vector<T>::pNorm(Real p) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::pNorm");
}

template <typename T>
T Vector<T>::sum() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::sum");
}

template <typename T>
T Vector<T>::product() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::product");
}

template <typename T>
Vector<T> Vector<T>::normalized() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::normalized");
}

template <typename T>
void Vector<T>::normalize()
{
    throw LinalgError("not implemented: linalg::Vector<T>::normalize");
}

template <typename T>
Vector<T> Vector<T>::segment(Index start, Index count) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::segment");
}

template <typename T>
Vector<T> Vector<T>::head(Index count) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::head");
}

template <typename T>
Vector<T> Vector<T>::tail(Index count) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::tail");
}

template <typename T>
Vector<T> Vector<T>::concat(const Vector& rhs) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::concat");
}

template <typename T>
Vector<T> Vector<T>::reversed() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::reversed");
}

template <typename T>
Matrix<T> Vector<T>::asColumnMatrix() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::asColumnMatrix");
}

template <typename T>
Matrix<T> Vector<T>::asRowMatrix() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::asRowMatrix");
}

template <typename T>
Matrix<T> Vector<T>::asDiagonalMatrix() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::asDiagonalMatrix");
}

template <typename T>
void Vector<T>::resize(Index size)
{
    throw LinalgError("not implemented: linalg::Vector<T>::resize");
}

template <typename T>
void Vector<T>::conservativeResize(Index size)
{
    throw LinalgError("not implemented: linalg::Vector<T>::conservativeResize");
}

template <typename T>
void Vector<T>::fill(const T& value)
{
    throw LinalgError("not implemented: linalg::Vector<T>::fill");
}

template <typename T>
void Vector<T>::setZero()
{
    throw LinalgError("not implemented: linalg::Vector<T>::setZero");
}

template <typename T>
void Vector<T>::setUnit(Index axis)
{
    throw LinalgError("not implemented: linalg::Vector<T>::setUnit");
}

template <typename T>
void Vector<T>::swap(Vector& other)
{
    throw LinalgError("not implemented: linalg::Vector<T>::swap");
}

template <typename T>
typename Vector<T>::Index Vector<T>::maxAbsIndex() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::maxAbsIndex");
}

template <typename T>
typename Vector<T>::Index Vector<T>::minAbsIndex() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::minAbsIndex");
}

template <typename T>
T Vector<T>::maxCoefficient() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::maxCoefficient");
}

template <typename T>
T Vector<T>::minCoefficient() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::minCoefficient");
}

template <typename T>
bool Vector<T>::isApprox(const Vector& other, Real tolerance) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::isApprox");
}

template <typename T>
bool Vector<T>::hasNaN() const
{
    throw LinalgError("not implemented: linalg::Vector<T>::hasNaN");
}

template <typename T>
std::string Vector<T>::toString(int precision) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::toString");
}

template <typename T>
void Vector<T>::checkBounds(Index i) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::checkBounds");
}

template <typename T>
void Vector<T>::checkSameSize(const Vector& other) const
{
    throw LinalgError("not implemented: linalg::Vector<T>::checkSameSize");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) \
    template class Vector<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
