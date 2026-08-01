#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"

namespace linalg {

template <typename T>
class Matrix;

// Owning dense vector. Kept distinct from an n x 1 Matrix so that
// dot/outer/norm have unambiguous meaning.
template <typename T>
class Vector {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // --- construction -----------------------------------------------------
    Vector();
    explicit Vector(Index size);
    Vector(Index size, const T& fill);
    Vector(std::initializer_list<T> values);
    explicit Vector(const std::vector<T>& values);
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;
    ~Vector();

    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other) noexcept;

    static Vector Zeros(Index size);
    static Vector Ones(Index size);
    static Vector Constant(Index size, const T& value);
    static Vector Unit(Index size, Index axis);   // e_axis: 1 at `axis`, 0 elsewhere
    static Vector Random(Index size, unsigned long seed); // deterministic per seed
    static Vector LinSpace(Index size, const T& begin, const T& end); // endpoints inclusive

    // --- element access ---------------------------------------------------
    T&       operator()(Index i);
    const T& operator()(Index i) const;
    T&       operator[](Index i);
    const T& operator[](Index i) const;
    T&       at(Index i);
    const T& at(Index i) const;

    T*       data();
    const T* data() const;

    Index size() const;
    bool  isEmpty() const;

    // --- arithmetic (members only) ----------------------------------------
    Vector operator+(const Vector& rhs) const;
    Vector operator-(const Vector& rhs) const;
    Vector operator*(const T& scalar) const;
    Vector operator/(const T& scalar) const;
    Vector operator-() const;

    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector& operator*=(const T& scalar);
    Vector& operator/=(const T& scalar);

    bool operator==(const Vector& rhs) const;
    bool operator!=(const Vector& rhs) const;

    Vector scaledBy(const T& scalar) const;
    Vector elementwiseProduct(const Vector& rhs) const;
    Vector elementwiseQuotient(const Vector& rhs) const;

    // --- products ---------------------------------------------------------
    // dot is the unconjugated bilinear form sum x_i y_i even for complex T;
    // hermitianDot conjugates *this* (the left argument): sum conj(x_i) y_i.
    // The two coincide for real scalars. Inner-product norms want
    // hermitianDot.
    T         dot(const Vector& rhs) const;
    T         hermitianDot(const Vector& rhs) const;
    Matrix<T> outer(const Vector& rhs) const;      // x * y^H: size() x rhs.size()
    Vector    cross(const Vector& rhs) const;      // size 3 only; DimensionMismatch otherwise
    Vector    axpy(const T& alpha, const Vector& y) const; // alpha * (*this) + y

    // --- norms ------------------------------------------------------------
    // norm() is the Euclidean norm, computed with scaling so entries near
    // the overflow threshold do not square to infinity; squaredNorm() is
    // the unscaled sum of |x_i|^2 and may overflow where norm() does not.
    Real norm() const;
    Real squaredNorm() const;
    Real oneNorm() const;
    Real infinityNorm() const;
    Real pNorm(Real p) const;
    T    sum() const;
    T    product() const;

    Vector normalized() const;  // zero vector throws LinalgError
    void   normalize();

    // --- shape ------------------------------------------------------------
    Vector    segment(Index start, Index count) const;
    Vector    head(Index count) const;
    Vector    tail(Index count) const;
    Vector    concat(const Vector& rhs) const;
    Vector    reversed() const;
    Matrix<T> asColumnMatrix() const;
    Matrix<T> asRowMatrix() const;
    Matrix<T> asDiagonalMatrix() const;

    void resize(Index size);
    void conservativeResize(Index size);
    void fill(const T& value);
    void setZero();
    void setUnit(Index axis);
    void swap(Vector& other);

    // --- reductions -------------------------------------------------------
    // Ties resolve to the lowest index. max/minCoefficient compare by real
    // part for complex T (there is no natural total order on C).
    Index maxAbsIndex() const;
    Index minAbsIndex() const;
    T     maxCoefficient() const;
    T     minCoefficient() const;
    bool  isApprox(const Vector& other, Real tolerance) const;
    bool  hasNaN() const;

    std::string toString(int precision) const;

private:
    void checkBounds(Index i) const;
    void checkSameSize(const Vector& other) const;

    std::vector<T> storage_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Vector<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
