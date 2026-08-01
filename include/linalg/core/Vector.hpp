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
    /// @brief Constructs an empty vector of length zero.
    Vector();
    /// @brief Constructs a vector of the given size with value-initialized
    /// (zero) elements.
    /// @param size Number of elements.
    explicit Vector(Index size);
    /// @brief Constructs a vector with every element set to a fill value.
    /// @param size Number of elements.
    /// @param fill Value assigned to every element.
    Vector(Index size, const T& fill);
    /// @brief Constructs a vector from a braced list of values.
    /// @param values Elements in order.
    Vector(std::initializer_list<T> values);
    /// @brief Constructs a vector by copying a std::vector of values.
    /// @param values Elements to copy.
    explicit Vector(const std::vector<T>& values);
    /// @brief Copy constructor; deep-copies another vector.
    /// @param other Vector to copy.
    Vector(const Vector& other);
    /// @brief Move constructor; takes over another vector's storage.
    /// @param other Vector to move from; left empty afterward.
    Vector(Vector&& other) noexcept;
    /// @brief Destructor; releases the owned storage.
    ~Vector();

    /// @brief Copy assignment; deep-copies another vector.
    /// @param other Vector to copy.
    /// @return Reference to this vector.
    Vector& operator=(const Vector& other);
    /// @brief Move assignment; takes over another vector's storage.
    /// @param other Vector to move from; left empty afterward.
    /// @return Reference to this vector.
    Vector& operator=(Vector&& other) noexcept;

    /// @brief Creates a vector of zeros.
    /// @param size Number of elements.
    /// @return A length-`size` vector of zeros.
    static Vector Zeros(Index size);
    /// @brief Creates a vector of ones.
    /// @param size Number of elements.
    /// @return A length-`size` vector of ones.
    static Vector Ones(Index size);
    /// @brief Creates a vector with every element set to a constant.
    /// @param size Number of elements.
    /// @param value Value assigned to every element.
    /// @return A length-`size` vector filled with value.
    static Vector Constant(Index size, const T& value);
    /// @brief Creates a standard basis vector e_axis (1 at `axis`, 0 elsewhere).
    /// @param size Number of elements.
    /// @param axis Index of the single nonzero entry (0-based).
    /// @return The requested unit vector.
    static Vector Unit(Index size, Index axis);
    /// @brief Creates a vector of pseudo-random entries, reproducible per seed.
    /// @param size Number of elements.
    /// @param seed Seed determining the entries.
    /// @return A length-`size` random vector.
    static Vector Random(Index size, unsigned long seed);
    /// @brief Creates a vector of equally spaced values with inclusive endpoints.
    /// @param size Number of elements.
    /// @param begin First value.
    /// @param end Last value (inclusive).
    /// @return The linearly spaced vector.
    static Vector LinSpace(Index size, const T& begin, const T& end);

    // --- element access ---------------------------------------------------
    /// @brief Unchecked mutable access to an element.
    /// @param i Element index (0-based).
    /// @return Reference to element i.
    T&       operator()(Index i);
    /// @brief Unchecked read-only access to an element.
    /// @param i Element index (0-based).
    /// @return Const reference to element i.
    const T& operator()(Index i) const;
    /// @brief Unchecked mutable access to an element.
    /// @param i Element index (0-based).
    /// @return Reference to element i.
    T&       operator[](Index i);
    /// @brief Unchecked read-only access to an element.
    /// @param i Element index (0-based).
    /// @return Const reference to element i.
    const T& operator[](Index i) const;
    /// @brief Bounds-checked mutable access to an element.
    /// @param i Element index (0-based).
    /// @return Reference to element i.
    /// @throws IndexOutOfRange if i >= size().
    T&       at(Index i);
    /// @brief Bounds-checked read-only access to an element.
    /// @param i Element index (0-based).
    /// @return Const reference to element i.
    /// @throws IndexOutOfRange if i >= size().
    const T& at(Index i) const;

    /// @brief Direct mutable access to the underlying buffer.
    /// @return Pointer to the first element.
    T*       data();
    /// @brief Direct read-only access to the underlying buffer.
    /// @return Const pointer to the first element.
    const T* data() const;

    /// @brief Number of elements.
    /// @return The length.
    Index size() const;
    /// @brief Reports whether the vector has no elements.
    /// @return True if size() is zero.
    bool  isEmpty() const;

    // --- arithmetic (members only) ----------------------------------------
    /// @brief Vector addition.
    /// @param rhs Vector of matching size.
    /// @return The elementwise sum.
    /// @throws DimensionMismatch if sizes differ.
    Vector operator+(const Vector& rhs) const;
    /// @brief Vector subtraction.
    /// @param rhs Vector of matching size.
    /// @return The elementwise difference.
    /// @throws DimensionMismatch if sizes differ.
    Vector operator-(const Vector& rhs) const;
    /// @brief Multiplies every element by a scalar.
    /// @param scalar Scalar multiplier.
    /// @return The scaled vector.
    Vector operator*(const T& scalar) const;
    /// @brief Divides every element by a scalar.
    /// @param scalar Scalar divisor.
    /// @return The scaled vector.
    Vector operator/(const T& scalar) const;
    /// @brief Unary negation.
    /// @return A vector with every element negated.
    Vector operator-() const;

    /// @brief In-place vector addition.
    /// @param rhs Vector of matching size.
    /// @return Reference to this vector.
    /// @throws DimensionMismatch if sizes differ.
    Vector& operator+=(const Vector& rhs);
    /// @brief In-place vector subtraction.
    /// @param rhs Vector of matching size.
    /// @return Reference to this vector.
    /// @throws DimensionMismatch if sizes differ.
    Vector& operator-=(const Vector& rhs);
    /// @brief In-place scalar multiplication.
    /// @param scalar Scalar multiplier.
    /// @return Reference to this vector.
    Vector& operator*=(const T& scalar);
    /// @brief In-place scalar division.
    /// @param scalar Scalar divisor.
    /// @return Reference to this vector.
    Vector& operator/=(const T& scalar);

    /// @brief Exact equality comparison.
    /// @param rhs Vector to compare against.
    /// @return True if same size and all elements are exactly equal.
    bool operator==(const Vector& rhs) const;
    /// @brief Exact inequality comparison.
    /// @param rhs Vector to compare against.
    /// @return True if the vectors differ in size or any element.
    bool operator!=(const Vector& rhs) const;

    /// @brief Scalar multiplication (mirror of operator* for readability).
    /// @param scalar Scalar multiplier.
    /// @return The scaled vector.
    Vector scaledBy(const T& scalar) const;
    /// @brief Elementwise (Hadamard) product.
    /// @param rhs Vector of matching size.
    /// @return The elementwise product.
    /// @throws DimensionMismatch if sizes differ.
    Vector elementwiseProduct(const Vector& rhs) const;
    /// @brief Elementwise division.
    /// @param rhs Vector of matching size.
    /// @return The elementwise quotient.
    /// @throws DimensionMismatch if sizes differ.
    Vector elementwiseQuotient(const Vector& rhs) const;

    // --- products ---------------------------------------------------------
    // dot is the unconjugated bilinear form sum x_i y_i even for complex T;
    // hermitianDot conjugates *this* (the left argument): sum conj(x_i) y_i.
    // The two coincide for real scalars. Inner-product norms want
    // hermitianDot.
    /// @brief Unconjugated bilinear dot product, sum x_i * y_i.
    /// @param rhs Vector of matching size.
    /// @return The dot product.
    /// @throws DimensionMismatch if sizes differ.
    T         dot(const Vector& rhs) const;
    /// @brief Hermitian inner product, sum conj(x_i) * y_i (conjugates *this).
    /// @param rhs Vector of matching size.
    /// @return The Hermitian inner product.
    /// @throws DimensionMismatch if sizes differ.
    T         hermitianDot(const Vector& rhs) const;
    /// @brief Outer product x * y^H.
    /// @param rhs Right operand.
    /// @return A size() x rhs.size() matrix.
    Matrix<T> outer(const Vector& rhs) const;
    /// @brief Cross product (3-vectors only).
    /// @param rhs Right operand; must have size 3.
    /// @return The cross product vector.
    /// @throws DimensionMismatch if either vector is not size 3.
    Vector    cross(const Vector& rhs) const;
    /// @brief Fused scale-and-add, alpha * (*this) + y.
    /// @param alpha Scalar multiplier applied to this vector.
    /// @param y Vector added to the scaled result; must match size.
    /// @return The vector alpha * (*this) + y.
    /// @throws DimensionMismatch if sizes differ.
    Vector    axpy(const T& alpha, const Vector& y) const;

    // --- norms ------------------------------------------------------------
    // norm() is the Euclidean norm, computed with scaling so entries near
    // the overflow threshold do not square to infinity; squaredNorm() is
    // the unscaled sum of |x_i|^2 and may overflow where norm() does not.
    /// @brief Euclidean (2-)norm, computed with overflow-safe scaling.
    /// @return The Euclidean norm.
    Real norm() const;
    /// @brief Sum of squared magnitudes, sum |x_i|^2 (may overflow).
    /// @return The squared Euclidean norm.
    Real squaredNorm() const;
    /// @brief 1-norm: the sum of element magnitudes.
    /// @return The 1-norm.
    Real oneNorm() const;
    /// @brief Infinity-norm: the largest element magnitude.
    /// @return The infinity-norm.
    Real infinityNorm() const;
    /// @brief General p-norm, (sum |x_i|^p)^(1/p).
    /// @param p The norm order (p >= 1).
    /// @return The p-norm.
    Real pNorm(Real p) const;
    /// @brief Sum of all elements.
    /// @return The total.
    T    sum() const;
    /// @brief Product of all elements.
    /// @return The product.
    T    product() const;

    /// @brief Returns a unit vector in the same direction.
    /// @return This vector scaled to unit Euclidean norm.
    /// @throws LinalgError if this vector is zero.
    Vector normalized() const;
    /// @brief Scales this vector in place to unit Euclidean norm.
    /// @throws LinalgError if this vector is zero.
    void   normalize();

    // --- shape ------------------------------------------------------------
    /// @brief Extracts a contiguous sub-vector.
    /// @param start Index of the first element (0-based).
    /// @param count Number of elements to take.
    /// @return The requested segment.
    /// @throws IndexOutOfRange if start + count exceeds size().
    Vector    segment(Index start, Index count) const;
    /// @brief Extracts the leading elements.
    /// @param count Number of elements to take from the front.
    /// @return The first `count` elements.
    /// @throws IndexOutOfRange if count exceeds size().
    Vector    head(Index count) const;
    /// @brief Extracts the trailing elements.
    /// @param count Number of elements to take from the back.
    /// @return The last `count` elements.
    /// @throws IndexOutOfRange if count exceeds size().
    Vector    tail(Index count) const;
    /// @brief Concatenates another vector after this one.
    /// @param rhs Vector to append.
    /// @return The concatenated vector.
    Vector    concat(const Vector& rhs) const;
    /// @brief Returns the elements in reverse order.
    /// @return The reversed vector.
    Vector    reversed() const;
    /// @brief Views the vector as a single-column matrix.
    /// @return A size() x 1 matrix copy.
    Matrix<T> asColumnMatrix() const;
    /// @brief Views the vector as a single-row matrix.
    /// @return A 1 x size() matrix copy.
    Matrix<T> asRowMatrix() const;
    /// @brief Builds a diagonal matrix with this vector on the diagonal.
    /// @return A size() x size() diagonal matrix.
    Matrix<T> asDiagonalMatrix() const;

    /// @brief Resizes the vector, discarding all existing contents.
    /// @param size New length.
    void resize(Index size);
    /// @brief Resizes the vector, keeping overlapping elements and
    /// zero-filling any growth.
    /// @param size New length.
    void conservativeResize(Index size);
    /// @brief Sets every element to the given value.
    /// @param value Value written to each element.
    void fill(const T& value);
    /// @brief Sets every element to zero.
    void setZero();
    /// @brief Overwrites the vector with the basis vector e_axis.
    /// @param axis Index of the single nonzero entry (0-based).
    void setUnit(Index axis);
    /// @brief Swaps the contents of this vector with another.
    /// @param other Vector to swap with.
    void swap(Vector& other);

    // --- reductions -------------------------------------------------------
    // Ties resolve to the lowest index. max/minCoefficient compare by real
    // part for complex T (there is no natural total order on C).
    /// @brief Index of the element with the largest magnitude (ties: lowest index).
    /// @return The index of the maximum-magnitude element.
    Index maxAbsIndex() const;
    /// @brief Index of the element with the smallest magnitude (ties: lowest index).
    /// @return The index of the minimum-magnitude element.
    Index minAbsIndex() const;
    /// @brief Largest coefficient (by real part for complex T; ties: lowest index).
    /// @return The maximum coefficient.
    T     maxCoefficient() const;
    /// @brief Smallest coefficient (by real part for complex T; ties: lowest index).
    /// @return The minimum coefficient.
    T     minCoefficient() const;
    /// @brief Tests approximate elementwise equality with another vector.
    /// @param other Vector to compare against.
    /// @param tolerance Absolute per-element tolerance.
    /// @return True if same size and all elements agree within tolerance.
    bool  isApprox(const Vector& other, Real tolerance) const;
    /// @brief Reports whether any element is NaN.
    /// @return True if at least one element is NaN.
    bool  hasNaN() const;

    /// @brief Formats the vector as a text string, for logging.
    /// @param precision Number of digits shown per element.
    /// @return A human-readable string.
    std::string toString(int precision) const;

private:
    /// @brief Validates that index i lies within the vector.
    /// @param i Element index (0-based).
    /// @throws IndexOutOfRange if i >= size().
    void checkBounds(Index i) const;
    /// @brief Validates that another vector has the same size as this one.
    /// @param other Vector to compare size against.
    /// @throws DimensionMismatch if sizes differ.
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
