#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

#include "linalg/core/MatrixView.hpp"
#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"

namespace linalg {

template <typename T>
class Vector;

// Owning dense matrix, row-major, contiguous.
//
// Every operator is a member, so `matrix * scalar` exists but
// `scalar * matrix` does not; use scaledBy() for the reversed form.
template <typename T>
class Matrix {
public:
    using Scalar = T;
    using Real   = typename NumericTraits<T>::Real;
    using Index  = std::size_t;

    // --- construction -----------------------------------------------------
    Matrix();
    Matrix(Index rows, Index cols);
    Matrix(Index rows, Index cols, const T& fill);
    Matrix(Index rows, Index cols, const std::vector<T>& rowMajorData);
    Matrix(std::initializer_list<std::initializer_list<T>> rows);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    explicit Matrix(const ConstMatrixView<T>& view);
    ~Matrix();

    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;

    // --- named constructors ----------------------------------------------
    static Matrix Zeros(Index rows, Index cols);
    static Matrix Ones(Index rows, Index cols);
    static Matrix Constant(Index rows, Index cols, const T& value);
    static Matrix Identity(Index size);
    static Matrix Diagonal(const Vector<T>& values);
    // The Random* constructors are deterministic for a given seed, so
    // tests are reproducible. RandomSymmetric is Hermitian for complex T;
    // RandomOrthogonal is unitary (a Q factor), so it is perfectly
    // conditioned. Hilbert (a(i,j) = 1/(i+j+1)) is the classic
    // ill-conditioned stress input; Vandermonde has a(i,j) = nodes[i]^j
    // for j = 0..degree.
    static Matrix Random(Index rows, Index cols, unsigned long seed);
    static Matrix RandomSymmetric(Index size, unsigned long seed);
    static Matrix RandomOrthogonal(Index size, unsigned long seed);
    static Matrix Hilbert(Index size);
    static Matrix Vandermonde(const Vector<T>& nodes, Index degree);
    static Matrix FromColumns(const std::vector<Vector<T>>& columns);
    static Matrix FromRows(const std::vector<Vector<T>>& rows);

    // --- element access ---------------------------------------------------
    // operator() is unchecked; at() throws IndexOutOfRange. data() exposes
    // the row-major buffer: element (i, j) is data()[i * cols() + j].
    T&       operator()(Index i, Index j);
    const T& operator()(Index i, Index j) const;
    T&       at(Index i, Index j);
    const T& at(Index i, Index j) const;

    T*       data();
    const T* data() const;

    Index rows() const;
    Index cols() const;
    Index size() const;
    bool  isEmpty() const;
    bool  isSquare() const;

    // --- views ------------------------------------------------------------
    // The *View members alias this matrix's storage: writes through them
    // are visible here, and they dangle after resize or destruction.
    // row/col/diagonal (below) return owning copies instead.
    MatrixView<T>      view();
    ConstMatrixView<T> view() const;
    MatrixView<T>      block(Index i, Index j, Index numRows, Index numCols);
    ConstMatrixView<T> block(Index i, Index j, Index numRows, Index numCols) const;
    MatrixView<T>      rowView(Index i);
    ConstMatrixView<T> rowView(Index i) const;
    MatrixView<T>      colView(Index j);
    ConstMatrixView<T> colView(Index j) const;

    Vector<T> row(Index i) const;
    Vector<T> col(Index j) const;
    Vector<T> diagonal() const;

    void setRow(Index i, const Vector<T>& values);
    void setCol(Index j, const Vector<T>& values);
    void setBlock(Index i, Index j, const Matrix& source);

    // --- arithmetic (members only) ----------------------------------------
    // Shape mismatches throw DimensionMismatch. operator== is exact
    // equality; use isApprox for floating-point comparison.
    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;
    Vector<T> operator*(const Vector<T>& rhs) const;
    Matrix operator*(const T& scalar) const;
    Matrix operator/(const T& scalar) const;
    Matrix operator-() const;

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator*=(const Matrix& rhs);
    Matrix& operator*=(const T& scalar);
    Matrix& operator/=(const T& scalar);

    bool operator==(const Matrix& rhs) const;
    bool operator!=(const Matrix& rhs) const;

    Matrix scaledBy(const T& scalar) const;       // scalar * (*this)
    Matrix elementwiseProduct(const Matrix& rhs) const;
    Matrix elementwiseQuotient(const Matrix& rhs) const;
    Matrix kroneckerProduct(const Matrix& rhs) const;
    Matrix power(Index exponent) const;           // square only; A^0 == I

    // --- shape manipulation ----------------------------------------------
    Matrix transpose() const;
    Matrix conjugateTranspose() const;            // equals transpose() for real T
    Matrix reshaped(Index rows, Index cols) const; // row-major reread; rows*cols must equal size()
    Matrix horizontalConcat(const Matrix& rhs) const;
    Matrix verticalConcat(const Matrix& rhs) const;
    Matrix withoutRow(Index i) const;
    Matrix withoutCol(Index j) const;

    void transposeInPlace();
    void resize(Index rows, Index cols);             // discards contents
    void conservativeResize(Index rows, Index cols); // keeps top-left block, zero-fills growth
    void swapRows(Index a, Index b);
    void swapCols(Index a, Index b);
    void fill(const T& value);
    void setZero();
    void setIdentity();
    void swap(Matrix& other);

    // --- scalar summaries -------------------------------------------------
    // determinant factorizes internally (LU); spectralNorm, conditionNumber,
    // and rank go through the SVD, so they are O(n^3) conveniences — hold a
    // decomposition object instead when you need more than one query.
    T    trace() const;
    T    sum() const;
    T    determinant() const;      // square only
    Real oneNorm() const;          // max absolute column sum
    Real infinityNorm() const;     // max absolute row sum
    Real frobeniusNorm() const;
    Real maxNorm() const;          // largest |a(i,j)|
    Real spectralNorm() const;     // largest singular value
    Real conditionNumber() const;  // sigma_max / sigma_min
    Index rank(Real tolerance) const; // singular values above tolerance * sigma_max

    // --- predicates -------------------------------------------------------
    // Entrywise comparisons against an absolute tolerance; pass 0 for exact
    // structure. isHermitian is the meaningful symmetry test for complex T.
    bool isSymmetric(Real tolerance) const;
    bool isHermitian(Real tolerance) const;
    bool isDiagonal(Real tolerance) const;
    bool isTriangular(Triangle::Kind which, Real tolerance) const;
    bool isOrthogonal(Real tolerance) const;
    bool isApprox(const Matrix& other, Real tolerance) const;
    bool hasNaN() const;

    // --- derived matrices -------------------------------------------------
    Matrix inverse() const;                      // via LU; SingularMatrix on failure
    Matrix pseudoInverse(Real tolerance) const;  // Moore-Penrose via SVD; tolerance as in rank()
    Matrix triangularPart(Triangle::Kind which) const; // other triangle zero-filled, diagonal kept
    Matrix symmetricPart() const;                // (A + A^H) / 2
    Matrix skewSymmetricPart() const;            // (A - A^H) / 2

    // --- serialization ----------------------------------------------------
    std::string toString(int precision) const;   // aligned rows, for logging
    std::string toMatlabLiteral() const;         // "[a b; c d]", pasteable into MATLAB/Octave
    static Matrix FromCsv(const std::string& path); // throws LinalgError on I/O or parse failure
    void         writeCsv(const std::string& path) const;

private:
    Index linearIndex(Index i, Index j) const;
    void  checkBounds(Index i, Index j) const;
    void  checkSameShape(const Matrix& other) const;

    Index          rows_;
    Index          cols_;
    std::vector<T> storage_;
};


// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR) \
    extern template class Matrix<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
