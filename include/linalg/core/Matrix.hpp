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
    /// @brief Constructs an empty 0 x 0 matrix.
    Matrix();
    /// @brief Constructs a matrix of the given shape with value-initialized
    /// (zero) elements.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    Matrix(Index rows, Index cols);
    /// @brief Constructs a matrix of the given shape with every element set
    /// to a fill value.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param fill Value assigned to every element.
    Matrix(Index rows, Index cols, const T& fill);
    /// @brief Constructs a matrix from a flat row-major list of elements.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param rowMajorData Elements in row-major order; size must equal rows*cols.
    Matrix(Index rows, Index cols, const std::vector<T>& rowMajorData);
    /// @brief Constructs a matrix from a braced list of rows.
    /// @param rows Nested initializer list; every inner list must have equal length.
    Matrix(std::initializer_list<std::initializer_list<T>> rows);
    /// @brief Copy constructor; deep-copies the elements of another matrix.
    /// @param other Matrix to copy.
    Matrix(const Matrix& other);
    /// @brief Move constructor; takes over another matrix's storage.
    /// @param other Matrix to move from; left empty afterward.
    Matrix(Matrix&& other) noexcept;
    /// @brief Deep-copies a view's elements into a fresh contiguous matrix.
    /// @param view Read-only view to copy.
    explicit Matrix(const ConstMatrixView<T>& view);
    /// @brief Destructor; releases the owned storage.
    ~Matrix();

    /// @brief Copy assignment; deep-copies the elements of another matrix.
    /// @param other Matrix to copy.
    /// @return Reference to this matrix.
    Matrix& operator=(const Matrix& other);
    /// @brief Move assignment; takes over another matrix's storage.
    /// @param other Matrix to move from; left empty afterward.
    /// @return Reference to this matrix.
    Matrix& operator=(Matrix&& other) noexcept;

    // --- named constructors ----------------------------------------------
    /// @brief Creates a matrix of zeros.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @return A rows x cols matrix with every element zero.
    static Matrix Zeros(Index rows, Index cols);
    /// @brief Creates a matrix of ones.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @return A rows x cols matrix with every element one.
    static Matrix Ones(Index rows, Index cols);
    /// @brief Creates a matrix with every element set to a constant.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param value Value assigned to every element.
    /// @return A rows x cols matrix filled with value.
    static Matrix Constant(Index rows, Index cols, const T& value);
    /// @brief Creates a square identity matrix.
    /// @param size Number of rows and columns.
    /// @return A size x size identity matrix.
    static Matrix Identity(Index size);
    /// @brief Creates a square diagonal matrix from a vector.
    /// @param values Diagonal entries; the result has this length on each side.
    /// @return A diagonal matrix with `values` on the main diagonal.
    static Matrix Diagonal(const Vector<T>& values);
    // The Random* constructors are deterministic for a given seed, so
    // tests are reproducible. RandomSymmetric is Hermitian for complex T;
    // RandomOrthogonal is unitary (a Q factor), so it is perfectly
    // conditioned. Hilbert (a(i,j) = 1/(i+j+1)) is the classic
    // ill-conditioned stress input; Vandermonde has a(i,j) = nodes[i]^j
    // for j = 0..degree.
    /// @brief Creates a matrix of pseudo-random entries, reproducible per seed.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param seed Seed determining the entries.
    /// @return A rows x cols random matrix.
    static Matrix Random(Index rows, Index cols, unsigned long seed);
    /// @brief Creates a random symmetric (Hermitian for complex T) matrix.
    /// @param size Number of rows and columns.
    /// @param seed Seed determining the entries.
    /// @return A size x size random symmetric/Hermitian matrix.
    static Matrix RandomSymmetric(Index size, unsigned long seed);
    /// @brief Creates a random orthogonal/unitary matrix (a Q factor).
    /// @param size Number of rows and columns.
    /// @param seed Seed determining the entries.
    /// @return A size x size perfectly conditioned orthogonal/unitary matrix.
    static Matrix RandomOrthogonal(Index size, unsigned long seed);
    /// @brief Creates a Hilbert matrix, a(i,j) = 1/(i+j+1); ill-conditioned.
    /// @param size Number of rows and columns.
    /// @return A size x size Hilbert matrix.
    static Matrix Hilbert(Index size);
    /// @brief Creates a Vandermonde matrix, a(i,j) = nodes[i]^j.
    /// @param nodes Node values, one per row.
    /// @param degree Highest power; the result has degree+1 columns.
    /// @return A Vandermonde matrix with the given nodes and degree.
    static Matrix Vandermonde(const Vector<T>& nodes, Index degree);
    /// @brief Assembles a matrix from column vectors.
    /// @param columns Columns of equal length, left to right.
    /// @return A matrix whose columns are the given vectors.
    static Matrix FromColumns(const std::vector<Vector<T>>& columns);
    /// @brief Assembles a matrix from row vectors.
    /// @param rows Rows of equal length, top to bottom.
    /// @return A matrix whose rows are the given vectors.
    static Matrix FromRows(const std::vector<Vector<T>>& rows);

    // --- element access ---------------------------------------------------
    // operator() is unchecked; at() throws IndexOutOfRange. data() exposes
    // the row-major buffer: element (i, j) is data()[i * cols() + j].
    /// @brief Unchecked mutable access to an element.
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @return Reference to element (i, j).
    T&       operator()(Index i, Index j);
    /// @brief Unchecked read-only access to an element.
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @return Const reference to element (i, j).
    const T& operator()(Index i, Index j) const;
    /// @brief Bounds-checked mutable access to an element.
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @return Reference to element (i, j).
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    T&       at(Index i, Index j);
    /// @brief Bounds-checked read-only access to an element.
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @return Const reference to element (i, j).
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    const T& at(Index i, Index j) const;

    /// @brief Direct mutable access to the row-major buffer; (i, j) is at
    /// data()[i * cols() + j].
    /// @return Pointer to the first element.
    T*       data();
    /// @brief Direct read-only access to the row-major buffer; (i, j) is at
    /// data()[i * cols() + j].
    /// @return Const pointer to the first element.
    const T* data() const;

    /// @brief Number of rows.
    /// @return The row count.
    Index rows() const;
    /// @brief Number of columns.
    /// @return The column count.
    Index cols() const;
    /// @brief Total number of elements (rows * cols).
    /// @return The element count.
    Index size() const;
    /// @brief Reports whether the matrix has no elements.
    /// @return True if rows() or cols() is zero.
    bool  isEmpty() const;
    /// @brief Reports whether the matrix is square.
    /// @return True if rows() equals cols().
    bool  isSquare() const;

    // --- views ------------------------------------------------------------
    // The *View members alias this matrix's storage: writes through them
    // are visible here, and they dangle after resize or destruction.
    // row/col/diagonal (below) return owning copies instead.
    /// @brief Creates a mutable view over the whole matrix.
    /// @return A view aliasing this matrix's storage.
    MatrixView<T>      view();
    /// @brief Creates a read-only view over the whole matrix.
    /// @return A read-only view aliasing this matrix's storage.
    ConstMatrixView<T> view() const;
    /// @brief Creates a mutable view over a rectangular block.
    /// @param i Row index of the block's top-left corner (0-based).
    /// @param j Column index of the block's top-left corner (0-based).
    /// @param numRows Number of rows in the block.
    /// @param numCols Number of columns in the block.
    /// @return A view onto the block, aliasing this matrix's storage.
    MatrixView<T>      block(Index i, Index j, Index numRows, Index numCols);
    /// @brief Creates a read-only view over a rectangular block.
    /// @param i Row index of the block's top-left corner (0-based).
    /// @param j Column index of the block's top-left corner (0-based).
    /// @param numRows Number of rows in the block.
    /// @param numCols Number of columns in the block.
    /// @return A read-only view onto the block, aliasing this matrix's storage.
    ConstMatrixView<T> block(Index i, Index j, Index numRows, Index numCols) const;
    /// @brief Creates a mutable view over a single row.
    /// @param i Row index (0-based).
    /// @return A 1 x cols view aliasing this matrix's storage.
    MatrixView<T>      rowView(Index i);
    /// @brief Creates a read-only view over a single row.
    /// @param i Row index (0-based).
    /// @return A read-only 1 x cols view aliasing this matrix's storage.
    ConstMatrixView<T> rowView(Index i) const;
    /// @brief Creates a mutable view over a single column.
    /// @param j Column index (0-based).
    /// @return A rows x 1 view aliasing this matrix's storage.
    MatrixView<T>      colView(Index j);
    /// @brief Creates a read-only view over a single column.
    /// @param j Column index (0-based).
    /// @return A read-only rows x 1 view aliasing this matrix's storage.
    ConstMatrixView<T> colView(Index j) const;

    /// @brief Copies a row into an owning vector.
    /// @param i Row index (0-based).
    /// @return A vector holding a copy of row i.
    Vector<T> row(Index i) const;
    /// @brief Copies a column into an owning vector.
    /// @param j Column index (0-based).
    /// @return A vector holding a copy of column j.
    Vector<T> col(Index j) const;
    /// @brief Copies the main diagonal into an owning vector.
    /// @return A vector holding the diagonal entries.
    Vector<T> diagonal() const;

    /// @brief Overwrites a row with the given values.
    /// @param i Row index (0-based).
    /// @param values Replacement values; length must equal cols().
    void setRow(Index i, const Vector<T>& values);
    /// @brief Overwrites a column with the given values.
    /// @param j Column index (0-based).
    /// @param values Replacement values; length must equal rows().
    void setCol(Index j, const Vector<T>& values);
    /// @brief Overwrites a block with the contents of another matrix.
    /// @param i Row index of the block's top-left corner (0-based).
    /// @param j Column index of the block's top-left corner (0-based).
    /// @param source Matrix copied into the block; defines the block shape.
    void setBlock(Index i, Index j, const Matrix& source);

    // --- arithmetic (members only) ----------------------------------------
    // Shape mismatches throw DimensionMismatch. operator== is exact
    // equality; use isApprox for floating-point comparison.
    /// @brief Matrix addition.
    /// @param rhs Matrix to add; must have this matrix's shape.
    /// @return The elementwise sum.
    /// @throws DimensionMismatch if shapes differ.
    Matrix operator+(const Matrix& rhs) const;
    /// @brief Matrix subtraction.
    /// @param rhs Matrix to subtract; must have this matrix's shape.
    /// @return The elementwise difference.
    /// @throws DimensionMismatch if shapes differ.
    Matrix operator-(const Matrix& rhs) const;
    /// @brief Matrix-matrix multiplication.
    /// @param rhs Right operand; its rows() must equal this matrix's cols().
    /// @return The product, of shape rows() x rhs.cols().
    /// @throws DimensionMismatch if the inner dimensions differ.
    Matrix operator*(const Matrix& rhs) const;
    /// @brief Matrix-vector multiplication.
    /// @param rhs Right operand; its size must equal this matrix's cols().
    /// @return The product vector, of length rows().
    /// @throws DimensionMismatch if the dimensions differ.
    Vector<T> operator*(const Vector<T>& rhs) const;
    /// @brief Multiplies every element by a scalar (matrix on the left).
    /// @param scalar Scalar multiplier.
    /// @return The scaled matrix.
    Matrix operator*(const T& scalar) const;
    /// @brief Divides every element by a scalar.
    /// @param scalar Scalar divisor.
    /// @return The scaled matrix.
    Matrix operator/(const T& scalar) const;
    /// @brief Unary negation.
    /// @return A matrix with every element negated.
    Matrix operator-() const;

    /// @brief In-place matrix addition.
    /// @param rhs Matrix to add; must have this matrix's shape.
    /// @return Reference to this matrix.
    /// @throws DimensionMismatch if shapes differ.
    Matrix& operator+=(const Matrix& rhs);
    /// @brief In-place matrix subtraction.
    /// @param rhs Matrix to subtract; must have this matrix's shape.
    /// @return Reference to this matrix.
    /// @throws DimensionMismatch if shapes differ.
    Matrix& operator-=(const Matrix& rhs);
    /// @brief In-place matrix-matrix multiplication.
    /// @param rhs Right operand; its rows() must equal this matrix's cols().
    /// @return Reference to this matrix.
    /// @throws DimensionMismatch if the inner dimensions differ.
    Matrix& operator*=(const Matrix& rhs);
    /// @brief In-place scalar multiplication.
    /// @param scalar Scalar multiplier.
    /// @return Reference to this matrix.
    Matrix& operator*=(const T& scalar);
    /// @brief In-place scalar division.
    /// @param scalar Scalar divisor.
    /// @return Reference to this matrix.
    Matrix& operator/=(const T& scalar);

    /// @brief Exact equality comparison.
    /// @param rhs Matrix to compare against.
    /// @return True if same shape and all elements are exactly equal.
    bool operator==(const Matrix& rhs) const;
    /// @brief Exact inequality comparison.
    /// @param rhs Matrix to compare against.
    /// @return True if the matrices differ in shape or any element.
    bool operator!=(const Matrix& rhs) const;

    /// @brief Scalar multiplication with the scalar on the left (scalar * A).
    /// @param scalar Scalar multiplier.
    /// @return The scaled matrix.
    Matrix scaledBy(const T& scalar) const;
    /// @brief Elementwise (Hadamard) product.
    /// @param rhs Matrix of matching shape.
    /// @return The elementwise product.
    /// @throws DimensionMismatch if shapes differ.
    Matrix elementwiseProduct(const Matrix& rhs) const;
    /// @brief Elementwise division.
    /// @param rhs Matrix of matching shape.
    /// @return The elementwise quotient.
    /// @throws DimensionMismatch if shapes differ.
    Matrix elementwiseQuotient(const Matrix& rhs) const;
    /// @brief Kronecker (tensor) product.
    /// @param rhs Right operand.
    /// @return The Kronecker product, of shape (rows*rhs.rows) x (cols*rhs.cols).
    Matrix kroneckerProduct(const Matrix& rhs) const;
    /// @brief Raises a square matrix to a non-negative integer power.
    /// @param exponent Power to raise to; exponent 0 yields the identity.
    /// @return This matrix raised to the given power.
    /// @throws DimensionMismatch if the matrix is not square.
    Matrix power(Index exponent) const;

    // --- shape manipulation ----------------------------------------------
    /// @brief Transpose.
    /// @return A new matrix that is the transpose of this one.
    Matrix transpose() const;
    /// @brief Conjugate (Hermitian) transpose; equals transpose() for real T.
    /// @return A new matrix that is the conjugate transpose of this one.
    Matrix conjugateTranspose() const;
    /// @brief Reinterprets the elements (row-major) with a new shape.
    /// @param rows New row count.
    /// @param cols New column count; rows*cols must equal size().
    /// @return A matrix with the new shape and the same elements.
    /// @throws DimensionMismatch if rows*cols != size().
    Matrix reshaped(Index rows, Index cols) const;
    /// @brief Concatenates another matrix to the right.
    /// @param rhs Matrix with the same number of rows.
    /// @return The horizontally stacked matrix.
    /// @throws DimensionMismatch if row counts differ.
    Matrix horizontalConcat(const Matrix& rhs) const;
    /// @brief Concatenates another matrix below.
    /// @param rhs Matrix with the same number of columns.
    /// @return The vertically stacked matrix.
    /// @throws DimensionMismatch if column counts differ.
    Matrix verticalConcat(const Matrix& rhs) const;
    /// @brief Returns a copy with one row removed.
    /// @param i Row index to drop (0-based).
    /// @return A matrix with one fewer row.
    /// @throws IndexOutOfRange if i >= rows().
    Matrix withoutRow(Index i) const;
    /// @brief Returns a copy with one column removed.
    /// @param j Column index to drop (0-based).
    /// @return A matrix with one fewer column.
    /// @throws IndexOutOfRange if j >= cols().
    Matrix withoutCol(Index j) const;

    /// @brief Transposes the matrix in place.
    void transposeInPlace();
    /// @brief Resizes to a new shape, discarding all existing contents.
    /// @param rows New row count.
    /// @param cols New column count.
    void resize(Index rows, Index cols);
    /// @brief Resizes to a new shape, keeping the overlapping top-left block
    /// and zero-filling any growth.
    /// @param rows New row count.
    /// @param cols New column count.
    void conservativeResize(Index rows, Index cols);
    /// @brief Swaps two rows in place.
    /// @param a First row index (0-based).
    /// @param b Second row index (0-based).
    void swapRows(Index a, Index b);
    /// @brief Swaps two columns in place.
    /// @param a First column index (0-based).
    /// @param b Second column index (0-based).
    void swapCols(Index a, Index b);
    /// @brief Sets every element to the given value.
    /// @param value Value written to each element.
    void fill(const T& value);
    /// @brief Sets every element to zero.
    void setZero();
    /// @brief Overwrites the matrix with the identity (square matrices only).
    void setIdentity();
    /// @brief Swaps the contents of this matrix with another.
    /// @param other Matrix to swap with.
    void swap(Matrix& other);

    // --- scalar summaries -------------------------------------------------
    // determinant factorizes internally (LU); spectralNorm, conditionNumber,
    // and rank go through the SVD, so they are O(n^3) conveniences — hold a
    // decomposition object instead when you need more than one query.
    /// @brief Sum of the main-diagonal entries.
    /// @return The trace.
    T    trace() const;
    /// @brief Sum of all elements.
    /// @return The total.
    T    sum() const;
    /// @brief Determinant of a square matrix, computed via LU factorization.
    /// @return The determinant.
    /// @throws DimensionMismatch if the matrix is not square.
    T    determinant() const;
    /// @brief 1-norm: the maximum absolute column sum.
    /// @return The 1-norm.
    Real oneNorm() const;
    /// @brief Infinity-norm: the maximum absolute row sum.
    /// @return The infinity-norm.
    Real infinityNorm() const;
    /// @brief Frobenius norm: the square root of the sum of squared magnitudes.
    /// @return The Frobenius norm.
    Real frobeniusNorm() const;
    /// @brief Max-norm: the largest element magnitude.
    /// @return The maximum |a(i,j)|.
    Real maxNorm() const;
    /// @brief Spectral norm: the largest singular value (via SVD).
    /// @return The spectral norm.
    Real spectralNorm() const;
    /// @brief Condition number in the 2-norm: sigma_max / sigma_min (via SVD).
    /// @return The condition number.
    Real conditionNumber() const;
    /// @brief Numerical rank: count of singular values above a relative cutoff.
    /// @param tolerance Relative cutoff; singular values above tolerance * sigma_max count.
    /// @return The numerical rank.
    Index rank(Real tolerance) const;

    // --- predicates -------------------------------------------------------
    // Entrywise comparisons against an absolute tolerance; pass 0 for exact
    // structure. isHermitian is the meaningful symmetry test for complex T.
    /// @brief Tests symmetry (A == A^T) within an absolute tolerance.
    /// @param tolerance Absolute per-element tolerance; 0 for exact.
    /// @return True if the matrix is symmetric within tolerance.
    bool isSymmetric(Real tolerance) const;
    /// @brief Tests Hermitian symmetry (A == A^H) within an absolute tolerance.
    /// @param tolerance Absolute per-element tolerance; 0 for exact.
    /// @return True if the matrix is Hermitian within tolerance.
    bool isHermitian(Real tolerance) const;
    /// @brief Tests whether off-diagonal entries are negligible.
    /// @param tolerance Absolute per-element tolerance; 0 for exact.
    /// @return True if the matrix is diagonal within tolerance.
    bool isDiagonal(Real tolerance) const;
    /// @brief Tests whether the matrix is triangular of the given kind.
    /// @param which Which triangle (upper or lower) is expected to be nonzero.
    /// @param tolerance Absolute per-element tolerance; 0 for exact.
    /// @return True if the matrix is triangular within tolerance.
    bool isTriangular(Triangle::Kind which, Real tolerance) const;
    /// @brief Tests orthogonality (A^H A == I) within an absolute tolerance.
    /// @param tolerance Absolute per-element tolerance; 0 for exact.
    /// @return True if the matrix is orthogonal/unitary within tolerance.
    bool isOrthogonal(Real tolerance) const;
    /// @brief Tests approximate elementwise equality with another matrix.
    /// @param other Matrix to compare against.
    /// @param tolerance Absolute per-element tolerance.
    /// @return True if same shape and all elements agree within tolerance.
    bool isApprox(const Matrix& other, Real tolerance) const;
    /// @brief Reports whether any element is NaN.
    /// @return True if at least one element is NaN.
    bool hasNaN() const;

    // --- derived matrices -------------------------------------------------
    /// @brief Matrix inverse, computed via LU factorization.
    /// @return The inverse matrix.
    /// @throws SingularMatrix if the matrix is singular.
    /// @throws DimensionMismatch if the matrix is not square.
    Matrix inverse() const;
    /// @brief Moore-Penrose pseudoinverse, computed via SVD.
    /// @param tolerance Relative singular-value cutoff, as in rank().
    /// @return The pseudoinverse.
    Matrix pseudoInverse(Real tolerance) const;
    /// @brief Extracts a triangular part, zero-filling the other triangle and
    /// keeping the diagonal.
    /// @param which Which triangle (upper or lower) to keep.
    /// @return The requested triangular matrix.
    Matrix triangularPart(Triangle::Kind which) const;
    /// @brief Symmetric part, (A + A^H) / 2.
    /// @return The symmetric (Hermitian) part.
    Matrix symmetricPart() const;
    /// @brief Skew-symmetric part, (A - A^H) / 2.
    /// @return The skew-symmetric (skew-Hermitian) part.
    Matrix skewSymmetricPart() const;

    // --- serialization ----------------------------------------------------
    /// @brief Formats the matrix as aligned rows of text, for logging.
    /// @param precision Number of digits shown per element.
    /// @return A human-readable multi-line string.
    std::string toString(int precision) const;
    /// @brief Formats the matrix as a MATLAB/Octave literal, e.g. "[a b; c d]".
    /// @return A string pasteable into MATLAB/Octave.
    std::string toMatlabLiteral() const;
    /// @brief Reads a matrix from a CSV file.
    /// @param path Filesystem path to the CSV file.
    /// @return The matrix parsed from the file.
    /// @throws LinalgError on I/O or parse failure.
    static Matrix FromCsv(const std::string& path);
    /// @brief Writes the matrix to a CSV file.
    /// @param path Filesystem path to write.
    /// @throws LinalgError on I/O failure.
    void         writeCsv(const std::string& path) const;

private:
    /// @brief Computes the flat storage offset for element (i, j).
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @return The row-major linear index i * cols() + j.
    Index linearIndex(Index i, Index j) const;
    /// @brief Validates that (i, j) lies within the matrix.
    /// @param i Row index (0-based).
    /// @param j Column index (0-based).
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    void  checkBounds(Index i, Index j) const;
    /// @brief Validates that another matrix has the same shape as this one.
    /// @param other Matrix to compare shape against.
    /// @throws DimensionMismatch if shapes differ.
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
