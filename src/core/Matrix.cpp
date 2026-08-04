#include "linalg/core/Matrix.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/MatrixView.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

#include <initializer_list>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace linalg {

template <typename T>
Matrix<T>::Matrix()
    : rows_(0),
      cols_(0),
      storage_() {}

template <typename T>
Matrix<T>::Matrix(Index rows,
                  Index cols)
    : rows_(rows),
      cols_(cols),
      storage_(rows_ * cols_) // value-initialized (zero) elements
{}

template <typename T>
Matrix<T>::Matrix(Index    rows,
                  Index    cols,
                  const T &fill)
    : rows_(rows),
      cols_(cols),
      storage_(rows_ * cols_,
               fill) // check
{}

template <typename T>
Matrix<T>::Matrix(Index                 rows,
                  Index                 cols,
                  const std::vector<T> &rowMajorData)
    : rows_(rows),
      cols_(cols),
      storage_(rowMajorData) {}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> rows)
    : rows_(rows.size()),
      cols_(rows.size() ? rows.begin()->size() : 0),
      storage_() {
    storage_.reserve(rows_ * cols_);
    for (const std::initializer_list<T> &row : rows) {
        // inefficient catch if(row.size() != cols_){throw LinalgError("ragged
        // rows: initializer-list linalg::Matrix<T> ctor");}
        storage_.insert(storage_.end(), row.begin(), row.end());
    }
}
template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : rows_(other.rows_),
      cols_(other.cols_),
      storage_(other.storage_) {}

template <typename T>
Matrix<T>::Matrix(Matrix &&other) noexcept
    : rows_(other.rows_),
      cols_(other.cols_),
      storage_(std::move(other.storage_)) {
    other.rows_ = 0;
    other.cols_ = 0;
}

template <typename T>
Matrix<T>::Matrix(const ConstMatrixView<T> &view)
    : Matrix(view.toMatrix()) {}

template <typename T> Matrix<T>::~Matrix() {}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &other) {
    if (this != &other) {
        std::vector<T> incoming(other.storage_);
        storage_ = std::move(incoming);
        rows_    = other.rows_;
        cols_    = other.cols_;
    }
    return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix &&other) noexcept {
    if (this != &other) {
        rows_    = other.rows_;
        cols_    = other.cols_;
        storage_ = std::move(other.storage_);

        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::Zeros(Index rows,
                           Index cols) {
    return Matrix(rows, cols);
}

template <typename T>
Matrix<T> Matrix<T>::Ones(Index rows,
                          Index cols) {
    return Matrix(rows, cols, T(1)); // alt
}

template <typename T>
Matrix<T> Matrix<T>::Constant(Index    rows,
                              Index    cols,
                              const T &value) {
    return Matrix(rows, cols, T(value));
}

template <typename T> Matrix<T> Matrix<T>::Identity(Index size) {
    Matrix      result(size, size);
    const Index stride = size + 1;
    for (Index k = 0; k < size * size; k += stride) {
        result.storage_[k] = T(1);
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::Diagonal(const Vector<T> &values) {
    const Index size = values.size();
    Matrix      result(size, size);
    for (Index i = 0; i < size; ++i) {
        result(i, i) = values(i);
    }
    return result;
}
template <typename T>
Matrix<T> Matrix<T>::Random(Index         rows,
                            Index         cols,
                            unsigned long seed) {
    Matrix                               result(rows, cols);
    std::mt19937_64                      gen(seed);
    std::uniform_real_distribution<Real> dist(Real(-1), Real(1));
    for (Index k = 0; k < rows * cols; ++k) {
        if constexpr (IsComplex<T>::value) {
            result.storage_[k] = T(dist(gen), dist(gen));
        } else {
            result.storage_[k] = dist(gen);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::RandomSymmetric(Index         size,
                                     unsigned long seed) {
    Matrix                               result(size, size);
    std::mt19937_64                      gen(seed);
    std::uniform_real_distribution<Real> dist(Real(-1), Real(1));
    for (Index i = 0; i < size; ++i) {
        // A Hermitian diagonal is real; an imaginary part here would make
        // A^H != A and fail isHermitian for complex T.
        if constexpr (IsComplex<T>::value) {
            result(i, i) = T(dist(gen), Real(0));
        } else {
            result(i, i) = dist(gen);
        }
        // Mirror with conj so the symmetry is exact rather than the
        // rounded result of averaging A with A^H.
        for (Index j = i + 1; j < size; ++j) {
            T value;
            if constexpr (IsComplex<T>::value) {
                value = T(dist(gen), dist(gen));
            } else {
                value = dist(gen);
            }
            result(i, j) = value;
            result(j, i) = NumericTraits<T>::conj(value);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::RandomOrthogonal(Index         size,
                                      unsigned long seed) {
    throw LinalgError("not implemented: linalg::Matrix<T>::RandomOrthogonal");
}

template <typename T> Matrix<T> Matrix<T>::Hilbert(Index size) {
    Matrix result(size, size);
    for (Index i = 0; i < size; ++i) {
        for (Index j = 0; j < size; ++j) {
            result(i, j) = T(1) / T(static_cast<Real>(i + j + 1));
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::Vandermonde(const Vector<T> &nodes,
                                 Index            degree) {
    const Index rows = nodes.size();
    const Index cols = degree + 1;
    Matrix      result(rows, cols);
    for (Index i = 0; i < rows; ++i) {
        // Running product rather than pow: exact for integer powers, and
        // needs no complex overload. Gives nodes[i]^0 == 1 for free.
        T power = T(1);
        for (Index j = 0; j < cols; ++j) {
            result(i, j) = power;
            power *= nodes(i);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::FromColumns(const std::vector<Vector<T>> &columns) {
    const Index cols = columns.size();
    const Index rows = cols ? columns[0].size() : 0;
    for (Index j = 1; j < cols; ++j) {
        if (columns[j].size() != rows) {
            throw DimensionMismatch(rows, 1, columns[j].size(), 1);
        }
    }
    Matrix result(rows, cols);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            result(i, j) = columns[j](i);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::FromRows(const std::vector<Vector<T>> &rows) {
    const Index numRows = rows.size();
    const Index numCols = numRows ? rows[0].size() : 0;
    for (Index i = 1; i < numRows; ++i) {
        if (rows[i].size() != numCols) {
            throw DimensionMismatch(1, numCols, 1, rows[i].size());
        }
    }
    Matrix result(numRows, numCols);
    for (Index i = 0; i < numRows; ++i) {
        for (Index j = 0; j < numCols; ++j) {
            result(i, j) = rows[i](j);
        }
    }
    return result;
}

// Element Access

template <typename T>
T &Matrix<T>::operator()(Index i,
                         Index j) {
    return storage_[linearIndex(i, j)];
}

template <typename T>
const T &Matrix<T>::operator()(Index i,
                               Index j) const {
    return storage_[linearIndex(i, j)];
}

template <typename T>
T &Matrix<T>::at(Index i,
                 Index j) {
    checkBounds(i, j);
    return storage_[linearIndex(i, j)];
}

template <typename T>
const T &Matrix<T>::at(Index i,
                       Index j) const {
    checkBounds(i, j);
    return storage_[linearIndex(i, j)];
}

template <typename T> T *Matrix<T>::data() { return storage_.data(); }

template <typename T> const T *Matrix<T>::data() const {
    return storage_.data();
}

template <typename T> typename Matrix<T>::Index Matrix<T>::rows() const {
    return rows_;
}

template <typename T> typename Matrix<T>::Index Matrix<T>::cols() const {
    return cols_;
}

template <typename T> typename Matrix<T>::Index Matrix<T>::size() const {
    return rows_ * cols_;
}

template <typename T> bool Matrix<T>::isEmpty() const {
    return rows_ == 0 || cols_ == 0;
}

template <typename T> bool Matrix<T>::isSquare() const {
    return rows_ == cols_;
}

// views

// Only view() touches the raw stride constructor: the storage is row-major,
// so moving down a row advances cols_ elements and moving right advances 1.
// Everything else delegates to MatrixView, which already rebases the
// pointer, keeps the parent's strides so the sub-view aliases rather than
// copies, and range-checks the block.

template <typename T> MatrixView<T> Matrix<T>::view() {
    return MatrixView<T>(storage_.data(), rows_, cols_, cols_, 1);
}

template <typename T> ConstMatrixView<T> Matrix<T>::view() const {
    return ConstMatrixView<T>(storage_.data(), rows_, cols_, cols_, 1);
}

template <typename T>
MatrixView<T> Matrix<T>::block(Index i,
                               Index j,
                               Index numRows,
                               Index numCols) {
    return view().block(i, j, numRows, numCols);
}

template <typename T>
ConstMatrixView<T> Matrix<T>::block(Index i,
                                    Index j,
                                    Index numRows,
                                    Index numCols) const {
    return view().block(i, j, numRows, numCols);
}

template <typename T> MatrixView<T> Matrix<T>::rowView(Index i) {
    return view().row(i);
}

template <typename T> ConstMatrixView<T> Matrix<T>::rowView(Index i) const {
    return view().row(i);
}

template <typename T> MatrixView<T> Matrix<T>::colView(Index j) {
    return view().col(j);
}

template <typename T> ConstMatrixView<T> Matrix<T>::colView(Index j) const {
    return view().col(j);
}

template <typename T> Vector<T> Matrix<T>::row(Index i) const {
    const ConstMatrixView<T> source = rowView(i);
    Vector<T>                result(cols_);
    for (Index j = 0; j < cols_; ++j) {
        result(j) = source(0, j);
    }
    return result;
}

template <typename T> Vector<T> Matrix<T>::col(Index j) const {
    const ConstMatrixView<T> source = colView(j);
    Vector<T>                result(rows_);
    for (Index i = 0; i < rows_; ++i) {
        result(i) = source(i, 0);
    }
    return result;
}

template <typename T> Vector<T> Matrix<T>::diagonal() const {
    // Rectangular matrices have a main diagonal of min(rows, cols).
    const ConstMatrixView<T> source = view().diagonal();
    Vector<T>                result(source.cols());
    for (Index k = 0; k < source.cols(); ++k) {
        result(k) = source(0, k);
    }
    return result;
}

template <typename T>
void Matrix<T>::setRow(Index            i,
                       const Vector<T> &values) {
    if (values.size() != cols_) {
        throw DimensionMismatch(1, cols_, 1, values.size());
    }
    MatrixView<T> target = rowView(i);
    for (Index j = 0; j < cols_; ++j) {
        target(0, j) = values(j);
    }
}

template <typename T>
void Matrix<T>::setCol(Index            j,
                       const Vector<T> &values) {
    if (values.size() != rows_) {
        throw DimensionMismatch(rows_, 1, values.size(), 1);
    }
    MatrixView<T> target = colView(j);
    for (Index i = 0; i < rows_; ++i) {
        target(i, 0) = values(i);
    }
}

template <typename T>
void Matrix<T>::setBlock(Index         i,
                         Index         j,
                         const Matrix &source) {
    // block() range-checks the destination; source defines the shape.
    MatrixView<T> target = block(i, j, source.rows_, source.cols_);
    for (Index r = 0; r < source.rows_; ++r) {
        for (Index c = 0; c < source.cols_; ++c) {
            target(r, c) = source(r, c);
        }
    }
}

// arithmetic
template <typename T> Matrix<T> Matrix<T>::operator+(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator+");
}

template <typename T> Matrix<T> Matrix<T>::operator-(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator-");
}

template <typename T> Matrix<T> Matrix<T>::operator*(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator*");
}

template <typename T>
Vector<T> Matrix<T>::operator*(const Vector<T> &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator*");
}

template <typename T> Matrix<T> Matrix<T>::operator*(const T &scalar) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator*");
}

template <typename T> Matrix<T> Matrix<T>::operator/(const T &scalar) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator/");
}

template <typename T> Matrix<T> Matrix<T>::operator-() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator-");
}

template <typename T> Matrix<T> &Matrix<T>::operator+=(const Matrix &rhs) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator+=");
}

template <typename T> Matrix<T> &Matrix<T>::operator-=(const Matrix &rhs) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator-=");
}

template <typename T> Matrix<T> &Matrix<T>::operator*=(const Matrix &rhs) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator*=");
}

template <typename T> Matrix<T> &Matrix<T>::operator*=(const T &scalar) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator*=");
}

template <typename T> Matrix<T> &Matrix<T>::operator/=(const T &scalar) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator/=");
}

template <typename T> bool Matrix<T>::operator==(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator==");
}

template <typename T> bool Matrix<T>::operator!=(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator!=");
}

template <typename T> Matrix<T> Matrix<T>::scaledBy(const T &scalar) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::scaledBy");
}

template <typename T>
Matrix<T> Matrix<T>::elementwiseProduct(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::elementwiseProduct");
}

template <typename T>
Matrix<T> Matrix<T>::elementwiseQuotient(const Matrix &rhs) const {
    throw LinalgError(
        "not implemented: linalg::Matrix<T>::elementwiseQuotient");
}

template <typename T>
Matrix<T> Matrix<T>::kroneckerProduct(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::kroneckerProduct");
}

template <typename T> Matrix<T> Matrix<T>::power(Index exponent) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::power");
}

// element manip
template <typename T> Matrix<T> Matrix<T>::transpose() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::transpose");
}

template <typename T> Matrix<T> Matrix<T>::conjugateTranspose() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::conjugateTranspose");
}

template <typename T>
Matrix<T> Matrix<T>::reshaped(Index rows,
                              Index cols) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::reshaped");
}

template <typename T>
Matrix<T> Matrix<T>::horizontalConcat(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::horizontalConcat");
}

template <typename T>
Matrix<T> Matrix<T>::verticalConcat(const Matrix &rhs) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::verticalConcat");
}

template <typename T> Matrix<T> Matrix<T>::withoutRow(Index i) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::withoutRow");
}

template <typename T> Matrix<T> Matrix<T>::withoutCol(Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::withoutCol");
}

template <typename T> void Matrix<T>::transposeInPlace() {
    throw LinalgError("not implemented: linalg::Matrix<T>::transposeInPlace");
}

template <typename T>
void Matrix<T>::resize(Index rows,
                       Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::resize");
}

template <typename T>
void Matrix<T>::conservativeResize(Index rows,
                                   Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::conservativeResize");
}

template <typename T>
void Matrix<T>::swapRows(Index a,
                         Index b) {
    throw LinalgError("not implemented: linalg::Matrix<T>::swapRows");
}

template <typename T>
void Matrix<T>::swapCols(Index a,
                         Index b) {
    throw LinalgError("not implemented: linalg::Matrix<T>::swapCols");
}

template <typename T> void Matrix<T>::fill(const T &value) {
    throw LinalgError("not implemented: linalg::Matrix<T>::fill");
}

template <typename T> void Matrix<T>::setZero() {
    throw LinalgError("not implemented: linalg::Matrix<T>::setZero");
}

template <typename T> void Matrix<T>::setIdentity() {
    throw LinalgError("not implemented: linalg::Matrix<T>::setIdentity");
}

template <typename T> void Matrix<T>::swap(Matrix &other) {
    throw LinalgError("not implemented: linalg::Matrix<T>::swap");
}

// scalar summaries
template <typename T> T Matrix<T>::trace() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::trace");
}

template <typename T> T Matrix<T>::sum() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::sum");
}

template <typename T> T Matrix<T>::determinant() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::determinant");
}

template <typename T> typename Matrix<T>::Real Matrix<T>::oneNorm() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::oneNorm");
}

template <typename T> typename Matrix<T>::Real Matrix<T>::infinityNorm() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::infinityNorm");
}

template <typename T>
typename Matrix<T>::Real Matrix<T>::frobeniusNorm() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::frobeniusNorm");
}

template <typename T> typename Matrix<T>::Real Matrix<T>::maxNorm() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::maxNorm");
}

template <typename T> typename Matrix<T>::Real Matrix<T>::spectralNorm() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::spectralNorm");
}

template <typename T>
typename Matrix<T>::Real Matrix<T>::conditionNumber() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::conditionNumber");
}

template <typename T>
typename Matrix<T>::Index Matrix<T>::rank(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::rank");
}

// predicates
template <typename T> bool Matrix<T>::isSymmetric(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isSymmetric");
}

template <typename T> bool Matrix<T>::isHermitian(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isHermitian");
}

template <typename T> bool Matrix<T>::isDiagonal(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isDiagonal");
}

template <typename T>
bool Matrix<T>::isTriangular(Triangle::Kind which,
                             Real           tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isTriangular");
}

template <typename T> bool Matrix<T>::isOrthogonal(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isOrthogonal");
}

template <typename T>
bool Matrix<T>::isApprox(const Matrix &other,
                         Real          tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isApprox");
}

template <typename T> bool Matrix<T>::hasNaN() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::hasNaN");
}

// derived matrices
template <typename T> Matrix<T> Matrix<T>::inverse() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::inverse");
}

template <typename T> Matrix<T> Matrix<T>::pseudoInverse(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::pseudoInverse");
}

template <typename T>
Matrix<T> Matrix<T>::triangularPart(Triangle::Kind which) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::triangularPart");
}

template <typename T> Matrix<T> Matrix<T>::symmetricPart() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::symmetricPart");
}

template <typename T> Matrix<T> Matrix<T>::skewSymmetricPart() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::skewSymmetricPart");
}

// serialization
template <typename T> std::string Matrix<T>::toString(int precision) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::toString");
}

template <typename T> std::string Matrix<T>::toMatlabLiteral() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::toMatlabLiteral");
}

template <typename T> Matrix<T> Matrix<T>::FromCsv(const std::string &path) {
    throw LinalgError("not implemented: linalg::Matrix<T>::FromCsv");
}

template <typename T> void Matrix<T>::writeCsv(const std::string &path) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::writeCsv");
}

// privates
template <typename T>
typename Matrix<T>::Index Matrix<T>::linearIndex(Index i,
                                                 Index j) const {
    return i * cols_ + j;
}

template <typename T>
void Matrix<T>::checkBounds(Index i,
                            Index j) const {
    // Reported one axis at a time: IndexOutOfRange carries a single
    // index/bound pair, so the row is checked before the column.
    if (i >= rows_) {
        throw IndexOutOfRange(i, rows_);
    }
    if (j >= cols_) {
        throw IndexOutOfRange(j, cols_);
    }
}

template <typename T>
void Matrix<T>::checkSameShape(const Matrix &other) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::checkSameShape");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Matrix<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
