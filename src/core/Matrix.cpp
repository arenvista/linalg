#include "linalg/core/Matrix.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/MatrixView.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Kernels.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <random>
#include <sstream>
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
    checkSameShape(rhs);
    Matrix result(*this);
    Kernels<T>::axpy(size(), T(1), rhs.storage_.data(), 1,
                     result.storage_.data(), 1);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator-(const Matrix &rhs) const {
    checkSameShape(rhs);
    Matrix result(*this);
    Kernels<T>::axpy(size(), T(-1), rhs.storage_.data(), 1,
                     result.storage_.data(), 1);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const Matrix &rhs) const {
    if (cols_ != rhs.rows_) {
        throw DimensionMismatch(rows_, cols_, rhs.rows_, rhs.cols_);
    }
    // beta == 0 makes the output write-only, so the constructor's zero fill
    // is not depended on; result is fresh, so it cannot alias either operand.
    Matrix result(rows_, rhs.cols_);
    Kernels<T>::gemm(Transposition::Kind::None, Transposition::Kind::None, T(1),
                     view(), rhs.view(), T{}, result.view());
    return result;
}

template <typename T>
Vector<T> Matrix<T>::operator*(const Vector<T> &rhs) const {
    if (cols_ != rhs.size()) {
        throw DimensionMismatch(rows_, cols_, rhs.size(), 1);
    }
    Vector<T> result(rows_);
    Kernels<T>::gemv(Transposition::Kind::None, T(1), view(), rhs.data(), 1,
                     T{}, result.data(), 1);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator*(const T &scalar) const {
    Matrix result(*this);
    Kernels<T>::scal(result.size(), scalar, result.storage_.data(), 1);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator/(const T &scalar) const {
    // Divides rather than scaling by 1/scalar: the reciprocal rounds twice
    // and overflows for subnormal divisors.
    Matrix result(rows_, cols_);
    for (Index k = 0; k < size(); ++k) {
        result.storage_[k] = storage_[k] / scalar;
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::operator-() const {
    Matrix result(*this);
    Kernels<T>::scal(result.size(), T(-1), result.storage_.data(), 1);
    return result;
}

template <typename T> Matrix<T> &Matrix<T>::operator+=(const Matrix &rhs) {
    checkSameShape(rhs);
    Kernels<T>::axpy(size(), T(1), rhs.storage_.data(), 1, storage_.data(), 1);
    return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator-=(const Matrix &rhs) {
    checkSameShape(rhs);
    Kernels<T>::axpy(size(), T(-1), rhs.storage_.data(), 1, storage_.data(), 1);
    return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator*=(const Matrix &rhs) {
    // gemm forbids the output aliasing an input, and the shape changes when
    // rhs is not square, so this cannot accumulate in place.
    *this = (*this) * rhs;
    return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator*=(const T &scalar) {
    Kernels<T>::scal(size(), scalar, storage_.data(), 1);
    return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator/=(const T &scalar) {
    for (Index k = 0; k < size(); ++k) {
        storage_[k] /= scalar;
    }
    return *this;
}

template <typename T> bool Matrix<T>::operator==(const Matrix &rhs) const {
    // Exact equality, shape included; isApprox is the tolerant comparison.
    return rows_ == rhs.rows_ && cols_ == rhs.cols_ && storage_ == rhs.storage_;
}

template <typename T> bool Matrix<T>::operator!=(const Matrix &rhs) const {
    return !(*this == rhs);
}

template <typename T> Matrix<T> Matrix<T>::scaledBy(const T &scalar) const {
    return (*this) * scalar;
}

template <typename T>
Matrix<T> Matrix<T>::elementwiseProduct(const Matrix &rhs) const {
    checkSameShape(rhs);
    Matrix result(rows_, cols_);
    for (Index k = 0; k < size(); ++k) {
        result.storage_[k] = storage_[k] * rhs.storage_[k];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::elementwiseQuotient(const Matrix &rhs) const {
    checkSameShape(rhs);
    Matrix result(rows_, cols_);
    for (Index k = 0; k < size(); ++k) {
        result.storage_[k] = storage_[k] / rhs.storage_[k];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::kroneckerProduct(const Matrix &rhs) const {

    Matrix result(rows_ * rhs.rows_, cols_ * rhs.cols_);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            const T factor = (*this)(i, j);
            for (Index r = 0; r < rhs.rows_; ++r) {
                for (Index c = 0; c < rhs.cols_; ++c) {
                    result(i * rhs.rows_ + r, j * rhs.cols_ + c) =
                        factor * rhs(r, c);
                }
            }
        }
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::power(Index exponent) const {
    if (!isSquare()) {
        throw DimensionMismatch(rows_, cols_, cols_, rows_);
    }
    // Binary exponentiation: ~log2(exponent) products rather than exponent
    // of them. Exponent 0 gives the identity, including for a 0 x 0 matrix.
    Matrix result = Identity(rows_);
    Matrix base(*this);
    for (Index e = exponent; e != 0; e >>= 1) {
        if (e & 1) {
            result *= base;
        }
        base *= base;
    }
    return result;
}

// element manip
template <typename T> Matrix<T> Matrix<T>::transpose() const {
    // transposed() swaps shape and strides without copying; toMatrix then
    // walks that view into fresh contiguous row-major storage.
    return view().transposed().toMatrix();
}

template <typename T> Matrix<T> Matrix<T>::conjugateTranspose() const {
    // Not transpose() + a conj pass: one traversal, and conj is the
    // identity for real T, so this collapses to a plain transpose there.
    Matrix result(cols_, rows_);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            result(j, i) = NumericTraits<T>::conj((*this)(i, j));
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::reshaped(Index rows,
                              Index cols) const {
    if (rows * cols != size()) {
        throw DimensionMismatch(rows_, cols_, rows, cols);
    }

    Matrix result(*this);
    result.rows_ = rows;
    result.cols_ = cols;
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::horizontalConcat(const Matrix &rhs) const {
    if (rows_ != rhs.rows_) {
        throw DimensionMismatch(rows_, cols_, rhs.rows_, rhs.cols_);
    }
    Matrix result(rows_, cols_ + rhs.cols_);
    result.setBlock(0, 0, *this);
    result.setBlock(0, cols_, rhs);
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::verticalConcat(const Matrix &rhs) const {
    if (cols_ != rhs.cols_) {
        throw DimensionMismatch(rows_, cols_, rhs.rows_, rhs.cols_);
    }
    Matrix result(rows_ + rhs.rows_, cols_);
    result.setBlock(0, 0, *this);
    result.setBlock(rows_, 0, rhs);
    return result;
}

template <typename T> Matrix<T> Matrix<T>::withoutRow(Index i) const {
    if (i >= rows_) {
        throw IndexOutOfRange(i, rows_);
    }
    Matrix result(rows_ - 1, cols_);
    for (Index r = 0; r < rows_; ++r) {
        if (r == i) {
            continue;
        }
        // Rows past the dropped one shift up by one.
        const Index dest = (r < i) ? r : r - 1;
        for (Index c = 0; c < cols_; ++c) {
            result(dest, c) = (*this)(r, c);
        }
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::withoutCol(Index j) const {
    if (j >= cols_) {
        throw IndexOutOfRange(j, cols_);
    }
    Matrix result(rows_, cols_ - 1);
    for (Index c = 0; c < cols_; ++c) {
        if (c == j) {
            continue;
        }
        const Index dest = (c < j) ? c : c - 1;
        for (Index r = 0; r < rows_; ++r) {
            result(r, dest) = (*this)(r, c);
        }
    }
    return result;
}

template <typename T> void Matrix<T>::transposeInPlace() {
    if (isSquare()) {
        // Square: swap across the diagonal, no allocation.
        for (Index i = 0; i < rows_; ++i) {
            for (Index j = i + 1; j < cols_; ++j) {
                std::swap(storage_[linearIndex(i, j)],
                          storage_[linearIndex(j, i)]);
            }
        }
        return;
    }
    Matrix result = transpose();
    swap(result);
}

template <typename T>
void Matrix<T>::resize(Index rows,
                       Index cols) {
    storage_.assign(rows * cols, T{});
    rows_ = rows;
    cols_ = cols;
}

template <typename T>
void Matrix<T>::conservativeResize(Index rows,
                                   Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::conservativeReSize");

    /*// Row-major means growing the column count moves every row, so this
    // rebuilds rather than resizing the buffer in place.
    Matrix       result(rows, cols);
    const Index  keptRows = (rows < rows_) ? rows : rows_;
    const Index  keptCols = (cols < cols_) ? cols : cols_;
    for (Index i = 0; i < keptRows; ++i) {
        for (Index j = 0; j < keptCols; ++j) {
            result(i, j) = (*this)(i, j);
        }
    }
    swap(result); // growth stays zero-filled from result's construction*/
}

template <typename T>
void Matrix<T>::swapRows(Index a,
                         Index b) {
    if (a == b || isEmpty()) {
        return;
    }
    // Row elements are adjacent in row-major storage, hence stride 1.
    Kernels<T>::swap(cols_, &storage_[linearIndex(a, 0)], 1,
                     &storage_[linearIndex(b, 0)], 1);
}

template <typename T>
void Matrix<T>::swapCols(Index a,
                         Index b) {
    if (a == b || isEmpty()) {
        return;
    }
    // Down a column, consecutive elements are one row apart: stride cols_.
    Kernels<T>::swap(rows_, &storage_[a], cols_, &storage_[b], cols_);
}

template <typename T> void Matrix<T>::fill(const T &value) {
    std::fill(storage_.begin(), storage_.end(), value);
}

template <typename T> void Matrix<T>::setZero() {
    std::fill(storage_.begin(), storage_.end(), T{});
}

template <typename T> void Matrix<T>::setIdentity() {
    setZero();
    const Index diagonalLength = (rows_ < cols_) ? rows_ : cols_;
    for (Index k = 0; k < diagonalLength; ++k) {
        (*this)(k, k) = T(1);
    }
}

template <typename T> void Matrix<T>::swap(Matrix &other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    storage_.swap(other.storage_);
}

// scalar summaries
template <typename T> T Matrix<T>::trace() const {
    // A rectangular matrix has a main diagonal of min(rows, cols).
    const Index diagonalLength = (rows_ < cols_) ? rows_ : cols_;
    T           total          = T{};
    for (Index k = 0; k < diagonalLength; ++k) {
        total += (*this)(k, k);
    }
    return total;
}

template <typename T> T Matrix<T>::sum() const {
    T total = T{};
    for (Index k = 0; k < size(); ++k) {
        total += storage_[k];
    }
    return total;
}

template <typename T> T Matrix<T>::determinant() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::determinant");
}

template <typename T> typename Matrix<T>::Real Matrix<T>::oneNorm() const {
    if (isEmpty()) {
        return Real{};
    }
    Real best = Real{};
    for (Index j = 0; j < cols_; ++j) {
        const Real columnSum = Kernels<T>::asum(rows_, &storage_[j], cols_);
        if (columnSum > best) {
            best = columnSum;
        }
    }
    return best;
}

template <typename T> typename Matrix<T>::Real Matrix<T>::infinityNorm() const {
    if (isEmpty()) {
        return Real{};
    }
    Real best = Real{};
    for (Index i = 0; i < rows_; ++i) {
        const Real rowSum =
            Kernels<T>::asum(cols_, &storage_[linearIndex(i, 0)], 1);
        if (rowSum > best) {
            best = rowSum;
        }
    }
    return best;
}

template <typename T>
typename Matrix<T>::Real Matrix<T>::frobeniusNorm() const {
    // The Frobenius norm is the 2-norm of the flattened buffer, so nrm2
    // covers it directly
    return Kernels<T>::nrm2(size(), storage_.data(), 1);
}

template <typename T> typename Matrix<T>::Real Matrix<T>::maxNorm() const {
    if (isEmpty()) {
        return Real{};
    }
    const Index k = Kernels<T>::iamax(size(), storage_.data(), 1);
    return NumericTraits<T>::abs(storage_[k]);
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
    if (!isSquare()) {
        return false; // a rectangular matrix cannot equal its transpose
    }
    // Compares against the transpose without forming it; only the strictly
    // upper triangle is walked, since the diagonal matches itself.
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = i + 1; j < cols_; ++j) {
            if (!NumericTraits<T>::isApproxZero((*this)(i, j) - (*this)(j, i),
                                                tolerance)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T> bool Matrix<T>::isHermitian(Real tolerance) const {
    if (!isSquare()) {
        return false;
    }
    for (Index i = 0; i < rows_; ++i) {
        // A Hermitian diagonal is real, so the diagonal is a genuine test
        // here rather than the tautology it is for isSymmetric.
        if (!NumericTraits<T>::isApproxZero(
                (*this)(i, i) - NumericTraits<T>::conj((*this)(i, i)),
                tolerance)) {
            return false;
        }
        for (Index j = i + 1; j < cols_; ++j) {
            if (!NumericTraits<T>::isApproxZero(
                    (*this)(i, j) - NumericTraits<T>::conj((*this)(j, i)),
                    tolerance)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T> bool Matrix<T>::isDiagonal(Real tolerance) const {
    // No square requirement: a rectangular matrix is diagonal when
    // everything off the main diagonal is negligible.
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            if (i != j &&
                !NumericTraits<T>::isApproxZero((*this)(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool Matrix<T>::isTriangular(Triangle::Kind which,
                             Real           tolerance) const {
    // `which` names the triangle allowed to hold data, so the test is that
    // the opposite one vanishes. The diagonal belongs to both and is
    // never checked.
    const bool upper = (which == Triangle::Kind::Upper);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            const bool mustVanish = upper ? (i > j) : (i < j);
            if (mustVanish &&
                !NumericTraits<T>::isApproxZero((*this)(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T> bool Matrix<T>::isOrthogonal(Real tolerance) const {
    // A^H A == I. For a tall matrix this is the columns-orthonormal test,
    // which is why the identity is sized by cols_ rather than rows_.
    const Matrix gram = conjugateTranspose() * (*this);
    return gram.isApprox(Identity(cols_), tolerance);
}

template <typename T>
bool Matrix<T>::isApprox(const Matrix &other,
                         Real          tolerance) const {
    // A shape mismatch is a false result, not an error: this is the
    // tolerant counterpart to operator==, which also just returns false.
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    for (Index k = 0; k < size(); ++k) {
        // isApproxZero(x, tol) is |x| <= tol; works for complex T too.
        if (!NumericTraits<T>::isApproxZero(storage_[k] - other.storage_[k],
                                            tolerance)) {
            return false;
        }
    }
    return true;
}

template <typename T> bool Matrix<T>::hasNaN() const {
    for (const T &value : storage_) {
        if (std::isnan(NumericTraits<T>::real(value)) ||
            std::isnan(NumericTraits<T>::imag(value))) {
            return true;
        }
    }
    return false;
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

    std::vector<std::string> cells(size());
    std::size_t              width = 0;
    for (Index k = 0; k < size(); ++k) {
        std::ostringstream cell;
        // std::complex carries its own operator<<, so real and complex T
        // both format through this one path.
        cell << std::setprecision(precision) << storage_[k];
        cells[k] = cell.str();
        if (cells[k].size() > width) {
            width = cells[k].size();
        }
    }

    std::ostringstream out;
    for (Index i = 0; i < rows_; ++i) {
        if (i > 0) {
            out << '\n'; // separator, not terminator: no trailing newline
        }
        for (Index j = 0; j < cols_; ++j) {
            if (j > 0) {
                out << ' ';
            }
            out << std::setw(static_cast<int>(width))
                << cells[linearIndex(i, j)];
        }
    }
    return out.str();
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
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw DimensionMismatch(rows_, cols_, other.rows_, other.cols_);
    }
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR) template class Matrix<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
