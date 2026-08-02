#include "linalg/core/MatrixView.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

#include <algorithm> // std::swap

namespace linalg {

template <typename T>
MatrixView<T>::MatrixView()
    : data_(nullptr), rows_(0), cols_(0), rowStride_(0), colStride_(0) {}

template <typename T>
MatrixView<T>::MatrixView(T *data, Index rows, Index cols, Index rowStride,
                          Index colStride)
    : data_(data), rows_(rows), cols_(cols), rowStride_(rowStride),
      colStride_(colStride) {}

template <typename T>
MatrixView<T>::MatrixView(const MatrixView &other)
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
      rowStride_(other.rowStride_), colStride_(other.colStride_) {}

template <typename T>
MatrixView<T>::MatrixView(MatrixView &&other) noexcept
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_),
      rowStride_(other.rowStride_), colStride_(other.colStride_) {
    // A view is non-owning, so moving is a shallow copy; clearing `other`
    // just leaves the moved-from view empty (a valid, unspecified state).
    other.data_      = nullptr;
    other.rows_      = 0;
    other.cols_      = 0;
    other.rowStride_ = 0;
    other.colStride_ = 0;
}

template <typename T> MatrixView<T>::~MatrixView() {}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(const MatrixView &other) {
    // Rebinds this view to `other`'s storage (shallow: shares pointer,
    // shape, strides). Does not copy elements; use copyFrom for that.
    if (this != &other) {
        data_      = other.data_;
        rows_      = other.rows_;
        cols_      = other.cols_;
        rowStride_ = other.rowStride_;
        colStride_ = other.colStride_;
    }
    return *this;
}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(MatrixView &&other) noexcept {
    // Rebinds this view to `other`'s storage (shallow; a view is
    // non-owning, so there is nothing to free). Guard against self-move,
    // then clear `other` to leave it empty.
    if (this != &other) {
        data_            = other.data_;
        rows_            = other.rows_;
        cols_            = other.cols_;
        rowStride_       = other.rowStride_;
        colStride_       = other.colStride_;
        other.data_      = nullptr;
        other.rows_      = 0;
        other.cols_      = 0;
        other.rowStride_ = 0;
        other.colStride_ = 0;
    }
    return *this;
}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(const Matrix<T> &source) {
    // Copies *elements* into the storage this view already refers to; the
    // binding (data_/strides) is untouched. Shapes must match exactly.
    if (rows_ != source.rows() || cols_ != source.cols()) {
        throw DimensionMismatch(rows_, cols_, source.rows(), source.cols());
    }
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            data_[i * rowStride_ + j * colStride_] = source(i, j);
        }
    }
    return *this;
}

template <typename T> T &MatrixView<T>::operator()(Index i, Index j) {
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T>
const T &MatrixView<T>::operator()(Index i, Index j) const {
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T> T &MatrixView<T>::at(Index i, Index j) {
    if (i >= rows_) {
        throw IndexOutOfRange(i, rows_);
    }
    if (j >= cols_) {
        throw IndexOutOfRange(j, cols_);
    }
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T> const T &MatrixView<T>::at(Index i, Index j) const {
    if (i >= rows_) {
        throw IndexOutOfRange(i, rows_);
    }
    if (j >= cols_) {
        throw IndexOutOfRange(j, cols_);
    }
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::rows() const {
    return rows_;
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::cols() const {
    return cols_;
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::rowStride() const {
    return rowStride_;
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::colStride() const {
    return colStride_;
}

template <typename T> bool MatrixView<T>::isContiguous() const {
    return (rowStride_ == cols_ && colStride_ == 1);
}

template <typename T> bool MatrixView<T>::isEmpty() const {
    return (rows_ == 0 || cols_ == 0);
}

template <typename T> T *MatrixView<T>::data() { return data_; }

template <typename T> const T *MatrixView<T>::data() const { return data_; }

template <typename T>
MatrixView<T> MatrixView<T>::block(Index i, Index j, Index numRows,
                                   Index numCols) {
    // Sub-view over the same storage: rebase the pointer to element (i, j)
    // and keep the strides, so the block sees the same layout. The block
    // must fit; comparisons are written to avoid unsigned overflow.
    if (i > rows_ || numRows > rows_ - i) {
        throw IndexOutOfRange(i + numRows, rows_);
    }
    if (j > cols_ || numCols > cols_ - j) {
        throw IndexOutOfRange(j + numCols, cols_);
    }
    return MatrixView(data_ + i * rowStride_ + j * colStride_, numRows, numCols,
                      rowStride_, colStride_);
}

template <typename T> MatrixView<T> MatrixView<T>::row(Index i) {
    return block(i, 0, 1, cols_); // a 1 x cols_ view
}

template <typename T> MatrixView<T> MatrixView<T>::col(Index j) {
    return block(0, j, rows_, 1); // a rows_ x 1 view
}

template <typename T> MatrixView<T> MatrixView<T>::diagonal() {
    // 1 x min(rows, cols) view along the main diagonal: stepping one
    // element advances a row and a column at once, so the column stride
    // is rowStride_ + colStride_. Base stays at (0, 0).
    const Index n = rows_ < cols_ ? rows_ : cols_;
    return MatrixView(data_, 1, n, rowStride_ + colStride_,
                      rowStride_ + colStride_);
}

template <typename T> MatrixView<T> MatrixView<T>::transposed() {
    // Swap the row and column counts and strides, but keep the same base.
    return MatrixView(data_, cols_, rows_, colStride_, rowStride_);
}

template <typename T> Matrix<T> MatrixView<T>::toMatrix() const {
    // Deep copy into fresh contiguous storage; the result does not alias
    // this view and is unaffected by later writes through it.
    Matrix<T> result(rows_, cols_);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            result(i, j) = data_[i * rowStride_ + j * colStride_];
        }
    }
    return result;
}

template <typename T> void MatrixView<T>::fill(const T &value) {
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            data_[i * rowStride_ + j * colStride_] = value;
        }
    }
}

template <typename T> void MatrixView<T>::setZero() { fill(T(0)); }

template <typename T> void MatrixView<T>::scale(const T &factor) {
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            data_[i * rowStride_ + j * colStride_] *= factor;
        }
    }
}

template <typename T> void MatrixView<T>::swapWith(MatrixView &other) {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw DimensionMismatch(rows_, cols_, other.rows(), other.cols());
    }
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            std::swap(
                data_[i * rowStride_ + j * colStride_],
                other.data_[i * other.rowStride() + j * other.colStride()]);
        }
    }
}

template <typename T> void MatrixView<T>::copyFrom(const MatrixView &source) {
    if (this == &source) {
        return; // no-op if the source is this view itself
    }
    if (rows_ != source.rows() || cols_ != source.cols()) {
        throw DimensionMismatch(rows_, cols_, source.rows(), source.cols());
    }
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            data_[i * rowStride_ + j * colStride_] =
                source.data_[i * source.rowStride() + j * source.colStride()];
        }
    }
}

template <typename T>
ConstMatrixView<T>::ConstMatrixView()
    : data_(nullptr), rows_(0), cols_(0), rowStride_(0), colStride_(0) {}

template <typename T>
ConstMatrixView<T>::ConstMatrixView(const T *data, Index rows, Index cols,
                                    Index rowStride, Index colStride)
    : data_(data), rows_(rows), cols_(cols), rowStride_(rowStride),
      colStride_(colStride) {}

template <typename T>
ConstMatrixView<T>::ConstMatrixView(const MatrixView<T> &view)
    : data_(view.data()), rows_(view.rows()), cols_(view.cols()),
      rowStride_(view.rowStride()), colStride_(view.colStride()) {}

template <typename T>
const T &ConstMatrixView<T>::operator()(Index i, Index j) const {
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T> const T &ConstMatrixView<T>::at(Index i, Index j) const {
    if (i >= rows_) {
        throw IndexOutOfRange(i, rows_);
    }
    if (j >= cols_) {
        throw IndexOutOfRange(j, cols_);
    }
    return data_[i * rowStride_ + j * colStride_];
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::rows() const {
    return rows_;
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::cols() const {
    return cols_;
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::rowStride() const {
    return rowStride_;
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::colStride() const {
    return colStride_;
}

template <typename T> bool ConstMatrixView<T>::isContiguous() const {
    return (rowStride_ == cols_ && colStride_ == 1);
}

template <typename T> bool ConstMatrixView<T>::isEmpty() const {
    return (rows_ == 0 || cols_ == 0);
}

template <typename T> const T *ConstMatrixView<T>::data() const {
    return data_;
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::block(Index i, Index j, Index numRows,
                                             Index numCols) const {
    // Read-only sub-view over the same storage; see MatrixView::block.
    if (i > rows_ || numRows > rows_ - i) {
        throw IndexOutOfRange(i + numRows, rows_);
    }
    if (j > cols_ || numCols > cols_ - j) {
        throw IndexOutOfRange(j + numCols, cols_);
    }
    return ConstMatrixView(data_ + i * rowStride_ + j * colStride_, numRows,
                           numCols, rowStride_, colStride_);
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::row(Index i) const {
    return block(i, 0, 1, cols_); // a 1 x cols_ view
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::col(Index j) const {
    return block(0, j, rows_, 1); // a rows_ x 1 view
}

template <typename T> ConstMatrixView<T> ConstMatrixView<T>::diagonal() const {
    // 1 x min(rows, cols) view along the main diagonal; see MatrixView.
    const Index n = rows_ < cols_ ? rows_ : cols_;
    return ConstMatrixView(data_, 1, n, rowStride_ + colStride_,
                           rowStride_ + colStride_);
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::transposed() const {
    // Swap the row and column counts and strides, but keep the same base.
    return ConstMatrixView(data_, cols_, rows_, colStride_, rowStride_);
}

template <typename T> Matrix<T> ConstMatrixView<T>::toMatrix() const {
    // Deep copy into fresh contiguous storage.
    Matrix<T> result(rows_, cols_);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            result(i, j) = data_[i * rowStride_ + j * colStride_];
        }
    }
    return result;
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)     \
    template class MatrixView<SCALAR>; \
    template class ConstMatrixView<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
