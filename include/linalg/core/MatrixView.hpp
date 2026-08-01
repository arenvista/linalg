#pragma once

#include <cstddef>

#include "linalg/Instantiate.hpp"
#include "linalg/core/Traits.hpp"

namespace linalg {

template <typename T> class Matrix;

// Non-owning, stride-aware window onto another matrix's storage.
// Blocked algorithms operate on these so panels are never copied.
//
// Element (i, j) lives at data[i * rowStride + j * colStride]; arbitrary
// strides make rows, columns, diagonals, and transposes all expressible
// as views. A view is invalidated when the owning matrix resizes or dies.
//
// Copy construction and copy assignment rebind the view (shallow: same
// pointer, shape, strides). Assigning from a Matrix copies *elements*
// into the viewed storage, as does copyFrom; both require equal shapes.
template <typename T> class MatrixView {
  public:
    using Scalar = T;
    using Index = std::size_t;

    MatrixView();
    MatrixView(T *data, Index rows, Index cols, Index rowStride,
               Index colStride);
    MatrixView(const MatrixView &other);     // Copy constructor
    MatrixView(MatrixView &&other) noexcept; // Move constructor
    ~MatrixView();

    MatrixView &operator=(const MatrixView &other);     // rebinds the view
    MatrixView &operator=(MatrixView &&other) noexcept; // rebinds the view
    MatrixView &operator=(const Matrix<T> &source);     // copies elements in

    T &operator()(Index i, Index j);
    const T &operator()(Index i, Index j) const;
    T &at(Index i, Index j);
    const T &at(Index i, Index j) const;

    Index rows() const;
    Index cols() const;
    Index rowStride() const;
    Index colStride() const;
    bool isContiguous()
        const; // row-major with no gaps: rowStride == cols, colStride == 1
    bool isEmpty() const;

    T *data();
    const T *data() const;

    // Sub-views share the same storage; indices are relative to this view,
    // not to the owning matrix. transposed() swaps shape and strides — no
    // elements move. diagonal() is a 1 x min(rows, cols) view.
    MatrixView block(Index i, Index j, Index numRows, Index numCols);
    MatrixView row(Index i);
    MatrixView col(Index j);
    MatrixView diagonal();
    MatrixView transposed();

    Matrix<T> toMatrix() const; // deep copy into fresh contiguous storage

    void fill(const T &value);
    void setZero();
    void scale(const T &factor);
    void swapWith(MatrixView &other); // swaps elements, not bindings
    void
    copyFrom(const MatrixView &source); // elementwise; source must not alias

  private:
    T *data_;
    Index rows_;
    Index cols_;
    Index rowStride_;
    Index colStride_;
};

// Read-only counterpart; converts implicitly from MatrixView so mutable
// views can be passed wherever only reads are needed.
template <typename T> class ConstMatrixView {
  public:
    using Scalar = T;
    using Index = std::size_t;

    ConstMatrixView();
    ConstMatrixView(const T *data, Index rows, Index cols, Index rowStride,
                    Index colStride);
    ConstMatrixView(const MatrixView<T> &view);

    const T &operator()(Index i, Index j) const;
    const T &at(Index i, Index j) const;

    Index rows() const;
    Index cols() const;
    Index rowStride() const;
    Index colStride() const;
    bool isContiguous() const;
    bool isEmpty() const;

    const T *data() const;

    ConstMatrixView block(Index i, Index j, Index numRows, Index numCols) const;
    ConstMatrixView row(Index i) const;
    ConstMatrixView col(Index j) const;
    ConstMatrixView diagonal() const;
    ConstMatrixView transposed() const;

    Matrix<T> toMatrix() const;

  private:
    const T *data_;
    Index rows_;
    Index cols_;
    Index rowStride_;
    Index colStride_;
};

// Declared extern so including this header does not instantiate
// anything; the definitions are compiled once in src/.
#define LINALG_EXTERN(SCALAR)                                                  \
    extern template class MatrixView<SCALAR>;                                  \
    extern template class ConstMatrixView<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_EXTERN)

#undef LINALG_EXTERN

} // namespace linalg
