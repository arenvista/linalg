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

    /// @brief Constructs an empty view (null data, zero shape and strides).
    MatrixView();
    /// @brief Constructs a view over caller-owned storage.
    /// @param data Pointer to the element at position (0, 0) of the view.
    /// @param rows Number of rows the view exposes.
    /// @param cols Number of columns the view exposes.
    /// @param rowStride Elements to advance in `data` to move down one row.
    /// @param colStride Elements to advance in `data` to move right one column.
    MatrixView(T *data, Index rows, Index cols, Index rowStride,
               Index colStride);
    /// @brief Copy constructor; rebinds this view to the same storage as
    /// `other` (shallow: shares the pointer, shape, and strides).
    /// @param other View to copy the binding from.
    MatrixView(const MatrixView &other);
    /// @brief Move constructor; takes over `other`'s binding and leaves it
    /// empty. Equivalent to a shallow copy since the view is non-owning.
    /// @param other View to move from; left empty afterward.
    MatrixView(MatrixView &&other) noexcept;
    /// @brief Destructor; a no-op because the view does not own its storage.
    ~MatrixView();

    /// @brief Copy assignment; rebinds this view to `other`'s storage
    /// (shallow). Does not copy elements.
    /// @param other View to rebind to.
    /// @return Reference to this view.
    MatrixView &operator=(const MatrixView &other);
    /// @brief Move assignment; rebinds this view to `other`'s storage and
    /// leaves `other` empty. Safe under self-assignment.
    /// @param other View to move from; left empty afterward.
    /// @return Reference to this view.
    MatrixView &operator=(MatrixView &&other) noexcept;
    /// @brief Copies the elements of `source` into the storage this view
    /// refers to; the binding is untouched. Shapes must match exactly.
    /// @param source Matrix whose element values are copied in.
    /// @return Reference to this view.
    MatrixView &operator=(const Matrix<T> &source);

    /// @brief Unchecked mutable access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Reference to element (i, j) in the viewed storage.
    T &operator()(Index i, Index j);
    /// @brief Unchecked read-only access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Const reference to element (i, j) in the viewed storage.
    const T &operator()(Index i, Index j) const;
    /// @brief Bounds-checked mutable access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Reference to element (i, j) in the viewed storage.
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    T &at(Index i, Index j);
    /// @brief Bounds-checked read-only access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Const reference to element (i, j) in the viewed storage.
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    const T &at(Index i, Index j) const;

    /// @brief Number of rows in the view.
    /// @return The row count.
    Index rows() const;
    /// @brief Number of columns in the view.
    /// @return The column count.
    Index cols() const;
    /// @brief Elements between the starts of consecutive rows in storage.
    /// @return The row stride.
    Index rowStride() const;
    /// @brief Elements between the starts of consecutive columns in storage.
    /// @return The column stride.
    Index colStride() const;
    /// @brief Reports whether the view is a dense row-major block with no
    /// gaps (rowStride == cols and colStride == 1).
    /// @return True if the storage can be treated as a contiguous buffer.
    bool isContiguous() const;
    /// @brief Reports whether the view has no elements.
    /// @return True if rows() or cols() is zero.
    bool isEmpty() const;

    /// @brief Direct mutable access to the underlying buffer. Honor the
    /// strides unless isContiguous() is true.
    /// @return Pointer to the element at position (0, 0).
    T *data();
    /// @brief Direct read-only access to the underlying buffer. Honor the
    /// strides unless isContiguous() is true.
    /// @return Const pointer to the element at position (0, 0).
    const T *data() const;

    // Sub-views share the same storage; indices are relative to this view,
    // not to the owning matrix.

    /// @brief Creates a sub-view over a rectangular block of this view.
    /// @param i Row index of the block's top-left corner (0-based).
    /// @param j Column index of the block's top-left corner (0-based).
    /// @param numRows Number of rows in the block.
    /// @param numCols Number of columns in the block.
    /// @return A view onto the block, sharing this view's storage.
    /// @throws IndexOutOfRange if the block does not fit within this view.
    MatrixView block(Index i, Index j, Index numRows, Index numCols);
    /// @brief Creates a 1 x cols view onto a single row.
    /// @param i Row index (0-based).
    /// @return A view onto row i, sharing this view's storage.
    /// @throws IndexOutOfRange if i >= rows().
    MatrixView row(Index i);
    /// @brief Creates a rows x 1 view onto a single column.
    /// @param j Column index (0-based).
    /// @return A view onto column j, sharing this view's storage.
    /// @throws IndexOutOfRange if j >= cols().
    MatrixView col(Index j);
    /// @brief Creates a 1 x min(rows, cols) view onto the main diagonal.
    /// @return A view onto the diagonal, sharing this view's storage.
    MatrixView diagonal();
    /// @brief Creates a transposed view by swapping shape and strides; no
    /// elements are moved.
    /// @return A cols x rows view onto the same storage.
    MatrixView transposed();

    /// @brief Deep-copies the viewed elements into a fresh contiguous Matrix.
    /// @return A newly allocated Matrix holding a copy of the elements.
    Matrix<T> toMatrix() const;

    /// @brief Assigns the given value to every element of the view.
    /// @param value Value written to each element.
    void fill(const T &value);
    /// @brief Sets every element of the view to zero.
    void setZero();
    /// @brief Multiplies every element of the view in place by a factor.
    /// @param factor Scalar multiplier applied to each element.
    void scale(const T &factor);
    /// @brief Swaps the elements of this view with those of another; the
    /// bindings are left unchanged. Shapes must match.
    /// @param other View whose elements are exchanged with this view's.
    void swapWith(MatrixView &other);
    /// @brief Copies elements from another view into this one. Shapes must
    /// match and the source must not alias this view's storage.
    /// @param source View to copy element values from.
    void copyFrom(const MatrixView &source);

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

    /// @brief Constructs an empty view (null data, zero shape and strides).
    ConstMatrixView();
    /// @brief Constructs a read-only view over caller-owned storage.
    /// @param data Pointer to the element at position (0, 0) of the view.
    /// @param rows Number of rows the view exposes.
    /// @param cols Number of columns the view exposes.
    /// @param rowStride Elements to advance in `data` to move down one row.
    /// @param colStride Elements to advance in `data` to move right one column.
    ConstMatrixView(const T *data, Index rows, Index cols, Index rowStride,
                    Index colStride);
    /// @brief Implicitly constructs a read-only view from a mutable one.
    /// @param view Mutable view to expose as read-only.
    ConstMatrixView(const MatrixView<T> &view);

    /// @brief Unchecked read-only access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Const reference to element (i, j) in the viewed storage.
    const T &operator()(Index i, Index j) const;
    /// @brief Bounds-checked read-only access to an element.
    /// @param i Row index, relative to this view (0-based).
    /// @param j Column index, relative to this view (0-based).
    /// @return Const reference to element (i, j) in the viewed storage.
    /// @throws IndexOutOfRange if i >= rows() or j >= cols().
    const T &at(Index i, Index j) const;

    /// @brief Number of rows in the view.
    /// @return The row count.
    Index rows() const;
    /// @brief Number of columns in the view.
    /// @return The column count.
    Index cols() const;
    /// @brief Elements between the starts of consecutive rows in storage.
    /// @return The row stride.
    Index rowStride() const;
    /// @brief Elements between the starts of consecutive columns in storage.
    /// @return The column stride.
    Index colStride() const;
    /// @brief Reports whether the view is a dense row-major block with no
    /// gaps (rowStride == cols and colStride == 1).
    /// @return True if the storage can be treated as a contiguous buffer.
    bool isContiguous() const;
    /// @brief Reports whether the view has no elements.
    /// @return True if rows() or cols() is zero.
    bool isEmpty() const;

    /// @brief Direct read-only access to the underlying buffer. Honor the
    /// strides unless isContiguous() is true.
    /// @return Const pointer to the element at position (0, 0).
    const T *data() const;

    /// @brief Creates a read-only sub-view over a rectangular block.
    /// @param i Row index of the block's top-left corner (0-based).
    /// @param j Column index of the block's top-left corner (0-based).
    /// @param numRows Number of rows in the block.
    /// @param numCols Number of columns in the block.
    /// @return A read-only view onto the block, sharing this view's storage.
    /// @throws IndexOutOfRange if the block does not fit within this view.
    ConstMatrixView block(Index i, Index j, Index numRows, Index numCols) const;
    /// @brief Creates a read-only 1 x cols view onto a single row.
    /// @param i Row index (0-based).
    /// @return A read-only view onto row i, sharing this view's storage.
    /// @throws IndexOutOfRange if i >= rows().
    ConstMatrixView row(Index i) const;
    /// @brief Creates a read-only rows x 1 view onto a single column.
    /// @param j Column index (0-based).
    /// @return A read-only view onto column j, sharing this view's storage.
    /// @throws IndexOutOfRange if j >= cols().
    ConstMatrixView col(Index j) const;
    /// @brief Creates a read-only 1 x min(rows, cols) view onto the diagonal.
    /// @return A read-only view onto the diagonal, sharing this view's storage.
    ConstMatrixView diagonal() const;
    /// @brief Creates a read-only transposed view by swapping shape and
    /// strides; no elements are moved.
    /// @return A read-only cols x rows view onto the same storage.
    ConstMatrixView transposed() const;

    /// @brief Deep-copies the viewed elements into a fresh contiguous Matrix.
    /// @return A newly allocated Matrix holding a copy of the elements.
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
