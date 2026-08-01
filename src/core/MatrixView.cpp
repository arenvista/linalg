#include "linalg/core/MatrixView.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T>
MatrixView<T>::MatrixView()
    : data_(nullptr), rows_(0), cols_(0), rowStride_(0), colStride_(0) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::MatrixView");
}

template <typename T>
MatrixView<T>::MatrixView(T *data, Index rows, Index cols, Index rowStride,
                          Index colStride)
    : data_(data), rows_(rows), cols_(cols), rowStride_(rowStride),
      colStride_(colStride) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::MatrixView");
}

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
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.rowStride_ = 0;
    other.colStride_ = 0;
}

template <typename T> MatrixView<T>::~MatrixView() {}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(const MatrixView &other) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::operator=");
}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(MatrixView &&other) noexcept {
    // TODO: transfer ownership from `other`. Declared noexcept,
    // so this stub cannot throw the way the others do.
    return *this;
}

template <typename T>
MatrixView<T> &MatrixView<T>::operator=(const Matrix<T> &source) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::operator=");
}

template <typename T> T &MatrixView<T>::operator()(Index i, Index j) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::operator()");
}

template <typename T>
const T &MatrixView<T>::operator()(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::operator()");
}

template <typename T> T &MatrixView<T>::at(Index i, Index j) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::at");
}

template <typename T> const T &MatrixView<T>::at(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::at");
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::rows() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::rows");
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::cols() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::cols");
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::rowStride() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::rowStride");
}

template <typename T>
typename MatrixView<T>::Index MatrixView<T>::colStride() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::colStride");
}

template <typename T> bool MatrixView<T>::isContiguous() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::isContiguous");
}

template <typename T> bool MatrixView<T>::isEmpty() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::isEmpty");
}

template <typename T> T *MatrixView<T>::data() {
    throw LinalgError("not implemented: linalg::MatrixView<T>::data");
}

template <typename T> const T *MatrixView<T>::data() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::data");
}

template <typename T>
MatrixView<T> MatrixView<T>::block(Index i, Index j, Index numRows,
                                   Index numCols) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::block");
}

template <typename T> MatrixView<T> MatrixView<T>::row(Index i) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::row");
}

template <typename T> MatrixView<T> MatrixView<T>::col(Index j) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::col");
}

template <typename T> MatrixView<T> MatrixView<T>::diagonal() {
    throw LinalgError("not implemented: linalg::MatrixView<T>::diagonal");
}

template <typename T> MatrixView<T> MatrixView<T>::transposed() {
    throw LinalgError("not implemented: linalg::MatrixView<T>::transposed");
}

template <typename T> Matrix<T> MatrixView<T>::toMatrix() const {
    throw LinalgError("not implemented: linalg::MatrixView<T>::toMatrix");
}

template <typename T> void MatrixView<T>::fill(const T &value) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::fill");
}

template <typename T> void MatrixView<T>::setZero() {
    throw LinalgError("not implemented: linalg::MatrixView<T>::setZero");
}

template <typename T> void MatrixView<T>::scale(const T &factor) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::scale");
}

template <typename T> void MatrixView<T>::swapWith(MatrixView &other) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::swapWith");
}

template <typename T> void MatrixView<T>::copyFrom(const MatrixView &source) {
    throw LinalgError("not implemented: linalg::MatrixView<T>::copyFrom");
}

template <typename T> ConstMatrixView<T>::ConstMatrixView() {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::ConstMatrixView");
}

template <typename T>
ConstMatrixView<T>::ConstMatrixView(const T *data, Index rows, Index cols,
                                    Index rowStride, Index colStride) {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::ConstMatrixView");
}

template <typename T>
ConstMatrixView<T>::ConstMatrixView(const MatrixView<T> &view) {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::ConstMatrixView");
}

template <typename T>
const T &ConstMatrixView<T>::operator()(Index i, Index j) const {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::operator()");
}

template <typename T> const T &ConstMatrixView<T>::at(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::at");
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::rows() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::rows");
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::cols() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::cols");
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::rowStride() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::rowStride");
}

template <typename T>
typename ConstMatrixView<T>::Index ConstMatrixView<T>::colStride() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::colStride");
}

template <typename T> bool ConstMatrixView<T>::isContiguous() const {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::isContiguous");
}

template <typename T> bool ConstMatrixView<T>::isEmpty() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::isEmpty");
}

template <typename T> const T *ConstMatrixView<T>::data() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::data");
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::block(Index i, Index j, Index numRows,
                                             Index numCols) const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::block");
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::row(Index i) const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::row");
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::col(Index j) const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::col");
}

template <typename T> ConstMatrixView<T> ConstMatrixView<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::diagonal");
}

template <typename T>
ConstMatrixView<T> ConstMatrixView<T>::transposed() const {
    throw LinalgError(
        "not implemented: linalg::ConstMatrixView<T>::transposed");
}

template <typename T> Matrix<T> ConstMatrixView<T>::toMatrix() const {
    throw LinalgError("not implemented: linalg::ConstMatrixView<T>::toMatrix");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)                                             \
    template class MatrixView<SCALAR>;                                         \
    template class ConstMatrixView<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
