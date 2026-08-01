#include "linalg/core/Matrix.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> Matrix<T>::Matrix() {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T> Matrix<T>::Matrix(Index rows, Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T> Matrix<T>::Matrix(Index rows, Index cols, const T &fill) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T>
Matrix<T>::Matrix(Index rows, Index cols, const std::vector<T> &rowMajorData) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> rows) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T> Matrix<T>::Matrix(const Matrix &other) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T> Matrix<T>::Matrix(Matrix &&other) noexcept {
    // TODO: transfer ownership from `other`. Declared noexcept,
    // so this stub cannot throw the way the others do.
}

template <typename T> Matrix<T>::Matrix(const ConstMatrixView<T> &view) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Matrix");
}

template <typename T> Matrix<T>::~Matrix() {}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &other) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator=");
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix &&other) noexcept {
    // TODO: transfer ownership from `other`. Declared noexcept,
    // so this stub cannot throw the way the others do.
    return *this;
}

template <typename T> Matrix<T> Matrix<T>::Zeros(Index rows, Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Zeros");
}

template <typename T> Matrix<T> Matrix<T>::Ones(Index rows, Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Ones");
}

template <typename T>
Matrix<T> Matrix<T>::Constant(Index rows, Index cols, const T &value) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Constant");
}

template <typename T> Matrix<T> Matrix<T>::Identity(Index size) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Identity");
}

template <typename T> Matrix<T> Matrix<T>::Diagonal(const Vector<T> &values) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Diagonal");
}

template <typename T>
Matrix<T> Matrix<T>::Random(Index rows, Index cols, unsigned long seed) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Random");
}

template <typename T>
Matrix<T> Matrix<T>::RandomSymmetric(Index size, unsigned long seed) {
    throw LinalgError("not implemented: linalg::Matrix<T>::RandomSymmetric");
}

template <typename T>
Matrix<T> Matrix<T>::RandomOrthogonal(Index size, unsigned long seed) {
    throw LinalgError("not implemented: linalg::Matrix<T>::RandomOrthogonal");
}

template <typename T> Matrix<T> Matrix<T>::Hilbert(Index size) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Hilbert");
}

template <typename T>
Matrix<T> Matrix<T>::Vandermonde(const Vector<T> &nodes, Index degree) {
    throw LinalgError("not implemented: linalg::Matrix<T>::Vandermonde");
}

template <typename T>
Matrix<T> Matrix<T>::FromColumns(const std::vector<Vector<T>> &columns) {
    throw LinalgError("not implemented: linalg::Matrix<T>::FromColumns");
}

template <typename T>
Matrix<T> Matrix<T>::FromRows(const std::vector<Vector<T>> &rows) {
    throw LinalgError("not implemented: linalg::Matrix<T>::FromRows");
}

template <typename T> T &Matrix<T>::operator()(Index i, Index j) {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator()");
}

template <typename T> const T &Matrix<T>::operator()(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::operator()");
}

template <typename T> T &Matrix<T>::at(Index i, Index j) {
    throw LinalgError("not implemented: linalg::Matrix<T>::at");
}

template <typename T> const T &Matrix<T>::at(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::at");
}

template <typename T> T *Matrix<T>::data() {
    throw LinalgError("not implemented: linalg::Matrix<T>::data");
}

template <typename T> const T *Matrix<T>::data() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::data");
}

template <typename T> typename Matrix<T>::Index Matrix<T>::rows() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::rows");
}

template <typename T> typename Matrix<T>::Index Matrix<T>::cols() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::cols");
}

template <typename T> typename Matrix<T>::Index Matrix<T>::size() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::size");
}

template <typename T> bool Matrix<T>::isEmpty() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isEmpty");
}

template <typename T> bool Matrix<T>::isSquare() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isSquare");
}

template <typename T> MatrixView<T> Matrix<T>::view() {
    throw LinalgError("not implemented: linalg::Matrix<T>::view");
}

template <typename T> ConstMatrixView<T> Matrix<T>::view() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::view");
}

template <typename T>
MatrixView<T> Matrix<T>::block(Index i, Index j, Index numRows, Index numCols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::block");
}

template <typename T>
ConstMatrixView<T> Matrix<T>::block(Index i, Index j, Index numRows,
                                    Index numCols) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::block");
}

template <typename T> MatrixView<T> Matrix<T>::rowView(Index i) {
    throw LinalgError("not implemented: linalg::Matrix<T>::rowView");
}

template <typename T> ConstMatrixView<T> Matrix<T>::rowView(Index i) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::rowView");
}

template <typename T> MatrixView<T> Matrix<T>::colView(Index j) {
    throw LinalgError("not implemented: linalg::Matrix<T>::colView");
}

template <typename T> ConstMatrixView<T> Matrix<T>::colView(Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::colView");
}

template <typename T> Vector<T> Matrix<T>::row(Index i) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::row");
}

template <typename T> Vector<T> Matrix<T>::col(Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::col");
}

template <typename T> Vector<T> Matrix<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::diagonal");
}

template <typename T> void Matrix<T>::setRow(Index i, const Vector<T> &values) {
    throw LinalgError("not implemented: linalg::Matrix<T>::setRow");
}

template <typename T> void Matrix<T>::setCol(Index j, const Vector<T> &values) {
    throw LinalgError("not implemented: linalg::Matrix<T>::setCol");
}

template <typename T>
void Matrix<T>::setBlock(Index i, Index j, const Matrix &source) {
    throw LinalgError("not implemented: linalg::Matrix<T>::setBlock");
}

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

template <typename T> Matrix<T> Matrix<T>::transpose() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::transpose");
}

template <typename T> Matrix<T> Matrix<T>::conjugateTranspose() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::conjugateTranspose");
}

template <typename T>
Matrix<T> Matrix<T>::reshaped(Index rows, Index cols) const {
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

template <typename T> void Matrix<T>::resize(Index rows, Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::resize");
}

template <typename T>
void Matrix<T>::conservativeResize(Index rows, Index cols) {
    throw LinalgError("not implemented: linalg::Matrix<T>::conservativeResize");
}

template <typename T> void Matrix<T>::swapRows(Index a, Index b) {
    throw LinalgError("not implemented: linalg::Matrix<T>::swapRows");
}

template <typename T> void Matrix<T>::swapCols(Index a, Index b) {
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
bool Matrix<T>::isTriangular(Triangle::Kind which, Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isTriangular");
}

template <typename T> bool Matrix<T>::isOrthogonal(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isOrthogonal");
}

template <typename T>
bool Matrix<T>::isApprox(const Matrix &other, Real tolerance) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::isApprox");
}

template <typename T> bool Matrix<T>::hasNaN() const {
    throw LinalgError("not implemented: linalg::Matrix<T>::hasNaN");
}

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

template <typename T>
typename Matrix<T>::Index Matrix<T>::linearIndex(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::linearIndex");
}

template <typename T> void Matrix<T>::checkBounds(Index i, Index j) const {
    throw LinalgError("not implemented: linalg::Matrix<T>::checkBounds");
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
