#include "linalg/decomp/Hessenberg.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> Hessenberg<T>::Hessenberg() {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::Hessenberg");
}

template <typename T> Hessenberg<T>::Hessenberg(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::Hessenberg");
}

template <typename T> void Hessenberg<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::compute");
}

template <typename T> bool Hessenberg<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::isComputed");
}

template <typename T> Matrix<T> Hessenberg<T>::matrixH() const {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::matrixH");
}

template <typename T> Matrix<T> Hessenberg<T>::matrixQ() const {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::matrixQ");
}

template <typename T>
const HouseholderSequence<T> &Hessenberg<T>::reflectors() const {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::reflectors");
}

template <typename T> bool Hessenberg<T>::isUnreduced(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Hessenberg<T>::isUnreduced");
}

template <typename T>
typename Hessenberg<T>::Index Hessenberg<T>::firstNegligibleSubdiagonal(
    Real tolerance
) const {
    throw LinalgError(
        "not implemented: linalg::Hessenberg<T>::firstNegligibleSubdiagonal"
    );
}

template <typename T> Tridiagonal<T>::Tridiagonal() {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::Tridiagonal");
}

template <typename T> Tridiagonal<T>::Tridiagonal(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::Tridiagonal");
}

template <typename T> void Tridiagonal<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::compute");
}

template <typename T> bool Tridiagonal<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::isComputed");
}

template <typename T>
Vector<typename Tridiagonal<T>::Real> Tridiagonal<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::diagonal");
}

template <typename T>
Vector<typename Tridiagonal<T>::Real> Tridiagonal<T>::subdiagonal() const {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::subdiagonal");
}

template <typename T> Matrix<T> Tridiagonal<T>::matrixT() const {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::matrixT");
}

template <typename T> Matrix<T> Tridiagonal<T>::matrixQ() const {
    throw LinalgError("not implemented: linalg::Tridiagonal<T>::matrixQ");
}

template <typename T>
typename Tridiagonal<T>::Index Tridiagonal<T>::eigenvalueCountBelow(
    Real shift
) const {
    throw LinalgError(
        "not implemented: linalg::Tridiagonal<T>::eigenvalueCountBelow"
    );
}

template <typename T> Bidiagonal<T>::Bidiagonal() {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::Bidiagonal");
}

template <typename T> Bidiagonal<T>::Bidiagonal(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::Bidiagonal");
}

template <typename T> void Bidiagonal<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::compute");
}

template <typename T> bool Bidiagonal<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::isComputed");
}

template <typename T>
Vector<typename Bidiagonal<T>::Real> Bidiagonal<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::diagonal");
}

template <typename T>
Vector<typename Bidiagonal<T>::Real> Bidiagonal<T>::superdiagonal() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::superdiagonal");
}

template <typename T> Matrix<T> Bidiagonal<T>::matrixB() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::matrixB");
}

template <typename T> Matrix<T> Bidiagonal<T>::matrixU() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::matrixU");
}

template <typename T> Matrix<T> Bidiagonal<T>::matrixV() const {
    throw LinalgError("not implemented: linalg::Bidiagonal<T>::matrixV");
}

template <typename T> bool Bidiagonal<T>::usedRPreprocessing() const {
    throw LinalgError(
        "not implemented: linalg::Bidiagonal<T>::usedRPreprocessing"
    );
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)      \
    template class Hessenberg<SCALAR>;  \
    template class Tridiagonal<SCALAR>; \
    template class Bidiagonal<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
