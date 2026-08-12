#include "linalg/decomp/Eigen.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> SymmetricEigen<T>::SymmetricEigen() {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::SymmetricEigen"
    );
}

template <typename T> SymmetricEigen<T>::SymmetricEigen(const Matrix<T> &a) {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::SymmetricEigen"
    );
}

template <typename T>
SymmetricEigen<T>::SymmetricEigen(
    const Matrix<T> &a,
    const Options   &options
) {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::SymmetricEigen"
    );
}

template <typename T> void SymmetricEigen<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::SymmetricEigen<T>::compute");
}

template <typename T> bool SymmetricEigen<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::SymmetricEigen<T>::isComputed");
}

template <typename T> bool SymmetricEigen<T>::converged() const {
    throw LinalgError("not implemented: linalg::SymmetricEigen<T>::converged");
}

template <typename T>
const Vector<typename SymmetricEigen<T>::Real> &SymmetricEigen<
    T>::eigenvalues() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::eigenvalues"
    );
}

template <typename T> const Matrix<T> &SymmetricEigen<T>::eigenvectors() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::eigenvectors"
    );
}

template <typename T> Vector<T> SymmetricEigen<T>::eigenvector(Index k) const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::eigenvector"
    );
}

template <typename T>
typename SymmetricEigen<T>::Real SymmetricEigen<T>::eigenvalue(Index k) const {
    throw LinalgError("not implemented: linalg::SymmetricEigen<T>::eigenvalue");
}

template <typename T>
Vector<typename SymmetricEigen<T>::Real> SymmetricEigen<T>::eigenvaluesInRange(
    Real lower,
    Real upper
) const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::eigenvaluesInRange"
    );
}

template <typename T>
Vector<typename SymmetricEigen<T>::Real> SymmetricEigen<T>::smallestEigenvalues(
    Index count
) const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::smallestEigenvalues"
    );
}

template <typename T>
Vector<typename SymmetricEigen<T>::Real> SymmetricEigen<T>::largestEigenvalues(
    Index count
) const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::largestEigenvalues"
    );
}

template <typename T>
typename SymmetricEigen<T>::Real SymmetricEigen<T>::spectralRadius() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::spectralRadius"
    );
}

template <typename T>
typename SymmetricEigen<T>::Real SymmetricEigen<T>::conditionNumber() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::conditionNumber"
    );
}

template <typename T>
typename SymmetricEigen<T>::Real SymmetricEigen<T>::residual(
    Index            k,
    const Matrix<T> &a
) const {
    throw LinalgError("not implemented: linalg::SymmetricEigen<T>::residual");
}

template <typename T>
Matrix<T> SymmetricEigen<T>::functionOfMatrix(
    const Vector<Real> &transformedEigenvalues
) const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::functionOfMatrix"
    );
}

template <typename T> Matrix<T> SymmetricEigen<T>::matrixSquareRoot() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::matrixSquareRoot"
    );
}

template <typename T> Matrix<T> SymmetricEigen<T>::matrixExponential() const {
    throw LinalgError(
        "not implemented: linalg::SymmetricEigen<T>::matrixExponential"
    );
}

template <typename T> Schur<T>::Schur() {
    throw LinalgError("not implemented: linalg::Schur<T>::Schur");
}

template <typename T> Schur<T>::Schur(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Schur<T>::Schur");
}

template <typename T> void Schur<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::Schur<T>::compute");
}

template <typename T> bool Schur<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Schur<T>::isComputed");
}

template <typename T> bool Schur<T>::converged() const {
    throw LinalgError("not implemented: linalg::Schur<T>::converged");
}

template <typename T> Matrix<T> Schur<T>::matrixT() const {
    throw LinalgError("not implemented: linalg::Schur<T>::matrixT");
}

template <typename T> Matrix<T> Schur<T>::matrixQ() const {
    throw LinalgError("not implemented: linalg::Schur<T>::matrixQ");
}

template <typename T>
Vector<typename Schur<T>::Complex> Schur<T>::eigenvalues() const {
    throw LinalgError("not implemented: linalg::Schur<T>::eigenvalues");
}

template <typename T>
typename Schur<T>::Index Schur<T>::iterationsUsed() const {
    throw LinalgError("not implemented: linalg::Schur<T>::iterationsUsed");
}

template <typename T>
void Schur<T>::reorder(const std::vector<Index> &desiredOrder) {
    throw LinalgError("not implemented: linalg::Schur<T>::reorder");
}

template <typename T>
void Schur<T>::francisDoubleShiftSweep(
    Index lo,
    Index hi
) {
    throw LinalgError(
        "not implemented: linalg::Schur<T>::francisDoubleShiftSweep"
    );
}

template <typename T>
typename Schur<T>::Index Schur<T>::detectDeflation(Real tolerance) const {
    throw LinalgError("not implemented: linalg::Schur<T>::detectDeflation");
}

template <typename T> GeneralEigen<T>::GeneralEigen() {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::GeneralEigen");
}

template <typename T> GeneralEigen<T>::GeneralEigen(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::GeneralEigen");
}

template <typename T> void GeneralEigen<T>::compute(const Matrix<T> &a) {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::compute");
}

template <typename T> bool GeneralEigen<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::isComputed");
}

template <typename T> bool GeneralEigen<T>::converged() const {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::converged");
}

template <typename T>
const Vector<typename GeneralEigen<T>::Complex> &GeneralEigen<
    T>::eigenvalues() const {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::eigenvalues");
}

template <typename T>
const Matrix<typename GeneralEigen<T>::Complex> &GeneralEigen<
    T>::rightEigenvectors() const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::rightEigenvectors"
    );
}

template <typename T>
const Matrix<typename GeneralEigen<T>::Complex> &GeneralEigen<
    T>::leftEigenvectors() const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::leftEigenvectors"
    );
}

template <typename T>
Vector<typename GeneralEigen<T>::Real> GeneralEigen<
    T>::conditionNumbers() const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::conditionNumbers"
    );
}

template <typename T>
bool GeneralEigen<T>::isDiagonalizable(Real tolerance) const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::isDiagonalizable"
    );
}

template <typename T>
typename GeneralEigen<T>::Real GeneralEigen<T>::spectralRadius() const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::spectralRadius"
    );
}

template <typename T>
typename GeneralEigen<T>::Real GeneralEigen<T>::spectralAbscissa() const {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::spectralAbscissa"
    );
}

template <typename T> void GeneralEigen<T>::balance() {
    throw LinalgError("not implemented: linalg::GeneralEigen<T>::balance");
}

template <typename T> void GeneralEigen<T>::solveTriangularEigenvectors() {
    throw LinalgError(
        "not implemented: linalg::GeneralEigen<T>::solveTriangularEigenvectors"
    );
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)         \
    template class SymmetricEigen<SCALAR>; \
    template class Schur<SCALAR>;          \
    template class GeneralEigen<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
