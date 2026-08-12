#include "linalg/iterative/KrylovSolver.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> KrylovSolver<T>::KrylovSolver() {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::KrylovSolver");
}

template <typename T> KrylovSolver<T>::KrylovSolver(const Options &options) {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::KrylovSolver");
}

template <typename T> KrylovSolver<T>::~KrylovSolver() {}

template <typename T>
void KrylovSolver<T>::setPreconditioner(
    const Preconditioner<T> *preconditioner
) {
    throw LinalgError(
        "not implemented: linalg::KrylovSolver<T>::setPreconditioner"
    );
}

template <typename T> void KrylovSolver<T>::setOptions(const Options &options) {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::setOptions");
}

template <typename T>
const typename KrylovSolver<T>::Options &KrylovSolver<T>::options() const {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::options");
}

template <typename T>
const typename KrylovSolver<T>::Report &KrylovSolver<T>::lastReport() const {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::lastReport");
}

template <typename T>
bool KrylovSolver<T>::hasConverged(
    Real residual,
    Real referenceResidual
) const {
    throw LinalgError("not implemented: linalg::KrylovSolver<T>::hasConverged");
}

template <typename T> void KrylovSolver<T>::recordResidual(Real residual) {
    throw LinalgError(
        "not implemented: linalg::KrylovSolver<T>::recordResidual"
    );
}

template <typename T> ConjugateGradient<T>::ConjugateGradient() {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::ConjugateGradient"
    );
}

template <typename T>
ConjugateGradient<T>::ConjugateGradient(
    const typename KrylovSolver<T>::Options &options
) {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::ConjugateGradient"
    );
}

template <typename T>
Vector<T> ConjugateGradient<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError("not implemented: linalg::ConjugateGradient<T>::solve");
}

template <typename T>
Vector<T> ConjugateGradient<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &initialGuess
) {
    throw LinalgError("not implemented: linalg::ConjugateGradient<T>::solve");
}

template <typename T>
Vector<typename ConjugateGradient<T>::Real> ConjugateGradient<
    T>::lanczosDiagonal() const {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::lanczosDiagonal"
    );
}

template <typename T>
Vector<typename ConjugateGradient<T>::Real> ConjugateGradient<
    T>::lanczosSubdiagonal() const {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::lanczosSubdiagonal"
    );
}

template <typename T>
typename ConjugateGradient<T>::Real ConjugateGradient<
    T>::conditionNumberEstimate() const {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::conditionNumberEstimate"
    );
}

template <typename T>
Vector<typename ConjugateGradient<T>::Real> ConjugateGradient<
    T>::energyNormHistory() const {
    throw LinalgError(
        "not implemented: linalg::ConjugateGradient<T>::energyNormHistory"
    );
}

template <typename T> LSQR<T>::LSQR() {
    throw LinalgError("not implemented: linalg::LSQR<T>::LSQR");
}

template <typename T>
LSQR<T>::LSQR(const typename KrylovSolver<T>::Options &options) {
    throw LinalgError("not implemented: linalg::LSQR<T>::LSQR");
}

template <typename T>
Vector<T> LSQR<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError("not implemented: linalg::LSQR<T>::solve");
}

template <typename T>
Vector<T> LSQR<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &initialGuess
) {
    throw LinalgError("not implemented: linalg::LSQR<T>::solve");
}

template <typename T> void LSQR<T>::setRegularization(Real lambda) {
    throw LinalgError("not implemented: linalg::LSQR<T>::setRegularization");
}

template <typename T>
typename LSQR<T>::Real LSQR<T>::matrixNormEstimate() const {
    throw LinalgError("not implemented: linalg::LSQR<T>::matrixNormEstimate");
}

template <typename T>
typename LSQR<T>::Real LSQR<T>::conditionNumberEstimate() const {
    throw LinalgError(
        "not implemented: linalg::LSQR<T>::conditionNumberEstimate"
    );
}

template <typename T> GMRES<T>::GMRES() {
    throw LinalgError("not implemented: linalg::GMRES<T>::GMRES");
}

template <typename T>
GMRES<T>::GMRES(const typename KrylovSolver<T>::Options &options) {
    throw LinalgError("not implemented: linalg::GMRES<T>::GMRES");
}

template <typename T>
Vector<T> GMRES<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError("not implemented: linalg::GMRES<T>::solve");
}

template <typename T>
Vector<T> GMRES<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &initialGuess
) {
    throw LinalgError("not implemented: linalg::GMRES<T>::solve");
}

template <typename T> void GMRES<T>::setRestart(Index restart) {
    throw LinalgError("not implemented: linalg::GMRES<T>::setRestart");
}

template <typename T>
void GMRES<T>::setOrthogonalization(typename Orthogonalization::Kind kind) {
    throw LinalgError(
        "not implemented: linalg::GMRES<T>::setOrthogonalization"
    );
}

template <typename T> typename GMRES<T>::Index GMRES<T>::restart() const {
    throw LinalgError("not implemented: linalg::GMRES<T>::restart");
}

template <typename T> const Matrix<T> &GMRES<T>::arnoldiBasis() const {
    throw LinalgError("not implemented: linalg::GMRES<T>::arnoldiBasis");
}

template <typename T> const Matrix<T> &GMRES<T>::hessenbergMatrix() const {
    throw LinalgError("not implemented: linalg::GMRES<T>::hessenbergMatrix");
}

template <typename T> typename GMRES<T>::Index GMRES<T>::restartCount() const {
    throw LinalgError("not implemented: linalg::GMRES<T>::restartCount");
}

template <typename T>
void GMRES<T>::arnoldiStep(
    const Matrix<T> &a,
    Index            k
) {
    throw LinalgError("not implemented: linalg::GMRES<T>::arnoldiStep");
}

template <typename T> void GMRES<T>::applyStoredRotations(Index k) {
    throw LinalgError(
        "not implemented: linalg::GMRES<T>::applyStoredRotations"
    );
}

template <typename T> void GMRES<T>::solveProjectedProblem(Index k) {
    throw LinalgError(
        "not implemented: linalg::GMRES<T>::solveProjectedProblem"
    );
}

template <typename T> BiCGSTAB<T>::BiCGSTAB() {
    throw LinalgError("not implemented: linalg::BiCGSTAB<T>::BiCGSTAB");
}

template <typename T>
BiCGSTAB<T>::BiCGSTAB(const typename KrylovSolver<T>::Options &options) {
    throw LinalgError("not implemented: linalg::BiCGSTAB<T>::BiCGSTAB");
}

template <typename T>
Vector<T> BiCGSTAB<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b
) {
    throw LinalgError("not implemented: linalg::BiCGSTAB<T>::solve");
}

template <typename T>
Vector<T> BiCGSTAB<T>::solve(
    const Matrix<T> &a,
    const Vector<T> &b,
    const Vector<T> &initialGuess
) {
    throw LinalgError("not implemented: linalg::BiCGSTAB<T>::solve");
}

template <typename T> bool BiCGSTAB<T>::breakdownDetected() const {
    throw LinalgError(
        "not implemented: linalg::BiCGSTAB<T>::breakdownDetected"
    );
}

template <typename T>
typename BiCGSTAB<T>::Real BiCGSTAB<T>::smallestOmega() const {
    throw LinalgError("not implemented: linalg::BiCGSTAB<T>::smallestOmega");
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)            \
    template class KrylovSolver<SCALAR>;      \
    template class ConjugateGradient<SCALAR>; \
    template class LSQR<SCALAR>;              \
    template class GMRES<SCALAR>;             \
    template class BiCGSTAB<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
