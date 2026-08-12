#include "linalg/iterative/EigenSolver.hpp"

#include "linalg/Instantiate.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

namespace linalg {

template <typename T> PowerIteration<T>::PowerIteration() {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::PowerIteration"
    );
}

template <typename T>
PowerIteration<T>::PowerIteration(const Options &options) {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::PowerIteration"
    );
}

template <typename T>
typename PowerIteration<T>::Real PowerIteration<T>::dominantEigenvalue(
    const Matrix<T> &a
) {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::dominantEigenvalue"
    );
}

template <typename T>
Vector<T> PowerIteration<T>::dominantEigenvector(const Matrix<T> &a) {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::dominantEigenvector"
    );
}

template <typename T>
typename PowerIteration<T>::Real PowerIteration<T>::shiftedInverseEigenvalue(
    const Matrix<T> &a,
    Real             shift
) {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::shiftedInverseEigenvalue"
    );
}

template <typename T>
typename PowerIteration<T>::Real PowerIteration<T>::rayleighQuotient(
    const Matrix<T> &a,
    const Vector<T> &x
) const {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::rayleighQuotient"
    );
}

template <typename T>
typename PowerIteration<T>::Index PowerIteration<T>::iterationsUsed() const {
    throw LinalgError(
        "not implemented: linalg::PowerIteration<T>::iterationsUsed"
    );
}

template <typename T> bool PowerIteration<T>::converged() const {
    throw LinalgError("not implemented: linalg::PowerIteration<T>::converged");
}

template <typename T> Lanczos<T>::Lanczos() {
    throw LinalgError("not implemented: linalg::Lanczos<T>::Lanczos");
}

template <typename T>
Lanczos<T>::Lanczos(
    Index                              krylovDimension,
    typename Reorthogonalization::Kind strategy
) {
    throw LinalgError("not implemented: linalg::Lanczos<T>::Lanczos");
}

template <typename T>
void Lanczos<T>::compute(
    const Matrix<T> &a,
    const Vector<T> &startVector
) {
    throw LinalgError("not implemented: linalg::Lanczos<T>::compute");
}

template <typename T> bool Lanczos<T>::isComputed() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::isComputed");
}

template <typename T>
Vector<typename Lanczos<T>::Real> Lanczos<T>::diagonal() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::diagonal");
}

template <typename T>
Vector<typename Lanczos<T>::Real> Lanczos<T>::subdiagonal() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::subdiagonal");
}

template <typename T> Matrix<T> Lanczos<T>::basis() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::basis");
}

template <typename T>
Vector<typename Lanczos<T>::Real> Lanczos<T>::ritzValues() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::ritzValues");
}

template <typename T> Matrix<T> Lanczos<T>::ritzVectors() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::ritzVectors");
}

template <typename T>
Vector<typename Lanczos<T>::Real> Lanczos<T>::ritzResidualBounds() const {
    throw LinalgError(
        "not implemented: linalg::Lanczos<T>::ritzResidualBounds"
    );
}

template <typename T>
typename Lanczos<T>::Real Lanczos<T>::orthogonalityLoss() const {
    throw LinalgError("not implemented: linalg::Lanczos<T>::orthogonalityLoss");
}

template <typename T>
ImplicitlyRestartedArnoldi<T>::ImplicitlyRestartedArnoldi() {
    throw LinalgError(
        "not implemented: "
        "linalg::ImplicitlyRestartedArnoldi<T>::ImplicitlyRestartedArnoldi"
    );
}

template <typename T>
ImplicitlyRestartedArnoldi<T>::ImplicitlyRestartedArnoldi(
    const Options &options
) {
    throw LinalgError(
        "not implemented: "
        "linalg::ImplicitlyRestartedArnoldi<T>::ImplicitlyRestartedArnoldi"
    );
}

template <typename T>
void ImplicitlyRestartedArnoldi<T>::compute(
    const Matrix<T> &a,
    const Vector<T> &startVector
) {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::compute"
    );
}

template <typename T> bool ImplicitlyRestartedArnoldi<T>::isComputed() const {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::isComputed"
    );
}

template <typename T> bool ImplicitlyRestartedArnoldi<T>::converged() const {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::converged"
    );
}

template <typename T>
Vector<T> ImplicitlyRestartedArnoldi<T>::ritzValues() const {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::ritzValues"
    );
}

template <typename T>
Matrix<T> ImplicitlyRestartedArnoldi<T>::ritzVectors() const {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::ritzVectors"
    );
}

template <typename T>
typename ImplicitlyRestartedArnoldi<T>::Index ImplicitlyRestartedArnoldi<
    T>::restartsUsed() const {
    throw LinalgError(
        "not implemented: linalg::ImplicitlyRestartedArnoldi<T>::restartsUsed"
    );
}

template <typename T>
void ImplicitlyRestartedArnoldi<T>::expandKrylovSpace(
    const Matrix<T> &a,
    Index            from,
    Index            to
) {
    throw LinalgError(
        "not implemented: "
        "linalg::ImplicitlyRestartedArnoldi<T>::expandKrylovSpace"
    );
}

template <typename T>
void ImplicitlyRestartedArnoldi<T>::applyShiftedQRRestart(
    const Vector<T> &shifts
) {
    throw LinalgError(
        "not implemented: "
        "linalg::ImplicitlyRestartedArnoldi<T>::applyShiftedQRRestart"
    );
}

// Explicit instantiation. Every scalar the library ships is
// compiled here once, rather than in each including translation unit.
#define LINALG_INSTANTIATE(SCALAR)         \
    template class PowerIteration<SCALAR>; \
    template class Lanczos<SCALAR>;        \
    template class ImplicitlyRestartedArnoldi<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)

#undef LINALG_INSTANTIATE

} // namespace linalg
