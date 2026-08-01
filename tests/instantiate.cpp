// Link check for the explicit instantiation lists.
//
// Each reference below odr-uses one member of a class template, so the
// symbol must exist in libLinalg. If a class is missing from a
// LINALG_INSTANTIATE list in src/, this fails to link and names the culprit.
//
// The definitions are no longer in the headers, so this is the only thing
// standing between a missing instantiation and a downstream user hitting it.

#include "linalg/linalg.hpp"

#include <complex>

namespace {

template <typename T>
void require(T)
{
}

template <typename S>
void checkScalar()
{
    using namespace linalg;

    require(&Matrix<S>::rows);
    require(&Vector<S>::size);
    require(&MatrixView<S>::rowStride);
    require(&ConstMatrixView<S>::rowStride);
    require(&NumericTraits<S>::epsilon);

    require(&Householder<S>::beta);
    require(&HouseholderSequence<S>::count);
    require(&Givens<S>::cosine);
    require(&GivensSequence<S>::count);
    require(&Norm<S>::matrixFrobenius);
    require(&Kernels<S>::tunedBlockSizes);

    require(&LU<S>::compute);
    require(&FullPivLU<S>::compute);
    require(&Cholesky<S>::compute);
    require(&LDLT<S>::compute);
    require(&QR<S>::compute);
    require(&ColPivQR<S>::compute);
    require(&GramSchmidt<S>::compute);
    require(&Hessenberg<S>::compute);
    require(&Tridiagonal<S>::compute);
    require(&Bidiagonal<S>::compute);
    require(&SVD<S>::compute);
    require(&SymmetricEigen<S>::compute);
    require(&Schur<S>::compute);
    require(&GeneralEigen<S>::compute);

    require(&TriangularSolver<S>::inverse);
    require(&LinearSolver<S>::selectMethod);
    require(&LeastSquares<S>::solveMinimumNorm);
    require(&GaussNewton<S>::minimize);

    require(&IdentityPreconditioner<S>::setup);
    require(&JacobiPreconditioner<S>::setup);
    require(&SSORPreconditioner<S>::setup);
    require(&IncompleteCholeskyPreconditioner<S>::setup);
    require(&KrylovSolver<S>::setPreconditioner);
    require(&ConjugateGradient<S>::lanczosDiagonal);
    require(&LSQR<S>::setRegularization);
    require(&GMRES<S>::setRestart);
    require(&BiCGSTAB<S>::breakdownDetected);
    require(&PowerIteration<S>::rayleighQuotient);
    require(&Lanczos<S>::compute);
    require(&ImplicitlyRestartedArnoldi<S>::compute);
}

} // namespace

int main()
{
    checkScalar<float>();
    checkScalar<double>();
    checkScalar<std::complex<float>>();
    checkScalar<std::complex<double>>();
    return 0;
}
