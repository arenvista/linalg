// Behavioural tests for linalg::SymmetricEigen / Schur / GeneralEigen
// (PLAN.md steps 14 and 16).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// Eigenvalues are compared as sets (sorted per test) so the tests do not
// depend on the Options::order default.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/Eigen.hpp"

#include "harness.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using C = std::complex<double>;

namespace {

std::vector<double> sortedValues(const V &v) {
    std::vector<double> out;
    for (std::size_t i = 0; i < v.size(); ++i)
        out.push_back(v(i));
    std::sort(out.begin(), out.end());
    return out;
}

// ---- SymmetricEigen --------------------------------------------------------

void test_symmetric_known_spectrum() {
    M a{{2, 1}, {1, 2}}; // eigenvalues 1 and 3
    SymmetricEigen<double> eig(a);
    assert(eig.isComputed());
    assert(eig.converged());
    auto vals = sortedValues(eig.eigenvalues());
    assert(std::abs(vals[0] - 1.0) < 1e-10);
    assert(std::abs(vals[1] - 3.0) < 1e-10);
    assert(std::abs(eig.spectralRadius() - 3.0) < 1e-10);
    assert(std::abs(eig.conditionNumber() - 3.0) < 1e-10);
}

void test_symmetric_eigenpairs() {
    M a = M::RandomSymmetric(6, 111);
    SymmetricEigen<double> eig(a);
    const M &vecs = eig.eigenvectors();
    assert(vecs.isOrthogonal(1e-8));
    // every pair satisfies A q == lambda q
    for (std::size_t k = 0; k < 6; ++k) {
        V q = eig.eigenvector(k);
        double lambda = eig.eigenvalue(k);
        assert((a * q - q * lambda).norm() < 1e-8);
        assert(eig.residual(k, a) < 1e-8);
    }
    // reconstruction with f = identity: Q diag(lambda) Q^H == A
    assert(eig.functionOfMatrix(eig.eigenvalues()).isApprox(a, 1e-8));
}

void test_symmetric_partial_spectra() {
    M a = M::Diagonal(V{-2, 1, 3, 7});
    SymmetricEigen<double> eig(a);
    auto smallest = sortedValues(eig.smallestEigenvalues(2));
    assert(std::abs(smallest[0] - (-2.0)) < 1e-10);
    assert(std::abs(smallest[1] - 1.0) < 1e-10);
    auto largest = sortedValues(eig.largestEigenvalues(2));
    assert(std::abs(largest[0] - 3.0) < 1e-10);
    assert(std::abs(largest[1] - 7.0) < 1e-10);
    auto inRange = eig.eigenvaluesInRange(0.0, 5.0);
    assert(inRange.size() == 2); // 1 and 3
}

void test_symmetric_matrix_functions() {
    // SPD matrix: square root exists and squares back
    M a{{5, 2}, {2, 5}};
    SymmetricEigen<double> eig(a);
    M root = eig.matrixSquareRoot();
    assert((root * root).isApprox(a, 1e-8));
    // exp(0) == I
    SymmetricEigen<double> zero(M::Zeros(3, 3));
    assert(zero.matrixExponential().isApprox(M::Identity(3), 1e-10));
}

void test_symmetric_complex_hermitian() {
    Matrix<C> a = Matrix<C>::RandomSymmetric(4, 112); // Hermitian
    SymmetricEigen<C> eig(a);
    // eigenvalues are real by type (Vector<Real>); pairs must check out
    for (std::size_t k = 0; k < 4; ++k) {
        Vector<C> q = eig.eigenvector(k);
        C lambda = C(eig.eigenvalue(k), 0.0);
        assert((a * q - q * lambda).norm() < 1e-8);
    }
    assert(eig.eigenvectors().isOrthogonal(1e-8)); // unitary
}

// ---- Schur -----------------------------------------------------------------

void test_schur_identity() {
    M a = M::Random(5, 5, 113);
    Schur<double> schur(a);
    assert(schur.isComputed());
    assert(schur.converged());
    M t = schur.matrixT();
    M q = schur.matrixQ();
    assert(q.isOrthogonal(1e-8));
    assert((q * t * q.transpose()).isApprox(a, 1e-7)); // A == Q T Q^H
    // quasi-triangular: zero below the first subdiagonal
    for (std::size_t i = 2; i < 5; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            assert(std::abs(t(i, j)) < 1e-8);
    assert(schur.iterationsUsed() > 0);
}

void test_schur_eigenvalues() {
    // rotation matrix: eigenvalues are exactly +/- i
    M rot{{0, -1}, {1, 0}};
    Schur<double> schur(rot);
    auto vals = schur.eigenvalues();
    assert(vals.size() == 2);
    // both on the unit circle, purely imaginary, conjugate pair
    assert(std::abs(std::abs(vals(0)) - 1.0) < 1e-8);
    assert(std::abs(vals(0).real()) < 1e-8);
    assert(std::abs((vals(0) + vals(1)).imag()) < 1e-8);
    // eigenvalue sum equals the trace for a general matrix
    M a = M::Random(4, 4, 114);
    Schur<double> s2(a);
    auto v2 = s2.eigenvalues();
    C sum(0, 0);
    for (std::size_t i = 0; i < v2.size(); ++i)
        sum += v2(i);
    assert(std::abs(sum - C(a.trace(), 0.0)) < 1e-7);
}

void test_schur_complex_scalar() {
    // complex T: strictly upper triangular, eigenvalues on the diagonal
    Matrix<C> a = Matrix<C>::Random(4, 4, 115);
    Schur<C> schur(a);
    Matrix<C> t = schur.matrixT();
    Matrix<C> q = schur.matrixQ();
    assert((q * t * q.conjugateTranspose()).isApprox(a, 1e-7));
    assert(t.isTriangular(Triangle::Kind::Upper, 1e-8));
}

// ---- GeneralEigen ----------------------------------------------------------

void test_general_known() {
    // diagonal matrix: spectrum read off directly
    M a = M::Diagonal(V{1, 2});
    GeneralEigen<double> eig(a);
    assert(eig.converged());
    auto vals = eig.eigenvalues();
    std::vector<double> re{vals(0).real(), vals(1).real()};
    std::sort(re.begin(), re.end());
    assert(std::abs(re[0] - 1.0) < 1e-8 && std::abs(re[1] - 2.0) < 1e-8);
    assert(std::abs(vals(0).imag()) < 1e-8);
    // rotation: complex pair, radius 1, abscissa 0
    GeneralEigen<double> rot(M{{0, -1}, {1, 0}});
    assert(std::abs(rot.spectralRadius() - 1.0) < 1e-8);
    assert(std::abs(rot.spectralAbscissa() - 0.0) < 1e-8);
}

void test_general_eigenpairs() {
    M a = M::Random(4, 4, 116);
    GeneralEigen<double> eig(a);
    // complexify A once to check A x == lambda x per column
    Matrix<C> ac(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            ac(i, j) = C(a(i, j), 0.0);
    const auto &vals = eig.eigenvalues();
    const auto &right = eig.rightEigenvectors();
    for (std::size_t k = 0; k < 4; ++k) {
        Vector<C> x = right.col(k);
        assert(std::abs(x.norm() - 1.0) < 1e-8); // unit columns
        assert((ac * x - x * vals(k)).norm() < 1e-7);
    }
    // left vectors: y^H A == lambda y^H  <=>  A^H y == conj(lambda) y
    const auto &left = eig.leftEigenvectors();
    for (std::size_t k = 0; k < 4; ++k) {
        Vector<C> y = left.col(k);
        assert((ac.conjugateTranspose() * y - y * std::conj(vals(k))).norm() <
               1e-7);
    }
    // a random matrix is diagonalizable almost surely, with modest
    // eigenvalue condition numbers
    assert(eig.isDiagonalizable(1e-8));
    auto conds = eig.conditionNumbers();
    for (std::size_t k = 0; k < conds.size(); ++k)
        assert(conds(k) >= 1.0 - 1e-10);
}

} // namespace

int main() {
    run("symmetric_known_spectrum", test_symmetric_known_spectrum);
    run("symmetric_eigenpairs", test_symmetric_eigenpairs);
    run("symmetric_partial_spectra", test_symmetric_partial_spectra);
    run("symmetric_matrix_functions", test_symmetric_matrix_functions);
    run("symmetric_complex_hermitian", test_symmetric_complex_hermitian);
    run("schur_identity", test_schur_identity);
    run("schur_eigenvalues", test_schur_eigenvalues);
    run("schur_complex_scalar", test_schur_complex_scalar);
    run("general_known", test_general_known);
    run("general_eigenpairs", test_general_eigenpairs);
    return finish("Eigen");
}
