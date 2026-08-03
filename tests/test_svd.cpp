// Behavioural tests for linalg::SVD (PLAN.md step 15).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// PLAN.md oracles: U*S*V^H == A, U/V orthogonal, singular values of a
// diagonal matrix are |diag| sorted, Hilbert condition number matches
// the literature.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/SVD.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using C = std::complex<double>;

namespace {

// U S V^H reconstruction from whatever vector mode compute() used.
M reconstruct(const SVD<double> &svd) {
    const M &u = svd.matrixU();
    const M &v = svd.matrixV();
    std::size_t k = svd.singularValues().size();
    M s = M::Zeros(u.cols(), v.cols());
    for (std::size_t i = 0; i < k && i < s.rows() && i < s.cols(); ++i)
        s(i, i) = svd.singularValues()(i);
    return u * s * v.transpose();
}

void test_decomposition_identity() {
    M a = M::Random(5, 4, 121);
    SVD<double> svd(a);
    assert(svd.isComputed());
    assert(svd.converged());
    assert(reconstruct(svd).isApprox(a, 1e-8)); // A == U S V^H
    const M &u = svd.matrixU();
    const M &v = svd.matrixV();
    assert((u.transpose() * u).isApprox(M::Identity(u.cols()), 1e-10));
    assert((v.transpose() * v).isApprox(M::Identity(v.cols()), 1e-10));
}

void test_singular_values_sorted_nonneg() {
    // singular values of a diagonal matrix are |diag|, sorted descending
    M a = M::Diagonal(V{3, -5, 1});
    SVD<double> svd(a);
    const V &s = svd.singularValues();
    assert(s.size() == 3);
    assert(std::abs(s(0) - 5.0) < 1e-10);
    assert(std::abs(s(1) - 3.0) < 1e-10);
    assert(std::abs(s(2) - 1.0) < 1e-10);
    for (std::size_t i = 0; i < 3; ++i)
        assert(s(i) >= 0.0);
    assert(std::abs(svd.largestSingularValue() - 5.0) < 1e-10);
    assert(std::abs(svd.smallestSingularValue() - 1.0) < 1e-10);
    assert(std::abs(svd.conditionNumber() - 5.0) < 1e-10);
    assert(std::abs(svd.spectralNorm() - 5.0) < 1e-10);
    assert(std::abs(svd.nuclearNorm() - 9.0) < 1e-10);
}

void test_hilbert_condition_number() {
    // literature value: cond_2(Hilbert(4)) ~ 1.5514e4
    SVD<double> svd(M::Hilbert(4));
    double cond = svd.conditionNumber();
    assert(cond > 1.5e4 && cond < 1.6e4);
}

void test_rank_and_spaces() {
    M rank1{{1, 2}, {2, 4}};
    SVD<double> svd(rank1);
    assert(svd.rank(1e-10) == 1);
    M ns = svd.nullSpace(1e-10);
    assert(ns.cols() == 1);
    assert((rank1 * ns).isApprox(M::Zeros(2, 1), 1e-8));
    assert((ns.transpose() * ns).isApprox(M::Identity(1), 1e-10));
    M cs = svd.columnSpace(1e-10);
    assert(cs.cols() == 1);
    // column space of [[1,2],[2,4]] is span{(1,2)}
    assert(std::abs(std::abs(cs(1, 0) / cs(0, 0)) - 2.0) < 1e-8);
}

void test_pseudo_inverse_and_solve() {
    M a = M::Random(6, 3, 122);
    SVD<double> svd(a);
    M p = svd.pseudoInverse(1e-12);
    assert((a * p * a).isApprox(a, 1e-8)); // Moore-Penrose
    assert((p * a * p).isApprox(p, 1e-8));
    // solve: least squares, gradient vanishes
    V b = V::Random(6, 123);
    V x = svd.solve(b);
    assert((a.transpose() * (b - a * x)).norm() < 1e-8);
    // Tikhonov with lambda = 0 matches the plain solve
    V xt = svd.solveTikhonov(b, 0.0);
    assert(xt.isApprox(x, 1e-8));
    // heavier damping shrinks the solution norm
    V xd = svd.solveTikhonov(b, 10.0);
    assert(xd.norm() < x.norm());
    // truncation to full rank changes nothing
    V xk = svd.solveTruncated(b, 3);
    assert(xk.isApprox(x, 1e-8));
}

void test_low_rank_approximation() {
    // Eckart-Young: the best rank-k error is sigma_{k+1}
    M a = M::Diagonal(V{3, 1});
    SVD<double> svd(a);
    M a1 = svd.lowRankApproximation(1);
    assert(std::abs((a - a1).frobeniusNorm() - 1.0) < 1e-8);
    // rank-1 approximation of a rank-1 matrix is the matrix itself
    M r1{{1, 2}, {2, 4}};
    SVD<double> s1(r1);
    assert(s1.lowRankApproximation(1).isApprox(r1, 1e-8));
}

void test_polar_decomposition() {
    M a = M::Random(4, 4, 124);
    SVD<double> svd(a);
    M w = svd.orthogonalPolarFactor();
    M p = svd.hermitianPolarFactor();
    assert(w.isOrthogonal(1e-8));
    assert(p.isSymmetric(1e-8));
    assert((w * p).isApprox(a, 1e-8)); // A == W P
    // P is positive semidefinite: x^H P x >= 0 on a few probes
    for (unsigned long seed = 1; seed <= 3; ++seed) {
        V x = V::Random(4, seed);
        assert(x.dot(p * x) >= -1e-8);
    }
}

void test_complex() {
    Matrix<C> a = Matrix<C>::Random(4, 3, 125);
    SVD<C> svd(a);
    const Vector<double> &s = svd.singularValues();
    for (std::size_t i = 0; i < s.size(); ++i)
        assert(s(i) >= 0.0);
    const Matrix<C> &u = svd.matrixU();
    const Matrix<C> &v = svd.matrixV();
    std::size_t k = s.size();
    Matrix<C> sm = Matrix<C>::Zeros(u.cols(), v.cols());
    for (std::size_t i = 0; i < k && i < sm.rows() && i < sm.cols(); ++i)
        sm(i, i) = C(s(i), 0.0);
    assert((u * sm * v.conjugateTranspose()).isApprox(a, 1e-8));
    assert((u.conjugateTranspose() * u)
               .isApprox(Matrix<C>::Identity(u.cols()), 1e-10));
}

} // namespace

int main() {
    run("decomposition_identity", test_decomposition_identity);
    run("singular_values_sorted_nonneg", test_singular_values_sorted_nonneg);
    run("hilbert_condition_number", test_hilbert_condition_number);
    run("rank_and_spaces", test_rank_and_spaces);
    run("pseudo_inverse_and_solve", test_pseudo_inverse_and_solve);
    run("low_rank_approximation", test_low_rank_approximation);
    run("polar_decomposition", test_polar_decomposition);
    run("complex", test_complex);
    return finish("SVD");
}
