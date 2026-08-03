// Behavioural tests for linalg::Preconditioner and its implementations
// (PLAN.md step 20).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/iterative/Preconditioner.hpp"

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

M spd(std::size_t n, unsigned long seed) {
    M b = M::Random(n, n, seed);
    return b.transpose() * b + M::Identity(n) * static_cast<double>(n);
}

void test_identity_preconditioner() {
    IdentityPreconditioner<double> p;
    p.setup(spd(4, 171));
    assert(p.dimension() == 4);
    assert(p.isSymmetric());
    V r = V::Random(4, 172);
    assert(p.apply(r).isApprox(r, 0.0)); // exactly r
    assert(p.applyTranspose(r).isApprox(r, 0.0));
}

void test_jacobi_preconditioner() {
    M a{{2, 5}, {7, 4}};
    JacobiPreconditioner<double> p;
    p.setup(a);
    assert(p.dimension() == 2);
    V r{6, 8};
    V y = p.apply(r); // diag(A)^-1 r, off-diagonals ignored
    assert(std::abs(y(0) - 3.0) < 1e-12);
    assert(std::abs(y(1) - 2.0) < 1e-12);
    assert(p.isSymmetric()); // real diagonal
    // complex: applyTranspose divides by the conjugate diagonal
    Matrix<C> ac{{C(0, 2)}};
    JacobiPreconditioner<C> pc;
    pc.setup(ac);
    Vector<C> rc{C(2, 0)};
    Vector<C> yc = pc.apply(rc); // 2 / 2i = -i
    assert(std::abs(yc(0) - C(0, -1)) < 1e-12);
    Vector<C> yt = pc.applyTranspose(rc); // 2 / conj(2i) = 2 / (-2i) = i
    assert(std::abs(yt(0) - C(0, 1)) < 1e-12);
}

void test_ssor_diagonal_matrix() {
    // for diagonal A the SSOR operator collapses to M = D, any relaxation
    M a = M::Diagonal(V{2, 4, 8});
    SSORPreconditioner<double> p(1.0);
    p.setup(a);
    assert(p.dimension() == 3);
    V r{2, 4, 8};
    V y = p.apply(r);
    assert(std::abs(y(0) - 1.0) < 1e-10);
    assert(std::abs(y(1) - 1.0) < 1e-10);
    assert(std::abs(y(2) - 1.0) < 1e-10);
}

void test_ssor_symmetric_operator() {
    // SSOR of a symmetric A is symmetric: <M^-1 e_i, e_j> == <e_i, M^-1 e_j>
    M a = spd(4, 173);
    SSORPreconditioner<double> p(1.2);
    p.setup(a);
    assert(p.isSymmetric());
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            V ei = V::Unit(4, i), ej = V::Unit(4, j);
            double mij = p.apply(ei).dot(ej);
            double mji = p.apply(ej).dot(ei);
            assert(std::abs(mij - mji) < 1e-10);
        }
    }
    // relaxation estimate stays in the valid SOR range
    double w = p.optimalRelaxationEstimate();
    assert(w > 0.0 && w < 2.0);
}

void test_incomplete_cholesky_dense() {
    // dense SPD input has no zero pattern to preserve, so IC(0) is the
    // exact Cholesky factor and apply() inverts A exactly
    M a = spd(4, 174);
    IncompleteCholeskyPreconditioner<double> p;
    p.setup(a);
    assert(!p.breakdownOccurred());
    assert(p.isSymmetric());
    assert(p.dimension() == 4);
    V x = V::Random(4, 175);
    V y = p.apply(a * x); // M^-1 A x == x when M == A
    assert(y.isApprox(x, 1e-8));
}

void test_incomplete_cholesky_breakdown() {
    // indefinite input: the pivot goes non-positive
    M indef{{1, 2}, {2, 1}};
    IncompleteCholeskyPreconditioner<double> p;
    p.setup(indef);
    assert(p.breakdownOccurred());
    // a large diagonal shift rescues it
    IncompleteCholeskyPreconditioner<double> shifted(4.0);
    shifted.setup(indef);
    assert(!shifted.breakdownOccurred());
}

void test_polymorphic_use() {
    // Krylov solvers hold a base-class pointer; exercise that path
    M a = spd(3, 176);
    JacobiPreconditioner<double> jacobi;
    jacobi.setup(a);
    Preconditioner<double> *base = &jacobi;
    V r = V::Random(3, 177);
    V direct = jacobi.apply(r);
    V viaBase = base->apply(r);
    assert(viaBase.isApprox(direct, 0.0));
    assert(base->dimension() == 3);
}

} // namespace

int main() {
    run("identity_preconditioner", test_identity_preconditioner);
    run("jacobi_preconditioner", test_jacobi_preconditioner);
    run("ssor_diagonal_matrix", test_ssor_diagonal_matrix);
    run("ssor_symmetric_operator", test_ssor_symmetric_operator);
    run("incomplete_cholesky_dense", test_incomplete_cholesky_dense);
    run("incomplete_cholesky_breakdown", test_incomplete_cholesky_breakdown);
    run("polymorphic_use", test_polymorphic_use);
    return finish("Preconditioner");
}
