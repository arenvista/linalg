// Behavioural tests for linalg::LinearSolver (PLAN.md step 18).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/solve/LinearSolver.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using LS = LinearSolver<double>;
using Method = LS::Method::Kind;

namespace {

M spd(std::size_t n, unsigned long seed) {
    M b = M::Random(n, n, seed);
    return b.transpose() * b + M::Identity(n) * static_cast<double>(n);
}

LS::Options opts(Method m) {
    LS::Options o;
    o.method = m;
    o.iterativeRefinement = false;
    o.maxRefinementSteps = 0;
    o.tolerance = 1e-12;
    o.equilibrate = false;
    return o;
}

void test_automatic_spd() {
    M a = spd(5, 131);
    V b = V::Random(5, 132);
    LS solver;
    V x = solver.solve(a, b);
    assert((a * x).isApprox(b, 1e-8));
    const auto &report = solver.lastReport();
    assert(report.succeeded);
    // the report's residual must match the actual residual
    double actual = (b - a * x).norm();
    assert(std::abs(report.residualNorm - actual) < 1e-10);
    assert(report.backwardError < 1e-12); // well-conditioned exact solve
    // Automatic recognizes SPD and picks Cholesky
    assert(report.chosenMethod == Method::Cholesky);
}

void test_method_selection() {
    LS solver;
    assert(solver.selectMethod(spd(4, 133)) == Method::Cholesky);
    // symmetric indefinite: Cholesky must not be the answer
    M indef{{1, 2}, {2, 1}};
    Method mi = solver.selectMethod(indef);
    assert(mi != Method::Cholesky);
    // general square, nonsymmetric
    M gen = M::Random(4, 4, 134);
    Method mg = solver.selectMethod(gen);
    assert(mg != Method::Cholesky && mg != Method::LDLT);
    // non-square never gets a square-only factorization
    M rect = M::Random(6, 3, 135);
    Method mr = solver.selectMethod(rect);
    assert(mr == Method::QR || mr == Method::ColPivQR || mr == Method::SVD);
}

void test_explicit_methods() {
    M a = M::Random(5, 5, 136);
    V b = V::Random(5, 137);
    for (Method m : {Method::LU, Method::FullPivLU, Method::QR,
                     Method::ColPivQR, Method::SVD}) {
        LS solver(opts(m));
        V x = solver.solve(a, b);
        assert((a * x).isApprox(b, 1e-7));
        assert(solver.lastReport().chosenMethod == m);
    }
    // Hermitian methods on an SPD input
    M s = spd(5, 138);
    for (Method m : {Method::Cholesky, Method::LDLT}) {
        LS solver(opts(m));
        V x = solver.solve(s, b);
        assert((s * x).isApprox(b, 1e-7));
    }
}

void test_matrix_rhs() {
    M a = M::Random(4, 4, 139);
    M b = M::Random(4, 3, 140);
    LS solver;
    M x = solver.solve(a, b);
    assert((a * x).isApprox(b, 1e-8));
}

void test_least_squares_dispatch() {
    // overdetermined: solve() minimizes the residual instead of failing
    M a = M::Random(8, 3, 141);
    V b = V::Random(8, 142);
    LS solver;
    V x = solver.solve(a, b);
    assert((a.transpose() * (b - a * x)).norm() < 1e-8);
}

void test_iterative_refinement() {
    M a = M::Hilbert(8);
    V xTrue = V::Ones(8);
    V b = a * xTrue;
    LS::Options o = opts(Method::Automatic);
    o.iterativeRefinement = true;
    o.maxRefinementSteps = 5;
    LS solver(o);
    V x = solver.solve(a, b);
    const auto &report = solver.lastReport();
    assert(report.succeeded);
    assert(report.refinementSteps <= 5);
    // refined solve of an ill-conditioned system keeps a tiny backward error
    assert(report.backwardError < 1e-12);
}

void test_refine() {
    M a = M::Random(5, 5, 143);
    V b = V::Random(5, 144);
    LS solver;
    V x = solver.solve(a, b);
    // perturb, then refine back
    V bad = x;
    bad(0) += 1e-4;
    V polished = solver.refine(a, b, bad, 3);
    assert((b - a * polished).norm() < (b - a * bad).norm());
}

void test_equilibrate() {
    // wildly scaled rows/columns
    M a{{1e8, 2e8}, {3e-8, 4e-8}};
    M scaled = a;
    V rowScale, colScale;
    LS solver;
    solver.equilibrate(scaled, rowScale, colScale);
    // reconstruct: scaled == D_r A D_c elementwise
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            assert(std::abs(scaled(i, j) - rowScale(i) * a(i, j) * colScale(j)) <
                   1e-8 * std::abs(scaled(i, j)) + 1e-300);
    // rows now have comparable norms (they differed by 16 orders before)
    double r0 = scaled.rowView(0).toMatrix().frobeniusNorm();
    double r1 = scaled.rowView(1).toMatrix().frobeniusNorm();
    assert(r0 / r1 < 1e3 && r1 / r0 < 1e3);
}

} // namespace

int main() {
    run("automatic_spd", test_automatic_spd);
    run("method_selection", test_method_selection);
    run("explicit_methods", test_explicit_methods);
    run("matrix_rhs", test_matrix_rhs);
    run("least_squares_dispatch", test_least_squares_dispatch);
    run("iterative_refinement", test_iterative_refinement);
    run("refine", test_refine);
    run("equilibrate", test_equilibrate);
    return finish("LinearSolver");
}
