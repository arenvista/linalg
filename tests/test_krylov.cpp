// Behavioural tests for linalg::KrylovSolver and its implementations
// (PLAN.md step 21).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// PLAN.md oracle: compare against the direct solvers on the same A, b;
// CG on SPD matrices converges in <= n iterations in exact arithmetic.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/iterative/KrylovSolver.hpp"
#include "linalg/iterative/Preconditioner.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;

namespace {

M spd(std::size_t n, unsigned long seed) {
    M b = M::Random(n, n, seed);
    return b.transpose() * b + M::Identity(n) * static_cast<double>(n);
}

// Diagonally dominant nonsymmetric matrix: safe territory for GMRES/BiCGSTAB.
M nonsym(std::size_t n, unsigned long seed) {
    M a = M::Random(n, n, seed);
    return a + M::Identity(n) * (2.0 * static_cast<double>(n));
}

KrylovSolver<double>::Options krylovOpts(bool history = false) {
    KrylovSolver<double>::Options o;
    o.maxIterations = 500;
    o.relativeTolerance = 1e-12;
    o.absoluteTolerance = 1e-14;
    o.recordHistory = history;
    o.useInitialGuess = false;
    return o;
}

void test_cg_solves_spd() {
    std::size_t n = 8;
    M a = spd(n, 181);
    V b = V::Random(n, 182);
    ConjugateGradient<double> cg(krylovOpts());
    V x = cg.solve(a, b);
    assert((a * x).isApprox(b, 1e-7));
    const auto &report = cg.lastReport();
    assert(report.converged);
    assert(!report.breakdown);
    // exact-arithmetic bound with a little floating-point slack
    assert(report.iterations <= n + 2);
    // report residual matches reality
    assert(std::abs(report.finalResidual - (b - a * x).norm()) < 1e-9);
}

void test_cg_history_and_diagnostics() {
    M a = spd(6, 183);
    V b = V::Random(6, 184);
    ConjugateGradient<double> cg(krylovOpts(true));
    V x = cg.solve(a, b);
    const auto &report = cg.lastReport();
    assert(!report.residualHistory.empty());
    // overall decrease
    assert(report.residualHistory.back() < report.residualHistory.front());
    // Lanczos diagnostics exist and give a plausible condition estimate
    V diag = cg.lanczosDiagonal();
    assert(diag.size() > 0);
    double cond = cg.conditionNumberEstimate();
    assert(cond >= 1.0);
}

void test_cg_initial_guess() {
    M a = spd(5, 185);
    V b = V::Random(5, 186);
    ConjugateGradient<double> direct(krylovOpts());
    V x = direct.solve(a, b);
    // starting at the answer: converges immediately
    auto o = krylovOpts();
    o.useInitialGuess = true;
    ConjugateGradient<double> warm(o);
    V x2 = warm.solve(a, b, x);
    assert((a * x2).isApprox(b, 1e-7));
    assert(warm.lastReport().iterations <= 1);
}

void test_cg_preconditioned() {
    // badly scaled SPD diagonal: Jacobi preconditioning fixes it outright
    std::size_t n = 8;
    V d(n);
    for (std::size_t i = 0; i < n; ++i)
        d(i) = std::pow(10.0, static_cast<double>(i));
    M a = M::Diagonal(d);
    V b = V::Ones(n);
    JacobiPreconditioner<double> jacobi;
    jacobi.setup(a);
    ConjugateGradient<double> cg(krylovOpts());
    cg.setPreconditioner(&jacobi);
    V x = cg.solve(a, b);
    assert((a * x).isApprox(b, 1e-6));
    assert(cg.lastReport().converged);
    // perfectly preconditioned diagonal system: a couple of iterations
    assert(cg.lastReport().iterations <= 3);
}

void test_gmres() {
    M a = nonsym(7, 187);
    V b = V::Random(7, 188);
    GMRES<double> gmres(krylovOpts());
    V x = gmres.solve(a, b);
    assert((a * x).isApprox(b, 1e-7));
    assert(gmres.lastReport().converged);
    // restart setter round-trips
    gmres.setRestart(3);
    assert(gmres.restart() == 3);
    V xr = gmres.solve(a, b);
    assert((a * xr).isApprox(b, 1e-6));
}

void test_bicgstab() {
    M a = nonsym(7, 189);
    V b = V::Random(7, 190);
    BiCGSTAB<double> solver(krylovOpts());
    V x = solver.solve(a, b);
    assert((a * x).isApprox(b, 1e-6));
    assert(solver.lastReport().converged);
    assert(!solver.breakdownDetected());
}

void test_lsqr() {
    // rectangular least squares without forming A^T A
    M a = M::Random(10, 4, 191);
    V b = V::Random(10, 192);
    LSQR<double> lsqr(krylovOpts());
    V x = lsqr.solve(a, b);
    assert((a.transpose() * (b - a * x)).norm() < 1e-6);
    // regularization shrinks the solution
    LSQR<double> damped(krylovOpts());
    damped.setRegularization(10.0);
    V xd = damped.solve(a, b);
    assert(xd.norm() < x.norm());
    // norm estimate is in the ballpark of the true frobenius bound
    assert(lsqr.matrixNormEstimate() > 0.0);
}

void test_options_and_report_bookkeeping() {
    ConjugateGradient<double> cg;
    auto o = krylovOpts();
    o.maxIterations = 2; // starve the solver
    cg.setOptions(o);
    assert(cg.options().maxIterations == 2);
    M a = spd(10, 193);
    V b = V::Random(10, 194);
    V x = cg.solve(a, b);
    const auto &report = cg.lastReport();
    // starved solve reports non-convergence rather than lying
    assert(!report.converged);
    assert(report.iterations <= 2);
}

} // namespace

int main() {
    run("cg_solves_spd", test_cg_solves_spd);
    run("cg_history_and_diagnostics", test_cg_history_and_diagnostics);
    run("cg_initial_guess", test_cg_initial_guess);
    run("cg_preconditioned", test_cg_preconditioned);
    run("gmres", test_gmres);
    run("bicgstab", test_bicgstab);
    run("lsqr", test_lsqr);
    run("options_and_report_bookkeeping", test_options_and_report_bookkeeping);
    return finish("KrylovSolver");
}
