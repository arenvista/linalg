// Behavioural tests for linalg::LeastSquares / GaussNewton (PLAN.md step 19).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// The least-squares optimality oracle is the normal-equations gradient:
// x minimizes ||A x - b|| iff A^H (b - A x) == 0.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/solve/LeastSquares.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using LSQ = LeastSquares<double>;
using Method = LSQ::Method::Kind;

namespace {

void test_all_methods_agree() {
    M a = M::Random(8, 3, 151);
    V b = V::Random(8, 152);
    V reference;
    for (Method m : {Method::HouseholderQR, Method::ColPivQR, Method::SVD,
                     Method::NormalEquations}) {
        LSQ solver(m);
        V x = solver.solve(a, b);
        // optimality: gradient of the normal equations vanishes
        assert((a.transpose() * (b - a * x)).norm() < 1e-8);
        if (m == Method::HouseholderQR)
            reference = x;
        else
            assert(x.isApprox(reference, 1e-6)); // same minimizer
    }
}

void test_report() {
    M a = M::Random(6, 2, 153);
    V b = V::Random(6, 154);
    LSQ solver;
    V x = solver.solve(a, b);
    const auto &report = solver.lastReport();
    assert(std::abs(report.residualNorm - (b - a * x).norm()) < 1e-10);
    assert(std::abs(report.solutionNorm - x.norm()) < 1e-10);
    assert(report.effectiveRank == 2);
    assert(!report.rankDeficient);
}

void test_minimum_norm() {
    // underdetermined consistent system: many minimizers, one smallest
    M a{{1, 0, 1}, {0, 1, 1}};
    V b{2, 3};
    LSQ solver;
    V x = solver.solveMinimumNorm(a, b);
    assert((a * x).isApprox(b, 1e-8)); // consistent: exact fit
    // minimum-norm solution lies in the row space: x orthogonal to the
    // kernel direction (1, 1, -1)
    V kernelDir{1, 1, -1};
    assert(std::abs(x.dot(kernelDir)) < 1e-8);
}

void test_weighted() {
    M a = M::Random(6, 2, 155);
    V b = V::Random(6, 156);
    LSQ solver;
    // unit weights match the plain solve
    V unweighted = solver.solve(a, b);
    V allOnes = solver.solveWeighted(a, b, V::Ones(6));
    assert(allOnes.isApprox(unweighted, 1e-8));
    // a huge weight on one row pins that row's residual to ~zero
    V w = V::Ones(6);
    w(0) = 1e12;
    V x = solver.solveWeighted(a, b, w);
    double row0Residual = std::abs(b(0) - (a * x)(0));
    assert(row0Residual < 1e-6);
}

void test_regularized() {
    M a = M::Random(6, 3, 157);
    V b = V::Random(6, 158);
    LSQ solver;
    // lambda = 0 is the plain problem
    V plain = solver.solve(a, b);
    V l0 = solver.solveRegularized(a, b, 0.0);
    assert(l0.isApprox(plain, 1e-8));
    // ridge optimality: (A^H A + lambda^2 I) x == A^H b
    double lambda = 0.7;
    V x = solver.solveRegularized(a, b, lambda);
    V lhs = a.transpose() * (a * x) + x * (lambda * lambda);
    V rhs = a.transpose() * b;
    assert(lhs.isApprox(rhs, 1e-8));
    // damping shrinks the solution
    assert(x.norm() < plain.norm() + 1e-12);
}

void test_constrained() {
    // minimize ||x - b|| subject to x1 + x2 = 0: the projection of b onto
    // the constraint plane. For b = (1, 1): x = (0, 0).
    M a = M::Identity(2);
    V b{1, 1};
    M c{{1, 1}};
    V d{0};
    LSQ solver;
    V x = solver.solveConstrained(a, b, c, d);
    assert((c * x).isApprox(d, 1e-8)); // constraint holds
    assert(std::abs(x(0)) < 1e-8 && std::abs(x(1)) < 1e-8);
    // general case: constraint satisfied and gradient orthogonal to the
    // constraint null space
    M a2 = M::Random(6, 3, 159);
    V b2 = V::Random(6, 160);
    M c2{{1, 1, 1}};
    V d2{1};
    V x2 = solver.solveConstrained(a2, b2, c2, d2);
    assert((c2 * x2).isApprox(d2, 1e-8));
    // KKT: A^H (A x - b) must lie in the row space of C, i.e. be parallel
    // to (1,1,1); check orthogonality to two directions spanning ker(C)
    V g = a2.transpose() * (a2 * x2 - b2);
    V k1{1, -1, 0}, k2{1, 0, -1};
    assert(std::abs(g.dot(k1)) < 1e-8 && std::abs(g.dot(k2)) < 1e-8);
}

void test_regression_statistics() {
    M a = M::Random(6, 2, 161);
    V b = V::Random(6, 162);
    LSQ solver;
    V x = solver.solve(a, b);
    // residual() is literally b - A x
    V r = solver.residual(a, b, x);
    assert(r.isApprox(b - a * x, 1e-10));
    // covariance: variance * (A^H A)^-1 — for A = I it is variance * I
    M cov = solver.covarianceMatrix(M::Identity(3), 2.5);
    assert(cov.isApprox(M::Identity(3) * 2.5, 1e-8));
    // standard errors: one per parameter, positive
    V se = solver.standardErrors(a, r);
    assert(se.size() == 2);
    for (std::size_t i = 0; i < se.size(); ++i)
        assert(se(i) >= 0.0);
    // hat matrix: symmetric idempotent projector onto range(A)
    M h = solver.hatMatrix(a);
    assert(h.isSymmetric(1e-8));
    assert((h * h).isApprox(h, 1e-8));
    assert((h * a).isApprox(a, 1e-8));
}

// ---- GaussNewton -----------------------------------------------------------

// Linear model: residual A p - b. Gauss-Newton must converge in one step.
class LinearModel : public GaussNewton<double>::Model {
public:
    LinearModel(const M &a, const V &b) : a_(a), b_(b) {}
    V residual(const V &p) const override { return a_ * p - b_; }
    M jacobian(const V &) const override { return a_; }
    std::size_t residualCount() const override { return a_.rows(); }
    std::size_t parameterCount() const override { return a_.cols(); }

private:
    M a_;
    V b_;
};

// Scalar nonlinear model: r(p) = p^2 - 4, minimized at p = +/-2.
class QuadraticModel : public GaussNewton<double>::Model {
public:
    V residual(const V &p) const override { return V{p(0) * p(0) - 4.0}; }
    M jacobian(const V &p) const override { return M{{2.0 * p(0)}}; }
    std::size_t residualCount() const override { return 1; }
    std::size_t parameterCount() const override { return 1; }
};

GaussNewton<double>::Options gnOpts(bool lm) {
    GaussNewton<double>::Options o;
    o.maxIterations = 100;
    o.gradientTolerance = 1e-12;
    o.stepTolerance = 1e-14;
    o.initialDamping = 1e-3;
    o.dampingIncrease = 10.0;
    o.dampingDecrease = 0.1;
    o.useLevenbergMarquardt = lm;
    return o;
}

void test_gauss_newton_linear() {
    M a = M::Random(8, 3, 163);
    V b = V::Random(8, 164);
    GaussNewton<double> gn(gnOpts(false));
    V p = gn.minimize(LinearModel(a, b), V::Zeros(3));
    // the minimizer of a linear model is the least-squares solution
    assert((a.transpose() * (b - a * p)).norm() < 1e-8);
    const auto &report = gn.lastReport();
    assert(report.converged);
    assert(report.gradientNorm < 1e-8);
}

void test_gauss_newton_nonlinear() {
    GaussNewton<double> gn(gnOpts(false));
    V p = gn.minimize(QuadraticModel(), V{1.0});
    assert(std::abs(p(0) - 2.0) < 1e-6); // nearest root from p0 = 1
    assert(gn.lastReport().converged);
    assert(std::abs(gn.lastReport().finalCost) < 1e-10);
}

void test_levenberg_marquardt() {
    // LM must reach the same minimum, damping notwithstanding
    GaussNewton<double> lm(gnOpts(true));
    V p = lm.minimize(QuadraticModel(), V{10.0});
    assert(std::abs(std::abs(p(0)) - 2.0) < 1e-6);
    assert(lm.lastReport().converged);
    M a = M::Random(6, 2, 165);
    V b = V::Random(6, 166);
    GaussNewton<double> lm2(gnOpts(true));
    V pl = lm2.minimize(LinearModel(a, b), V::Zeros(2));
    assert((a.transpose() * (b - a * pl)).norm() < 1e-6);
}

} // namespace

int main() {
    run("all_methods_agree", test_all_methods_agree);
    run("report", test_report);
    run("minimum_norm", test_minimum_norm);
    run("weighted", test_weighted);
    run("regularized", test_regularized);
    run("constrained", test_constrained);
    run("regression_statistics", test_regression_statistics);
    run("gauss_newton_linear", test_gauss_newton_linear);
    run("gauss_newton_nonlinear", test_gauss_newton_nonlinear);
    run("levenberg_marquardt", test_levenberg_marquardt);
    return finish("LeastSquares");
}
