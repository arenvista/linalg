// Behavioural tests for linalg::TriangularSolver (PLAN.md step 8).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// Solutions are verified by residual (op(A) * x == b) so the tests do not
// bake in any one substitution order.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/solve/TriangularSolver.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <complex>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using C = std::complex<double>;
using TS = TriangularSolver<double>;
using Ul = Triangle::Kind;
using Dg = Diagonal::Kind;
using Tr = Transposition::Kind;

namespace {

constexpr double kTol = 1e-10;

TS::Options opts(Ul uplo, Dg diag = Dg::NonUnit, Tr trans = Tr::None) {
    TS::Options o;
    o.uplo = uplo;
    o.diagonal = diag;
    o.transposition = trans;
    o.useBlocked = false;
    o.blockSize = 0;
    return o;
}

// Built lazily: a file-scope Matrix would construct before main() and its
// stub throw would abort the process instead of skipping.
M lower3() { return M{{2, 0, 0}, {1, 3, 0}, {1, 1, 4}}; }
M upper3() { return M{{2, 1, 1}, {0, 3, 1}, {0, 0, 4}}; }

void test_lower_solve() {
    M l = lower3();
    TS s(opts(Ul::Lower));
    V b{2, 5, 10};
    V x = s.solve(l, b);
    assert((l * x).isApprox(b, kTol));
}

void test_upper_solve() {
    M u = upper3();
    TS s(opts(Ul::Upper));
    V b{4, 5, 8};
    V x = s.solve(u, b);
    assert((u * x).isApprox(b, kTol));
}

void test_opposite_triangle_never_read() {
    // junk in the unused triangle must not affect the answer
    M l = lower3();
    M lowerWithJunk = l;
    lowerWithJunk(0, 1) = 999;
    lowerWithJunk(0, 2) = -999;
    lowerWithJunk(1, 2) = 999;
    TS s(opts(Ul::Lower));
    V b{2, 5, 10};
    V clean = s.solve(l, b);
    V junky = s.solve(lowerWithJunk, b);
    assert(junky.isApprox(clean, kTol));
}

void test_unit_diagonal_not_read() {
    // Unit: stored diagonal is ignored, assumed all ones
    M l{{9, 0}, {1, 9}}; // effective matrix [[1,0],[1,1]]
    TS s(opts(Ul::Lower, Dg::Unit));
    V b{5, 7};
    V x = s.solve(l, b);
    assert(std::abs(x(0) - 5.0) < kTol && std::abs(x(1) - 2.0) < kTol);
}

void test_transpose_solve() {
    // solves op(A) x = b with op = transpose: A^T x == b
    M l = lower3();
    TS s(opts(Ul::Lower, Dg::NonUnit, Tr::Transpose));
    V b{3, 4, 5};
    V x = s.solve(l, b);
    assert((l.transpose() * x).isApprox(b, kTol));
}

void test_conjugate_transpose_solve() {
    Matrix<C> l{{C(2, 1), C(0, 0)}, {C(1, -1), C(3, 0)}};
    TriangularSolver<C>::Options o;
    o.uplo = Ul::Lower;
    o.diagonal = Dg::NonUnit;
    o.transposition = Tr::ConjugateTranspose;
    o.useBlocked = false;
    o.blockSize = 0;
    TriangularSolver<C> s(o);
    Vector<C> b{C(1, 1), C(2, -1)};
    Vector<C> x = s.solve(l, b);
    assert((l.conjugateTranspose() * x).isApprox(b, kTol));
}

void test_matrix_rhs() {
    // each column of B is an independent right-hand side
    M l = lower3();
    TS s(opts(Ul::Lower));
    M b{{2, 4}, {5, 7}, {10, 8}};
    M x = s.solve(l, b);
    assert((l * x).isApprox(b, kTol));
    // column consistency with the vector overload
    V x0 = s.solve(l, b.col(0));
    assert(x.col(0).isApprox(x0, kTol));
}

void test_solve_in_place() {
    M u = upper3();
    TS s(opts(Ul::Upper));
    V b{4, 5, 8};
    V expected = s.solve(u, b);
    V x = b;
    s.solveInPlace(u, x);
    assert(x.isApprox(expected, kTol));
    M bm{{4, 1}, {5, 2}, {8, 3}};
    M expectedM = s.solve(u, bm);
    M xm = bm;
    s.solveInPlace(u, xm);
    assert(xm.isApprox(expectedM, kTol));
}

void test_singular_throws() {
    M l{{2, 0}, {1, 0}}; // zero diagonal entry
    TS s(opts(Ul::Lower));
    bool threw = false;
    try {
        (void)s.solve(l, V{1, 1});
    } catch (const SingularMatrix &e) {
        threw = true;
        assert(e.pivotIndex() == 1); // the offending diagonal entry
    }
    assert(threw);
    // Unit diagonal never reads the diagonal, so no throw
    TS su(opts(Ul::Lower, Dg::Unit));
    V x = su.solve(l, V{1, 3});
    assert(std::abs(x(0) - 1.0) < kTol && std::abs(x(1) - 2.0) < kTol);
}

void test_inverse() {
    M l = lower3();
    TS s(opts(Ul::Lower));
    M inv = s.inverse(l);
    assert((l * inv).isApprox(M::Identity(3), kTol));
    // structure preserved: inverse of lower triangular is lower triangular
    assert(inv.isTriangular(Ul::Lower, 1e-14));
}

void test_is_nonsingular() {
    M l = lower3();
    TS s(opts(Ul::Lower));
    assert(s.isNonsingular(l, 1e-12));
    M sing{{2, 0}, {1, 0}};
    assert(!s.isNonsingular(sing, 1e-12));
    // tolerance is a cutoff on diagonal magnitudes
    assert(!s.isNonsingular(l, 2.5)); // diagonal entry 2 is below 2.5
}

void test_reciprocal_condition_estimate() {
    TS s(opts(Ul::Lower));
    // identity is perfectly conditioned
    double r = s.reciprocalConditionEstimate(M::Identity(3));
    assert(r > 0.5 && r <= 1.0 + 1e-12);
    // a nearly singular triangle has a tiny estimate
    M bad{{1, 0}, {1, 1e-12}};
    double rb = s.reciprocalConditionEstimate(bad);
    assert(rb >= 0.0 && rb < 1e-6);
}

} // namespace

int main() {
    run("lower_solve", test_lower_solve);
    run("upper_solve", test_upper_solve);
    run("opposite_triangle_never_read", test_opposite_triangle_never_read);
    run("unit_diagonal_not_read", test_unit_diagonal_not_read);
    run("transpose_solve", test_transpose_solve);
    run("conjugate_transpose_solve", test_conjugate_transpose_solve);
    run("matrix_rhs", test_matrix_rhs);
    run("solve_in_place", test_solve_in_place);
    run("singular_throws", test_singular_throws);
    run("inverse", test_inverse);
    run("is_nonsingular", test_is_nonsingular);
    run("reciprocal_condition_estimate", test_reciprocal_condition_estimate);
    return finish("TriangularSolver");
}
