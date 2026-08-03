// Behavioural tests for linalg::LU / FullPivLU (PLAN.md step 9).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// PLAN.md oracles: P*A == L*U on random matrices, solve residual small,
// singular matrix triggers the throwOnSingular path.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/LU.hpp"

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

constexpr double kTol = 1e-10;

void test_factorization_identity() {
    M a = M::Random(5, 5, 11);
    LU<double> lu(a);
    assert(lu.isComputed());
    M p = lu.permutationMatrix();
    M l = lu.lower();
    M u = lu.upper();
    assert((p * a).isApprox(l * u, kTol)); // P*A == L*U
    // structure: L unit lower, U upper
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-14));
    assert(u.isTriangular(Triangle::Kind::Upper, 1e-14));
    for (std::size_t i = 0; i < 5; ++i)
        assert(std::abs(l(i, i) - 1.0) < 1e-14);
    // P is a permutation: orthogonal with entries 0/1
    assert(p.isOrthogonal(1e-14));
}

void test_packed_factors() {
    M a = M::Random(4, 4, 3);
    LU<double> lu(a);
    M packed = lu.packedFactors();
    M l = lu.lower(), u = lu.upper();
    // packed holds U on and above the diagonal, L strictly below
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            assert(std::abs(packed(i, j) - (i > j ? l(i, j) : u(i, j))) <
                   1e-14);
}

void test_solve() {
    M a = M::Random(6, 6, 21);
    V b = V::Random(6, 22);
    LU<double> lu(a);
    V x = lu.solve(b);
    assert((a * x).isApprox(b, 1e-8));
    // multiple right-hand sides
    M bm = M::Random(6, 3, 23);
    M xm = lu.solve(bm);
    assert((a * xm).isApprox(bm, 1e-8));
    // transpose solve reuses the same factors
    V xt = lu.solveTranspose(b);
    assert((a.transpose() * xt).isApprox(b, 1e-8));
}

void test_determinant() {
    LU<double> lu(M{{1, 2}, {3, 4}});
    assert(std::abs(lu.determinant() - (-2.0)) < kTol);
    assert(std::abs(lu.logAbsDeterminant() - std::log(2.0)) < kTol);
    // triangular matrix: determinant is the diagonal product
    LU<double> lu2(M{{2, 5}, {0, 3}});
    assert(std::abs(lu2.determinant() - 6.0) < kTol);
    // permutation parity: swapping rows flips the sign
    LU<double> lu3(M{{0, 1}, {1, 0}});
    assert(std::abs(lu3.determinant() - (-1.0)) < kTol);
}

void test_inverse_rank() {
    M a = M::Random(4, 4, 31);
    LU<double> lu(a);
    assert((a * lu.inverse()).isApprox(M::Identity(4), 1e-8));
    assert(lu.rank(1e-10) == 4);
    assert(!lu.isSingular(1e-10));
}

void test_singular_paths() {
    M sing{{1, 2}, {2, 4}}; // rank 1
    // default: record, do not throw
    LU<double> lu(sing);
    assert(lu.isSingular(1e-12));
    assert(lu.rank(1e-10) == 1);
    // opted-in throw
    LU<double>::Options o;
    o.useBlocked = false;
    o.blockSize = 0;
    o.pivotTolerance = 1e-12;
    o.throwOnSingular = true;
    bool threw = false;
    try {
        LU<double> lu2(sing, o);
    } catch (const SingularMatrix &) {
        threw = true;
    }
    assert(threw);
}

void test_not_computed() {
    LU<double> lu;
    assert(!lu.isComputed());
    bool threw = false;
    try {
        (void)lu.solve(V{1, 2});
    } catch (const NotComputed &) {
        threw = true;
    }
    assert(threw);
}

void test_refinement() {
    M a = M::Hilbert(8); // ill-conditioned on purpose
    V xTrue = V::Ones(8);
    V b = a * xTrue;
    LU<double> lu(a);
    V x0 = lu.solve(b);
    V x1 = lu.refineSolution(a, b, x0, 3);
    // refinement must not make the residual worse
    double r0 = (a * x0 - b).norm();
    double r1 = (a * x1 - b).norm();
    assert(r1 <= r0 + 1e-14);
}

void test_condition_estimate() {
    LU<double> good(M::Identity(4));
    double rg = good.reciprocalConditionEstimate();
    assert(rg > 0.1 && rg <= 1.0 + 1e-12);
    LU<double> bad(M::Hilbert(10));
    double rb = bad.reciprocalConditionEstimate();
    assert(rb >= 0.0 && rb < 1e-6); // Hilbert(10): cond ~ 1e13
}

void test_complex() {
    Matrix<C> a = Matrix<C>::Random(4, 4, 41);
    LU<C> lu(a);
    Matrix<C> p = lu.permutationMatrix();
    assert((p * a).isApprox(lu.lower() * lu.upper(), kTol));
    Vector<C> b = Vector<C>::Random(4, 42);
    Vector<C> x = lu.solve(b);
    assert((a * x).isApprox(b, 1e-8));
}

// ---- FullPivLU -------------------------------------------------------------

void test_full_piv_identity() {
    M a = M::Random(4, 4, 51);
    FullPivLU<double> lu(a);
    assert(lu.isComputed());
    // P*A*Q == L*U with both permutations
    M pa = lu.rowPermutation() * a * lu.columnPermutation();
    assert(pa.isApprox(lu.lower() * lu.upper(), kTol));
    V b = V::Random(4, 52);
    V x = lu.solve(b);
    assert((a * x).isApprox(b, 1e-8));
}

void test_full_piv_rank_kernel_image() {
    // rank-2 matrix: third row/column are combinations
    M a{{1, 0, 1}, {0, 1, 1}, {1, 1, 2}};
    FullPivLU<double> lu(a);
    assert(lu.rank(1e-10) == 2);
    M k = lu.kernel();
    assert(k.cols() >= 1);
    // kernel columns are genuinely null directions
    assert((a * k).isApprox(M::Zeros(3, k.cols()), 1e-8));
    M img = lu.image();
    assert(img.cols() == 2);
    // image columns lie in the column space: each is A * something, so
    // rank([A | img]) stays 2
    FullPivLU<double> aug(a.horizontalConcat(img));
    assert(aug.rank(1e-8) == 2);
}

} // namespace

int main() {
    run("factorization_identity", test_factorization_identity);
    run("packed_factors", test_packed_factors);
    run("solve", test_solve);
    run("determinant", test_determinant);
    run("inverse_rank", test_inverse_rank);
    run("singular_paths", test_singular_paths);
    run("not_computed", test_not_computed);
    run("refinement", test_refinement);
    run("condition_estimate", test_condition_estimate);
    run("complex", test_complex);
    run("full_piv_identity", test_full_piv_identity);
    run("full_piv_rank_kernel_image", test_full_piv_rank_kernel_image);
    return finish("LU");
}
