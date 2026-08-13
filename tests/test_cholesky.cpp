// Behavioural tests for linalg::Cholesky / LDLT (PLAN.md step 10).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp). update/downdate are deferred in the plan and will
// simply stay skipped until implemented.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/Cholesky.hpp"

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

// Random SPD matrix: B^T B + n I is comfortably positive definite.
M spd(std::size_t n, unsigned long seed) {
    M b = M::Random(n, n, seed);
    return b.transpose() * b + M::Identity(n) * static_cast<double>(n);
}

void test_factorization_identity() {
    M a = spd(5, 61);
    Cholesky<double> chol(a);
    assert(chol.isComputed());
    assert(chol.isPositiveDefinite());
    M l = chol.lower();
    assert((l * l.transpose()).isApprox(a, 1e-8)); // A == L L^H
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-14));
    // diagonal of L is positive
    for (std::size_t i = 0; i < 5; ++i)
        assert(l(i, i) > 0.0);
    // upper() is the conjugate transpose of lower()
    assert(chol.upper().isApprox(l.transpose(), 1e-14));
}

void test_solve() {
    M a = spd(6, 62);
    V b = V::Random(6, 63);
    Cholesky<double> chol(a);
    V x = chol.solve(b);
    assert((a * x).isApprox(b, 1e-8));
    M bm = M::Random(6, 2, 64);
    M xm = chol.solve(bm);
    assert((a * xm).isApprox(bm, 1e-8));
    assert((a * chol.inverse()).isApprox(M::Identity(6), 1e-8));
}

void test_determinant() {
    M a{{4, 2}, {2, 3}}; // det 8, SPD
    Cholesky<double> chol(a);
    assert(std::abs(chol.determinant() - 8.0) < kTol);
    assert(std::abs(chol.logDeterminant() - std::log(8.0)) < kTol);
}

void test_read_triangle_option() {
    // junk in the triangle not selected by readFrom must be ignored
    M a = spd(4, 65);
    M junky = a;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i + 1; j < 4; ++j)
            junky(i, j) = 999.0; // poison the upper triangle
    Cholesky<double>::Options o;
    o.readFrom = Triangle::Kind::Lower;
    o.useBlocked = false;
    o.blockSize = 0;
    o.throwOnIndefinite = false;
    Cholesky<double> chol(junky, o);
    assert(chol.isPositiveDefinite());
    assert((chol.lower() * chol.upper()).isApprox(a, 1e-8));
}

void test_read_upper_triangle() {
    // readFrom == Upper must read only the upper triangle and still return a
    // genuinely lower-triangular L from lower(); junk below the diagonal is
    // ignored. (Regression: lower()/upper() used to transpose for Upper.)
    M a = spd(4, 65);
    M junky = a;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < i; ++j)
            junky(i, j) = 999.0; // poison the strictly-lower triangle
    Cholesky<double>::Options o;
    o.readFrom = Triangle::Kind::Upper;
    o.useBlocked = false;
    o.blockSize = 0;
    o.throwOnIndefinite = false;
    Cholesky<double> chol(junky, o);
    assert(chol.isPositiveDefinite());
    M l = chol.lower();
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-12)); // L is lower, not L^H
    assert(chol.upper().isApprox(l.transpose(), 1e-14));
    assert((l * chol.upper()).isApprox(a, 1e-8)); // recovers the true A
}

void test_update() {
    // Standalone (no downdate) so it runs rather than skipping: a rank-one
    // update must yield a factor of A + x x^H that is still lower-triangular
    // with a positive diagonal, and still solves the modified system.
    M a = spd(4, 68);
    V x = V::Random(4, 69);
    Cholesky<double> chol(a);
    chol.update(x);
    M updated = a + x.outer(x); // A + x x^H
    M l = chol.lower();
    assert((l * chol.upper()).isApprox(updated, 1e-8));
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-12));
    for (std::size_t i = 0; i < 4; ++i)
        assert(l(i, i) > 0.0);
    V b = V::Random(4, 70);
    assert((updated * chol.solve(b)).isApprox(b, 1e-8));
}

void test_blocked_matches_unblocked() {
    // useBlocked currently falls back to the unblocked algorithm, but it must
    // no longer be a silent no-op: the blocked path must produce a valid factor
    // matching the unblocked one. (Regression: compute() ignored useBlocked.)
    M a = spd(6, 74);
    Cholesky<double>::Options blocked;
    blocked.readFrom = Triangle::Kind::Lower;
    blocked.useBlocked = true;
    blocked.blockSize = 2;
    blocked.throwOnIndefinite = false;
    Cholesky<double> cb(a, blocked);
    assert(cb.isPositiveDefinite());
    M l = cb.lower();
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-12));
    assert((l * cb.upper()).isApprox(a, 1e-8));
    // identical to the default unblocked factor
    Cholesky<double> cu(a);
    assert(l.isApprox(cu.lower(), 1e-12));
}

void test_indefinite_is_a_result() {
    M indef{{1, 2}, {2, 1}}; // eigenvalues 3, -1
    // default: recorded, not thrown
    Cholesky<double> chol(indef);
    assert(!chol.isPositiveDefinite());
    // opted-in throw
    Cholesky<double>::Options o;
    o.readFrom = Triangle::Kind::Lower;
    o.useBlocked = false;
    o.blockSize = 0;
    o.throwOnIndefinite = true;
    bool threw = false;
    try {
        Cholesky<double> chol2(indef, o);
    } catch (const NotPositiveDefinite &) {
        threw = true;
    }
    assert(threw);
}

void test_condition_estimate() {
    Cholesky<double> good(M::Identity(4));
    double rg = good.reciprocalConditionEstimate();
    assert(rg > 0.1 && rg <= 1.0 + 1e-12);
    Cholesky<double> bad(M::Hilbert(8)); // SPD but terribly conditioned
    double rb = bad.reciprocalConditionEstimate();
    assert(rb >= 0.0 && rb < 1e-4);
}

void test_complex_hermitian() {
    Matrix<C> b = Matrix<C>::Random(4, 4, 66);
    Matrix<C> a =
        b.conjugateTranspose() * b + Matrix<C>::Identity(4) * C(4, 0);
    Cholesky<C> chol(a);
    assert(chol.isPositiveDefinite());
    Matrix<C> l = chol.lower();
    assert((l * l.conjugateTranspose()).isApprox(a, 1e-8));
    Vector<C> rhs = Vector<C>::Random(4, 67);
    assert((a * chol.solve(rhs)).isApprox(rhs, 1e-8));
}

void test_update_downdate() { // deferred in the plan; skips until built
    M a = spd(4, 68);
    V x = V::Random(4, 69);
    Cholesky<double> chol(a);
    chol.update(x);
    M updated = a + x.outer(x); // A + x x^H
    assert((chol.lower() * chol.upper()).isApprox(updated, 1e-8));
    chol.downdate(x); // back to A
    assert((chol.lower() * chol.upper()).isApprox(a, 1e-8));
    // a downdate that destroys positive definiteness throws and leaves
    // the factor unchanged
    V big = V::Constant(4, 100.0);
    bool threw = false;
    try {
        chol.downdate(big);
    } catch (const NotPositiveDefinite &) {
        threw = true;
    }
    assert(threw);
    assert((chol.lower() * chol.upper()).isApprox(a, 1e-8));
}

// ---- LDLT ------------------------------------------------------------------

void test_ldlt_identity() {
    M a = spd(5, 71);
    LDLT<double> ldlt(a);
    assert(ldlt.isComputed());
    M l = ldlt.lower();
    M d = M::Diagonal(ldlt.diagonal());
    M p = ldlt.permutationMatrix();
    // P^T A P == L D L^H
    assert((p.transpose() * a * p).isApprox(l * d * l.transpose(), 1e-8));
    // L unit lower triangular
    assert(l.isTriangular(Triangle::Kind::Lower, 1e-14));
    for (std::size_t i = 0; i < 5; ++i)
        assert(std::abs(l(i, i) - 1.0) < 1e-14);
}

void test_ldlt_solve_and_inertia() {
    M a = spd(5, 72);
    V b = V::Random(5, 73);
    LDLT<double> ldlt(a);
    assert((a * ldlt.solve(b)).isApprox(b, 1e-8));
    // SPD: all D entries positive
    assert(ldlt.isPositiveSemiDefinite(1e-12));
    assert(!ldlt.isNegativeSemiDefinite(1e-12));
    assert(ldlt.rank(1e-10) == 5);
    // indefinite: LDLT still factors, and D's signs reveal the inertia
    M indef = M::Diagonal(V{2, -3, 1});
    LDLT<double> li(indef);
    assert(!li.isPositiveSemiDefinite(1e-12));
    assert(!li.isNegativeSemiDefinite(1e-12));
    V d = li.diagonal();
    std::size_t negatives = 0;
    for (std::size_t i = 0; i < d.size(); ++i)
        if (d(i) < 0.0)
            ++negatives;
    assert(negatives == 1);
    // semidefinite: rank drops
    M semi{{1, 1}, {1, 1}};
    LDLT<double> ls(semi);
    assert(ls.rank(1e-10) == 1);
    assert(ls.isPositiveSemiDefinite(1e-10));
}

} // namespace

int main() {
    run("factorization_identity", test_factorization_identity);
    run("solve", test_solve);
    run("determinant", test_determinant);
    run("read_triangle_option", test_read_triangle_option);
    run("read_upper_triangle", test_read_upper_triangle);
    run("update", test_update);
    run("blocked_matches_unblocked", test_blocked_matches_unblocked);
    run("indefinite_is_a_result", test_indefinite_is_a_result);
    run("condition_estimate", test_condition_estimate);
    run("complex_hermitian", test_complex_hermitian);
    run("update_downdate", test_update_downdate);
    run("ldlt_identity", test_ldlt_identity);
    run("ldlt_solve_and_inertia", test_ldlt_solve_and_inertia);
    return finish("Cholesky");
}
