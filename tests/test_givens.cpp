// Behavioural tests for linalg::Givens / GivensSequence (PLAN.md step 7).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// FromPair must be dlartg-style robust: exact near the overflow and
// underflow thresholds, never squaring a or b.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Givens.hpp"

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
using G = Givens<double>;
using GC = Givens<C>;

namespace {

constexpr double kTol = 1e-12;

bool close(double a, double b) { return std::abs(a - b) < kTol; }

void test_identity() {
    G g = G::Identity(0, 1);
    assert(g.cosine() == 1.0 && g.sine() == 0.0);
    V x{3, 4};
    g.apply(x);
    assert(x(0) == 3 && x(1) == 4); // no-op
}

void test_from_pair_zeroes_b() {
    G g = G::FromPair(3.0, 4.0, 0, 1);
    assert(g.firstIndex() == 0 && g.secondIndex() == 1);
    V x{3, 4};
    g.apply(x);
    assert(close(std::abs(x(0)), 5.0)); // rotated onto (r, 0)
    assert(close(x(1), 0.0));
    assert(close(std::abs(g.radius()), 5.0));
    assert(close(x(0), g.radius())); // radius is what (a, b) lands on
    // rotation coefficients satisfy c^2 + |s|^2 == 1
    assert(close(g.cosine() * g.cosine() + g.sine() * g.sine(), 1.0));
}

void test_from_pair_edge_cases() {
    // b == 0: nothing to zero, rotation is (possibly signed) identity
    G g = G::FromPair(7.0, 0.0, 0, 1);
    V x{7, 0};
    g.apply(x);
    assert(close(std::abs(x(0)), 7.0) && close(x(1), 0.0));
    // a == 0: pure swap
    G g2 = G::FromPair(0.0, 5.0, 0, 1);
    V y{0, 5};
    g2.apply(y);
    assert(close(std::abs(y(0)), 5.0) && close(y(1), 0.0));
}

void test_from_pair_overflow_safe() {
    // naive hypot a^2 + b^2 would overflow; dlartg-style scaling must not
    double big = 1e200;
    G g = G::FromPair(big, big, 0, 1);
    assert(std::isfinite(g.radius()));
    assert(std::abs(std::abs(g.radius()) - std::sqrt(2.0) * big) /
               (std::sqrt(2.0) * big) <
           1e-12);
    // and no underflow to zero for tiny inputs
    double tiny = 1e-200;
    G g2 = G::FromPair(tiny, tiny, 0, 1);
    assert(g2.radius() != 0.0);
    assert(std::abs(std::abs(g2.radius()) - std::sqrt(2.0) * tiny) /
               (std::sqrt(2.0) * tiny) <
           1e-12);
}

void test_apply_left_touches_two_rows() {
    G g = G::FromPair(1.0, 1.0, 0, 2);
    M a{{1, 2}, {100, 200}, {3, 4}};
    M before = a;
    g.applyLeft(a.view());
    // row 1 (not in the plane) is untouched
    assert(a(1, 0) == 100 && a(1, 1) == 200);
    // rows 0 and 2 changed, and the transform is norm-preserving per column
    for (std::size_t j = 0; j < 2; ++j) {
        double n0 = before(0, j) * before(0, j) + before(2, j) * before(2, j);
        double n1 = a(0, j) * a(0, j) + a(2, j) * a(2, j);
        assert(close(n0, n1));
    }
    // agrees with the dense form
    M dense = g.toMatrix(3) * before;
    assert(a.isApprox(dense, kTol));
}

void test_apply_right() {
    G g = G::FromPair(2.0, -1.0, 0, 1);
    M a{{1, 2, 5}, {3, 4, 6}};
    M byMatrix = a * g.toMatrix(3);
    M inPlace = a;
    g.applyRight(inPlace.view());
    assert(inPlace.isApprox(byMatrix, kTol));
    // column 2 lies outside the (0, 1) plane and stays fixed
    assert(inPlace(0, 2) == 5 && inPlace(1, 2) == 6);
}

void test_to_matrix_orthogonal() {
    G g = G::FromPair(3.0, -4.0, 0, 1);
    M d = g.toMatrix(2);
    assert(d.isOrthogonal(1e-12));
    // rotations have determinant +1 (Phase 2 back-fill on determinant)
    assert(close(d.determinant(), 1.0));
}

void test_transpose_inverse() {
    G g = G::FromPair(3.0, 4.0, 0, 1);
    // inverse equals transpose; applying g then its inverse restores x
    V x{3, 4};
    g.apply(x);
    G inv = g.inverse();
    inv.apply(x);
    assert(close(x(0), 3.0) && close(x(1), 4.0));
    M gd = g.toMatrix(2), td = g.transposed().toMatrix(2);
    assert((gd * td).isApprox(M::Identity(2), kTol));
}

void test_complex_rotation() {
    // c stays real for complex T; the rotation is unitary
    GC g = GC::FromPair(C(3, 4), C(1, -2), 0, 1);
    assert(std::abs(NumericTraits<C>::imag(g.cosine())) < 1e-12);
    Vector<C> x{C(3, 4), C(1, -2)};
    double normBefore = x.norm();
    g.apply(x);
    assert(std::abs(x(1)) < 1e-12); // b zeroed
    assert(std::abs(std::abs(x(0)) - normBefore) < 1e-12); // norm kept
    Matrix<C> d = g.toMatrix(2);
    assert(d.isOrthogonal(1e-12)); // unitary
}

// ---- GivensSequence --------------------------------------------------------

void test_sequence_basics() {
    GivensSequence<double> seq;
    assert(seq.count() == 0);
    assert(seq.toMatrix(2).isApprox(M::Identity(2), kTol));
    G g0 = G::FromPair(1.0, 2.0, 0, 1);
    G g1 = G::FromPair(3.0, 1.0, 1, 2);
    seq.append(g0);
    seq.append(g1);
    assert(seq.count() == 2);
    assert(seq[0].firstIndex() == 0 && seq[1].secondIndex() == 2);
    seq.clear();
    assert(seq.count() == 0);
}

void test_sequence_apply_order() {
    // applyLeft is append order: target <- G1 * (G0 * target)
    G g0 = G::FromPair(1.0, 2.0, 0, 1);
    G g1 = G::FromPair(3.0, 1.0, 1, 2);
    GivensSequence<double> seq;
    seq.append(g0);
    seq.append(g1);
    M a{{1, 0}, {0, 1}, {1, 1}};
    M expected = g1.toMatrix(3) * (g0.toMatrix(3) * a);
    M inPlace = a;
    seq.applyLeft(inPlace.view());
    assert(inPlace.isApprox(expected, kTol));
    // toMatrix is the same accumulated product
    assert(seq.toMatrix(3).isApprox(g1.toMatrix(3) * g0.toMatrix(3), kTol));
    // reversed order
    M expectedRev = g0.toMatrix(3) * (g1.toMatrix(3) * a);
    M inPlaceRev = a;
    seq.applyLeftReversed(inPlaceRev.view());
    assert(inPlaceRev.isApprox(expectedRev, kTol));
    assert(seq.reversed().toMatrix(3).isApprox(
        g0.toMatrix(3) * g1.toMatrix(3), kTol));
    // right application
    M b{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    M inPlaceR = b;
    seq.applyRight(inPlaceR.view());
    assert(inPlaceR.isApprox(b * g0.toMatrix(3) * g1.toMatrix(3), kTol));
}

} // namespace

int main() {
    run("identity", test_identity);
    run("from_pair_zeroes_b", test_from_pair_zeroes_b);
    run("from_pair_edge_cases", test_from_pair_edge_cases);
    run("from_pair_overflow_safe", test_from_pair_overflow_safe);
    run("apply_left_touches_two_rows", test_apply_left_touches_two_rows);
    run("apply_right", test_apply_right);
    run("to_matrix_orthogonal", test_to_matrix_orthogonal);
    run("transpose_inverse", test_transpose_inverse);
    run("complex_rotation", test_complex_rotation);
    run("sequence_basics", test_sequence_basics);
    run("sequence_apply_order", test_sequence_apply_order);
    return finish("Givens");
}
