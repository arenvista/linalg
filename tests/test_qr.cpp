// Behavioural tests for linalg::QR / ColPivQR / GramSchmidt
// (PLAN.md step 11).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp). The QR updating members (appendRow/appendColumn/
// removeRow/rankOneUpdate) are deferred in the plan and stay skipped.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/QR.hpp"

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
    M a = M::Random(4, 4, 81);
    QR<double> qr(a);
    assert(qr.isComputed());
    M q = qr.matrixQ();
    M r = qr.matrixR();
    assert((q * r).isApprox(a, 1e-8)); // A == Q R
    // Q has orthonormal columns (thin or full): Q^H Q == I
    assert((q.transpose() * q).isApprox(M::Identity(q.cols()), 1e-10));
    assert(r.isTriangular(Triangle::Kind::Upper, 1e-12));
}

void test_tall_matrix() {
    M a = M::Random(6, 3, 82);
    QR<double> qr(a);
    M q = qr.matrixQ();
    M r = qr.matrixR();
    assert(r.rows() == 3 && r.cols() == 3); // R is min(m,n) x n
    assert(q.cols() == r.rows());           // shapes chain up
    assert((q * r).isApprox(a, 1e-8));
    assert((q.transpose() * q).isApprox(M::Identity(q.cols()), 1e-10));
}

void test_full_mode() {
    QR<double>::Options o;
    o.mode = QR<double>::Mode::Kind::Full;
    o.useBlocked = false;
    o.blockSize = 0;
    o.accumulateQ = false;
    M a = M::Random(5, 3, 83);
    QR<double> qr(a, o);
    M q = qr.matrixQ();
    assert(q.rows() == 5 && q.cols() == 5); // full square Q
    assert(q.isOrthogonal(1e-10));
    // leading columns still reproduce A
    assert((q.block(0, 0, 5, 3).toMatrix() * qr.matrixR()).isApprox(a, 1e-8));
}

void test_solve() {
    M a = M::Random(5, 5, 84);
    V b = V::Random(5, 85);
    QR<double> qr(a);
    V x = qr.solve(b);
    assert((a * x).isApprox(b, 1e-8));
    M bm = M::Random(5, 2, 86);
    assert((a * qr.solve(bm)).isApprox(bm, 1e-8));
}

void test_least_squares() {
    // overdetermined: the residual must be orthogonal to range(A)
    M a = M::Random(8, 3, 87);
    V b = V::Random(8, 88);
    QR<double> qr(a);
    V x = qr.solveLeastSquares(b);
    V residual = b - a * x;
    V gradient = a.transpose() * residual; // normal equations: A^H r == 0
    assert(gradient.norm() < 1e-8);
}

void test_apply_q() {
    M a = M::Random(4, 4, 89);
    QR<double> qr(a);
    M target = M::Random(4, 2, 90);
    M expected = qr.matrixQ() * target;
    M inPlace = target;
    qr.applyQOnLeft(inPlace);
    assert(inPlace.isApprox(expected, 1e-8));
    // Q^H undoes Q
    qr.applyQTransposeOnLeft(inPlace);
    assert(inPlace.isApprox(target, 1e-8));
}

void test_determinant() {
    QR<double> qr(M{{1, 2}, {3, 4}}); // det -2
    assert(std::abs(qr.absDeterminant() - 2.0) < kTol);
    assert(std::abs(qr.logAbsDeterminant() - std::log(2.0)) < kTol);
    QR<double> full(M::Random(4, 4, 91));
    assert(full.rank(1e-10) == 4);
}

void test_not_computed() {
    QR<double> qr;
    assert(!qr.isComputed());
    bool threw = false;
    try {
        (void)qr.solve(V{1, 2});
    } catch (const NotComputed &) {
        threw = true;
    }
    assert(threw);
}

void test_complex() {
    Matrix<C> a = Matrix<C>::Random(4, 4, 92);
    QR<C> qr(a);
    Matrix<C> q = qr.matrixQ();
    assert((q * qr.matrixR()).isApprox(a, 1e-8));
    assert((q.conjugateTranspose() * q)
               .isApprox(Matrix<C>::Identity(q.cols()), 1e-10));
}

void test_updating() { // deferred in the plan; skips until built
    M a = M::Random(5, 3, 93);
    QR<double> qr(a);
    V newRow{1, 2, 3};
    qr.appendRow(newRow);
    M grown = a.verticalConcat(newRow.asRowMatrix());
    assert((qr.matrixQ() * qr.matrixR()).isApprox(grown, 1e-8));
}

// ---- ColPivQR --------------------------------------------------------------

void test_colpiv_identity() {
    M a = M::Random(5, 4, 94);
    ColPivQR<double> qr(a);
    assert(qr.isComputed());
    M q = qr.matrixQ();
    M r = qr.matrixR();
    M p = qr.permutationMatrix();
    assert((a * p).isApprox(q * r, 1e-8)); // A P == Q R
    assert((q.transpose() * q).isApprox(M::Identity(q.cols()), 1e-10));
    // pivoting makes |R(i,i)| non-increasing
    for (std::size_t i = 1; i < 4; ++i)
        assert(std::abs(r(i, i)) <= std::abs(r(i - 1, i - 1)) + 1e-12);
}

void test_colpiv_rank_kernel() {
    M rank2{{1, 0, 1}, {0, 1, 1}, {1, 1, 2}, {2, 1, 3}};
    ColPivQR<double> qr(rank2);
    assert(qr.rank(1e-10) == 2);
    M k = qr.kernel(1e-10);
    assert(k.cols() == 1);
    assert((rank2 * k).isApprox(M::Zeros(4, 1), 1e-8));
}

void test_colpiv_solve() {
    M a = M::Random(7, 3, 95);
    V b = V::Random(7, 96);
    ColPivQR<double> qr(a);
    V x = qr.solve(b);
    assert((a.transpose() * (b - a * x)).norm() < 1e-8);
    // minimum-norm solution: also a minimizer, with no larger norm
    V xm = qr.solveMinimumNorm(b);
    assert((a.transpose() * (b - a * xm)).norm() < 1e-8);
    assert(xm.norm() <= x.norm() + 1e-10);
    double rc = qr.reciprocalConditionEstimate();
    assert(rc >= 0.0 && rc <= 1.0 + 1e-12);
}

// ---- GramSchmidt -----------------------------------------------------------

void test_gram_schmidt_variants() {
    using GS = GramSchmidt<double>;
    M a = M::Random(6, 4, 97);
    for (auto variant : {GS::Variant::Kind::Classical,
                         GS::Variant::Kind::Modified,
                         GS::Variant::Kind::ClassicalTwice}) {
        GS gs(a, variant);
        assert(gs.isComputed());
        M q = gs.matrixQ();
        M r = gs.matrixR();
        assert((q * r).isApprox(a, 1e-8));
        assert(r.isTriangular(Triangle::Kind::Upper, 1e-12));
        // well-conditioned input: every variant stays orthogonal
        assert(gs.orthogonalityLoss() < 1e-10);
    }
}

void test_gram_schmidt_loss_ordering() {
    // the point of the class: on ill-conditioned input, reorthogonalization
    // beats a single classical pass
    using GS = GramSchmidt<double>;
    M h = M::Hilbert(8);
    GS classical(h, GS::Variant::Kind::Classical);
    GS twice(h, GS::Variant::Kind::ClassicalTwice);
    assert(twice.orthogonalityLoss() <= classical.orthogonalityLoss() + 1e-14);
    // "twice is enough": reorthogonalized Q is orthogonal to roundoff
    assert(twice.orthogonalityLoss() < 1e-10);
}

} // namespace

int main() {
    run("factorization_identity", test_factorization_identity);
    run("tall_matrix", test_tall_matrix);
    run("full_mode", test_full_mode);
    run("solve", test_solve);
    run("least_squares", test_least_squares);
    run("apply_q", test_apply_q);
    run("determinant", test_determinant);
    run("not_computed", test_not_computed);
    run("complex", test_complex);
    run("updating", test_updating);
    run("colpiv_identity", test_colpiv_identity);
    run("colpiv_rank_kernel", test_colpiv_rank_kernel);
    run("colpiv_solve", test_colpiv_solve);
    run("gram_schmidt_variants", test_gram_schmidt_variants);
    run("gram_schmidt_loss_ordering", test_gram_schmidt_loss_ordering);
    return finish("QR");
}
