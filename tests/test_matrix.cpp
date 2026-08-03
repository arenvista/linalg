// Behavioural tests for linalg::Matrix (PLAN.md step 3).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp); implement a method and its tests start enforcing.
// Members deferred to Phase 2/4 (determinant, inverse, rank, spectralNorm,
// conditionNumber, pseudoInverse, RandomOrthogonal) have tests here too —
// they stay skipped until those phases land.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using C = std::complex<double>;
using MC = Matrix<C>;

namespace {

constexpr double kTol = 1e-10;

bool close(double a, double b) { return std::abs(a - b) < kTol; }

// ---- construction ----------------------------------------------------------

void test_ctor_shape() {
    M a(2, 3);
    assert(a.rows() == 2 && a.cols() == 3 && a.size() == 6);
    assert(!a.isEmpty() && !a.isSquare());
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            assert(a(i, j) == 0.0); // value-initialized
    M e;
    assert(e.rows() == 0 && e.cols() == 0 && e.isEmpty());
    M sq(3, 3);
    assert(sq.isSquare());
}

void test_ctor_fill_and_data() {
    M a(2, 2, 7.5);
    assert(a(0, 0) == 7.5 && a(1, 1) == 7.5);
    M b(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
    assert(b(0, 0) == 1 && b(0, 2) == 3 && b(1, 0) == 4 && b(1, 2) == 6);
    // row-major data(): (i,j) at data()[i*cols + j]
    assert(b.data()[1 * 3 + 2] == 6);
}

void test_ctor_initializer_list() {
    M a{{1, 2, 3}, {4, 5, 6}};
    assert(a.rows() == 2 && a.cols() == 3);
    assert(a(0, 1) == 2 && a(1, 2) == 6);
}

void test_copy_move() {
    M a{{1, 2}, {3, 4}};
    M b(a); // deep copy
    b(0, 0) = 99;
    assert(a(0, 0) == 1 && b(0, 0) == 99);
    M c(std::move(b));
    assert(c(0, 0) == 99);
    assert(b.isEmpty()); // moved-from is empty
    M d;
    d = a;
    d(1, 1) = -1;
    assert(a(1, 1) == 4 && d(1, 1) == -1);
    M e;
    e = std::move(d);
    assert(e(1, 1) == -1 && d.isEmpty());
}

void test_named_constructors() {
    assert(M::Zeros(2, 3)(1, 2) == 0.0);
    assert(M::Ones(2, 2)(0, 1) == 1.0);
    assert(M::Constant(2, 2, 3.5)(1, 0) == 3.5);
    M i = M::Identity(3);
    assert(i(0, 0) == 1 && i(1, 1) == 1 && i(0, 1) == 0 && i(2, 0) == 0);
    M d = M::Diagonal(V{5, 7});
    assert(d.rows() == 2 && d.cols() == 2);
    assert(d(0, 0) == 5 && d(1, 1) == 7 && d(0, 1) == 0 && d(1, 0) == 0);
}

void test_random_deterministic() {
    M a = M::Random(3, 4, 42), b = M::Random(3, 4, 42);
    assert(a.rows() == 3 && a.cols() == 4);
    assert(a == b); // same seed -> same entries
    M c = M::Random(3, 4, 43);
    assert(a != c); // different seed -> different entries
}

void test_random_symmetric() {
    M s = M::RandomSymmetric(4, 1);
    assert(s.isSymmetric(0.0)); // exactly symmetric for real T
    MC h = MC::RandomSymmetric(4, 1);
    assert(h.isHermitian(1e-14));
}

void test_random_orthogonal() { // Phase 2 back-fill
    M q = M::RandomOrthogonal(4, 7);
    assert(q.isOrthogonal(1e-12));
    MC u = MC::RandomOrthogonal(3, 7);
    assert(u.isOrthogonal(1e-12));
}

void test_hilbert_vandermonde() {
    M h = M::Hilbert(3); // a(i,j) = 1/(i+j+1)
    assert(close(h(0, 0), 1.0) && close(h(0, 1), 0.5));
    assert(close(h(1, 1), 1.0 / 3.0) && close(h(2, 2), 0.2));
    assert(h.isSymmetric(0.0));
    M v = M::Vandermonde(V{1, 2, 3}, 2); // a(i,j) = nodes[i]^j
    assert(v.rows() == 3 && v.cols() == 3);
    assert(close(v(1, 0), 1) && close(v(1, 1), 2) && close(v(1, 2), 4));
    assert(close(v(2, 2), 9));
}

void test_from_columns_rows() {
    M a = M::FromColumns({V{1, 2}, V{3, 4}});
    assert(a.rows() == 2 && a.cols() == 2);
    assert(a(0, 0) == 1 && a(1, 0) == 2 && a(0, 1) == 3 && a(1, 1) == 4);
    M b = M::FromRows({V{1, 2}, V{3, 4}});
    assert(b(0, 0) == 1 && b(0, 1) == 2 && b(1, 0) == 3 && b(1, 1) == 4);
}

// ---- element access --------------------------------------------------------

void test_at_bounds() {
    M a{{1, 2}, {3, 4}};
    assert(a.at(1, 1) == 4);
    a.at(0, 1) = 9;
    assert(a(0, 1) == 9);
    bool threw = false;
    try {
        a.at(2, 0);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        a.at(0, 2);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
}

// ---- views -----------------------------------------------------------------

void test_views_alias() {
    M a{{0, 1, 2}, {10, 11, 12}, {20, 21, 22}};
    auto v = a.view();
    assert(v.rows() == 3 && v(1, 2) == 12);
    v(0, 0) = -5; // writes visible in the matrix
    assert(a(0, 0) == -5);
    auto blk = a.block(1, 1, 2, 2);
    assert(blk(0, 0) == 11 && blk(1, 1) == 22);
    blk.fill(0.0);
    assert(a(1, 1) == 0 && a(2, 2) == 0 && a(0, 1) == 1); // outside untouched
    assert(a.rowView(0).cols() == 3 && a.rowView(0)(0, 1) == 1);
    assert(a.colView(2).rows() == 3 && a.colView(2)(0, 0) == 2);
}

void test_row_col_diagonal_copies() {
    M a{{1, 2}, {3, 4}};
    V r = a.row(1);
    assert(r.size() == 2 && r(0) == 3 && r(1) == 4);
    r(0) = 99; // owning copy: matrix unaffected
    assert(a(1, 0) == 3);
    V c = a.col(0);
    assert(c(0) == 1 && c(1) == 3);
    V d = a.diagonal();
    assert(d.size() == 2 && d(0) == 1 && d(1) == 4);
}

void test_set_row_col_block() {
    M a = M::Zeros(3, 3);
    a.setRow(0, V{1, 2, 3});
    assert(a(0, 0) == 1 && a(0, 2) == 3);
    a.setCol(2, V{7, 8, 9});
    assert(a(0, 2) == 7 && a(2, 2) == 9);
    a.setBlock(1, 0, M{{5, 6}});
    assert(a(1, 0) == 5 && a(1, 1) == 6);
}

// ---- arithmetic ------------------------------------------------------------

void test_add_sub() {
    M a{{1, 2}, {3, 4}}, b{{10, 20}, {30, 40}};
    M s = a + b;
    assert(s(0, 0) == 11 && s(1, 1) == 44);
    M d = b - a;
    assert(d(0, 0) == 9 && d(1, 1) == 36);
    M n = -a;
    assert(n(0, 1) == -2);
    a += b;
    assert(a(1, 0) == 33);
    a -= b;
    assert(a(1, 0) == 3);
    bool threw = false;
    try {
        (void)(a + M::Zeros(3, 3));
    } catch (const DimensionMismatch &) {
        threw = true;
    }
    assert(threw);
}

void test_matmul() {
    M a{{1, 2, 3}, {4, 5, 6}};      // 2x3
    M b{{7, 8}, {9, 10}, {11, 12}}; // 3x2
    M p = a * b;
    assert(p.rows() == 2 && p.cols() == 2);
    assert(close(p(0, 0), 58) && close(p(0, 1), 64));
    assert(close(p(1, 0), 139) && close(p(1, 1), 154));
    // identity is neutral
    assert((a * M::Identity(3)).isApprox(a, kTol));
    bool threw = false;
    try {
        (void)(a * a); // inner dims 3 vs 2
    } catch (const DimensionMismatch &) {
        threw = true;
    }
    assert(threw);
    M sq{{1, 1}, {0, 1}};
    M sq2 = sq;
    sq2 *= sq;
    assert((sq2).isApprox(sq * sq, kTol));
}

void test_matvec_distributes() {
    // (A+B)*x == A*x + B*x — the PLAN.md checkpoint identity
    M a{{1, 2}, {3, 4}}, b{{5, 6}, {7, 8}};
    V x{1, -1};
    V lhs = (a + b) * x;
    V rhs = (a * x).axpy(1.0, b * x);
    assert(lhs.isApprox(rhs, kTol));
    V ax = a * x;
    assert(close(ax(0), -1) && close(ax(1), -1));
}

void test_scalar_ops() {
    M a{{1, 2}, {3, 4}};
    M m = a * 2.0;
    assert(m(1, 1) == 8);
    M d = a / 2.0;
    assert(d(0, 0) == 0.5);
    M s = a.scaledBy(3.0); // scalar * A form
    assert(s(0, 1) == 6);
    a *= 10.0;
    assert(a(0, 0) == 10);
    a /= 10.0;
    assert(a(0, 0) == 1);
}

void test_elementwise_kronecker_power() {
    M a{{1, 2}, {3, 4}}, b{{2, 2}, {2, 2}};
    assert(a.elementwiseProduct(b)(1, 1) == 8);
    assert(a.elementwiseQuotient(b)(1, 0) == 1.5);
    M k = M{{1, 2}}.kroneckerProduct(M{{0, 1}, {1, 0}});
    assert(k.rows() == 2 && k.cols() == 4);
    assert(k(0, 0) == 0 && k(0, 1) == 1 && k(0, 2) == 0 && k(0, 3) == 2);
    assert(k(1, 0) == 1 && k(1, 2) == 2);
    assert(a.power(0).isApprox(M::Identity(2), kTol));
    assert(a.power(1).isApprox(a, kTol));
    assert(a.power(3).isApprox(a * a * a, kTol));
}

void test_equality() {
    M a{{1, 2}, {3, 4}};
    M b = a;
    assert(a == b && !(a != b));
    b(0, 0) += 1e-15;
    assert(a != b); // exact comparison
    assert(a.isApprox(b, 1e-12));
    assert(!(a == M::Zeros(2, 3))); // shape mismatch is inequality, not throw
}

// ---- shape manipulation ----------------------------------------------------

void test_transpose() {
    M a{{1, 2, 3}, {4, 5, 6}};
    M t = a.transpose();
    assert(t.rows() == 3 && t.cols() == 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            assert(t(j, i) == a(i, j));
    assert(t.transpose() == a); // round-trip
    M sq{{1, 2}, {3, 4}};
    sq.transposeInPlace();
    assert(sq(0, 1) == 3 && sq(1, 0) == 2);
}

void test_conjugate_transpose() {
    MC a{{C(1, 2), C(3, -1)}, {C(0, 1), C(2, 0)}};
    MC h = a.conjugateTranspose();
    assert(h(0, 0) == C(1, -2) && h(1, 0) == C(3, 1) && h(0, 1) == C(0, -1));
    assert(h.conjugateTranspose() == a);
    // real T: conjugateTranspose == transpose
    M r{{1, 2}, {3, 4}};
    assert(r.conjugateTranspose() == r.transpose());
}

void test_reshape_concat_without() {
    M a{{1, 2, 3}, {4, 5, 6}};
    M r = a.reshaped(3, 2); // row-major order preserved
    assert(r(0, 0) == 1 && r(0, 1) == 2 && r(1, 0) == 3 && r(2, 1) == 6);
    bool threw = false;
    try {
        (void)a.reshaped(4, 2);
    } catch (const DimensionMismatch &) {
        threw = true;
    }
    assert(threw);
    M h = M{{1}, {3}}.horizontalConcat(M{{2}, {4}});
    assert(h.rows() == 2 && h.cols() == 2 && h(0, 1) == 2 && h(1, 0) == 3);
    M v = M{{1, 2}}.verticalConcat(M{{3, 4}});
    assert(v.rows() == 2 && v(1, 0) == 3);
    M wr = a.withoutRow(0);
    assert(wr.rows() == 1 && wr(0, 0) == 4);
    M wc = a.withoutCol(1);
    assert(wc.cols() == 2 && wc(0, 1) == 3 && wc(1, 1) == 6);
}

void test_resize_swap_fill() {
    M a{{1, 2}, {3, 4}};
    a.conservativeResize(3, 3); // keeps top-left, zero-fills growth
    assert(a.rows() == 3 && a(0, 0) == 1 && a(1, 1) == 4);
    assert(a(2, 2) == 0 && a(0, 2) == 0);
    a.resize(2, 2); // discards contents
    assert(a.rows() == 2 && a.cols() == 2);
    a.fill(5.0);
    assert(a(1, 1) == 5);
    a.setZero();
    assert(a(0, 0) == 0);
    a.setIdentity();
    assert(a(0, 0) == 1 && a(0, 1) == 0);
    M b{{9, 8}, {7, 6}};
    a.swap(b);
    assert(a(0, 0) == 9 && b(0, 0) == 1);
    M c{{1, 2}, {3, 4}};
    c.swapRows(0, 1);
    assert(c(0, 0) == 3 && c(1, 0) == 1);
    c.swapCols(0, 1);
    assert(c(0, 0) == 4 && c(0, 1) == 3);
}

// ---- scalar summaries ------------------------------------------------------

void test_trace_sum_norms() {
    M a{{1, -2}, {3, 4}};
    assert(close(a.trace(), 5));
    assert(close(a.sum(), 6));
    assert(close(a.oneNorm(), 6));      // max abs column sum: |−2|+|4|
    assert(close(a.infinityNorm(), 7)); // max abs row sum: |3|+|4|
    assert(close(a.frobeniusNorm(), std::sqrt(30.0)));
    assert(close(a.maxNorm(), 4));
    // complex: magnitudes, not real parts
    MC c{{C(3, 4)}};
    assert(close(c.frobeniusNorm(), 5.0));
    assert(close(c.maxNorm(), 5.0));
}

void test_determinant() { // Phase 2 back-fill (LU)
    assert(close(M{{2, 0}, {0, 3}}.determinant(), 6));
    assert(close(M{{1, 2}, {3, 4}}.determinant(), -2));
    assert(close(M{{1, 2}, {2, 4}}.determinant(), 0)); // singular
    assert(close(M::Identity(4).determinant(), 1));
}

void test_inverse() { // Phase 2 back-fill (LU)
    M a{{4, 7}, {2, 6}};
    M inv = a.inverse();
    assert((a * inv).isApprox(M::Identity(2), 1e-10));
    assert((inv * a).isApprox(M::Identity(2), 1e-10));
    bool threw = false;
    try {
        (void)M{{1, 2}, {2, 4}}.inverse();
    } catch (const SingularMatrix &) {
        threw = true;
    }
    assert(threw);
}

void test_rank_spectral_condition() { // Phase 2/4 back-fill (ColPivQR/SVD)
    M full{{1, 0}, {0, 2}};
    assert(full.rank(1e-12) == 2);
    M rank1{{1, 2}, {2, 4}}; // second row = 2 * first
    assert(rank1.rank(1e-12) == 1);
    assert(M::Zeros(3, 3).rank(1e-12) == 0);
    M d = M::Diagonal(V{3, -5});
    assert(close(d.spectralNorm(), 5.0));
    assert(close(d.conditionNumber(), 5.0 / 3.0));
    assert(close(M::Identity(3).conditionNumber(), 1.0));
}

void test_pseudo_inverse() { // Phase 4 back-fill (SVD)
    M a{{1, 0}, {0, 2}, {0, 0}}; // tall, full column rank
    M p = a.pseudoInverse(1e-12);
    assert(p.rows() == 2 && p.cols() == 3);
    assert((p * a).isApprox(M::Identity(2), 1e-10)); // left inverse
    assert((a * p * a).isApprox(a, 1e-10));          // Moore-Penrose
}

// ---- predicates ------------------------------------------------------------

void test_predicates() {
    M s{{1, 2}, {2, 3}};
    assert(s.isSymmetric(0.0));
    assert(!M{{1, 2}, {3, 4}}.isSymmetric(0.0));
    assert(M{{1, 2}, {3, 4}}.isSymmetric(1.5)); // within loose tolerance

    MC h{{C(1, 0), C(2, 3)}, {C(2, -3), C(5, 0)}};
    assert(h.isHermitian(0.0));
    MC nh{{C(1, 1), C(0, 0)}, {C(0, 0), C(1, 0)}}; // complex diagonal
    assert(!nh.isHermitian(1e-12));

    assert(M::Diagonal(V{1, 2}).isDiagonal(0.0));
    assert(!M{{1, 1}, {0, 1}}.isDiagonal(0.0));

    M up{{1, 2}, {0, 3}};
    assert(up.isTriangular(Triangle::Kind::Upper, 0.0));
    assert(!up.isTriangular(Triangle::Kind::Lower, 0.0));
    assert(up.transpose().isTriangular(Triangle::Kind::Lower, 0.0));

    assert(M::Identity(3).isOrthogonal(0.0));
    // rotation by 90 degrees is orthogonal
    assert(M{{0, -1}, {1, 0}}.isOrthogonal(1e-14));
    assert(!M{{2, 0}, {0, 1}}.isOrthogonal(1e-6));

    assert(!M::Ones(2, 2).hasNaN());
    M nan{{1, 2}, {3, 4}};
    nan(0, 1) = std::nan("");
    assert(nan.hasNaN());
}

// ---- derived matrices ------------------------------------------------------

void test_parts() {
    M a{{1, 2}, {3, 4}};
    M up = a.triangularPart(Triangle::Kind::Upper);
    assert(up(0, 1) == 2 && up(1, 0) == 0 && up(1, 1) == 4);
    M lo = a.triangularPart(Triangle::Kind::Lower);
    assert(lo(1, 0) == 3 && lo(0, 1) == 0);
    M sym = a.symmetricPart(), skew = a.skewSymmetricPart();
    assert(sym.isSymmetric(1e-14));
    assert(close(skew(0, 1), -skew(1, 0)));
    assert((sym + skew).isApprox(a, kTol)); // decomposition identity
}

void test_matrixview_interop() {
    // the two MatrixView members deferred until Matrix existed
    M a{{1, 2, 3}, {4, 5, 6}};
    // toMatrix deep-copies: mutating the copy leaves the source alone
    M blockCopy = a.block(0, 1, 2, 2).toMatrix();
    assert(blockCopy.rows() == 2 && blockCopy.cols() == 2);
    assert(blockCopy(0, 0) == 2 && blockCopy(1, 1) == 6);
    blockCopy(0, 0) = 99;
    assert(a(0, 1) == 2);
    // toMatrix through a transposed (strided) view
    M t = a.view().transposed().toMatrix();
    assert(t.rows() == 3 && t(2, 1) == 6);
    // view = Matrix copies elements into the viewed storage
    auto blk = a.block(0, 0, 2, 2);
    blk = M{{-1, -2}, {-4, -5}};
    assert(a(0, 0) == -1 && a(1, 1) == -5);
    assert(a(0, 2) == 3); // outside the block untouched
}

// ---- serialization ---------------------------------------------------------

void test_to_string() {
    M a{{1.5, 2}, {3, 4}};
    std::string s = a.toString(3);
    assert(!s.empty());
    assert(s.find("1.5") != std::string::npos);
    std::string ml = a.toMatlabLiteral();
    assert(ml.front() == '[' && ml.back() == ']');
    assert(ml.find(';') != std::string::npos); // row separator
}

void test_csv_roundtrip() {
    M a{{1.25, -2}, {3, 4.5}};
    const std::string path = "test_matrix_roundtrip.csv";
    a.writeCsv(path);
    M b = M::FromCsv(path);
    std::remove(path.c_str());
    assert(b.rows() == 2 && b.cols() == 2);
    assert(b.isApprox(a, 1e-12));
}

} // namespace

int main() {
    run("ctor_shape", test_ctor_shape);
    run("ctor_fill_and_data", test_ctor_fill_and_data);
    run("ctor_initializer_list", test_ctor_initializer_list);
    run("copy_move", test_copy_move);
    run("named_constructors", test_named_constructors);
    run("random_deterministic", test_random_deterministic);
    run("random_symmetric", test_random_symmetric);
    run("random_orthogonal", test_random_orthogonal);
    run("hilbert_vandermonde", test_hilbert_vandermonde);
    run("from_columns_rows", test_from_columns_rows);
    run("at_bounds", test_at_bounds);
    run("views_alias", test_views_alias);
    run("row_col_diagonal_copies", test_row_col_diagonal_copies);
    run("set_row_col_block", test_set_row_col_block);
    run("add_sub", test_add_sub);
    run("matmul", test_matmul);
    run("matvec_distributes", test_matvec_distributes);
    run("scalar_ops", test_scalar_ops);
    run("elementwise_kronecker_power", test_elementwise_kronecker_power);
    run("equality", test_equality);
    run("transpose", test_transpose);
    run("conjugate_transpose", test_conjugate_transpose);
    run("reshape_concat_without", test_reshape_concat_without);
    run("resize_swap_fill", test_resize_swap_fill);
    run("trace_sum_norms", test_trace_sum_norms);
    run("determinant", test_determinant);
    run("inverse", test_inverse);
    run("rank_spectral_condition", test_rank_spectral_condition);
    run("pseudo_inverse", test_pseudo_inverse);
    run("predicates", test_predicates);
    run("parts", test_parts);
    run("matrixview_interop", test_matrixview_interop);
    run("to_string", test_to_string);
    run("csv_roundtrip", test_csv_roundtrip);
    return finish("Matrix");
}
