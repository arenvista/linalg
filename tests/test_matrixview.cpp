// Behavioural tests for linalg::MatrixView / ConstMatrixView.
//
// Convention (matches tests/): exits 0 on success, aborts via assert() on the
// first failure, which ctest reports as a failure.
//
// toMatrix() and operator=(const Matrix&) are exercised elsewhere once
// Matrix is implemented (PLAN.md step 3); everything else is covered here.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/MatrixView.hpp"

#include <cassert>
#include <complex>
#include <cstdio>
#include <utility>

using namespace linalg;
using C = std::complex<double>;

namespace {

// A 3x4 row-major buffer: element (i,j) == 10*i + j.
struct Buffer34 {
    double a[12];
    Buffer34() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                a[i * 4 + j] = 10.0 * i + j;
    }
    MatrixView<double> view() { return MatrixView<double>(a, 3, 4, 4, 1); }
};

// ---- construction and accessors ------------------------------------------

void test_empty() {
    MatrixView<double> v;
    assert(v.rows() == 0 && v.cols() == 0);
    assert(v.isEmpty());
    assert(v.data() == nullptr);
    ConstMatrixView<double> cv;
    assert(cv.isEmpty() && cv.data() == nullptr);
    // zero rows or zero cols is empty even with a live pointer
    double x = 1;
    assert(MatrixView<double>(&x, 0, 5, 5, 1).isEmpty());
    assert(MatrixView<double>(&x, 5, 0, 1, 1).isEmpty());
}

void test_element_access() {
    Buffer34 b;
    auto v = b.view();
    assert(v.rows() == 3 && v.cols() == 4);
    assert(v.rowStride() == 4 && v.colStride() == 1);
    assert(!v.isEmpty());
    assert(v.data() == b.a);
    assert(v(0, 0) == 0.0 && v(1, 2) == 12.0 && v(2, 3) == 23.0);
    v(1, 2) = -5.0;
    assert(b.a[6] == -5.0); // writes land in the underlying buffer
}

void test_strided_access() {
    // Column-major view over the same buffer: (i,j) at data[i + 3*j]
    double a[6] = {1, 2, 3, 4, 5, 6}; // columns (1,2,3) and (4,5,6)
    MatrixView<double> v(a, 3, 2, 1, 3);
    assert(v(0, 0) == 1 && v(2, 0) == 3 && v(0, 1) == 4 && v(2, 1) == 6);
    assert(!v.isContiguous()); // contiguous means row-major, gap-free
}

void test_isContiguous() {
    Buffer34 b;
    assert(b.view().isContiguous());
    // a block that skips columns has rowStride != cols
    auto blk = b.view().block(0, 0, 3, 2);
    assert(!blk.isContiguous());
    // a full-width block stays contiguous
    auto full = b.view().block(1, 0, 2, 4);
    assert(full.isContiguous());
}

void test_at_bounds() {
    Buffer34 b;
    auto v = b.view();
    assert(v.at(2, 3) == 23.0);
    v.at(0, 1) = 99.0;
    assert(b.a[1] == 99.0);
    bool threw = false;
    try {
        v.at(3, 0);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        v.at(0, 4);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
    // caught as the base class too
    threw = false;
    try {
        v.at(7, 7);
    } catch (const LinalgError &) {
        threw = true;
    }
    assert(threw);
}

// ---- shallow copy / move semantics ---------------------------------------

void test_copy_rebinds() {
    Buffer34 b;
    auto v = b.view();
    MatrixView<double> w(v); // shallow: same storage
    assert(w.data() == v.data());
    w(0, 0) = 42.0;
    assert(v(0, 0) == 42.0);

    double other[4] = {7, 7, 7, 7};
    MatrixView<double> o(other, 2, 2, 2, 1);
    o = v; // copy assignment rebinds, does not copy elements
    assert(o.data() == b.a);
    assert(o.rows() == 3 && o.cols() == 4);
    assert(other[0] == 7.0); // old target untouched
}

void test_move_empties_source() {
    Buffer34 b;
    auto v = b.view();
    MatrixView<double> m(std::move(v));
    assert(m.data() == b.a && m.rows() == 3);
    assert(v.isEmpty() && v.data() == nullptr);

    MatrixView<double> m2;
    m2 = std::move(m);
    assert(m2.data() == b.a);
    assert(m.isEmpty());
}

// ---- sub-views ------------------------------------------------------------

void test_block() {
    Buffer34 b;
    auto v = b.view();
    auto blk = v.block(1, 1, 2, 3);
    assert(blk.rows() == 2 && blk.cols() == 3);
    assert(blk(0, 0) == 11.0 && blk(1, 2) == 23.0);
    blk(0, 1) = -1.0; // writes visible in the parent
    assert(v(1, 2) == -1.0);
    // nested block indices are relative to the block, not the root
    auto inner = blk.block(1, 1, 1, 2);
    assert(inner(0, 0) == 22.0 && inner(0, 1) == 23.0);
    // a block that overruns the view throws
    bool threw = false;
    try {
        v.block(1, 1, 3, 3);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
}

void test_row_col() {
    Buffer34 b;
    auto v = b.view();
    auto r = v.row(1);
    assert(r.rows() == 1 && r.cols() == 4);
    assert(r(0, 0) == 10.0 && r(0, 3) == 13.0);
    auto c = v.col(2);
    assert(c.rows() == 3 && c.cols() == 1);
    assert(c(0, 0) == 2.0 && c(2, 0) == 22.0);
    c(1, 0) = 77.0;
    assert(v(1, 2) == 77.0);
    bool threw = false;
    try {
        v.row(3);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        v.col(4);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
}

void test_diagonal() {
    Buffer34 b;
    auto v = b.view();
    auto d = v.diagonal(); // 1 x min(3,4)
    assert(d.rows() == 1 && d.cols() == 3);
    assert(d(0, 0) == 0.0 && d(0, 1) == 11.0 && d(0, 2) == 22.0);
    d(0, 1) = -3.0;
    assert(v(1, 1) == -3.0);
    // diagonal of a tall block
    auto blk = v.block(0, 1, 3, 2); // 3x2 -> diagonal length 2
    auto d2 = blk.diagonal();
    assert(d2.cols() == 2);
    assert(d2(0, 0) == v(0, 1) && d2(0, 1) == v(1, 2));
}

void test_transposed() {
    Buffer34 b;
    auto v = b.view();
    auto t = v.transposed();
    assert(t.rows() == 4 && t.cols() == 3);
    assert(t.rowStride() == v.colStride() && t.colStride() == v.rowStride());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            assert(t(j, i) == v(i, j));
    t(3, 0) = -9.0; // shares storage
    assert(v(0, 3) == -9.0);
    // transpose round-trip is the original shape and values
    auto tt = t.transposed();
    assert(tt.rows() == 3 && tt.cols() == 4);
    assert(tt(0, 3) == -9.0);
}

// ---- element mutation ------------------------------------------------------

void test_fill_setZero_scale() {
    Buffer34 b;
    auto v = b.view();
    auto blk = v.block(0, 0, 2, 2);
    blk.fill(5.0);
    assert(v(0, 0) == 5.0 && v(1, 1) == 5.0);
    assert(v(2, 0) == 20.0 && v(0, 2) == 2.0); // outside the block untouched
    blk.scale(3.0);
    assert(v(0, 0) == 15.0 && v(1, 1) == 15.0);
    assert(v(2, 2) == 22.0);
    blk.setZero();
    assert(v(0, 0) == 0.0 && v(1, 1) == 0.0);
    assert(v(2, 3) == 23.0);
}

void test_swapWith() {
    double a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8};
    MatrixView<double> va(a, 2, 2, 2, 1), vb(b, 2, 2, 2, 1);
    va.swapWith(vb);
    // elements exchanged...
    assert(a[0] == 5 && a[3] == 8 && b[0] == 1 && b[3] == 4);
    // ...but the bindings stay put
    assert(va.data() == a && vb.data() == b);
}

void test_copyFrom() {
    double src[4] = {1, 2, 3, 4}, dst[4] = {0, 0, 0, 0};
    MatrixView<double> vs(src, 2, 2, 2, 1), vd(dst, 2, 2, 2, 1);
    vd.copyFrom(vs);
    assert(dst[0] == 1 && dst[1] == 2 && dst[2] == 3 && dst[3] == 4);
    assert(src[0] == 1); // source untouched
    // copy across different strides: transposed source
    double d2[4] = {0, 0, 0, 0};
    MatrixView<double> vd2(d2, 2, 2, 2, 1);
    vd2.copyFrom(vs.transposed());
    assert(d2[0] == 1 && d2[1] == 3 && d2[2] == 2 && d2[3] == 4);
}

// ---- ConstMatrixView -------------------------------------------------------

void test_const_view() {
    Buffer34 b;
    ConstMatrixView<double> cv(b.a, 3, 4, 4, 1);
    assert(cv.rows() == 3 && cv.cols() == 4);
    assert(cv(1, 2) == 12.0 && cv.at(2, 3) == 23.0);
    assert(cv.isContiguous() && !cv.isEmpty());
    assert(cv.data() == b.a);

    // implicit conversion from a mutable view
    auto mv = b.view();
    ConstMatrixView<double> converted = mv;
    assert(converted.data() == b.a && converted.rows() == 3);

    // sub-views mirror the mutable API
    auto blk = cv.block(1, 1, 2, 3);
    assert(blk(0, 0) == 11.0 && blk(1, 2) == 23.0);
    assert(cv.row(2)(0, 1) == 21.0);
    assert(cv.col(3)(1, 0) == 13.0);
    auto d = cv.diagonal();
    assert(d.cols() == 3 && d(0, 2) == 22.0);
    auto t = cv.transposed();
    assert(t.rows() == 4 && t(3, 1) == 13.0);

    bool threw = false;
    try {
        cv.at(3, 0);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        cv.block(0, 2, 2, 3);
    } catch (const IndexOutOfRange &) {
        threw = true;
    }
    assert(threw);
}

// ---- complex scalar --------------------------------------------------------

void test_complex_scalar() {
    C a[4] = {C(1, 1), C(2, -1), C(0, 3), C(-1, 0)};
    MatrixView<C> v(a, 2, 2, 2, 1);
    assert(v(0, 1) == C(2, -1));
    v.scale(C(0, 1)); // multiply by i
    assert(a[0] == C(-1, 1) && a[3] == C(0, -1));
    auto t = v.transposed();
    assert(t(1, 0) == v(0, 1));
}

} // namespace

int main() {
    test_empty();
    test_element_access();
    test_strided_access();
    test_isContiguous();
    test_at_bounds();
    test_copy_rebinds();
    test_move_empties_source();
    test_block();
    test_row_col();
    test_diagonal();
    test_transposed();
    test_fill_setZero_scale();
    test_swapWith();
    test_copyFrom();
    test_const_view();
    test_complex_scalar();
    std::puts("all MatrixView behavioural tests passed");
    return 0;
}
