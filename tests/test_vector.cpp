// Behavioural tests for linalg::Vector.
//
// Convention (matches the rest of tests/): the program exits 0 on success and
// aborts via assert() on the first failure, which ctest reports as a failure.
// The Matrix-returning methods (outer/as*Matrix) and conservativeResize
// depend on code not yet implemented, so those tests run through the skip
// harness and report as Skipped until it lands.

#include "linalg/core/Exceptions.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

using namespace linalg;
using C = std::complex<double>;

namespace {

// Runs `expr`; asserts it throws a linalg::LinalgError (covers all subclasses).
#define EXPECT_LINALG_THROW(expr)                        \
    do {                                                 \
        bool threw = false;                              \
        try {                                            \
            (void)(expr);                                \
        } catch (const LinalgError &) {                  \
            threw = true;                                \
        }                                                \
        assert(threw && "expected LinalgError: " #expr); \
    } while (0)

void test_constructors_and_assignment() {
    Vector<double> a{1.0, 2.0, 3.0};
    assert(a.size() == 3 && !a.isEmpty());
    assert(a(0) == 1.0 && a(2) == 3.0);

    Vector<double> empty;
    assert(empty.size() == 0 && empty.isEmpty());

    Vector<double> filled(4, 7.0);
    assert(filled.size() == 4 && filled(0) == 7.0 && filled(3) == 7.0);

    Vector<double> copy(a); // copy ctor
    assert(copy == a);

    Vector<double> moved(std::move(copy)); // move ctor
    assert(moved == a && moved.size() == 3);

    Vector<double> assigned;
    assigned = a; // copy assign
    assert(assigned == a);
    Vector<double> massign;
    massign = std::move(assigned); // move assign
    assert(massign == a);

    std::vector<double> raw{5.0, 6.0};
    Vector<double>      fromStd(raw);
    assert(fromStd.size() == 2 && fromStd(1) == 6.0);
}

void test_factories() {
    assert(Vector<double>::Zeros(3) == Vector<double>({0.0, 0.0, 0.0}));
    assert(Vector<double>::Ones(2) == Vector<double>({1.0, 1.0}));
    assert(Vector<double>::Constant(2, 9.0) == Vector<double>({9.0, 9.0}));
    assert(Vector<double>::Unit(3, 1) == Vector<double>({0.0, 1.0, 0.0}));
    EXPECT_LINALG_THROW(Vector<double>::Unit(3, 3)); // axis out of range

    // LinSpace: inclusive endpoints, even spacing
    auto ls = Vector<double>::LinSpace(5, 0.0, 1.0);
    for (std::size_t i = 0; i < 5; ++i)
        assert(std::abs(ls(i) - 0.25 * static_cast<double>(i)) < 1e-12);
    assert(ls(4) == 1.0);
    assert(Vector<double>::LinSpace(1, 2.0, 9.0) == Vector<double>({2.0}));

    // Random: reproducible per seed, entries in [-1, 1]
    auto r1 = Vector<double>::Random(4, 123);
    auto r2 = Vector<double>::Random(4, 123);
    assert(r1 == r2);
    for (std::size_t i = 0; i < 4; ++i)
        assert(r1(i) >= -1.0 && r1(i) <= 1.0);
}

void test_access_bounds() {
    Vector<double> v{10.0, 20.0, 30.0};
    v(1) = 99.0;
    assert(v[1] == 99.0 && v.at(1) == 99.0);
    v.at(2) = 42.0;
    assert(v(2) == 42.0);

    // data() is contiguous and writes through
    double *p = v.data();
    p[0]      = -1.0;
    assert(v(0) == -1.0);

    // at() is bounds-checked with the library's own exception type
    bool caught = false;
    try {
        v.at(5);
    } catch (const IndexOutOfRange &e) {
        caught = true;
        assert(e.index() == 5 && e.bound() == 3);
    }
    assert(caught);
}

void test_arithmetic() {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};

    assert(a + b == Vector<double>({5.0, 7.0, 9.0}));
    assert(b - a == Vector<double>({3.0, 3.0, 3.0}));
    assert(a * 2.0 == Vector<double>({2.0, 4.0, 6.0}));
    assert(b / 2.0 == Vector<double>({2.0, 2.5, 3.0}));
    assert(-a == Vector<double>({-1.0, -2.0, -3.0}));
    assert(a.scaledBy(3.0) == Vector<double>({3.0, 6.0, 9.0}));
    assert(a.elementwiseProduct(b) == Vector<double>({4.0, 10.0, 18.0}));

    // dimension mismatch throws
    Vector<double> shorter{1.0, 2.0};
    EXPECT_LINALG_THROW(a + shorter);
    EXPECT_LINALG_THROW(a - shorter);

    // compound assignment returns *this and mutates in place
    Vector<double>  c   = a;
    Vector<double> &ref = (c += b);
    assert(&ref == &c && c == Vector<double>({5.0, 7.0, 9.0}));
    c -= b;
    assert(c == a);
    c *= 10.0;
    assert(c == Vector<double>({10.0, 20.0, 30.0}));
    c /= 10.0;
    assert(c == a);
    EXPECT_LINALG_THROW(c += shorter);
}

void test_products() {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};
    assert(a.dot(b) == 32.0);          // 4 + 10 + 18
    assert(a.hermitianDot(b) == 32.0); // real: same as dot

    // hermitianDot conjugates the left argument
    Vector<C> x{C(1, 2), C(3, -1)};
    Vector<C> y{C(2, 0), C(1, 1)};
    assert(std::abs(x.hermitianDot(y) - C(4, 0)) < 1e-12);
    assert(std::abs(x.hermitianDot(x) - C(15, 0)) < 1e-12); // == ||x||^2, real

    // cross product (3-vectors)
    Vector<double> e1{1.0, 0.0, 0.0}, e2{0.0, 1.0, 0.0};
    assert(e1.cross(e2) == Vector<double>({0.0, 0.0, 1.0}));
    assert(e2.cross(e1) == Vector<double>({0.0, 0.0, -1.0})); // anticommutative
    EXPECT_LINALG_THROW(a.cross(Vector<double>({1.0, 2.0}))); // non-3 throws
}

void test_norms() {
    Vector<double> v{3.0, -4.0};
    assert(std::abs(v.norm() - 5.0) < 1e-12);
    assert(std::abs(v.squaredNorm() - 25.0) < 1e-12);
    assert(std::abs(v.oneNorm() - 7.0) < 1e-12);
    assert(std::abs(v.infinityNorm() - 4.0) < 1e-12);
    assert(std::abs(v.pNorm(2.0) - 5.0) < 1e-12);
    assert(std::abs(v.pNorm(1.0) - 7.0) < 1e-12);

    // norm() uses overflow-safe scaling; squaredNorm() may overflow
    Vector<double> big{1e200, 1e200};
    assert(std::isfinite(big.norm()));
    assert(std::abs(big.norm() - std::sqrt(2.0) * 1e200) /
               (std::sqrt(2.0) * 1e200) <
           1e-12);
    assert(std::isinf(big.squaredNorm()));

    assert(Vector<double>().norm() == 0.0); // empty -> 0

    // complex magnitudes
    Vector<C> c{C(3, 4), C(0, 12)};
    assert(std::abs(c.norm() - 13.0) < 1e-12); // sqrt(25 + 144)
    assert(std::abs(c.oneNorm() - 17.0) < 1e-12);
    assert(std::abs(c.infinityNorm() - 12.0) < 1e-12);
}

void test_reductions() {
    // maxCoefficient/minCoefficient compare by value (real part), not magnitude
    Vector<double> v{-5.0, 2.0, -1.0};
    assert(v.maxCoefficient() == 2.0);
    assert(v.minCoefficient() == -5.0);
    // maxAbsIndex/minAbsIndex compare by magnitude
    assert(v.maxAbsIndex() == 0); // |-5|
    assert(v.minAbsIndex() == 2); // |-1|

    // ties resolve to lowest index
    Vector<double> t{3.0, 1.0, 3.0};
    assert(t.maxCoefficient() == 3.0 && t.maxAbsIndex() == 0);

    // complex: coefficients by real part, abs-index by modulus
    Vector<C> c{C(1, 10), C(3, 0), C(2, -1)};
    assert(c.maxCoefficient() == C(3, 0));
    assert(c.minCoefficient() == C(1, 10));
    assert(c.maxAbsIndex() == 0); // |1+10i| largest

    // empty vector throws (would otherwise be UB)
    Vector<double> e;
    EXPECT_LINALG_THROW(e.maxCoefficient());
    EXPECT_LINALG_THROW(e.minCoefficient());
    EXPECT_LINALG_THROW(e.maxAbsIndex());
    EXPECT_LINALG_THROW(e.minAbsIndex());
}

void test_predicates_and_misc() {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{1.0 + 1e-10, 2.0 - 1e-10, 3.0};
    assert(a.isApprox(b, 1e-9));
    assert(!a.isApprox(b, 1e-12));
    assert(
        !a.isApprox(Vector<double>({1.0, 2.0}), 1e9)); // size mismatch -> false

    // hasNaN
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    assert(!a.hasNaN());
    assert(Vector<double>({1.0, nan, 3.0}).hasNaN());
    assert(!Vector<double>({inf}).hasNaN()); // inf is not NaN
    assert(Vector<C>({C(1, nan)}).hasNaN()); // NaN in imaginary part

    // resize discards contents and zero-fills
    Vector<double> r{1.0, 2.0, 3.0};
    r.resize(2);
    assert(r == Vector<double>({0.0, 0.0}));
    r.resize(0);
    assert(r.isEmpty());

    // equality / inequality
    assert(a == Vector<double>({1.0, 2.0, 3.0}));
    assert(a != b);
    assert(a != Vector<double>({1.0, 2.0})); // different size

    // toString
    assert(Vector<double>().toString(3) == "[]");
    assert(Vector<double>({1.0}).toString(3) == "[1]");
}

void test_conservative_resize() {
    Vector<double> v{1.0, 2.0, 3.0};
    v.conservativeResize(5); // keeps the head, zero-fills the growth
    assert(v == Vector<double>({1.0, 2.0, 3.0, 0.0, 0.0}));
    v.conservativeResize(2); // truncates
    assert(v == Vector<double>({1.0, 2.0}));
    v.conservativeResize(0);
    assert(v.isEmpty());
}

void test_matrix_bridging() {
    Vector<double> x{1.0, 2.0}, y{10.0, 20.0, 30.0};
    // outer product: x y^H is 2x3 with (i,j) = x_i * y_j
    Matrix<double> o = x.outer(y);
    assert(o.rows() == 2 && o.cols() == 3);
    assert(o(0, 0) == 10.0 && o(0, 2) == 30.0 && o(1, 1) == 40.0);
    // complex outer conjugates the right factor
    Vector<C> cx{C(0, 1)}, cy{C(0, 1)};
    assert(cx.outer(cy)(0, 0) == C(1, 0)); // i * conj(i) = 1
    // as*Matrix shapes and values
    Matrix<double> col = x.asColumnMatrix();
    assert(col.rows() == 2 && col.cols() == 1 && col(1, 0) == 2.0);
    Matrix<double> row = x.asRowMatrix();
    assert(row.rows() == 1 && row.cols() == 2 && row(0, 1) == 2.0);
    Matrix<double> diag = x.asDiagonalMatrix();
    assert(diag.rows() == 2 && diag.cols() == 2);
    assert(diag(0, 0) == 1.0 && diag(1, 1) == 2.0 && diag(0, 1) == 0.0);
}

} // namespace

int main() {
    test_constructors_and_assignment();
    test_factories();
    test_access_bounds();
    test_arithmetic();
    test_products();
    test_norms();
    test_reductions();
    test_predicates_and_misc();
    std::puts("all Vector behavioural tests passed");
    testharness::run("conservative_resize", test_conservative_resize);
    testharness::run("matrix_bridging", test_matrix_bridging);
    return testharness::finish("Vector stragglers");
}
