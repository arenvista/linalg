// Behavioural tests for linalg::Kernels (the BLAS-style compute kernels).
//
// Convention (matches tests/): exits 0 on success, aborts via assert() on the
// first failure, which ctest reports as a failure.
//
// Level-2/3 kernels are cross-checked against each other (e.g. trsv/trsm invert
// gemv/gemm) so a round-trip returning the input is the correctness signal.

#include "linalg/core/MatrixView.hpp"
#include "linalg/ops/Kernels.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>

using namespace linalg;
using C  = std::complex<double>;
using Tr = Transposition::Kind;
using Ul = Triangle::Kind;
using Dg = Diagonal::Kind;

namespace {

bool close(double a, double b) { return std::abs(a - b) < 1e-10; }
bool close(C a, C b) { return std::abs(a - b) < 1e-10; }

// Row-major views over a raw buffer: rowStride = cols, colStride = 1.
template <typename T>
ConstMatrixView<T> cview(const T *p, std::size_t r, std::size_t c) {
    return ConstMatrixView<T>(p, r, c, c, 1);
}
template <typename T> MatrixView<T> mview(T *p, std::size_t r, std::size_t c) {
    return MatrixView<T>(p, r, c, c, 1);
}

// ---- Level 1 -------------------------------------------------------------

void test_scal() {
    double a[4] = {1, 2, 3, 4};
    Kernels<double>::scal(4, 2.0, a, 1);
    assert(close(a[0], 2) && close(a[3], 8));
    // strided: odd indices untouched
    double b[5] = {1, 10, 2, 20, 3};
    Kernels<double>::scal(3, 10.0, b, 2);
    assert(close(b[0], 10) && close(b[2], 20) && close(b[4], 30));
    assert(close(b[1], 10) && close(b[3], 20));
    // complex * i
    C c[2] = {C(1, 1), C(2, -2)};
    Kernels<C>::scal(2, C(0, 1), c, 1);
    assert(close(c[0], C(-1, 1)) && close(c[1], C(2, 2)));
}

void test_axpy() {
    double x[3] = {1, 2, 3}, y[3] = {10, 20, 30};
    Kernels<double>::axpy(3, 2.0, x, 1, y, 1);
    assert(close(y[0], 12) && close(y[1], 24) && close(y[2], 36));
    assert(close(x[0], 1)); // x untouched
    // strided x
    double xs[5] = {1, 0, 2, 0, 3}, ys[3] = {100, 200, 300};
    Kernels<double>::axpy(3, 10.0, xs, 2, ys, 1);
    assert(close(ys[0], 110) && close(ys[2], 330));
    // alpha = 0 leaves y unchanged; n = 0 is a no-op
    double x0[2] = {9, 9}, y0[2] = {1, 2};
    Kernels<double>::axpy(2, 0.0, x0, 1, y0, 1);
    assert(close(y0[0], 1) && close(y0[1], 2));
    Kernels<double>::axpy(0, 1.0, x0, 1, y0, 1);
    assert(close(y0[0], 1));
}

void test_dot_dotc() {
    double x[3] = {1, 2, 3}, y[3] = {4, 5, 6};
    assert(close(Kernels<double>::dot(3, x, 1, y, 1), 32.0));
    assert(close(Kernels<double>::dotc(3, x, 1, y, 1), 32.0)); // real: same
    assert(close(Kernels<double>::dot(0, x, 1, y, 1), 0.0));
    // complex: dot is unconjugated, dotc conjugates x
    C xc[1] = {C(1, 1)}, yc[1] = {C(1, 1)};
    assert(close(Kernels<C>::dot(1, xc, 1, yc, 1), C(0, 2)));  // (1+i)^2 = 2i
    assert(close(Kernels<C>::dotc(1, xc, 1, yc, 1), C(2, 0))); // (1-i)(1+i)=2
    // dotc(v,v) == ||v||^2 (real, nonneg)
    C v[2] = {C(3, 4), C(0, 1)};
    assert(close(Kernels<C>::dotc(2, v, 1, v, 1), C(26, 0)));
}

void test_nrm2() {
    double x[2] = {3, -4};
    assert(close(Kernels<double>::nrm2(2, x, 1), 5.0));
    double xs[3] = {3, 0, -4};
    assert(close(Kernels<double>::nrm2(2, xs, 2), 5.0));
    assert(close(Kernels<double>::nrm2(0, x, 1), 0.0));
    // overflow-safe: entries 1e200 -> finite sqrt(2)*1e200
    double big[2] = {1e200, 1e200};
    double nb     = Kernels<double>::nrm2(2, big, 1);
    assert(std::isfinite(nb));
    assert(std::abs(nb - std::sqrt(2.0) * 1e200) / (std::sqrt(2.0) * 1e200) <
           1e-12);
    C c[2] = {C(3, 4), C(0, 12)};
    assert(close(Kernels<C>::nrm2(2, c, 1), 13.0));
}

void test_asum() {
    double x[3] = {3, -4, 1};
    assert(close(Kernels<double>::asum(3, x, 1), 8.0));
    double xs[5] = {3, 0, -4, 0, 1};
    assert(close(Kernels<double>::asum(3, xs, 2), 8.0));
    assert(close(Kernels<double>::asum(0, x, 1), 0.0));
    // complex: sum of moduli (5 + 12), not |re|+|im|
    C c[2] = {C(3, 4), C(0, 12)};
    assert(close(Kernels<C>::asum(2, c, 1), 17.0));
}

void test_iamax() {
    double x[4] = {1, -9, 3, 9};
    assert(Kernels<double>::iamax(4, x, 1) == 1); // first max magnitude
    double xs[5] = {1, 0, -9, 0, 3};
    assert(Kernels<double>::iamax(3, xs, 2) == 1); // element index, not offset
    double t[3] = {5, 5, 1};
    assert(Kernels<double>::iamax(3, t, 1) == 0); // ties -> lowest
    assert(Kernels<double>::iamax(0, x, 1) == 0); // n=0 convention
    C c[3] = {C(3, 0), C(4, 3), C(1, 10)};
    assert(Kernels<C>::iamax(3, c, 1) == 2); // by modulus
}

void test_swap_copy() {
    double x[3] = {1, 2, 3}, y[3] = {4, 5, 6};
    Kernels<double>::swap(3, x, 1, y, 1);
    assert(close(x[0], 4) && close(y[0], 1) && close(x[2], 6));
    double src[3] = {7, 8, 9}, dst[3] = {0, 0, 0};
    Kernels<double>::copy(3, src, 1, dst, 1);
    assert(close(dst[0], 7) && close(dst[2], 9));
    // strided copy
    double s2[3] = {1, 2, 3}, d2[5] = {0, 0, 0, 0, 0};
    Kernels<double>::copy(3, s2, 1, d2, 2);
    assert(close(d2[0], 1) && close(d2[2], 2) && close(d2[4], 3));
    assert(close(d2[1], 0)); // gap untouched
}

// ---- Level 2 -------------------------------------------------------------

void test_gemv() {
    double A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    auto   a    = cview(A, 2, 3);
    // None: y = 2*A*x, beta=0 ignores garbage in y
    {
        double x[3] = {1, 1, 1}, y[2] = {-999, -999};
        Kernels<double>::gemv(Tr::None, 2.0, a, x, 1, 0.0, y, 1);
        assert(close(y[0], 12) && close(y[1], 30));
    }
    // beta accumulate
    {
        double x[3] = {1, 1, 1}, y[2] = {1, 2};
        Kernels<double>::gemv(Tr::None, 1.0, a, x, 1, 10.0, y, 1);
        assert(close(y[0], 16) && close(y[1], 35));
    }
    // transpose: A^T x
    {
        double x[2] = {1, 1}, y[3] = {0, 0, 0};
        Kernels<double>::gemv(Tr::Transpose, 1.0, a, x, 1, 0.0, y, 1);
        assert(close(y[0], 5) && close(y[1], 7) && close(y[2], 9));
    }
    // complex conjugate-transpose
    C M[4] = {C(1, 1), C(2, 0), C(3, 0), C(4, -1)};
    auto cm = cview(M, 2, 2);
    {
        C x[2] = {C(1, 0), C(1, 0)}, y[2] = {C(0, 0), C(0, 0)};
        Kernels<C>::gemv(Tr::ConjugateTranspose, C(1, 0), cm, x, 1, C(0, 0), y,
                         1);
        assert(close(y[0], C(4, -1)) && close(y[1], C(6, 1)));
    }
}

void test_ger() {
    double A[6] = {0, 0, 0, 0, 0, 0};
    auto   a    = mview(A, 2, 3);
    double x[2] = {1, 2}, y[3] = {10, 20, 30};
    Kernels<double>::ger(1.0, x, 1, y, 1, a); // x*y^T
    assert(close(A[0], 10) && close(A[4], 40) && close(A[5], 60));
    // accumulates with alpha
    double B[4] = {1, 1, 1, 1};
    auto   b    = mview(B, 2, 2);
    double bx[2] = {1, 1}, by[2] = {2, 3};
    Kernels<double>::ger(10.0, bx, 1, by, 1, b);
    assert(close(B[0], 21) && close(B[1], 31));
    // complex unconjugated
    C M[1] = {C(0, 0)};
    auto cm = mview(M, 1, 1);
    C mx[1] = {C(1, 1)}, my[1] = {C(1, 1)};
    Kernels<C>::ger(C(1, 0), mx, 1, my, 1, cm);
    assert(close(M[0], C(0, 2))); // (1+i)^2, not |1+i|^2
}

void test_trsv() {
    // Lower nonunit: solve then multiply back with gemv
    double L[9] = {2, 0, 0, 1, 3, 0, 1, 1, 4};
    auto   lv   = cview(L, 3, 3);
    {
        double b[3] = {2, 5, 10}, x[3] = {2, 5, 10};
        Kernels<double>::trsv(Ul::Lower, Tr::None, Dg::NonUnit, lv, x, 1);
        double chk[3] = {0, 0, 0};
        Kernels<double>::gemv(Tr::None, 1.0, lv, x, 1, 0.0, chk, 1);
        for (int i = 0; i < 3; ++i)
            assert(close(chk[i], b[i]));
    }
    // Transpose round-trip
    {
        double b[3] = {3, 4, 5}, x[3] = {3, 4, 5};
        Kernels<double>::trsv(Ul::Lower, Tr::Transpose, Dg::NonUnit, lv, x, 1);
        double chk[3] = {0, 0, 0};
        Kernels<double>::gemv(Tr::Transpose, 1.0, lv, x, 1, 0.0, chk, 1);
        for (int i = 0; i < 3; ++i)
            assert(close(chk[i], b[i]));
    }
    // Unit diagonal ignores stored diagonal entries
    double LU[4] = {9, 0, 1, 9};
    auto   luv   = cview(LU, 2, 2);
    {
        double x[2] = {5, 7}; // [[1,0],[1,1]] x = [5,7] -> [5,2]
        Kernels<double>::trsv(Ul::Lower, Tr::None, Dg::Unit, luv, x, 1);
        assert(close(x[0], 5) && close(x[1], 2));
    }
    // complex conjugate-transpose round-trip
    C A[4] = {C(2, 1), C(0, 0), C(1, -1), C(3, 0)};
    auto cv = cview(A, 2, 2);
    {
        C b[2] = {C(1, 1), C(2, -1)}, x[2] = {C(1, 1), C(2, -1)};
        Kernels<C>::trsv(Ul::Lower, Tr::ConjugateTranspose, Dg::NonUnit, cv, x,
                         1);
        C chk[2] = {C(0, 0), C(0, 0)};
        Kernels<C>::gemv(Tr::ConjugateTranspose, C(1, 0), cv, x, 1, C(0, 0),
                         chk, 1);
        for (int i = 0; i < 2; ++i)
            assert(close(chk[i], b[i]));
    }
}

void test_symv() {
    // Symmetric A = [[2,1,0],[1,3,1],[0,1,4]] stored upper, junk in lower
    double S[9] = {2, 1, 0, 999, 3, 1, 999, 999, 4};
    auto   sv   = cview(S, 3, 3);
    double x[3] = {1, 2, 3}, y[3] = {0, 0, 0};
    Kernels<double>::symv(Ul::Upper, 1.0, sv, x, 1, 0.0, y, 1);
    assert(close(y[0], 4) && close(y[1], 10) && close(y[2], 14)); // lower ignored
    // Lower-stored variant gives the same product
    double Slo[9] = {2, -1, -1, 1, 3, -1, 0, 1, 4}; // lower valid, upper junk
    auto   svlo   = cview(Slo, 3, 3);
    double ylo[3] = {0, 0, 0};
    Kernels<double>::symv(Ul::Lower, 1.0, svlo, x, 1, 0.0, ylo, 1);
    assert(close(ylo[0], 4) && close(ylo[1], 10) && close(ylo[2], 14));
}

// ---- Level 3 -------------------------------------------------------------

void test_gemm() {
    double A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    double B[6] = {7, 8, 9, 10, 11, 12}; // 3x2
    double Cc[4] = {0, 0, 0, 0};
    auto   a = cview(A, 2, 3), b = cview(B, 3, 2);
    auto   c = mview(Cc, 2, 2);
    Kernels<double>::gemm(Tr::None, Tr::None, 1.0, a, b, 0.0, c);
    assert(close(Cc[0], 58) && close(Cc[1], 64) && close(Cc[2], 139) &&
           close(Cc[3], 154));
    // beta accumulate
    Kernels<double>::gemm(Tr::None, Tr::None, 2.0, a, b, 1.0, c);
    assert(close(Cc[0], 58 + 116) && close(Cc[3], 154 + 308));
    // transB: A(2x3) * (B2^T where B2 is 2x3) -> 2x2
    double B2[6] = {1, 0, 1, 0, 1, 0}; // 2x3
    double C2[4] = {0, 0, 0, 0};
    auto   b2 = cview(B2, 2, 3);
    auto   c2 = mview(C2, 2, 2);
    Kernels<double>::gemm(Tr::None, Tr::Transpose, 1.0, a, b2, 0.0, c2);
    // A row0=(1,2,3): dot B2row0(1,0,1)=4, dot B2row1(0,1,0)=2
    assert(close(C2[0], 4) && close(C2[1], 2));
}

void test_syrk() {
    double A[6] = {1, 2, 3, 4, 5, 6}; // 2x3
    auto   a    = cview(A, 2, 3);
    double Sc[4] = {0, 0, 0, 0};
    auto   sc    = mview(Sc, 2, 2);
    Kernels<double>::syrk(Ul::Upper, Tr::None, 1.0, a, 0.0, sc); // A*A^T
    assert(close(Sc[0], 14) && close(Sc[1], 32) && close(Sc[3], 77));
    assert(Sc[2] == 0); // lower not written
    // complex Hermitian: diagonal must come out real
    C AH[2] = {C(3, 4), C(0, 1)}; // 1x2
    auto ahv = cview(AH, 1, 2);
    C HC[1] = {C(0, 0)};
    auto hc = mview(HC, 1, 1);
    Kernels<C>::syrk(Ul::Upper, Tr::None, C(1, 0), ahv, C(0, 0), hc);
    assert(close(HC[0], C(26, 0))); // 25 + 1
}

void test_trsm() {
    // Solve op(A) X = alpha*B for multiple RHS; verify via gemm.
    double L[9] = {2, 0, 0, 1, 3, 0, 1, 1, 4};
    auto   lv   = cview(L, 3, 3);
    double Bm[6] = {2, 4, 5, 7, 10, 8}; // 3x2
    double Borig[6];
    for (int i = 0; i < 6; ++i)
        Borig[i] = Bm[i];
    auto bm = mview(Bm, 3, 2);
    Kernels<double>::trsm(Ul::Lower, Tr::None, Dg::NonUnit, 3.0, lv, bm);
    // L * X should equal 3 * Borig
    double chk[6] = {0, 0, 0, 0, 0, 0};
    auto   xv     = cview(Bm, 3, 2);
    auto   chkv   = mview(chk, 3, 2);
    Kernels<double>::gemm(Tr::None, Tr::None, 1.0, lv, xv, 0.0, chkv);
    for (int i = 0; i < 6; ++i)
        assert(close(chk[i], 3.0 * Borig[i]));
}

} // namespace

int main() {
    test_scal();
    test_axpy();
    test_dot_dotc();
    test_nrm2();
    test_asum();
    test_iamax();
    test_swap_copy();
    test_gemv();
    test_ger();
    test_trsv();
    test_symv();
    test_gemm();
    test_syrk();
    test_trsm();
    std::puts("all Kernels behavioural tests passed");
    return 0;
}
