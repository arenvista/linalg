// Behavioural tests for linalg::Hessenberg / Tridiagonal / Bidiagonal
// (PLAN.md step 13).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// PLAN.md oracle: Q * H * Q^H == A and structure (zeros where zeros belong).

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/decomp/Hessenberg.hpp"

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

void test_hessenberg_identity() {
    M a = M::Random(5, 5, 101);
    Hessenberg<double> hess(a);
    assert(hess.isComputed());
    M h = hess.matrixH();
    M q = hess.matrixQ();
    assert(q.isOrthogonal(1e-10));
    assert((q * h * q.transpose()).isApprox(a, 1e-8)); // A == Q H Q^H
    // structure: zero below the first subdiagonal
    for (std::size_t i = 2; i < 5; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            assert(std::abs(h(i, j)) < 1e-12);
    // a random dense matrix is unreduced almost surely
    assert(hess.isUnreduced(1e-12));
}

void test_hessenberg_similarity() {
    // similarity transform preserves trace (eigenvalue sum)
    M a = M::Random(6, 6, 102);
    Hessenberg<double> hess(a);
    assert(std::abs(hess.matrixH().trace() - a.trace()) < 1e-8);
}

void test_hessenberg_complex() {
    Matrix<C> a = Matrix<C>::Random(4, 4, 103);
    Hessenberg<C> hess(a);
    Matrix<C> h = hess.matrixH();
    Matrix<C> q = hess.matrixQ();
    assert(q.isOrthogonal(1e-10)); // unitary
    assert((q * h * q.conjugateTranspose()).isApprox(a, 1e-8));
    for (std::size_t i = 2; i < 4; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            assert(std::abs(h(i, j)) < 1e-12);
}

// ---- Tridiagonal -----------------------------------------------------------

void test_tridiagonal_identity() {
    M a = M::RandomSymmetric(5, 104);
    Tridiagonal<double> tri(a);
    assert(tri.isComputed());
    M t = tri.matrixT();
    M q = tri.matrixQ();
    assert(q.isOrthogonal(1e-10));
    assert((q * t * q.transpose()).isApprox(a, 1e-8)); // A == Q T Q^H
    // structure: tridiagonal and symmetric
    assert(t.isSymmetric(1e-10));
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            if (i > j + 1 || j > i + 1)
                assert(std::abs(t(i, j)) < 1e-12);
    // diagonal()/subdiagonal() agree with matrixT()
    V d = tri.diagonal(), s = tri.subdiagonal();
    assert(d.size() == 5 && s.size() == 4);
    for (std::size_t i = 0; i < 5; ++i)
        assert(std::abs(d(i) - t(i, i)) < 1e-12);
    for (std::size_t i = 0; i < 4; ++i)
        assert(std::abs(s(i) - t(i + 1, i)) < 1e-12);
}

void test_tridiagonal_complex_hermitian() {
    Matrix<C> a = Matrix<C>::RandomSymmetric(4, 105); // Hermitian
    Tridiagonal<C> tri(a);
    Matrix<C> q = tri.matrixQ();
    assert((q * tri.matrixT() * q.conjugateTranspose()).isApprox(a, 1e-8));
    // the tridiagonal factor is real even for complex Hermitian input
    Matrix<C> t = tri.matrixT();
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            assert(std::abs(NumericTraits<C>::imag(t(i, j))) < 1e-10);
}

void test_sturm_count() {
    // a diagonal matrix is its own tridiagonal form: eigenvalues 1, 2, 3
    M a = M::Diagonal(V{1, 2, 3});
    Tridiagonal<double> tri(a);
    assert(tri.eigenvalueCountBelow(0.0) == 0);
    assert(tri.eigenvalueCountBelow(1.5) == 1);
    assert(tri.eigenvalueCountBelow(2.5) == 2);
    assert(tri.eigenvalueCountBelow(10.0) == 3);
}

// ---- Bidiagonal ------------------------------------------------------------

void test_bidiagonal_identity() {
    M a = M::Random(5, 4, 106);
    Bidiagonal<double> bid(a);
    assert(bid.isComputed());
    M b = bid.matrixB();
    M u = bid.matrixU();
    M v = bid.matrixV();
    // A == U B V^H
    assert((u * b * v.transpose()).isApprox(a, 1e-8));
    // orthonormal columns on both sides
    assert((u.transpose() * u).isApprox(M::Identity(u.cols()), 1e-10));
    assert((v.transpose() * v).isApprox(M::Identity(v.cols()), 1e-10));
    // structure: upper bidiagonal — only (i,i) and (i,i+1) may be nonzero
    for (std::size_t i = 0; i < b.rows(); ++i)
        for (std::size_t j = 0; j < b.cols(); ++j)
            if (j != i && j != i + 1)
                assert(std::abs(b(i, j)) < 1e-12);
    // diagonal()/superdiagonal() agree with matrixB()
    V d = bid.diagonal(), s = bid.superdiagonal();
    for (std::size_t i = 0; i < d.size(); ++i)
        assert(std::abs(d(i) - b(i, i)) < 1e-12);
    for (std::size_t i = 0; i < s.size(); ++i)
        assert(std::abs(s(i) - b(i, i + 1)) < 1e-12);
}

void test_bidiagonal_tall() {
    // m >> n: the QR-first path may kick in; the identity must hold either way
    M a = M::Random(12, 3, 107);
    Bidiagonal<double> bid(a);
    assert((bid.matrixU() * bid.matrixB() * bid.matrixV().transpose())
               .isApprox(a, 1e-8));
    (void)bid.usedRPreprocessing(); // whichever path, it must answer
}

void test_bidiagonal_complex() {
    Matrix<C> a = Matrix<C>::Random(4, 3, 108);
    Bidiagonal<C> bid(a);
    assert((bid.matrixU() * bid.matrixB() * bid.matrixV().conjugateTranspose())
               .isApprox(a, 1e-8));
    // B is real even for complex input
    Matrix<C> b = bid.matrixB();
    for (std::size_t i = 0; i < b.rows(); ++i)
        for (std::size_t j = 0; j < b.cols(); ++j)
            assert(std::abs(NumericTraits<C>::imag(b(i, j))) < 1e-10);
}

} // namespace

int main() {
    run("hessenberg_identity", test_hessenberg_identity);
    run("hessenberg_similarity", test_hessenberg_similarity);
    run("hessenberg_complex", test_hessenberg_complex);
    run("tridiagonal_identity", test_tridiagonal_identity);
    run("tridiagonal_complex_hermitian", test_tridiagonal_complex_hermitian);
    run("sturm_count", test_sturm_count);
    run("bidiagonal_identity", test_bidiagonal_identity);
    run("bidiagonal_tall", test_bidiagonal_tall);
    run("bidiagonal_complex", test_bidiagonal_complex);
    return finish("Hessenberg");
}
