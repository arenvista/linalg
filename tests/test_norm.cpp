// Behavioural tests for linalg::Norm (PLAN.md step 5).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp). matrixTwo / matrixNuclear need the SVD and stay
// skipped until Phase 4.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Norm.hpp"

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
using N = Norm<double>;
using NC = Norm<C>;

namespace {

constexpr double kTol = 1e-12;

bool close(double a, double b) { return std::abs(a - b) < kTol; }

// ---- vector norms ----------------------------------------------------------

void test_vector_one() {
    assert(close(N::vectorOne(V{3, -4, 1}), 8.0));
    assert(close(N::vectorOne(V{}), 0.0));
    // complex: sum of moduli
    assert(close(NC::vectorOne(Vector<C>{C(3, 4), C(0, -12)}), 17.0));
}

void test_vector_two() {
    assert(close(N::vectorTwo(V{3, -4}), 5.0));
    assert(close(N::vectorTwo(V{}), 0.0));
    assert(close(NC::vectorTwo(Vector<C>{C(3, 4), C(0, 12)}), 13.0));
}

void test_vector_infinity() {
    assert(close(N::vectorInfinity(V{1, -9, 3}), 9.0));
    assert(close(NC::vectorInfinity(Vector<C>{C(3, 4), C(1, 0)}), 5.0));
}

void test_vector_p() {
    V x{3, -4};
    assert(close(N::vectorP(x, 1.0), N::vectorOne(x)));
    assert(close(N::vectorP(x, 2.0), N::vectorTwo(x)));
    // p = 3: (27 + 64)^(1/3)
    assert(close(N::vectorP(x, 3.0), std::cbrt(91.0)));
}

void test_vector_two_scaled() {
    // agrees with the plain norm on benign input
    assert(close(N::vectorTwoScaled(V{3, -4}), 5.0));
    // the point of the function: no overflow near the exponent ceiling
    double big = 1e200;
    double r = N::vectorTwoScaled(V{big, big});
    assert(std::isfinite(r));
    assert(std::abs(r - std::sqrt(2.0) * big) / (std::sqrt(2.0) * big) < 1e-12);
    // and no underflow to zero for tiny entries
    double tiny = 1e-200;
    double rt = N::vectorTwoScaled(V{tiny, tiny});
    assert(rt > 0.0);
    assert(std::abs(rt - std::sqrt(2.0) * tiny) / (std::sqrt(2.0) * tiny) <
           1e-12);
}

// ---- matrix norms ----------------------------------------------------------

void test_matrix_one_infinity() {
    M a{{1, -2}, {3, 4}};
    assert(close(N::matrixOne(a), 6.0));      // max abs column sum
    assert(close(N::matrixInfinity(a), 7.0)); // max abs row sum
    // consistency: ||A^T||_1 == ||A||_inf
    assert(close(N::matrixOne(a.transpose()), N::matrixInfinity(a)));
}

void test_matrix_frobenius_max() {
    M a{{1, -2}, {3, 4}};
    assert(close(N::matrixFrobenius(a), std::sqrt(30.0)));
    assert(close(N::matrixMax(a), 4.0));
    Matrix<C> c{{C(3, 4)}};
    assert(close(NC::matrixFrobenius(c), 5.0));
    assert(close(NC::matrixMax(c), 5.0));
}

void test_matrix_two() { // Phase 4 (SVD)
    // largest singular value of a diagonal matrix is max |d_i|
    assert(close(N::matrixTwo(M::Diagonal(V{3, -5})), 5.0));
    assert(close(N::matrixTwo(M::Identity(3)), 1.0));
    // submultiplicative sanity: ||A||_2 <= ||A||_F
    M a{{1, 2}, {3, 4}};
    assert(N::matrixTwo(a) <= N::matrixFrobenius(a) + kTol);
}

void test_matrix_nuclear() { // Phase 4 (SVD)
    assert(close(N::matrixNuclear(M::Diagonal(V{3, -5})), 8.0));
    assert(close(N::matrixNuclear(M::Identity(3)), 3.0));
}

// ---- diagnostics -----------------------------------------------------------

void test_distance_relative_error() {
    assert(close(N::distance(V{1, 2}, V{4, 6}), 5.0));
    assert(close(N::distance(V{1, 2}, V{1, 2}), 0.0));
    // ||approx - exact|| / ||exact||
    assert(close(N::relativeError(V{1.1, 0}, V{1, 0}),
                 0.1 / 1.0));
    assert(close(N::relativeError(V{3, 4}, V{3, 4}), 0.0));
}

void test_residual_norm() {
    M a{{2, 0}, {0, 3}};
    V b{4, 9};
    assert(close(N::residualNorm(a, V{2, 3}, b), 0.0)); // exact solution
    // x = 0: residual is ||b||
    assert(close(N::residualNorm(a, V{0, 0}, b), N::vectorTwo(b)));
}

void test_backward_error() {
    M a{{2, 0}, {0, 3}};
    V b{4, 9};
    // exact solution -> backward error 0
    assert(close(N::backwardError(a, V{2, 3}, b), 0.0));
    // a wrong answer has a strictly positive backward error, bounded by ~1
    double be = N::backwardError(a, V{1, 1}, b);
    assert(be > 0.0 && be < 2.0);
}

void test_orthogonality_defect() {
    assert(close(N::orthogonalityDefect(M::Identity(3)), 0.0));
    // Q = 2I: Q^H Q - I = 3I, Frobenius norm 3*sqrt(2) for 2x2
    assert(close(N::orthogonalityDefect(M::Identity(2) * 2.0),
                 3.0 * std::sqrt(2.0)));
    // permutation matrices are orthogonal
    assert(close(N::orthogonalityDefect(M{{0, 1}, {1, 0}}), 0.0));
}

} // namespace

int main() {
    run("vector_one", test_vector_one);
    run("vector_two", test_vector_two);
    run("vector_infinity", test_vector_infinity);
    run("vector_p", test_vector_p);
    run("vector_two_scaled", test_vector_two_scaled);
    run("matrix_one_infinity", test_matrix_one_infinity);
    run("matrix_frobenius_max", test_matrix_frobenius_max);
    run("matrix_two", test_matrix_two);
    run("matrix_nuclear", test_matrix_nuclear);
    run("distance_relative_error", test_distance_relative_error);
    run("residual_norm", test_residual_norm);
    run("backward_error", test_backward_error);
    run("orthogonality_defect", test_orthogonality_defect);
    return finish("Norm");
}
