// Behavioural tests for linalg::PowerIteration / Lanczos /
// ImplicitlyRestartedArnoldi (PLAN.md step 22).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp).
//
// Diagonal test matrices keep the spectra exact, so Ritz values can be
// compared against known eigenvalues directly.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/iterative/EigenSolver.hpp"

#include "harness.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;

namespace {

PowerIteration<double>::Options powerOpts(bool rayleigh = false) {
    PowerIteration<double>::Options o;
    o.maxIterations = 2000;
    o.tolerance = 1e-12;
    o.useRayleighQuotient = rayleigh;
    return o;
}

void test_power_dominant() {
    M a = M::Diagonal(V{1, 2, 10});
    PowerIteration<double> power(powerOpts());
    double lambda = power.dominantEigenvalue(a);
    assert(std::abs(lambda - 10.0) < 1e-8);
    assert(power.converged());
    assert(power.iterationsUsed() > 0);
    // dominant eigenvector is +/- e_3, unit norm
    V v = power.dominantEigenvector(a);
    assert(std::abs(v.norm() - 1.0) < 1e-8);
    assert(std::abs(std::abs(v(2)) - 1.0) < 1e-6);
    assert(std::abs(v(0)) < 1e-6 && std::abs(v(1)) < 1e-6);
}

void test_power_rayleigh_quotient() {
    M a{{2, 1}, {1, 2}};
    PowerIteration<double> power(powerOpts(true));
    // exact quotient on an exact eigenvector: (1,1)/sqrt(2) -> lambda = 3
    V q{1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)};
    assert(std::abs(power.rayleighQuotient(a, q) - 3.0) < 1e-10);
    // and the iteration itself agrees
    assert(std::abs(power.dominantEigenvalue(a) - 3.0) < 1e-8);
}

void test_power_shifted_inverse() {
    M a = M::Diagonal(V{1, 2, 10});
    PowerIteration<double> power(powerOpts());
    // nearest eigenvalue to 1.9 is 2
    double lambda = power.shiftedInverseEigenvalue(a, 1.9);
    assert(std::abs(lambda - 2.0) < 1e-8);
    // nearest to 0.2 is 1
    assert(std::abs(power.shiftedInverseEigenvalue(a, 0.2) - 1.0) < 1e-8);
}

// ---- Lanczos ---------------------------------------------------------------

void test_lanczos_full_dimension() {
    // krylovDimension == n with full reorthogonalization: Ritz values are
    // the exact spectrum
    std::size_t n = 6;
    V d(n);
    for (std::size_t i = 0; i < n; ++i)
        d(i) = static_cast<double>(i + 1); // eigenvalues 1..6
    M a = M::Diagonal(d);
    Lanczos<double> lanczos(n, Lanczos<double>::Reorthogonalization::Kind::Full);
    lanczos.compute(a, V::Ones(n)); // start touches every eigendirection
    assert(lanczos.isComputed());
    V ritz = lanczos.ritzValues();
    std::vector<double> sorted;
    for (std::size_t i = 0; i < ritz.size(); ++i)
        sorted.push_back(ritz(i));
    std::sort(sorted.begin(), sorted.end());
    assert(sorted.size() == n);
    for (std::size_t i = 0; i < n; ++i)
        assert(std::abs(sorted[i] - static_cast<double>(i + 1)) < 1e-8);
    // the basis stays orthonormal under full reorthogonalization
    assert(lanczos.orthogonalityLoss() < 1e-8);
    M basis = lanczos.basis();
    assert((basis.transpose() * basis)
               .isApprox(M::Identity(basis.cols()), 1e-8));
}

void test_lanczos_projection_structure() {
    M a = M::RandomSymmetric(8, 201);
    Lanczos<double> lanczos(5, Lanczos<double>::Reorthogonalization::Kind::Full);
    lanczos.compute(a, V::Ones(8));
    // projected tridiagonal: diagonal length m, subdiagonal m-1
    V diag = lanczos.diagonal();
    V sub = lanczos.subdiagonal();
    assert(diag.size() == 5);
    assert(sub.size() == 4);
    // Lanczos relation: diag/sub equal V^H A V for the built basis
    M v = lanczos.basis();
    M proj = v.transpose() * a * v;
    for (std::size_t i = 0; i < 5; ++i)
        assert(std::abs(proj(i, i) - diag(i)) < 1e-8);
    for (std::size_t i = 0; i < 4; ++i)
        assert(std::abs(std::abs(proj(i + 1, i)) - std::abs(sub(i))) < 1e-8);
}

void test_lanczos_ritz_pairs() {
    M a = M::Diagonal(V{1, 3, 5, 7});
    Lanczos<double> lanczos(4, Lanczos<double>::Reorthogonalization::Kind::Full);
    lanczos.compute(a, V::Ones(4));
    V ritz = lanczos.ritzValues();
    M vectors = lanczos.ritzVectors();
    V bounds = lanczos.ritzResidualBounds();
    // full-dimension run: every Ritz pair is an eigenpair with tiny bound
    for (std::size_t k = 0; k < ritz.size(); ++k) {
        V x = vectors.col(k);
        assert((a * x - x * ritz(k)).norm() < 1e-7);
        assert(bounds(k) < 1e-7);
    }
}

// ---- ImplicitlyRestartedArnoldi --------------------------------------------

void test_arnoldi_largest() {
    // nonsymmetric upper-triangular: eigenvalues on the diagonal, 1..8
    std::size_t n = 8;
    M a = M::Random(n, n, 202).triangularPart(Triangle::Kind::Upper);
    for (std::size_t i = 0; i < n; ++i)
        a(i, i) = static_cast<double>(i + 1);
    ImplicitlyRestartedArnoldi<double>::Options o;
    o.wantedCount = 2;
    o.krylovDimension = 5;
    o.maxRestarts = 200;
    o.tolerance = 1e-10;
    o.target =
        ImplicitlyRestartedArnoldi<double>::Target::Kind::LargestMagnitude;
    ImplicitlyRestartedArnoldi<double> arnoldi(o);
    arnoldi.compute(a, V::Ones(n));
    assert(arnoldi.isComputed());
    assert(arnoldi.converged());
    V ritz = arnoldi.ritzValues();
    assert(ritz.size() >= 2);
    std::vector<double> got;
    for (std::size_t i = 0; i < ritz.size(); ++i)
        got.push_back(std::abs(ritz(i)));
    std::sort(got.rbegin(), got.rend());
    assert(std::abs(got[0] - 8.0) < 1e-6); // two largest: 8 and 7
    assert(std::abs(got[1] - 7.0) < 1e-6);
    // Ritz vectors actually solve the eigenproblem
    M vectors = arnoldi.ritzVectors();
    for (std::size_t k = 0; k < 2; ++k) {
        V x = vectors.col(k);
        assert((a * x - x * ritz(k)).norm() < 1e-6);
    }
}

} // namespace

int main() {
    run("power_dominant", test_power_dominant);
    run("power_rayleigh_quotient", test_power_rayleigh_quotient);
    run("power_shifted_inverse", test_power_shifted_inverse);
    run("lanczos_full_dimension", test_lanczos_full_dimension);
    run("lanczos_projection_structure", test_lanczos_projection_structure);
    run("lanczos_ritz_pairs", test_lanczos_ritz_pairs);
    run("arnoldi_largest", test_arnoldi_largest);
    return finish("EigenSolver");
}
