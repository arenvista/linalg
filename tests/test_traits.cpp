// Behavioural tests for linalg::NumericTraits and the IsComplex trait.
//
// Convention (matches tests/): exits 0 on success, aborts via assert() on the
// first failure, which ctest reports as a failure.

#include "linalg/core/Traits.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <type_traits>

using namespace linalg;
using Cd = std::complex<double>;
using Cf = std::complex<float>;

namespace {

bool close(double a, double b) { return std::abs(a - b) < 1e-12; }
bool close(Cd a, Cd b) { return std::abs(a - b) < 1e-12; }

// ---- compile-time traits -------------------------------------------------

static_assert(!IsComplex<float>::value);
static_assert(!IsComplex<double>::value);
static_assert(IsComplex<Cf>::value);
static_assert(IsComplex<Cd>::value);

// Real == Scalar for real T; Real is the component type for complex T.
static_assert(std::is_same_v<NumericTraits<double>::Real, double>);
static_assert(std::is_same_v<NumericTraits<float>::Real, float>);
static_assert(std::is_same_v<NumericTraits<Cd>::Real, double>);
static_assert(std::is_same_v<NumericTraits<Cf>::Real, float>);

// ---- real scalars --------------------------------------------------------

void test_real_constants() {
    assert(NumericTraits<double>::epsilon() ==
           std::numeric_limits<double>::epsilon());
    assert(NumericTraits<float>::epsilon() ==
           std::numeric_limits<float>::epsilon());
    assert(NumericTraits<double>::zero() == 0.0);
    assert(NumericTraits<double>::one() == 1.0);

    // safeMin: positive, and its reciprocal must not overflow.
    double sm = NumericTraits<double>::safeMin();
    assert(sm > 0.0);
    assert(std::isfinite(1.0 / sm));
    float smf = NumericTraits<float>::safeMin();
    assert(smf > 0.0f);
    assert(std::isfinite(1.0f / smf));
}

void test_real_abs() {
    assert(close(NumericTraits<double>::abs(-3.5), 3.5));
    assert(close(NumericTraits<double>::abs(2.0), 2.0));
    assert(close(NumericTraits<double>::abs(0.0), 0.0));
    assert(close(NumericTraits<double>::absSquared(-3.0), 9.0));
    assert(close(NumericTraits<double>::absSquared(0.5), 0.25));
    // absSquared must be x*x, not sqrt-based: exact for representable squares
    assert(NumericTraits<double>::absSquared(-4.0) == 16.0);
}

void test_real_conj_real_imag() {
    // conj is the identity for real T; imag is zero
    assert(NumericTraits<double>::conj(-7.25) == -7.25);
    assert(NumericTraits<double>::real(-7.25) == -7.25);
    assert(NumericTraits<double>::imag(-7.25) == 0.0);
    assert(NumericTraits<float>::conj(2.5f) == 2.5f);
    assert(NumericTraits<float>::imag(2.5f) == 0.0f);
}

void test_real_sqrt() {
    assert(close(NumericTraits<double>::sqrt(4.0), 2.0));
    assert(close(NumericTraits<double>::sqrt(2.0), std::sqrt(2.0)));
    assert(NumericTraits<double>::sqrt(0.0) == 0.0);
}

void test_real_isApproxZero() {
    assert(NumericTraits<double>::isApproxZero(0.0, 0.0));
    assert(NumericTraits<double>::isApproxZero(1e-15, 1e-12));
    assert(NumericTraits<double>::isApproxZero(-1e-15, 1e-12)); // by magnitude
    assert(!NumericTraits<double>::isApproxZero(1e-10, 1e-12));
    assert(NumericTraits<double>::isApproxZero(1e-12, 1e-12)); // |x| <= tol
}

// ---- complex scalars -----------------------------------------------------

void test_complex_constants() {
    // epsilon/safeMin come from the component type
    assert(NumericTraits<Cd>::epsilon() ==
           std::numeric_limits<double>::epsilon());
    assert(NumericTraits<Cf>::epsilon() ==
           std::numeric_limits<float>::epsilon());
    double sm = NumericTraits<Cd>::safeMin();
    assert(sm > 0.0);
    assert(std::isfinite(1.0 / sm));
    assert(NumericTraits<Cd>::zero() == Cd(0, 0));
    assert(NumericTraits<Cd>::one() == Cd(1, 0));
}

void test_complex_abs() {
    assert(close(NumericTraits<Cd>::abs(Cd(3, 4)), 5.0));
    assert(close(NumericTraits<Cd>::abs(Cd(0, -2)), 2.0));
    // absSquared: |z|^2 without sqrt — exact for integer components
    assert(NumericTraits<Cd>::absSquared(Cd(3, 4)) == 25.0);
    assert(NumericTraits<Cd>::absSquared(Cd(-1, 1)) == 2.0);
    assert(NumericTraits<Cd>::absSquared(Cd(0, 0)) == 0.0);
}

void test_complex_conj_real_imag() {
    assert(NumericTraits<Cd>::conj(Cd(1, 2)) == Cd(1, -2));
    assert(NumericTraits<Cd>::conj(Cd(-3, -4)) == Cd(-3, 4));
    assert(NumericTraits<Cd>::real(Cd(1, 2)) == 1.0);
    assert(NumericTraits<Cd>::imag(Cd(1, 2)) == 2.0);
    // conj(conj(z)) == z
    Cd z(0.5, -1.5);
    assert(NumericTraits<Cd>::conj(NumericTraits<Cd>::conj(z)) == z);
}

void test_complex_sqrt() {
    // principal branch: sqrt(-1) = +i, not -i
    assert(close(NumericTraits<Cd>::sqrt(Cd(-1, 0)), Cd(0, 1)));
    // sqrt(2i) = 1 + i
    assert(close(NumericTraits<Cd>::sqrt(Cd(0, 2)), Cd(1, 1)));
    // square of the result recovers the input
    Cd z(3, -4);
    Cd r = NumericTraits<Cd>::sqrt(z);
    assert(close(r * r, z));
    assert(NumericTraits<Cd>::real(r) >= 0.0); // principal: Re >= 0
}

void test_complex_isApproxZero() {
    // judged by modulus: |3e-3 + 4e-3 i| = 5e-3
    assert(NumericTraits<Cd>::isApproxZero(Cd(3e-3, 4e-3), 6e-3));
    assert(!NumericTraits<Cd>::isApproxZero(Cd(3e-3, 4e-3), 4e-3));
    assert(NumericTraits<Cd>::isApproxZero(Cd(0, 0), 0.0));
    // purely imaginary is not "zero" just because Re == 0
    assert(!NumericTraits<Cd>::isApproxZero(Cd(0, 1), 1e-3));
}

} // namespace

int main() {
    test_real_constants();
    test_real_abs();
    test_real_conj_real_imag();
    test_real_sqrt();
    test_real_isApproxZero();
    test_complex_constants();
    test_complex_abs();
    test_complex_conj_real_imag();
    test_complex_sqrt();
    test_complex_isApproxZero();
    std::puts("all NumericTraits behavioural tests passed");
    return 0;
}
