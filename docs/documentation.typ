// Reference documentation for template.typ.
// This file is both a usage guide and a live demonstration: it is rendered
// with the very template it documents, so every callout and shortcut below
// shows its real output.
//
// Compile with:
//   typst compile documentation.typ

#import "template.typ": *

#show: notes.with(
	title: "template.typ — Reference & Usage Guide",
	course: "linalg/docs",
	author: "Aren Vista",
	date: "August 5, 2026",
)

#outline(title: [Contents], indent: auto)

= Traits

`NumericTraits<T>` is the scalar-type abstraction the rest of the library
queries instead of assuming `double`. Each method below is `static` and exists
for both the real primary template and the `std::complex<T>` specialization,
which share the same contract; where the two differ, the difference is noted.
Throughout, `Scalar` is `T`, and `Real` is the magnitude type — equal to
`Scalar` for real `T`, and the component type for complex `T`.

#method(
	"static Real epsilon()",
	example: ```cpp
	assert(NumericTraits<double>::epsilon() ==
	       std::numeric_limits<double>::epsilon());
	// component type for complex: Real is double
	assert(NumericTraits<std::complex<double>>::epsilon() ==
	       std::numeric_limits<double>::epsilon());
	```,
	description: [
		Machine epsilon: the gap between $1$ and the next representable `Real`,
		i.e. $epsilon := min { delta > 0 : "fl"(1 + delta) != 1 }$. For a complex
		`T` it is the epsilon of the component type — it characterizes the
		floating-point format, not the complex number.
	],
	params: (),
	returns: [
		`std::numeric_limits<Real>::epsilon()` as `Real`.
	],
)

#method(
	"static Real safeMin()",
	example: ```cpp
	double sm = NumericTraits<double>::safeMin();
	assert(sm > 0.0);
	assert(std::isfinite(1.0 / sm)); // reciprocal never overflows
	```,
	description: [
		Smallest positive value $s$ such that $1 slash s$ is finite — the scaling
		threshold used to avoid overflow (LAPACK `dlamch('S')`). On IEEE formats
		this reduces to `std::numeric_limits<Real>::min()`; the guard covers
		formats where $1 slash "min"()$ would overflow.
	],
	params: (),
	returns: [
		The safe minimum as `Real`; always positive with a finite reciprocal.
	],
)

#method(
	"static Real abs(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::abs(-3.5) == 3.5);
	// complex modulus: |3 + 4i| = 5
	assert(NumericTraits<std::complex<double>>::abs({3, 4}) == 5.0);
	```,
	description: [
		Magnitude of a scalar, $abs(x)$. For real `T` this is `std::abs` from
		`<cmath>` (chosen to avoid signed-zero issues); for complex `T` it is the
		modulus, computed via `std::abs` (hypot-based, so $"re"^2 + "im"^2$ cannot
		overflow prematurely).
	],
	params: (("x", [the scalar whose magnitude is taken]),),
	returns: [
		$abs(x)$ as `Real`.
	],
)

#method(
	"static Real absSquared(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::absSquared(-4.0) == 16.0);
	// |3 + 4i|^2 = 25, computed without a square root
	assert(NumericTraits<std::complex<double>>::absSquared({3, 4}) == 25.0);
	```,
	description: [
		Squared magnitude $abs(x)^2$, computed without a square root — so it is
		exact for representable squares. Real `T` returns $x dot x$; complex `T`
		returns `std::norm(x)` $= "re"^2 + "im"^2$.
	],
	params: (("x", [the scalar whose squared magnitude is taken]),),
	returns: [
		$abs(x)^2$ as `Real`.
	],
)

#method(
	"static Scalar conj(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::conj(-7.25) == -7.25);        // identity
	assert(NumericTraits<std::complex<double>>::conj({1, 2}) ==
	       std::complex<double>(1, -2));                        // 1 - 2i
	```,
	description: [
		Complex conjugate $overline(x)$. For real `T` this is the identity; for
		complex `T` it flips the sign of the imaginary part. Involutive:
		$overline(overline(x)) = x$.
	],
	params: (("x", [the scalar to conjugate]),),
	returns: [
		$overline(x)$ as `Scalar`.
	],
)

#method(
	"static Real real(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::real(-7.25) == -7.25);
	assert(NumericTraits<std::complex<double>>::real({1, 2}) == 1.0);
	```,
	description: [
		Real part $Re(x)$. For real `T` this returns `x` unchanged; for complex
		`T` it returns `x.real()`.
	],
	params: (("x", [the scalar whose real part is taken]),),
	returns: [
		$Re(x)$ as `Real`.
	],
)

#method(
	"static Real imag(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::imag(-7.25) == 0.0);          // zero for real T
	assert(NumericTraits<std::complex<double>>::imag({1, 2}) == 2.0);
	```,
	description: [
		Imaginary part $Im(x)$. Always zero for real `T`; for complex `T` it
		returns `x.imag()`.
	],
	params: (("x", [the scalar whose imaginary part is taken]),),
	returns: [
		$Im(x)$ as `Real`.
	],
)

#method(
	"static Scalar sqrt(const Scalar& x)",
	example: ```cpp
	assert(NumericTraits<double>::sqrt(4.0) == 2.0);
	// principal branch: sqrt(-1) = +i, not -i
	auto r = NumericTraits<std::complex<double>>::sqrt({-1, 0});
	assert(std::abs(r - std::complex<double>(0, 1)) < 1e-12);
	```,
	description: [
		Principal square root $sqrt(x)$. Real `T` uses `std::sqrt`; complex `T`
		uses the principal branch (`std::sqrt`), so the result has non-negative
		real part and squaring it recovers the input.
	],
	params: (("x", [the scalar whose principal square root is taken]),),
	returns: [
		$sqrt(x)$ as `Scalar`.
	],
)

#method(
	"static Scalar zero()",
	example: ```cpp
	assert(NumericTraits<double>::zero() == 0.0);
	assert(NumericTraits<std::complex<double>>::zero() ==
	       std::complex<double>(0, 0));
	```,
	description: [
		The additive identity for the scalar type — `Scalar(0)`, i.e. $0 + 0i$
		for complex `T`.
	],
	params: (),
	returns: [
		Zero as `Scalar`.
	],
)

#method(
	"static Scalar one()",
	example: ```cpp
	assert(NumericTraits<double>::one() == 1.0);
	assert(NumericTraits<std::complex<double>>::one() ==
	       std::complex<double>(1, 0));
	```,
	description: [
		The multiplicative identity for the scalar type — `Scalar(1)`, i.e.
		$1 + 0i$ for complex `T`.
	],
	params: (),
	returns: [
		One as `Scalar`.
	],
)

#method(
	"static bool isApproxZero(const Scalar& x, Real tol)",
	example: ```cpp
	assert(NumericTraits<double>::isApproxZero(1e-15, 1e-12));
	assert(!NumericTraits<double>::isApproxZero(1e-10, 1e-12));
	// complex: judged by modulus, |3e-3 + 4e-3 i| = 5e-3
	assert(NumericTraits<std::complex<double>>::isApproxZero({3e-3, 4e-3}, 6e-3));
	```,
	description: [
		Tests whether a scalar is negligible against an absolute tolerance,
		returning $abs(x) <= "tol"$. The test is by magnitude, so a purely
		imaginary value is not "zero" merely because its real part is.
	],
	params: (
		("x", [the scalar to test]),
		("tol", [absolute magnitude tolerance (a `Real`)]),
	),
	returns: [
		`true` if $abs(x) <= "tol"$, otherwise `false`.
	],
)

#include "exceptions.typ"
#include "vector.typ"
#include "matrixview.typ"
#include "matrix.typ"
#include "givens.typ"
