// Documentation for include/linalg/core/Exceptions.hpp.
// Included from documentation.typ; imports the template for the `method` env.
#import "template.typ": *

= Exceptions

The library reports every contract violation by throwing a type derived from
`LinalgError`, which itself derives from `std::runtime_error`. A single
`catch (const LinalgError&)` therefore catches everything the library throws,
while the specific subclasses carry structured detail (offending indices,
pivot positions, iteration counts) for callers that want it. In an ordered
`try`/`catch`, the specific handler wins over the `LinalgError` base.

== LinalgError

#method(
	"LinalgError(const std::string& message)",
	example: ```cpp
	LinalgError e("something broke");
	assert(std::string(e.what()) == "something broke");
	// it is a std::runtime_error
	try { throw LinalgError("msg"); }
	catch (const std::runtime_error& re) { assert(std::string(re.what()) == "msg"); }
	```,
	description: [
		Root of the exception hierarchy. Thrown directly by unimplemented stubs
		and for one-off contract violations that have no dedicated subclass;
		every other library exception derives from it.
	],
	params: (("message", [human-readable description of the failure]),),
	returns: [
		Constructs the error; `what()` returns `message`.
	],
)

== DimensionMismatch

#method(
	"DimensionMismatch(size_t lhsRows, size_t lhsCols, size_t rhsRows, size_t rhsCols)",
	example: ```cpp
	DimensionMismatch e(3, 4, 5, 6);
	assert(e.lhsRows() == 3 && e.lhsCols() == 4);
	assert(e.rhsRows() == 5 && e.rhsCols() == 6);
	```,
	description: [
		Thrown when operand shapes are incompatible for the requested operation.
		Vectors report their size as an $n times 1$ shape. Records both operands'
		shapes so the handler can report exactly what failed to line up.
	],
	params: (
		("lhsRows", [rows of the left operand]),
		("lhsCols", [columns of the left operand]),
		("rhsRows", [rows of the right operand]),
		("rhsCols", [columns of the right operand]),
	),
	returns: [
		Constructs the error; `what()` names all four dimensions.
	],
)

#method(
	"size_t lhsRows() / lhsCols() / rhsRows() / rhsCols() const",
	example: ```cpp
	try { throw DimensionMismatch(2, 2, 7, 1); }
	catch (const LinalgError& le) {
	    auto* dm = dynamic_cast<const DimensionMismatch*>(&le);
	    assert(dm && dm->rhsRows() == 7); // detail survives a base-ref catch
	}
	```,
	description: [
		The four recorded operand dimensions. They remain accessible after the
		exception is caught by base reference and `dynamic_cast` back down.
	],
	params: (),
	returns: [
		The corresponding recorded row/column count as `size_t`.
	],
)

== IndexOutOfRange

#method(
	"IndexOutOfRange(size_t index, size_t bound)",
	example: ```cpp
	IndexOutOfRange e(9, 4);
	assert(e.index() == 9 && e.bound() == 4);
	```,
	description: [
		Thrown by the checked accessors (`at`, checked element access) when an
		index is not less than its bound. The unchecked `operator()` never throws
		this.
	],
	params: (
		("index", [the index that was requested]),
		("bound", [the exclusive upper bound; valid indices are `< bound`]),
	),
	returns: [
		Constructs the error; `what()` names the index and bound.
	],
)

#method(
	"size_t index() / bound() const",
	example: ```cpp
	IndexOutOfRange e(9, 4);
	assert(e.index() == 9);  // offending index
	assert(e.bound() == 4);  // exclusive upper bound
	```,
	description: [
		The offending index and the exclusive bound it violated.
	],
	params: (),
	returns: [
		The recorded index / bound as `size_t`.
	],
)

== SingularMatrix

#method(
	"SingularMatrix(size_t pivotIndex)",
	example: ```cpp
	SingularMatrix e(2);
	assert(e.pivotIndex() == 2);
	// thrown by factorizations/solves that hit a zero pivot, e.g.
	try { (void)Matrix<double>{{1, 2}, {2, 4}}.inverse(); }
	catch (const SingularMatrix&) { /* rank-deficient */ }
	```,
	description: [
		A factorization or solve met an exactly (or numerically) zero pivot.
		`pivotIndex` is the elimination step that failed.
	],
	params: (("pivotIndex", [the elimination step whose pivot was (near) zero]),),
	returns: [
		Constructs the error; `what()` contains "singular" and the pivot index.
	],
)

#method(
	"size_t pivotIndex() const",
	example: ```cpp
	SingularMatrix e(2);
	assert(e.pivotIndex() == 2);
	```,
	description: [
		The elimination step at which the (near-)zero pivot appeared.
	],
	params: (),
	returns: [
		The recorded pivot index as `size_t`.
	],
)

== NotPositiveDefinite

#method(
	"NotPositiveDefinite(size_t pivotIndex)",
	example: ```cpp
	NotPositiveDefinite e(1);
	assert(e.pivotIndex() == 1);
	```,
	description: [
		A Cholesky-family factorization met a non-positive pivot; `pivotIndex`
		is the first such pivot. Only raised when the caller opted into
		`throwOnIndefinite`.
	],
	params: (("pivotIndex", [index of the first non-positive pivot]),),
	returns: [
		Constructs the error; `what()` contains "positive definite".
	],
)

#method(
	"size_t pivotIndex() const",
	example: ```cpp
	NotPositiveDefinite e(1);
	assert(e.pivotIndex() == 1);
	```,
	description: [
		The index of the first non-positive pivot encountered.
	],
	params: (),
	returns: [
		The recorded pivot index as `size_t`.
	],
)

== ConvergenceFailure

#method(
	"ConvergenceFailure(const std::string& algorithm, size_t iterations)",
	example: ```cpp
	ConvergenceFailure e("GMRES", 500);
	assert(e.algorithm() == "GMRES");
	assert(e.iterations() == 500);
	```,
	description: [
		An iterative algorithm exhausted its iteration budget. Solvers that report
		convergence through `converged()` / a `Report` object do not throw this.
	],
	params: (
		("algorithm", [name of the algorithm that failed to converge]),
		("iterations", [number of iterations performed before giving up]),
	),
	returns: [
		Constructs the error; `what()` names the algorithm and iteration count.
	],
)

#method(
	"const std::string& algorithm() / size_t iterations() const",
	example: ```cpp
	ConvergenceFailure e("GMRES", 500);
	assert(e.algorithm() == "GMRES" && e.iterations() == 500);
	```,
	description: [
		The algorithm name and the number of iterations performed before failure.
	],
	params: (),
	returns: [
		The recorded algorithm name (`const std::string&`) / iteration count
		(`size_t`).
	],
)

== NotComputed

#method(
	"NotComputed(const std::string& factorization)",
	example: ```cpp
	NotComputed e("QR");
	assert(std::string(e.what()).find("QR") != std::string::npos);
	assert(std::string(e.what()).find("compute()") != std::string::npos);
	```,
	description: [
		A factorization accessor (`solve`, `factors`, `eigenvalues`, ...) was
		called before `compute()`, or after a `compute()` that failed.
	],
	params: (("factorization", [name of the factorization queried before `compute()`]),),
	returns: [
		Constructs the error; `what()` names the factorization and mentions
		`compute()`.
	],
)
