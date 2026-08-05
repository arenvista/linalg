// Documentation for src/ops/Givens.cpp (include/linalg/ops/Givens.hpp).
// Included from documentation.typ; imports the template for the `method` env.
#import "template.typ": *

= Givens rotations

A `Givens<T>` is a plane rotation that zeros a single entry — preferred over a
Householder reflector where the pattern is sparse (bidiagonal sweeps, QR
updating, Hessenberg reduction inside GMRES). Acting on rows $p$ and $q$ it
applies the LAPACK `lartg` convention

$ mat(c, s; -overline(s), c) vec("row"_p, "row"_q), $

with the cosine $c$ always real, so the rotation is unitary even for complex
`T`. `FromPair` chooses $(c, s)$ without ever squaring its inputs
(`dlartg`-style scaling), so it stays exact near the overflow and underflow
thresholds. `Real` is `NumericTraits<T>::Real`.

== Construction

#method(
	"Givens()  /  Givens(const T& cosine, const T& sine, Index p, Index q)",
	example: ```cpp
	Givens<double> uninit;                 // coefficients value-initialized
	Givens<double> g(1.0, 0.0, 0, 1);      // explicit c, s and plane (p, q)
	assert(g.cosine() == 1.0 && g.sine() == 0.0);
	```,
	description: [
		The default constructor makes an uninitialized rotation. The four-argument
		constructor sets the coefficients $c$ and $s$ directly, together with the
		two plane indices $p$ and $q$ it acts on.
	],
	params: (
		("cosine", [the cosine coefficient $c$]),
		("sine", [the sine coefficient $s$]),
		("p", [first plane index]),
		("q", [second plane index]),
	),
	returns: [
		The constructed rotation.
	],
)

#method(
	"static Givens FromPair(const T& a, const T& b, Index p, Index q)",
	example: ```cpp
	G g = G::FromPair(3.0, 4.0, 0, 1);
	V x{3, 4};
	g.apply(x);
	assert(close(std::abs(x(0)), 5.0) && close(x(1), 0.0)); // (a,b) -> (r,0)
	assert(close(x(0), g.radius()));            // lands on the radius
	assert(close(g.cosine()*g.cosine() + g.sine()*g.sine(), 1.0)); // c^2+|s|^2=1
	// dlartg-style scaling: neither a^2 nor b^2 is ever materialized
	G big = G::FromPair(1e200, 1e200, 0, 1);
	assert(std::isfinite(big.radius()));
	```,
	description: [
		Builds the rotation that maps the pair $(a, b)$ to $(r, 0)$, i.e. the
		rotation that zeros the second component. The coefficients are formed with
		a scale factor so that neither $a^2$ nor $b^2$ is materialized, keeping the
		result exact near the overflow and underflow thresholds. Two edge cases:
		$b = 0$ gives a (possibly phase-carrying) identity, and $a = 0$ gives a
		pure swap.
	],
	params: (
		("a", [first component to rotate]),
		("b", [second component (the one zeroed)]),
		("p", [first plane index]),
		("q", [second plane index]),
	),
	returns: [
		The rotation mapping $(a, b)$ to $(r, 0)$; the landing radius $r$ is
		available via `radius()`.
	],
)

#method(
	"static Givens Identity(Index p, Index q)",
	example: ```cpp
	G g = G::Identity(0, 1);
	assert(g.cosine() == 1.0 && g.sine() == 0.0);
	V x{3, 4};
	g.apply(x);
	assert(x(0) == 3 && x(1) == 4); // no-op on the (p, q) plane
	```,
	description: [
		Builds the identity rotation ($c = 1$, $s = 0$): a no-op on the $(p, q)$
		plane.
	],
	params: (
		("p", [first plane index]),
		("q", [second plane index]),
	),
	returns: [
		The identity rotation on the $(p, q)$ plane.
	],
)

== Accessors

#method(
	"const T& cosine() / sine()  /  Index firstIndex() / secondIndex()  /  T radius()",
	example: ```cpp
	G g = G::FromPair(3.0, 4.0, 0, 1);
	assert(g.firstIndex() == 0 && g.secondIndex() == 1);
	assert(close(std::abs(g.radius()), 5.0)); // what (a, b) rotated onto
	```,
	description: [
		The rotation's stored data: the coefficients $c$ (`cosine`) and $s$
		(`sine`); the two plane indices $p$ (`firstIndex`) and $q$
		(`secondIndex`); and the radius $r$ (`radius`) — the value $(a, b)$ lands
		on, set by `FromPair` (zero otherwise).
	],
	params: (),
	returns: [
		A reference to $c$ / $s$, the index $p$ / $q$, or the radius $r$
		(by value).
	],
)

== Transpose & inverse

#method(
	"Givens transposed()  /  Givens inverse()",
	example: ```cpp
	G g = G::FromPair(3.0, 4.0, 0, 1);
	V x{3, 4};
	g.apply(x);
	g.inverse().apply(x);                 // undo: restores the input
	assert(close(x(0), 3.0) && close(x(1), 4.0));
	M gd = g.toMatrix(2), td = g.transposed().toMatrix(2);
	assert((gd * td).isApprox(M::Identity(2), 1e-12));
	```,
	description: [
		Both return the rotation that reverses this one. Because the rotation is
		unitary, its `inverse` is the conjugate transpose $G^H$ ($c$ unchanged,
		$s -> -s$), while `transposed` is $G^T$ ($c$ unchanged,
		$s -> -overline(s)$). The two coincide for real `T` and differ for
		complex `T`.
	],
	params: (),
	returns: [
		The transposed / inverse rotation.
	],
)

== Application

#method(
	"void applyLeft(MatrixView<T> target)  /  void applyRight(MatrixView<T> target)",
	example: ```cpp
	G g = G::FromPair(1.0, 1.0, 0, 2);
	M a{{1, 2}, {100, 200}, {3, 4}};
	g.applyLeft(a.view());
	assert(a(1, 0) == 100 && a(1, 1) == 200); // row 1 (off-plane) untouched
	assert(a.isApprox(g.toMatrix(3) * before, 1e-12)); // matches the dense form
	// applyRight rotates the two indexed columns instead
	g2.applyRight(m.view());
	```,
	description: [
		Apply the rotation in place. `applyLeft` rotates rows $p$ and $q$
		($"target" <- G dot "target"$); `applyRight` rotates columns $p$ and $q$
		($"target" <- "target" dot G$). Each is $O("cols")$ / $O("rows")$: only
		the two indexed rows/columns are touched.
	],
	params: (("target", [matrix view overwritten with the rotated rows / columns]),),
	returns: [
		Nothing (in place).
	],
)

#method(
	"void apply(Vector<T>& x)",
	example: ```cpp
	G g = G::FromPair(3.0, 4.0, 0, 1);
	V x{3, 4};
	g.apply(x);
	assert(close(std::abs(x(0)), 5.0) && close(x(1), 0.0));
	```,
	description: [
		Applies the rotation to a vector in place, rotating entries $p$ and $q$;
		every other entry is untouched.
	],
	params: (("x", [vector overwritten with the rotated entries]),),
	returns: [
		Nothing (in place).
	],
)

#method(
	"Matrix<T> toMatrix(Index dimension)",
	example: ```cpp
	G g = G::FromPair(3.0, -4.0, 0, 1);
	M d = g.toMatrix(2);
	assert(d.isOrthogonal(1e-12));       // rotation is orthogonal/unitary
	assert(close(d.determinant(), 1.0)); // proper rotation
	```,
	description: [
		Materializes the dense rotation: the $2 times 2$ block embedded in an
		identity of the given size, with every entry off the $(p, q)$ plane left
		as the identity. Intended mainly for tests and cross-checks.
	],
	params: (("dimension", [size of the surrounding identity matrix]),),
	returns: [
		The dense rotation matrix.
	],
)

== GivensSequence

`GivensSequence<T>` is an ordered product of rotations — the accumulated sweep
of a Jacobi eigenvalue pass or a bidiagonal chase. `applyLeft` applies the
rotations in append order ($"target" <- G_(k-1) dots.h.c G_1 G_0 dot
"target"$); `applyLeftReversed` uses the opposite order (the inverse sequence
pairs `reversed` with `transposed`).

#method(
	"GivensSequence()  /  void append(const Givens<T>&) / clear()  /  Index count()",
	example: ```cpp
	GivensSequence<double> seq;
	assert(seq.count() == 0);
	seq.append(G::FromPair(1.0, 2.0, 0, 1));
	seq.append(G::FromPair(3.0, 1.0, 1, 2));
	assert(seq.count() == 2);
	seq.clear();
	assert(seq.count() == 0);
	```,
	description: [
		Construct an empty sequence, then grow it: `append` adds a rotation at the
		end, `clear` removes all rotations, `count` reports how many it holds.
	],
	params: (("rotation", [the rotation to add (`append`)]),),
	returns: [
		Nothing / the rotation count (`count`).
	],
)

#method(
	"const Givens<T>& operator[](Index k)",
	example: ```cpp
	seq.append(g0); seq.append(g1);
	assert(seq[0].firstIndex() == 0 && seq[1].secondIndex() == 2);
	```,
	description: [
		Read-only access to a rotation by position in the sequence.
	],
	params: (("k", [position in the sequence (0-based)]),),
	returns: [
		Reference to rotation $k$.
	],
)

#method(
	"void applyLeft / applyRight / applyLeftReversed(MatrixView<T> target)",
	example: ```cpp
	// applyLeft is append order: target <- G1 * (G0 * target)
	M expected = g1.toMatrix(3) * (g0.toMatrix(3) * a);
	M inPlace = a;
	seq.applyLeft(inPlace.view());
	assert(inPlace.isApprox(expected, 1e-12));
	// applyLeftReversed applies them the other way round
	seq.applyLeftReversed(other.view());
	```,
	description: [
		Apply the whole sequence in place. `applyLeft` composes the rotations on
		the left in append order; `applyRight` composes them on the right in
		append order; `applyLeftReversed` composes on the left in reverse append
		order (used to apply the inverse sweep together with `transposed`).
	],
	params: (("target", [matrix view overwritten with the result]),),
	returns: [
		Nothing (in place).
	],
)

#method(
	"GivensSequence reversed()  /  Matrix<T> toMatrix(Index dimension)",
	example: ```cpp
	// toMatrix is the accumulated product; reversed() flips the order
	assert(seq.toMatrix(3).isApprox(g1.toMatrix(3) * g0.toMatrix(3), 1e-12));
	assert(seq.reversed().toMatrix(3).isApprox(
	    g0.toMatrix(3) * g1.toMatrix(3), 1e-12));
	```,
	description: [
		`reversed` returns the sequence in reverse order; `toMatrix` materializes
		the accumulated rotation as a dense matrix of the given size.
	],
	params: (("dimension", [size of the identity the rotations act within (`toMatrix`)]),),
	returns: [
		The reversed sequence / the dense product matrix.
	],
)
