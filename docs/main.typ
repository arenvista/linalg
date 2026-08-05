#import "template.typ": *

#show: notes.with(
	title: "linalg: A Dense Numerical Linear Algebra Library in C++17",
	author: "Aren Vista, Micah Havens, Sina Roomi, Max Monson, Jingson Guo, Dominic Mello",
	date: "August 1, 2026",
)

// Prose document: no automatic equation numbers.
#set math.equation(numbering: none)

// ---- Figure helpers ----------------------------------------------------

// Small module chip for the architecture diagram.
#let chip(name) = box(
	fill: luma(248),
	stroke: 0.5pt + luma(190),
	radius: 2pt,
	inset: (x: 5pt, y: 3.5pt),
	text(size: 8.5pt, font: "DejaVu Sans Mono", name),
)

// One dependency tier: directory name, one-line role, accent color, modules.
#let layer(name, desc, color, modules) = block(
	width: 100%,
	fill: color.lighten(93%),
	stroke: (left: 2.5pt + color, rest: 0.5pt + color.lighten(55%)),
	radius: 2pt,
	inset: (x: 10pt, y: 7pt),
	above: 4pt,
	below: 4pt,
)[
	#grid(
		columns: (86pt, 1fr),
		align: horizon,
		[
			#text(weight: "bold", fill: color.darken(20%), size: 9.5pt, font: "DejaVu Sans Mono", name) \
			#text(size: 8pt, fill: luma(100), desc)
		], modules.map(chip).join(h(4pt)),
	)
]

#let depends-arrow = align(center, text(size: 8pt, fill: luma(130))[$arrow.b$ #h(2pt) _depends on_])

#let architecture = {
	grid(
		columns: (1fr, 1fr),
		column-gutter: 6pt,
		layer("solve/", "direct front doors", rgb("#a51d2d"), ("LinearSolver", "LeastSquares", "TriangularSolver")), layer("iterative/", "large-scale methods", rgb("#6c5b00"), ("KrylovSolver", "EigenSolver", "Preconditioner")),
	)
	depends-arrow
	layer("decomp/", "factorizations", rgb("#613583"), ("LU", "Cholesky", "QR", "Hessenberg", "SVD", "Eigen"))
	depends-arrow
	layer("ops/", "kernels & primitives", rgb("#2e7d32"), ("Kernels", "Norm", "Householder", "Givens"))
	depends-arrow
	layer("core/", "types & scalars", rgb("#1a5fb4"), ("Matrix", "Vector", "MatrixView", "NumericTraits", "Exceptions"))
}

// A matrix outline with highlighted sub-panels, for the blocking figure.
#let blocked(w, h, overlays, label) = box(width: 96pt)[
	#align(center)[
		#box(width: w, height: h, fill: luma(252), stroke: 0.8pt + luma(80), {
			for o in overlays {
				place(dx: o.at(0), dy: o.at(1), rect(width: o.at(2), height: o.at(3), fill: o.at(4), stroke: o.at(5)))
			}
		})
	]
	#v(3pt)
	#align(center, text(size: 8pt, fill: luma(95), label))
]

#let gemm-fig = align(center, grid(
	columns: 5,
	column-gutter: 8pt,
	align: horizon,
	blocked(64pt, 56pt, (
		(12pt, 12pt, 40pt, 28pt, rgb("#a51d2d").lighten(88%), 0.5pt + rgb("#a51d2d").lighten(40%)),
		(12pt, 12pt, 12pt, 9pt, rgb("#a51d2d").lighten(55%), 0.7pt + rgb("#a51d2d")),
	), [$C$: $m_r times n_r$ tile in an $m_c times n_c$ block]), [$=$], blocked(40pt, 56pt, ((8pt, 12pt, 16pt, 28pt, rgb("#2e7d32").lighten(60%), 0.7pt + rgb("#2e7d32")),), [$A$: packed $m_c times k_c$ panel]), [$times$], blocked(64pt, 40pt, ((12pt, 8pt, 40pt, 16pt, rgb("#1a5fb4").lighten(60%), 0.7pt + rgb("#1a5fb4")),), [$B$: packed $k_c times n_c$ panel]),
))

// ------------------------------------------------------------------------

// Compact API-reference table (Method | Description) and a full-width
// category separator row, used throughout the Module Reference. `rows` is a
// flat stream of cells: api-group("...") starts a labelled group.
#let api-methods(rows) = {
	set text(size: 8.5pt)
	table(
		columns: (auto, 1fr, 1fr),
		stroke: 0.5pt + luma(214),
		inset: (x: 7pt, y: 4pt),
		align: (left + horizon, left + horizon, left + horizon),
		table.header([*Member*], [*Description*], [*Example*]),
		..rows,
	)
}
#let api-group(name) = table.cell(colspan: 3, fill: luma(240))[
	#text(size: 8pt, weight: "bold", fill: luma(70), tracking: 0.4pt, upper(name))
]

#outline(title: [Contents], depth: 2, indent: auto)
#v(0.6em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#v(0.6em)

This project is a C++ library for numerical linear algebra. It provides a
set of classes for performing operations on dense matrices and vectors —
arithmetic, norms, factorization, linear solving, least squares, and
eigenvalue computation — implemented from first principles rather than by
wrapping BLAS/LAPACK. The library is designed to be efficient and easy to
use, with a focus on exploring the techniques commonly taught in numerical
linear algebra, and to serve as a demonstration of ability to design and
build robust numerical software: choosing algorithms for stability rather
than convenience, handling floating-point edge cases (overflow-safe norms,
cancellation-free reflectors, deflation criteria), and verifying every
component against the mathematical identities it must satisfy.

Complementary to the library is a set of notes that explain the algorithms
and the rationale behind their computational methods. These notes can be
accessed here:
#link("http://ec2-54-146-246-77.compute-1.amazonaws.com/notes/Numerical-Linear-Algebra/L01-Basic-Matrix-Algebra")[Numerical Linear Algebra — L01: Basic Matrix Algebra].

= What the Library Covers <sec-overview>

- *Core types.* Owning `Matrix`/`Vector` classes over a scalar
	abstraction (`NumericTraits`) that supports `float`, `double`, and their
	complex counterparts uniformly, plus non-owning stride-aware views
	(`MatrixView`) so blocked algorithms can operate on panels without
	copying.

- *Computational kernels.* BLAS-shaped level 1/2/3 routines (`gemm`,
	`gemv`, `trsm`, ...) that serve as the single tuning surface for
	performance work, and the two workhorse orthogonal primitives —
	Householder reflectors and Givens rotations — stored implicitly and
	applied in $O(m n)$.

- *Direct factorizations.* LU with partial and full pivoting, Cholesky
	and $L D L^H$, Householder QR with and without column pivoting, and the
	condensed forms (Hessenberg, tridiagonal, bidiagonal) that feed the
	eigenvalue and singular value algorithms.

- *Eigenvalues and the SVD.* Symmetric eigensolving via implicit-shift
	QR with Wilkinson shifts, the real Schur form by Francis double-shift
	sweeps, general eigenvectors on top of it, and the Golub–Kahan SVD.

- *High-level solvers.* A dispatching `LinearSolver` that inspects the
	matrix and chooses a factorization, least squares by QR/SVD (with the
	normal equations included only as a cautionary comparison), and
	Gauss–Newton / Levenberg–Marquardt for nonlinear problems.

#figure(
	architecture,
	caption: [The dependency tiers of the library. Each tier is written only
	in terms of the tiers below it, so tuning the `ops` kernels
	accelerates every factorization and solver above without touching
	their code.],
)

== The Kernels as the Tuning Surface <sec-tuning-surface>

"BLAS-shaped" means the kernels follow the interface conventions of the
Basic Linear Algebra Subprograms, the de facto standard vocabulary of
dense computation since the 1970s: strided vector arguments, an
`alpha`/`beta` scaling pair, and in-place updates like
$C arrow.l alpha A B + beta C$ (`gemm`) rather than functions that
allocate and return results. The _levels_ classify routines by the ratio
of arithmetic to data touched:

- *Level 1* — vector–vector, $O(n)$ work on $O(n)$ data: `axpy`, `dot`,
	`nrm2`. Memory-bound; there is little to tune.

- *Level 2* — matrix–vector, $O(n^2)$ work on $O(n^2)$ data: `gemv`,
	`ger`, `trsv`. Still memory-bound: every matrix element is read once
	and used once.

- *Level 3* — matrix–matrix, $O(n^3)$ work on $O(n^2)$ data: `gemm`,
	`syrk`, `trsm`. The only level with real data reuse — each element
	participates in $tilde n$ operations — which is what cache blocking,
	packing, and SIMD micro-kernels can exploit to approach the hardware's
	peak floating-point throughput.

This is why the kernels are the _single tuning surface_: the
factorizations above are written as thin orchestration around kernel
calls (a blocked LU is a panel factorization plus `trsm` and `gemm`
updates on the trailing submatrix). Optimizing `gemm` alone therefore
accelerates LU, Cholesky, QR, and the condensed-form reductions at once,
with no change to their code — mirroring how LAPACK gets its speed from
whatever tuned BLAS it is linked against. It also bounds the risk of
performance work: the naive triple-loop kernels stay as the reference
implementation, and a tuned variant must merely reproduce their results
to be correct everywhere.

#figure(
	gemm-fig,
	caption: [Level-3 blocking: `gemm` packs a panel of $A$ and a panel of
	$B$ into contiguous buffers sized for the cache hierarchy, then a
	register-tiled micro-kernel sweeps small tiles of $C$, reusing every
	packed element $tilde n$ times.],
)

= A Worked Kernel: Givens Rotations <sec-givens-kernel>

The file `src/ops/Givens.cpp` is a good place to see the library's guiding
principle — _choose the algorithm for numerical stability, then let the code
mirror the math it implements_ — at the scale of a single kernel. A Givens
rotation is the second of the two orthogonal primitives (the other being the
Householder reflector); it is the tool of choice when only one entry must be
zeroed at a time, as in bidiagonal sweeps, QR updating, and the Hessenberg
reduction inside GMRES.

#definition(title: "Givens rotation")[
	A _Givens rotation_ acting on the plane spanned by coordinates $p < q$ is
	the identity everywhere except in that $2 times 2$ block, where it is
	$ G = mat(delim: "[", c, s; -overline(s), c), quad c in RR, quad c^2 + abs(s)^2 = 1. $
	Keeping $c$ real while allowing $s in CC$ makes $G$ unitary for complex
	scalars too, so the same object serves the real and complex builds.
]

Because $G$ differs from the identity in only two rows and two columns, the
class never stores the dense matrix: the members are just `cosine_` ($c$),
`sine_` ($s$), the plane indices `p_`, `q_`, and a cached `radius_`
(explained below). Applying it is therefore $O(m n)$, not $O(m n^2)$ — the
implicit representation _is_ the performance story.

== Constructing the rotation

The core routine `FromPair(a, b, p, q)` chooses $(c, s)$ so that the rotation
maps the pair $(a, b)$ onto $(r, 0)$, zeroing the second component:
$ mat(delim: "[", c, s; -overline(s), c) vec(a, b) = vec(r, 0). $

#theorem(title: "closed form")[
	With $d = sqrt(abs(a)^2 + abs(b)^2)$, the solution is
	$ c = abs(a) / d, quad s = a / abs(a) dot overline(b) / d, quad r = a / abs(a) dot d. $
	One checks directly that $c$ is real, $c^2 + abs(s)^2 = 1$, and the lower
	entry $-overline(s) a + c b = 0$.
]

Translating the closed form into code line-for-line would be a mistake: forming
$abs(a)^2$ overflows for $a approx 10^(200)$ in double precision and underflows
to zero for $a approx 10^(-200)$, even though the true $d$ is perfectly
representable in both cases. The implementation follows LAPACK's `dlartg` and
never squares the inputs directly. It scales by the larger magnitude first,

$ sigma = max(abs(a), abs(b)), quad
  d = sigma sqrt((abs(a) / sigma)^2 + (abs(b) / sigma)^2), $

so the squared terms live in $[0, 1]$ and the lone $sqrt(dot)$ cannot over- or
underflow. The code names these quantities exactly: `magA`, `magB` are
$abs(a), abs(b)$; `scale` is $sigma$; `scaledNorm` is the bracketed root;
`norm` is $d$; `phase` is the unit-modulus direction $a slash abs(a)$; and the
results land in `cosCoeff`, `sinCoeff`, and `radius`. Two exact-zero branches
handle the degenerate planes: $b = 0$ needs no rotation (identity, with $r = a$
carrying $a$'s sign), and $a = 0$ is a pure swap.

#remark(title: "Why cache the radius")[
	`radius_` stores the $r$ that $(a, b)$ lands on. It is exactly the new
	diagonal entry a factorization writes after the rotation zeroes the
	subdiagonal one, so caching it during construction saves recomputing
	$abs(dot)$ at the call site. The accessor `radius()` simply returns it;
	the constructors set it to zero since a rotation built from raw
	coefficients has no associated pair.
]

== Applying and undoing the rotation

The three `apply*` methods are the same $2 times 2$ update specialized to
different targets. Left application `applyLeft` rotates two _rows_,

$ "row"_p <- c dot "row"_p + s dot "row"_q, quad
  "row"_q <- -overline(s) dot "row"_p + c dot "row"_q, $

`apply` is this update on a single vector, and `applyRight` is the
right-multiplication $M G$, which rotates two _columns_ with the transposed
coefficient pattern. Each loop reads both old values before writing either, so
the second assignment never sees a value the first one already overwrote.

Finally, `transposed` and `inverse` look identical for real scalars but are
deliberately distinct for complex ones, which is where the "$c$ real, $s$
complex" convention earns its keep:

$ G^T = mat(delim: "[", c, -overline(s); s, c) quad ("store" c, -overline(s)),
  quad quad
  G^(-1) = G^H = mat(delim: "[", c, -s; overline(s), c) quad ("store" c, -s). $

Since $G$ is unitary, its inverse is the conjugate transpose $G^H$, so
`inverse` negates $s$ while `transposed` negates $overline(s)$. For real
$s$ the two coincide, matching the familiar fact that a real rotation's
inverse is its transpose — but using `transposed` where `inverse` is meant
would silently give a non-unitary "undo" on complex data.

== Method Reference

`Givens<T>` — a single plane rotation on coordinates $(p, q)$:

#{
	set text(size: 8.5pt)
	table(
		columns: (auto, 1fr, 1fr),
		stroke: 0.5pt + luma(210),
		inset: (x: 7pt, y: 5pt),
		align: (left + horizon, left + horizon, left + horizon),
		table.header([*Method*], [*Description*], [*Example*]),
		[`Givens()`], [Default constructor: an uninitialized rotation ($c = s = 0$, radius $0$).], [`Givens<double> g;`],
		[`Givens(c, s, p, q)`], [Construct directly from coefficients and plane indices.], [`Givens<double>(0.6, 0.8, 0, 1)`],
		[`FromPair(a, b, p, q)`], [_(static)_ The rotation carrying $(a, b)$ to $(r, 0)$; overflow-safe, `dlartg`-style.], [`Givens<double>::FromPair(3.0, 4.0, 0, 1)`],
		[`Identity(p, q)`], [_(static)_ The no-op rotation $c = 1, s = 0$ on the plane $(p, q)$.], [`Givens<double>::Identity(0, 1)`],
		[`cosine()`], [The real cosine coefficient $c$.], [`g.cosine()` $-> 0.6$],
		[`sine()`], [The sine coefficient $s$.], [`g.sine()` $-> 0.8$],
		[`firstIndex()`], [The lower plane index $p$.], [`g.firstIndex()` $-> 0$],
		[`secondIndex()`], [The upper plane index $q$.], [`g.secondIndex()` $-> 1$],
		[`radius()`], [The $r$ that `FromPair` mapped $(a, b)$ onto.], [`g.radius()` $-> 5.0$],
		[`transposed()`], [The transpose $G^T$ (stored as $c, -overline(s)$).], [`g.transposed()`],
		[`inverse()`], [The inverse $G^(-1) = G^H$ (stored as $c, -s$); unitary.], [`g.inverse()`],
		[`applyLeft(view)`], [In place: overwrites `target` with $G dot$ `target`, rotating rows $p, q$. $O("cols")$.], [`g.applyLeft(A.view())`],
		[`applyRight(view)`], [In place: overwrites `target` with `target` $dot G$, rotating columns $p, q$. $O("rows")$.], [`g.applyRight(A.view())`],
		[`apply(x)`], [In place rotation of vector entries $p$ and $q$.], [`g.apply(x)`],
		[`toMatrix(n)`], [The dense $n times n$ rotation embedded in the identity (tests only).], [`g.toMatrix(2)`],
	)
}

`GivensSequence<T>` — an ordered product $G_(k-1) dots.h G_1 G_0$ of rotations:

#{
	set text(size: 8.5pt)
	table(
		columns: (auto, 1fr, 1fr),
		stroke: 0.5pt + luma(210),
		inset: (x: 7pt, y: 5pt),
		align: (left + horizon, left + horizon, left + horizon),
		table.header([*Method*], [*Description*], [*Example*]),
		[`GivensSequence()`], [Construct an empty sequence ($"count" = 0$).], [`GivensSequence<double> seq;`],
		[`append(g)`], [Add a rotation to the end of the sequence.], [`seq.append(g);`],
		[`clear()`], [Remove all rotations.], [`seq.clear();`],
		[`count()`], [The number of rotations held.], [`seq.count()`],
		[`operator[](k)`], [Read-only access to rotation $k$ (append order).], [`seq[0]`],
		[`applyLeft(view)`], [Apply all on the left in append order: `target` $<- G_(k-1) dots.h G_0 dot$ `target`.], [`seq.applyLeft(A.view())`],
		[`applyRight(view)`], [Apply all on the right in append order.], [`seq.applyRight(A.view())`],
		[`applyLeftReversed(view)`], [Apply all on the left in _reverse_ append order (used by the inverse sweep).], [`seq.applyLeftReversed(A.view())`],
		[`reversed()`], [A copy of the sequence in reverse order.], [`seq.reversed()`],
		[`toMatrix(n)`], [The dense $n times n$ accumulated product.], [`seq.toMatrix(3)`],
	)
}

== Example: one step of Givens QR

A Givens rotation triangularizes a matrix one subdiagonal entry at a time.
The snippet below zeroes $A_(1,0)$ of a $2 times 2$ matrix and accumulates the
rotation so the orthogonal factor can be recovered — the atom from which a
full Givens QR is built by looping over the subdiagonal.

```cpp
#include "linalg/core/Matrix.hpp"
#include "linalg/ops/Givens.hpp"
using namespace linalg;

Matrix<double> A{{4.0, 3.0},
                 {3.0, 2.0}};

// Choose the rotation on rows (0, 1) that sends (A(0,0), A(1,0)) to (r, 0).
Givens<double> g = Givens<double>::FromPair(A(0, 0), A(1, 0), 0, 1);

g.applyLeft(A.view());   // A <- G * A; now A(1,0) == 0 and A(0,0) == g.radius()

// Collect rotations so Q^T = ... G_1 G_0 can be materialized or reapplied.
GivensSequence<double> qt;
qt.append(g);
Matrix<double> Qt = qt.toMatrix(2);   // the accumulated orthogonal factor

// A single vector rotates the same way:
Vector<double> x{4.0, 3.0};
g.apply(x);              // x(1) == 0, x(0) == g.radius()
```

After `applyLeft`, `A` is upper-triangular and its new diagonal entry equals
`g.radius()` — the value `FromPair` cached precisely because it is the entry a
QR step writes next. Extending the loop to larger matrices (zeroing
$A_(i,0)$ for $i = m-1, dots.h, 1$, then moving to the next column) reduces $A$
to $R$ while `qt` accumulates $Q^H$.

= Module Reference

The two lowest tiers of the dependency graph — `core/` (the types every
other file is written against) and `ops/` (the kernels and orthogonal
primitives built directly on them) — are the foundation the factorizations
and solvers orchestrate. Each header declares; the matching `src/` file
defines and explicitly instantiates for the supported scalar list.

== Core Types (`core/`)

=== `Traits` — the scalar abstraction

`NumericTraits<T>` is the single place the library asks questions about a
scalar instead of assuming `double`, and the reason one template body serves
`float`, `double`, and their `complex` counterparts. It draws the line
between a scalar `T` and its magnitude type `Real` (equal for real `T`, the
component type for `complex`). All members are `static`.

#api-methods((
	api-group("queries"),
	[`epsilon()`], [Machine epsilon of `Real` — the gap between $1$ and the next value.], [`epsilon()` $approx 2.2 times 10^(-16)$ for `double`],
	[`safeMin()`], [Smallest `s` with $1 slash s$ finite; the scaling threshold for overflow-safe code.], [`safeMin()` $approx 2.2 times 10^(-308)$],
	api-group("arithmetic"),
	[`abs(x)`], [Magnitude $abs(x)$ as a `Real`.], [`abs({3.0, 4.0})` $-> 5.0$],
	[`absSquared(x)`], [$abs(x)^2$, computed without a square root.], [`absSquared({3.0, 4.0})` $-> 25.0$],
	[`conj(x)`], [Complex conjugate (the identity for real `T`).], [`conj({3.0, 4.0})` $-> (3, -4)$],
	[`real(x)` / `imag(x)`], [Real and imaginary parts ($"imag"$ is $0$ for real `T`).], [`imag({3.0, 4.0})` $-> 4.0$],
	[`sqrt(x)`], [Principal square root.], [`sqrt(2.0)` $approx 1.414$],
	[`zero()` / `one()`], [The additive / multiplicative identity for `T`.], [`one()` $-> (1, 0)$ for `complex`],
	[`isApproxZero(x, tol)`], [Whether $abs(x) <= "tol"$.], [`isApproxZero(1e-20, 1e-9)` $->$ `true`],
))

The same header carries `IsComplex<T>::value` (the compile-time category
test) and the tag enums — `StorageOrder`, `Triangle`, `Transposition`,
`Diagonal` — that kernels and solvers select behavior with.

```cpp
using T = std::complex<double>;
using R = NumericTraits<T>::Real;              // double
R m  = NumericTraits<T>::abs({3.0, 4.0});      // 5.0
T zc = NumericTraits<T>::conj({3.0, 4.0});     // (3, -4)
constexpr bool cplx = IsComplex<T>::value;     // true
```

=== `Exceptions` — the error hierarchy

Everything derives from `LinalgError` (itself a `std::runtime_error`), so a
single `catch (const LinalgError&)` handles the whole family; the typed
subclasses carry structured context for callers that want it.

#{
	set text(size: 8.5pt)
	table(
		columns: (auto, 1fr, 1fr),
		stroke: 0.5pt + luma(214),
		inset: (x: 7pt, y: 4pt),
		align: (left + horizon, left + horizon, left + horizon),
		table.header([*Type*], [*Meaning and payload*], [*Example*]),
		[`LinalgError`], [Base type; thrown by every unimplemented stub and by generic contract violations.], [`throw LinalgError("not implemented: ...")`],
		[`DimensionMismatch`], [Incompatible operand shapes; `lhsRows()`, `lhsCols()`, `rhsRows()`, `rhsCols()`.], [Adding a $2 times 3$ to a $3 times 2$ matrix.],
		[`IndexOutOfRange`], [Bad index from a checked `at()`; `index()`, `bound()`.], [`v.at(9)` on a length-$3$ vector.],
		[`SingularMatrix`], [A (near-)zero pivot in a factorization or solve; `pivotIndex()`.], [`Matrix::Zeros(3,3).inverse()`],
		[`NotPositiveDefinite`], [First non-positive pivot in a Cholesky-family factorization; `pivotIndex()`.], [Cholesky of an indefinite matrix.],
		[`ConvergenceFailure`], [An iterative method exhausted its budget; `algorithm()`, `iterations()`.], [GMRES that never meets its tolerance.],
		[`NotComputed`], [A factorization accessor called before `compute()`, or after one that failed.], [`lu.solve(b)` before `lu.compute(A)`.],
	)
}

```cpp
try {
    Matrix<double> Ainv = A.inverse();          // may throw
} catch (const SingularMatrix& e) {
    std::cerr << "zero pivot at step " << e.pivotIndex() << "\n";
} catch (const LinalgError& e) {                // base catches the rest
    std::cerr << e.what() << "\n";
}
```

=== `Vector` — owning dense vector

Kept a distinct type from an $n times 1$ `Matrix` precisely so that `dot`,
`outer`, and `norm` have one unambiguous meaning rather than a
row-versus-column ambiguity. Arithmetic operators are members only.

#api-methods((
	api-group("construction"),
	[`Vector()`], [Empty, length $0$.], [`Vector<double> v;`],
	[`Vector(n)` / `Vector(n, fill)`], [`n` zeros, or `n` copies of `fill`.], [`Vector<double>(3, 1.0)` $-> (1,1,1)$],
	[`Vector({...})` / `Vector(std::vector)`], [From a braced list or a `std::vector`.], [`Vector<double>{3.0, 4.0}`],
	[copy/move, `~Vector`, `operator=`], [Standard deep-copy value semantics.], [`Vector<double> w = v;`],
	[`Zeros` / `Ones` / `Constant`], [_(static)_ Filled vectors of a given length.], [`Vector<double>::Ones(3)`],
	[`Unit(n, axis)`], [_(static)_ The basis vector $e_"axis"$.], [`Unit(3, 0)` $-> (1,0,0)$],
	[`Random(n, seed)`], [_(static)_ Reproducible pseudo-random entries.], [`Vector<double>::Random(4, 42)`],
	[`LinSpace(n, begin, end)`], [_(static)_ Equally spaced values, endpoints inclusive.], [`LinSpace(3, 0.0, 1.0)` $-> (0, .5, 1)$],
	api-group("element access"),
	[`operator()(i)` / `operator[](i)`], [Unchecked element access (mutable and const).], [`v(0) = 5.0;`],
	[`at(i)`], [Bounds-checked access; throws `IndexOutOfRange`.], [`v.at(0)`],
	[`data()`], [Raw buffer pointer.], [`v.data()`],
	[`size()` / `isEmpty()`], [Length, and whether it is $0$.], [`v.size()` $-> 2$],
	api-group("arithmetic"),
	[`+` `-` `*` `/` (unary `-`)], [Elementwise add/sub and scalar mul/div; negation.], [`v * 2.0`],
	[`+=` `-=` `*=` `/=`], [In-place forms.], [`v += w;`],
	[`==` / `!=`], [Exact equality / inequality.], [`v == w`],
	[`scaledBy(s)`], [Scalar multiple (readable mirror of `operator*`).], [`v.scaledBy(2.0)`],
	[`elementwiseProduct` / `elementwiseQuotient`], [Hadamard product / quotient.], [`v.elementwiseProduct(w)`],
	api-group("products"),
	[`dot(y)`], [Unconjugated bilinear form $sum x_i y_i$.], [`x.dot(y)` $-> 11.0$],
	[`hermitianDot(y)`], [Hermitian inner product $sum overline(x_i) y_i$ (conjugates `*this`).], [`x.hermitianDot(y)`],
	[`outer(y)`], [Outer product $x y^H$ as a matrix.], [`x.outer(y)` $-> $ matrix],
	[`cross(y)`], [Cross product (size-$3$ vectors only).], [`e1.cross(e2)` $-> e_3$],
	[`axpy(alpha, y)`], [Fused $alpha x + y$.], [`x.axpy(2.0, y)`],
	api-group("norms & scalar summaries"),
	[`norm()`], [Euclidean $2$-norm, overflow-safe scaled.], [`x.norm()` $-> 5.0$],
	[`squaredNorm()`], [$sum abs(x_i)^2$ (unscaled; may overflow).], [`x.squaredNorm()` $-> 25.0$],
	[`oneNorm()` / `infinityNorm()` / `pNorm(p)`], [The $1$-, $infinity$-, and general $p$-norms.], [`x.infinityNorm()` $-> 4.0$],
	[`sum()` / `product()`], [Sum and product of all elements.], [`x.sum()` $-> 7.0$],
	[`normalized()` / `normalize()`], [Unit-norm copy / in-place; throw on a zero vector.], [`x.normalized()` $-> (.6, .8)$],
	api-group("shape"),
	[`segment(start, count)` / `head` / `tail`], [Contiguous sub-vectors.], [`v.head(2)`],
	[`concat(y)` / `reversed()`], [Concatenation; reverse order.], [`v.reversed()`],
	[`asColumnMatrix` / `asRowMatrix` / `asDiagonalMatrix`], [Reinterpret as a matrix.], [`v.asDiagonalMatrix()`],
	[`resize` / `conservativeResize`], [Resize, discarding or preserving overlap.], [`v.resize(5)`],
	[`fill` / `setZero` / `setUnit(axis)` / `swap`], [In-place mutation.], [`v.setZero();`],
	api-group("reductions & diagnostics"),
	[`maxAbsIndex` / `minAbsIndex`], [Index of largest / smallest magnitude (ties: lowest index).], [`x.maxAbsIndex()` $-> 1$],
	[`maxCoefficient` / `minCoefficient`], [Largest / smallest coefficient (by real part for complex `T`).], [`x.maxCoefficient()` $-> 4.0$],
	[`isApprox(y, tol)` / `hasNaN()`], [Approximate equality; NaN check.], [`x.isApprox(y, 1e-9)`],
	[`toString(precision)`], [Human-readable formatting.], [`x.toString(3)`],
))

```cpp
Vector<double> x{3.0, 4.0};
double n = x.norm();                  // 5.0 (overflow-safe)
Vector<double> u = x.normalized();    // (0.6, 0.8)
Vector<double> y{1.0, 2.0};
double d = x.dot(y);                  // 11.0
Vector<double> z = x.axpy(2.0, y);    // 2*x + y = (7, 10)
```

=== `Matrix` — owning dense matrix

Row-major and contiguous. Every operator is a member, so `matrix * scalar`
exists but `scalar * matrix` does not (use `scaledBy()`) — a deliberate
asymmetry that keeps overload resolution unsurprising. The `O(n^3)`
convenience summaries (`determinant`, `spectralNorm`, `conditionNumber`,
`rank`, `inverse`, `pseudoInverse`) factorize internally; hold a decomposition
object instead when you need more than one query.

#api-methods((
	api-group("construction"),
	[`Matrix()`], [Empty $0 times 0$.], [`Matrix<double> A;`],
	[`Matrix(r, c[, fill])`], [Shape $r times c$, zero- or `fill`-initialized.], [`Matrix<double>(2, 2, 1.0)`],
	[`Matrix(r, c, rowMajorData)`], [From a flat row-major `std::vector`.], [`Matrix<double>(2, 2, {1,2,3,4})`],
	[`Matrix({{...}})`], [From a braced list of equal-length rows.], [`Matrix<double>{{4,1},{1,3}}`],
	[`Matrix(ConstMatrixView)`], [Deep-copy a view (explicit).], [`Matrix<double>(A.block(0,0,2,2))`],
	[copy/move, `~Matrix`, `operator=`], [Standard deep-copy value semantics.], [`Matrix<double> B = A;`],
	[`Zeros` / `Ones` / `Constant`], [_(static)_ Filled matrices.], [`Matrix<double>::Zeros(2, 3)`],
	[`Identity(n)` / `Diagonal(v)`], [_(static)_ Identity; diagonal from a vector.], [`Matrix<double>::Identity(3)`],
	[`Random` / `RandomSymmetric` / `RandomOrthogonal`], [_(static)_ Reproducible random, Hermitian, or unitary matrices.], [`Matrix<double>::RandomOrthogonal(4, 7)`],
	[`Hilbert(n)` / `Vandermonde(nodes, deg)`], [_(static)_ Classic ill-conditioned / structured inputs.], [`Matrix<double>::Hilbert(5)`],
	[`FromColumns` / `FromRows`], [_(static)_ Assemble from vectors.], [`Matrix<double>::FromColumns({u, v})`],
	api-group("element access & shape queries"),
	[`operator()(i, j)` / `at(i, j)`], [Unchecked / bounds-checked element access.], [`A(0, 1) = 5.0;`],
	[`data()`], [Row-major buffer; $(i, j)$ at `data()[i*cols()+j]`.], [`A.data()`],
	[`rows()` / `cols()` / `size()`], [Dimensions and element count.], [`A.rows()` $-> 2$],
	[`isEmpty()` / `isSquare()`], [Shape predicates.], [`A.isSquare()` $->$ `true`],
	api-group("views & row/col access"),
	[`view()` / `block(...)`], [Mutable and const aliasing views (whole or sub-block).], [`A.block(0, 0, 2, 2)`],
	[`rowView(i)` / `colView(j)`], [Aliasing views of a single row / column.], [`A.colView(0)`],
	[`row(i)` / `col(j)` / `diagonal()`], [Owning *copies* into a `Vector`.], [`A.diagonal()`],
	[`setRow` / `setCol` / `setBlock`], [Overwrite a row, column, or block.], [`A.setRow(0, v)`],
	api-group("arithmetic"),
	[`+` `-` `*` `/` (unary `-`)], [Matrix $plus.minus$; matrix$times$matrix, matrix$times$vector, matrix$times$scalar; scalar div; negation.], [`A * B`, `A * x`, `A * 2.0`],
	[`+=` `-=` `*=` `/=`], [In-place forms.], [`A += B;`],
	[`==` / `!=`], [Exact equality / inequality.], [`A == B`],
	[`scaledBy(s)`], [$s A$ (scalar on the left).], [`A.scaledBy(2.0)`],
	[`elementwiseProduct` / `elementwiseQuotient`], [Hadamard product / quotient.], [`A.elementwiseProduct(B)`],
	[`kroneckerProduct(B)`], [Kronecker (tensor) product.], [`A.kroneckerProduct(B)`],
	[`power(k)`], [Integer matrix power ($k = 0$ gives $I$).], [`A.power(3)`],
	api-group("shape manipulation"),
	[`transpose()` / `conjugateTranspose()`], [$A^T$ / $A^H$ (equal for real `T`).], [`A.transpose()`],
	[`reshaped(r, c)`], [Reinterpret row-major with a new shape.], [`A.reshaped(1, 4)`],
	[`horizontalConcat` / `verticalConcat`], [Stack side by side / on top.], [`A.horizontalConcat(B)`],
	[`withoutRow(i)` / `withoutCol(j)`], [Copy with one row / column dropped.], [`A.withoutRow(0)`],
	[`transposeInPlace()`], [In-place transpose.], [`A.transposeInPlace();`],
	[`resize` / `conservativeResize`], [Resize, discarding or preserving the top-left block.], [`A.conservativeResize(3, 3)`],
	[`swapRows` / `swapCols`], [In-place row / column swap.], [`A.swapRows(0, 1);`],
	[`fill` / `setZero` / `setIdentity` / `swap`], [In-place mutation.], [`A.setIdentity();`],
	api-group("scalar summaries"),
	[`trace()` / `sum()`], [Diagonal sum; sum of all elements.], [`A.trace()` $-> 7.0$],
	[`determinant()`], [Determinant via LU.], [`A.determinant()` $-> 11.0$],
	[`oneNorm` / `infinityNorm` / `frobeniusNorm` / `maxNorm`], [Entrywise / induced norms (no factorization).], [`A.frobeniusNorm()`],
	[`spectralNorm` / `conditionNumber` / `rank(tol)`], [SVD-based, $O(n^3)$.], [`A.conditionNumber()`],
	api-group("predicates"),
	[`isSymmetric` / `isHermitian` / `isDiagonal`], [Structure tests within an absolute tolerance.], [`A.isSymmetric(0.0)`],
	[`isTriangular(which, tol)` / `isOrthogonal(tol)`], [Triangularity; $A^H A = I$.], [`A.isOrthogonal(1e-12)`],
	[`isApprox(B, tol)` / `hasNaN()`], [Approximate equality; NaN check.], [`A.isApprox(B, 1e-9)`],
	api-group("derived matrices & serialization"),
	[`inverse()`], [Inverse via LU; throws `SingularMatrix`.], [`A.inverse()`],
	[`pseudoInverse(tol)`], [Moore–Penrose pseudoinverse via SVD.], [`A.pseudoInverse(1e-12)`],
	[`triangularPart(which)`], [Keep one triangle, zero the other.], [`A.triangularPart(Triangle::Kind::Upper)`],
	[`symmetricPart()` / `skewSymmetricPart()`], [$(A plus.minus A^H) slash 2$.], [`A.symmetricPart()`],
	[`toString` / `toMatlabLiteral`], [Human-readable / MATLAB-literal formatting.], [`A.toMatlabLiteral()`],
	[`FromCsv(path)` / `writeCsv(path)`], [CSV read _(static)_ / write.], [`A.writeCsv("A.csv")`],
))

```cpp
Matrix<double> A{{4.0, 1.0},
                 {1.0, 3.0}};
double det = A.determinant();          // 11 (via LU)
Matrix<double> Ainv = A.inverse();     // via LU
bool sym   = A.isSymmetric(0.0);       // true
Matrix<double> P = A * Ainv;           // ~ Identity(2)
```

=== `MatrixView` / `ConstMatrixView` — non-owning windows

Element $(i, j)$ lives at `data[i * rowStride + j * colStride]`, so rows,
columns, diagonals, transposes, and rectangular blocks are all expressible as
views over the same storage — which is what lets blocked algorithms operate on
panels without copying. Copy assignment *rebinds* the view; `copyFrom` and
`operator=(const Matrix&)` copy *elements*. `ConstMatrixView` is the read-only
counterpart a mutable view converts to implicitly.

#api-methods((
	api-group("construction & binding"),
	[`MatrixView()` / `MatrixView(data, r, c, rs, cs)`], [Empty view; view over caller storage with explicit strides.], [`MatrixView<double>(p, 2, 2, 2, 1)`],
	[copy/move, `operator=`], [Rebind to another view's storage (shallow).], [`view = A.view();`],
	[`operator=(const Matrix&)` / `copyFrom(v)`], [Copy *elements* into the viewed storage (shapes must match).], [`A.block(0,0,2,2) = B;`],
	api-group("access & shape"),
	[`operator()(i, j)` / `at(i, j)`], [Unchecked / bounds-checked element access.], [`view(0, 1)`],
	[`rows` / `cols` / `rowStride` / `colStride`], [Shape and strides.], [`view.colStride()`],
	[`isContiguous()` / `isEmpty()`], [Dense-block test; empty test.], [`view.isContiguous()`],
	[`data()`], [Pointer to element $(0, 0)$; honor strides unless contiguous.], [`view.data()`],
	api-group("sub-views"),
	[`block(i, j, nr, nc)`], [Rectangular sub-view.], [`view.block(1, 1, 2, 2)`],
	[`row(i)` / `col(j)` / `diagonal()`], [$1 times$cols, rows$times 1$, and main-diagonal views.], [`view.diagonal()`],
	[`transposed()`], [Shape/stride-swapped view; no elements move.], [`view.transposed()`],
	api-group("bulk operations"),
	[`toMatrix()`], [Deep-copy the elements into a fresh `Matrix`.], [`view.toMatrix()`],
	[`fill` / `setZero` / `scale(f)`], [Write every element (mutable view only).], [`view.scale(2.0);`],
	[`swapWith(other)`], [Exchange elements with another view.], [`v1.swapWith(v2);`],
))

```cpp
Matrix<double> A = Matrix<double>::Identity(4);
MatrixView<double> blk = A.block(1, 1, 2, 2);  // aliases A's storage
blk.fill(5.0);                                  // writes through to A
MatrixView<double> d = A.view().diagonal();     // 1 x 4 view of the diagonal
```

== Operations and Primitives (`ops/`)

=== `Kernels` — the BLAS-shaped tuning surface

Level 1/2/3 routines as `static` members, following BLAS conventions: strided
raw-pointer vectors, matrix operands as views (their strides playing the role
of the leading dimension), and a `beta == 0` output treated as write-only.
The level-3 routines are where blocking, packing, and the register-tiled
micro-kernel (parameterized by `BlockSizes`) live, per
@sec-tuning-surface[the earlier discussion].

#api-methods((
	api-group("level 1 — vector/vector, O(n)"),
	[`scal(n, alpha, x, incx)`], [$x <- alpha x$.], [`scal(n, 2.0, x, 1)`],
	[`axpy(n, alpha, x, .., y, ..)`], [$y <- alpha x + y$.], [`axpy(n, 1.0, x, 1, y, 1)`],
	[`dot` / `dotc`], [Unconjugated $sum x_i y_i$ / conjugated $sum overline(x_i) y_i$.], [`dot(n, x, 1, y, 1)`],
	[`nrm2` / `asum`], [$2$-norm (overflow-safe) / $sum abs(x_i)$.], [`nrm2(n, x, 1)`],
	[`iamax`], [Index of the largest-magnitude element.], [`iamax(n, x, 1)`],
	[`swap` / `copy`], [Exchange / copy two strided vectors.], [`copy(n, x, 1, y, 1)`],
	api-group("level 2 — matrix/vector, O(n²)"),
	[`gemv(trans, ..)`], [$y <- alpha "op"(A) x + beta y$.], [`gemv(None, 1.0, A, x, 1, 0.0, y, 1)`],
	[`ger(..)`], [Rank-one update $A <- alpha x y^T + A$.], [`ger(1.0, x, 1, y, 1, A)`],
	[`trsv(uplo, trans, diag, ..)`], [Triangular solve $x <- "op"(A)^(-1) x$.], [`trsv(Upper, None, NonUnit, A, x, 1)`],
	[`symv(uplo, ..)`], [Symmetric $y <- alpha A x + beta y$.], [`symv(Lower, 1.0, A, x, 1, 0.0, y, 1)`],
	api-group("level 3 — matrix/matrix, O(n³)"),
	[`gemm(transA, transB, ..)`], [$C <- alpha "op"(A) "op"(B) + beta C$.], [`gemm(None, None, 1.0, A, B, 0.0, C)`],
	[`syrk(uplo, trans, ..)`], [Symmetric rank-$k$ update $C <- alpha "op"(A) "op"(A)^H + beta C$.], [`syrk(Upper, None, 1.0, A, 0.0, C)`],
	[`trsm(uplo, trans, diag, ..)`], [Triangular solve, many right-hand sides.], [`trsm(Left, None, NonUnit, 1.0, A, B)`],
	api-group("blocked-gemm internals (benchmarking)"),
	[`tunedBlockSizes()`], [The `BlockSizes` chosen for the current scalar.], [`Kernels<double>::tunedBlockSizes()`],
	[`packPanelA` / `packPanelB`], [Pack a panel into a contiguous, kernel-friendly layout.], [`packPanelA(a, buffer)`],
	[`microKernel(kc, ..)`], [The $m_r times n_r$ rank-$k_c$ register-tile update.], [`microKernel(kc, pa, pb, C)`],
))

```cpp
// C <- 1.0 * A * B + 0.0 * C   (beta == 0: C is write-only)
Matrix<double> C(A.rows(), B.cols());
Kernels<double>::gemm(Transposition::Kind::None, Transposition::Kind::None,
                      1.0, A.view(), B.view(), 0.0, C.view());
```

=== `Norm` — vector and matrix norms

Static routines separating the cheap entrywise/induced norms from the
$O(n^3)$ SVD-based ones, plus the diagnostics the correctness suite leans on.
The scaled $2$-norm avoids an intermediate $sum abs(x_i)^2$ that would overflow
for large entries — the same concern that shapes `Givens::FromPair`.

#api-methods((
	api-group("vector norms"),
	[`vectorOne` / `vectorInfinity`], [$sum abs(x_i)$ / $max abs(x_i)$.], [`Norm<double>::vectorOne(x)`],
	[`vectorTwo` / `vectorTwoScaled`], [Euclidean norm; the overflow-safe scaled variant.], [`Norm<double>::vectorTwoScaled(x)`],
	[`vectorP(x, p)`], [General $p$-norm.], [`Norm<double>::vectorP(x, 3.0)`],
	api-group("matrix norms"),
	[`matrixOne` / `matrixInfinity`], [Max absolute column / row sum.], [`Norm<double>::matrixOne(A)`],
	[`matrixFrobenius` / `matrixMax`], [Frobenius; largest entry magnitude.], [`Norm<double>::matrixFrobenius(A)`],
	[`matrixTwo` / `matrixNuclear`], [Spectral / nuclear norm (SVD; $O(n^3)$).], [`Norm<double>::matrixTwo(A)`],
	api-group("diagnostics"),
	[`distance(x, y)` / `relativeError(approx, exact)`], [$norm(x - y)$; $norm("approx" - "exact") slash norm("exact")$.], [`Norm<double>::distance(x, y)`],
	[`residualNorm(A, x, b)`], [$norm(b - A x)$.], [`Norm<double>::residualNorm(A, x, b)`],
	[`backwardError(A, x, b)`], [$norm(b - A x) slash (norm(A) norm(x) + norm(b))$.], [`Norm<double>::backwardError(A, x, b)`],
	[`orthogonalityDefect(Q)`], [$norm(Q^H Q - I)_F$.], [`Norm<double>::orthogonalityDefect(Q)`],
))

```cpp
double bwd = Norm<double>::backwardError(A, x, b);
// bwd ~ machine epsilon means x exactly solves a nearby system,
// regardless of how ill-conditioned A is.
```

=== `Householder` — reflectors

A reflector $H = I - beta v v^H$ stored implicitly as its essential vector
(the leading $1$ of $v$ is omitted) and a real $beta$. `FromVector` /
`FromColumn` build the reflector mapping $x$ to $plus.minus norm(x) e_1$,
choosing the sign _opposite_ $x_1$ (opposite its phase for complex `T`) so the
leading subtraction cannot cancel. Application is a two-pass $O(m n)$ update
that never forms $H$.

#api-methods((
	api-group("Householder — single reflector"),
	[`Householder()` / `Householder(essential, beta)`], [Identity reflector; construct from stored parts.], [`Householder<double>(tail, beta)`],
	[`FromVector(x)` / `FromColumn(a, col, startRow)`], [_(static)_ Build $H$ zeroing all but the first entry.], [`Householder<double>::FromVector(x)`],
	[`essential()` / `beta()` / `size()` / `isIdentity()`], [Stored tail, coefficient, dimension, no-op test.], [`h.beta()`],
	[`applyLeft` / `applyRight` / `applyLeftConjugate`], [$H A$, $A H$, $H^H A$ in place ($H$ never formed).], [`h.applyLeft(A.view())`],
	[`apply(x)`], [Reflect a vector in place.], [`h.apply(x)` $-> (-5, 0, 0)$],
	[`toMatrix(n)`], [Dense $H$ embedded in an identity (tests).], [`h.toMatrix(3)`],
	api-group("HouseholderSequence — aggregated Q"),
	[`HouseholderSequence()` / `(reflectors, betas)`], [Empty; or from packed reflector columns and betas.], [`HouseholderSequence<double>(V, betas)`],
	[`append(h)` / `count()`], [Add a reflector; how many are held.], [`seq.append(h);`],
	[`applyLeft` / `applyRight` / `applyLeftTranspose`], [$Q A$, $A Q$, $Q^H A$, applied through level-3 kernels.], [`seq.applyLeftTranspose(A.view())`],
	[`toMatrix(n)` / `firstColumns(n, k)`], [Full $Q$; its leading $k$ columns (thin $Q$).], [`seq.firstColumns(5, 3)`],
	[`blockV` / `blockT` / `buildBlockRepresentation(bs)`], [Compact-WY factors $Q = I - V T V^H$ and their (re)build.], [`seq.buildBlockRepresentation(32)`],
))

```cpp
Vector<double> x{3.0, 4.0, 0.0};
Householder<double> h = Householder<double>::FromVector(x);
h.apply(x);            // x -> (-5, 0, 0): the column is reflected onto e_1
```

=== `Givens` — plane rotations

The other orthogonal primitive, whose full per-method reference and worked
example are in @sec-givens-kernel[the worked-kernel section]: a unitary
$2 times 2$ rotation stored as $(c, s)$ plus its plane indices, used to zero
one entry at a time. `GivensSequence` is the ordered product of such rotations
— the accumulated sweep of a Jacobi eigenvalue pass or a bidiagonal chase.

= Design

Headers declare, source files define. Every class template is explicitly
instantiated in `src/` for the supported scalar list, so the project
builds as an ordinary compiled library with real separate compilation
instead of a header-only template dump. Unimplemented members throw a
`LinalgError` naming themselves, which makes the remaining work greppable
and turns the test suite into a red-to-green progress meter.

#remark(title: "Progress at a glance")[
	Every stub names itself when it throws, so the remaining work is one
	grep away — the per-file count doubles as a burndown chart:

	```sh
	grep -rc "not implemented" src/ | sort -t: -k2 -rn
	```
]

Correctness is checked against identities rather than golden files:
$P A = L U$, $Q^H Q = I$, residual and backward-error bounds sized by
machine epsilon and the condition number, ill-conditioned stress inputs
like the Hilbert matrix.

= Documentation

The project is documented at three levels, each aimed at a different reader
and kept close to the thing it describes.

- *API reference — in the headers.* Every declaration in `include/` carries
	a Doxygen-style comment block (`@brief`, `@param`, `@return`, `@throws`),
	so the headers double as the reference manual: the contract for a
	function sits directly above its signature, where it is hardest to let
	drift out of sync with the code. Because the headers are declaration-only,
	reading one is a tour of _what_ a component offers without the noise of
	_how_ it is implemented.

- *Mathematical notes — the "why".* A companion set of notes derives the
	algorithms and the rationale behind their computational choices —
	overflow-safe scaling, sign choices that avoid cancellation, deflation
	criteria — at a depth the source comments deliberately leave out. They are
	linked from the top of this document and cross-referenced from the code
	where a derivation is load-bearing (see, for instance, the `dlartg`
	reference in `Givens::FromPair` and the structured-decomposition note in
	`Givens::inverse`).

- *This report — the design narrative.* The high-level story of how the
	tiers fit together, why the kernels are the single tuning surface, and how
	a representative kernel maps to its math. It is written in Typst and lives
	beside the source it describes:

	```sh
	typst compile docs/main.typ
	```

#remark(title: "One convention, three views")[
	The three layers share a single organizing idea — _the header layout is
	the table of contents_. The directory tiers of @sec-overview[the
	architecture diagram] name the API sections, the module reference walks those same
	files, and the notes are indexed by the same components. A reader can enter
	at any level and cross to the others without relearning the map.
]

= Directions for Expansion

Expanding this library to include methods of large-scale and structured
computation is the natural next step, in rough order of ambition:

- *Iterative Krylov methods* (already scaffolded): conjugate gradients,
	GMRES with restarting, BiCGSTAB, and LSQR, with Jacobi, SSOR, and
	incomplete-Cholesky preconditioners — the bridge from dense $O(n^3)$
	factorizations to problems where only matrix–vector products are
	affordable.

- *Iterative eigensolvers:* power iteration, Lanczos with selective
	reorthogonalization, and implicitly restarted Arnoldi for a few
	eigenpairs of large matrices.

- *Performance engineering:* cache-blocked and packed `gemm` with a
	register-tiled micro-kernel, blocked variants of LU/Cholesky/QR, and
	compact-WY application of reflector sequences, benchmarked against the
	naive implementations the correctness tests already validate.

- *Sparse storage and operations,* which the iterative methods are
	deliberately shaped to exploit (they touch the matrix only through
	products and preconditioner applications).

= Building

```sh
cmake -B build -DLINALG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```
