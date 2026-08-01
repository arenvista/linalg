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

= What the Library Covers

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

== The Kernels as the Tuning Surface

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
