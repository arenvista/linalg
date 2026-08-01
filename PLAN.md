# Implementation Plan

The headers are done and every body in `src/` is a throwing stub, so "building
the library" means filling in `src/*.cpp` files in an order where each file
only depends on things you have already implemented. This is that order, with
notes on what inside each file to defer.

Workflow for every step:

```sh
cmake --build build && ctest --test-dir build
grep -rc "not implemented" src/ | sort -t: -k2 -rn   # what's left
```

`tests/instantiate.cpp` only checks that everything links. Add a real test
file per phase (register it in `tests/CMakeLists.txt`) and verify with
mathematical identities: `P*A == L*U`, `Q^H*Q == I`, `A*x - b` small, etc.
`Matrix::Hilbert` gives you a famously ill-conditioned matrix for stress
tests; `RandomSymmetric` / `RandomOrthogonal` give well-behaved inputs.

---

## Phase 0 — Foundations

### 1. `src/core/Traits.cpp`

Leaf dependency; everything asks it questions. Two specializations
(real, `std::complex`), all one-liners over `<cmath>`/`<complex>` and
`std::numeric_limits`. Watch: for real `T`, `conj` is identity and `imag`
is zero; `absSquared` should not call `sqrt` (it's `x*x` / `norm(z)`).

### 2. `src/core/MatrixView.cpp`

`MatrixView` + `ConstMatrixView`. Pure pointer-and-stride arithmetic:
element `(i,j)` lives at `data_[i*rowStride_ + j*colStride_]`. Get
`transposed()` right (swap rows/cols _and_ strides) — the decompositions
lean on it. `toMatrix()` needs `Matrix`, so leave that one member stubbed
until step 3; everything else here is self-contained.

### 3. `src/core/Vector.cpp` and `src/core/Matrix.cpp` (together)

They reference each other (`outer`, `row`, `asDiagonalMatrix`, ...), so
implement in one sitting. This is the bulk of the mechanical work: storage,
element access, elementwise arithmetic, naive `operator*`, blocks/views,
concat/reshape, predicates, `toString`/CSV.

Defer these members — they depend on decompositions you don't have yet
(leave the stub throw in place, come back in Phase 3/4):

- `Matrix`: `determinant`, `inverse`, `pseudoInverse`, `rank`,
  `spectralNorm`, `conditionNumber`, `RandomOrthogonal` (build it as QR of
  `Random` later).
- `Vector`: nothing — fully implementable now. Do `norm()` with scaling
  (or via `hypot`-style accumulation) so `Vector{1e200, 1e200}` doesn't
  overflow; complex `hermitianDot` conjugates the _left_ argument —
  pick a convention and test it.

Checkpoint test: construction, arithmetic identities
(`(A+B)*x == A*x + B*x`), transpose round-trip, block writes visible in
the parent matrix.

---

## Phase 1 — Kernels and reflectors

### 4. `src/ops/Kernels.cpp`

Naive versions only: triple loop `gemm`, straightforward level 1/2. Skip
`tunedBlockSizes` / `packPanelA` / `packPanelB` / `microKernel` — that's
Phase 6 tuning surface. Watch: `dotc` conjugates `x`; `gemv` with
`ConjugateTranspose` must conjugate as it goes; respect `beta == 0`
(don't read possibly-uninitialized `y`/`C`).

Once `gemm` works, go back and reroute `Matrix::operator*` through it so
every later speedup is free.

### 5. `src/ops/Norm.cpp`

Mostly reductions over what you built in step 3. `vectorTwoScaled` is the
point of the file: two-pass (find max, then scale) or LAPACK-style running
rescale. Defer `matrixTwo` and `matrixNuclear` (need SVD, Phase 4).

### 6. `src/ops/Householder.cpp`

The single most load-bearing algorithm in the tree — QR, Hessenberg,
Tridiagonal, Bidiagonal, and SVD all reduce to it. `FromVector`: choose the
sign of `||x||` to avoid cancellation (`-sign(x_1)||x||`); for complex
scalars beta is real but the phase matters — test with complex inputs
early. `applyLeft` is `target -= beta * v * (v^H * target)` — two passes,
never form `H`. `HouseholderSequence::buildBlockRepresentation` (WY form)
can wait until Phase 6.

Test: `H*H == I`, `H*x` lands on `±||x|| e_1`, `toMatrix` agrees with
`applyLeft` on the identity.

### 7. `src/ops/Givens.cpp`

`FromPair` must be robust to overflow (LAPACK `dlartg`-style, not naive
`a/sqrt(a²+b²)`). `applyLeft` touches exactly two rows — keep it O(n).

---

## Phase 2 — Triangular solves and the first factorizations

### 8. `src/solve/TriangularSolver.cpp`

Forward/back substitution with the four option axes (uplo, unit diagonal,
transposition, blocked — skip blocked for now). Every direct solver bottoms
out here, so test all option combinations against explicit inverses of
small triangular matrices.

### 9. `src/decomp/LU.cpp`

`unblockedFactorize` with partial pivoting; `solve` = permute + two
triangular solves; determinant = product of diagonal × pivot sign. Defer
`blockedFactorize` (Phase 6) and `reciprocalConditionEstimate` (needs a
condition estimator — Hager/Higham 1-norm estimate, can also wait).
`FullPivLU` afterward: same loop, two pivot searches, gives you `rank`,
`kernel`, `image`.

Test: `P*A == L*U` on random matrices, `solve` residual ~ eps·cond,
singular matrix triggers the `throwOnSingular` path.

### 10. `src/decomp/Cholesky.cpp`

Easier than LU (no pivoting) — order is just because LU tests your
triangular solver harder first. Failure of the positive-definite check is
a _result_ (`isPositiveDefinite`), only optionally a throw. `LDLT` next.
Defer `update`/`downdate` (rank-one modification, hyperbolic rotations)
until you want them; they're independent of everything else.

### 11. `src/decomp/QR.cpp`

The payoff of Phase 1: the Householder loop over columns, factors stored
packed, Q implicit in the reflector sequence. `solve`/`solveLeastSquares`
= apply `Q^H`, back-substitute against `R`. Then `ColPivQR` (norm-tracking
pivot selection, with norm-downdate re-check) and `GramSchmidt` (easy; do
all three variants, compare `orthogonalityLoss` — that's the point of it).
Defer `appendRow`/`appendColumn`/`removeRow`/`rankOneUpdate` (QR updating
is fiddly Givens work; nothing downstream needs it).

### 12. Back-fill `Matrix` members that needed LU/QR

`determinant`, `inverse` (LU), `rank` (ColPivQR), `RandomOrthogonal`
(Q of QR of Random). Still deferred: `pseudoInverse`, `spectralNorm`,
`conditionNumber` (SVD, next phase).

---

## Phase 3 — Condensed forms

### 13. `src/decomp/Hessenberg.cpp`

Three similarity/orthogonal reductions, all the same Householder pattern
with two-sided application: `Hessenberg` (general → upper Hessenberg),
`Tridiagonal` (Hermitian → tridiagonal; the Sturm-count
`eigenvalueCountBelow` is a small standalone loop), `Bidiagonal`
(alternate left/right reflectors; `usedRPreprocessing` = QR-first when
m ≫ n). Test: `Q * H * Q^H == A` and structure (zeros where zeros belong).

---

## Phase 4 — Eigenvalues and SVD (the hard part)

### 14. `src/decomp/Eigen.cpp` — `SymmetricEigen` first

Tridiagonalize (step 13), then implicit QL/QR with Wilkinson shifts on the
tridiagonal — bulge-chasing with Givens rotations (step 7), accumulating
into the eigenvector matrix. This is the gentlest introduction to
implicit-shift iterations; get it solid before Schur. Start with
`Algorithm::ImplicitQR` only; Jacobi is a nice cross-check second
implementation, defer DivideAndConquer/Bisection unless you want them.

### 15. `src/decomp/SVD.cpp`

Golub–Kahan: bidiagonalize (step 13), then implicit-shift QR sweeps on the
bidiagonal, with the standard deflation tests. Singular values sorted
descending, non-negative — that's `sortSingularValues`. Everything else in
the class (`pseudoInverse`, `solveTikhonov`, `lowRankApproximation`,
`nullSpace`, polar factors) is easy arithmetic on U, S, V once `compute`
converges. Jacobi SVD as optional cross-check.

Test: `U*S*V^H == A`, `U`/`V` orthogonal, singular values of a diagonal
matrix are `|diag|` sorted, Hilbert matrix condition number matches
literature.

### 16. `src/decomp/Eigen.cpp` — `Schur`, then `GeneralEigen`

The hardest code in the tree. Reduce to Hessenberg, then Francis
double-shift QR with deflation. Note the trap: the double-shift trick is
for _real_ scalars; for the complex instantiations use single-shift
complex QR (same `francisDoubleShiftSweep` entry point, branch on
`IsComplex`). `GeneralEigen` then back-solves the quasi-triangular T for
eigenvectors and un-balances. `balance()` and `reorder()` can come last.

### 17. Back-fill the rest

`Matrix::spectralNorm` / `conditionNumber` / `pseudoInverse`,
`Norm::matrixTwo` / `matrixNuclear`, `reciprocalConditionEstimate`s.
After this: `grep "not implemented" src/decomp src/core src/ops` should
be near-silent.

---

## Phase 5 — High-level solvers and iterative methods

### 18. `src/solve/LinearSolver.cpp`

Pure dispatch over Phases 2–4: probe symmetry/Hermitian-ness, try
Cholesky, fall back to LDLT/LU/QR; fill in the `Report`; iterative
refinement is a short residual loop reusing the factorization.

### 19. `src/solve/LeastSquares.cpp`

`LeastSquares` = thin wrappers over QR/ColPivQR/SVD (NormalEquations
included only to demonstrate why not). `solveConstrained` is the one real
algorithm (nullspace method via QR of the constraints). `GaussNewton` /
Levenberg–Marquardt: a damping loop where each step is a `LeastSquares`
solve on the Jacobian.

### 20. `src/iterative/Preconditioner.cpp`

Identity and Jacobi are trivial; SSOR and IC(0) are short factorization
loops + triangular applies. Needed before the Krylov solvers since they
hold a `Preconditioner*`.

### 21. `src/iterative/KrylovSolver.cpp`

Base-class bookkeeping first (convergence check, history). Then in order
of difficulty: `ConjugateGradient` (textbook PCG; store the alphas/betas
for the Lanczos-diagnostics accessors), `BiCGSTAB`, `GMRES` (Arnoldi +
stored Givens rotations on the growing Hessenberg — reuse step 7 —

- restarting), `LSQR` (Golub–Kahan bidiagonalization recurrence).

Test against your own direct solvers: same `A`, `b`, compare answers; CG
on SPD matrices converges in ≤ n iterations in exact arithmetic.

### 22. `src/iterative/EigenSolver.cpp`

`PowerIteration` (trivial), `Lanczos` (three-term recurrence, full
reorthogonalization first — `None`/`Selective`/`Partial` are refinements;
Ritz values via `SymmetricEigen` on the projected tridiagonal),
`ImplicitlyRestartedArnoldi` (uses `Schur` on the projected Hessenberg;
the implicit-restart QR sweep is the tricky bit — do it last).

---

## Phase 6 — Performance (optional, orthogonal)

Only after everything is green, and driven by `benchmarks/`:

1. `Kernels::tunedBlockSizes` / `packPanelA` / `packPanelB` /
   `microKernel`, then a blocked `gemm` on top.
2. `blockedFactorize` in LU, Cholesky, QR (panel factorization + trailing
   `gemm`/`trsm` update) and blocked `TriangularSolver`.
3. `HouseholderSequence::buildBlockRepresentation` (compact WY) so QR and
   the condensed forms apply Q with level-3 kernels.

Nothing else in the tree changes: correctness tests from earlier phases
are your regression suite for all of this.

---

## Dependency map (why this order)

```
Traits ─→ MatrixView ─→ Matrix/Vector ─→ Kernels ─→ Norm
│
├─→ Householder, Givens
│        │
├─→ TriangularSolver
│        │
│   LU, Cholesky, QR ──→ LinearSolver,
│        │               LeastSquares
│   Hessenberg/Tridiag/Bidiag
│        │
│   SymmetricEigen, SVD, Schur
│
└─→ Preconditioner ─→ KrylovSolver
(EigenSolver uses SymmetricEigen/Schur)
```

## Standing reminders

- Every file's `LINALG_INSTANTIATE` block already exists — you're only
  replacing function bodies, never touching headers or instantiation lists.
- Everything is instantiated for `std::complex<float>/<double>` too. Write
  `NumericTraits<T>::conj(...)` instead of assuming symmetry, and test each
  phase with a complex scalar before moving on — retrofitting conjugates
  into a working real-only decomposition is miserable.
- Compare against a reference when in doubt: NumPy/SciPy in a throwaway
  script gives you expected factors and eigenvalues for any small test
  matrix.
- Golub & Van Loan, _Matrix Computations_, is effectively the spec for
  Phases 1–4; the LAPACK Users' Guide documents the edge-case handling
  (scaling in `nrm2`/`lartg`, deflation criteria) these headers imply.
