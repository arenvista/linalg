# linalg

A dense numerical linear algebra library in C++17, implemented from first principles rather than by wrapping
BLAS/LAPACK. It supports `float`, `double`, and their complex counterparts uniformly, through a single scalar
abstraction.

The design goal is a correct, readable, and reasonably efficient library that works through the algorithms taught in a
numerical linear algebra course — and gets the hard parts right: stability-driven algorithm choices, floating-point edge
cases (overflow-safe norms, cancellation-free reflectors, deflation criteria), and verification against the
mathematical identities each component must satisfy. The design narrative lives in
[`docs/main.typ`](docs/main.typ) (`typst compile docs/main.typ`).

## Status

The foundation tiers are implemented and green under `ctest`; the higher decompositions and iterative solvers are
scaffolded and being filled in. Concretely:

**Implemented and tested**

- **Core types** — owning `Matrix` / `Vector` and non-owning stride-aware `MatrixView`, over a `NumericTraits` scalar
  abstraction that handles real and complex uniformly.
- **Kernels** — reference BLAS-shaped level 1/2/3 routines (`gemm`, `gemv`, `trsm`, `syrk`, …).
- **Orthogonal primitives** — Householder reflectors and Givens rotations, stored implicitly and applied in `O(mn)`,
  with the numerical care described [below](#numerical-care).
- **Triangular solves** (`TriangularSolver`) and **Cholesky** (`LDL^H` in progress).

**In progress** — LU, QR, the condensed forms (Hessenberg/tridiagonal/bidiagonal), the symmetric/general eigen and SVD
routines, and the iterative Krylov and eigen solvers. Their members throw `LinalgError` until implemented, so an
unfinished call fails loudly rather than returning garbage. The remaining work is one grep away — the per-file count
doubles as a burndown chart:

```sh
grep -rc "not implemented" src/ | sort -t: -k2 -rn
```

See the [roadmap](#roadmap) for the per-component state.

## Example

```cpp
#include "linalg/linalg.hpp"
using namespace linalg;

Matrix<double> A{{4.0, 1.0},
                 {1.0, 3.0}};
Vector<double> x{3.0, 4.0};

double n = x.norm();            // 5.0, computed overflow-safe (no intermediate sum of squares)
Vector<double> b = A * x;       // matrix-vector product routed through the gemv kernel

// Zero A(1,0) with a Givens rotation; the new diagonal entry is exactly g.radius().
Givens<double> g = Givens<double>::FromPair(A(0, 0), A(1, 0), 0, 1);
g.applyLeft(A.view());          // A(1,0) == 0; A(0,0) == g.radius()

// Householder reflection onto e1, with the sign chosen so the leading subtraction can't cancel.
Vector<double> v{3.0, 4.0, 0.0};
Householder<double> h = Householder<double>::FromVector(v);
h.apply(v);                     // v -> (-5, 0, 0)
```

Every declaration in `include/` carries a Doxygen-style comment (`@brief`, `@param`, `@throws`), so the headers double
as the reference manual.

## Numerical care

Two examples of the stability work that separates this from a naive triple-loop implementation:

- **`Householder::FromVector`** accumulates the tail norm directly instead of forming `‖x‖² − |α|²`, and chooses the
  reflection sign *opposite* the leading entry's phase (for complex scalars too) so the subtraction `α − r` never loses
  significance to cancellation.
- **`Givens::FromPair`** follows LAPACK's `dlartg`: it scales by the larger magnitude before squaring, so the one square
  root cannot over- or underflow even for inputs near `1e±200` in double precision, where the naive `sqrt(a² + b²)`
  would.

The same overflow-safe scaling shapes `Norm::vectorTwoScaled`. These choices are derived in the companion notes and
cross-referenced from the code where a derivation is load-bearing.

## Architecture

The library is organized in dependency tiers, each written only in terms of the tiers below it:

```
solve/       direct front doors        LinearSolver  LeastSquares  TriangularSolver
iterative/   large-scale methods       KrylovSolver  EigenSolver   Preconditioner
      │  depends on
decomp/      factorizations            LU  Cholesky  QR  Hessenberg  SVD  Eigen
      │  depends on
ops/         kernels & primitives      Kernels  Norm  Householder  Givens
      │  depends on
core/         types & scalars          Matrix  Vector  MatrixView  NumericTraits  Exceptions
```

Because each tier depends only on the ones below, tuning the `ops` kernels accelerates every factorization and solver
above without touching their code — the kernels are the single tuning surface.

### Layout

```
include/linalg/    declarations only, mirroring the tiers above
src/               definitions + explicit instantiation, one .cpp per header
tests/             instantiate.cpp and the identity-based correctness suite
docs/main.typ      the design narrative
```

## Design: real separate compilation

Headers declare, source files define. Every declaration lives inside a class or struct — no free functions, no
namespace-scope enums — and every class template is explicitly instantiated in the matching `src/*.cpp` for the
supported scalar list. So the project compiles as an ordinary library with real separate compilation rather than a
header-only template dump: touching one body rebuilds one object file, not every translation unit that includes the
header.

```
include/linalg/decomp/QR.hpp     declarations
src/decomp/QR.cpp                definitions + explicit instantiation
```

The instantiation list is driven by one macro, `LINALG_FOR_EACH_SCALAR`, defined in `include/linalg/Instantiate.hpp`,
currently covering `float`, `double`, `std::complex<float>`, and `std::complex<double>`. Adding a scalar is a one-line
change there; anything outside the list compiles but fails to *link*. Each header also carries the matching
`extern template` declarations, so including a header never triggers an implicit instantiation and the supported scalar
set is visible where you read the interface.

`tests/instantiate.cpp` odr-uses one member of every class template for all four scalars, which keeps the instantiation
lists honest: a class that is declared and defined but never instantiated fails the test link, naming the missing
symbol.

## Correctness

Correctness is checked against identities, not golden files: `PA = LU`, `Q^H Q = I`, residual and backward-error bounds
sized by machine epsilon and the condition number, and ill-conditioned stress inputs like the Hilbert matrix. The CI
matrix builds under gcc and clang; note that `RelWithDebInfo` defines `NDEBUG`, which would make the assert-based tests
pass vacuously, so the test builds keep asserts live.

## Build

```sh
cmake -B build -DLINALG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

`LINALG_BUILD_SELF_CONTAINED` defaults to ON, so a bare `cmake --build build` already compiles every header in
isolation. Tests for unimplemented components report `SKIP` (exit 77) rather than failing, so the suite stays green as
the stubs are filled in.

## Roadmap

Build order, tracked in [`PLAN.md`](PLAN.md); each item depends only on the ones above it. Status: `[x]` done ·
`[o]` in progress · `[ ]` not started.

### Phase 0 — Foundations

| Status | Component  | File                      | Notes                                          |
| ------ | ---------- | ------------------------- | ---------------------------------------------- |
| `[x]`  | Traits     | `src/core/Traits.cpp`     | —                                              |
| `[x]`  | MatrixView | `src/core/MatrixView.cpp` | —                                              |
| `[x]`  | Exceptions | `src/core/Exceptions.cpp` | —                                              |
| `[x]`  | Vector     | `src/core/Vector.cpp`     | —                                              |
| `[o]`  | Matrix     | `src/core/Matrix.cpp`     | core members done; decomposition-backed summaries deferred |

### Phase 1 — Kernels and reflectors

| Status | Component   | File                      | Notes                                                   |
| ------ | ----------- | ------------------------- | ------------------------------------------------------- |
| `[x]`  | Kernels     | `src/ops/Kernels.cpp`     | reference level 1/2/3; blocked internals are Phase 6    |
| `[o]`  | Norm        | `src/ops/Norm.cpp`        | vector & entrywise done; `matrixTwo`/`matrixNuclear` need SVD |
| `[x]`  | Householder | `src/ops/Householder.cpp` | reflector done; compact-WY (Phase 6) deferred           |
| `[x]`  | Givens      | `src/ops/Givens.cpp`      | —                                                       |

### Phase 2 — Triangular solves and first factorizations

| Status | Component        | File                             | Notes                                                |
| ------ | ---------------- | -------------------------------- | ---------------------------------------------------- |
| `[x]`  | TriangularSolver | `src/solve/TriangularSolver.cpp` | —                                                    |
| `[o]`  | Cholesky         | `src/decomp/Cholesky.cpp`        | factorization done; `LDLT` in progress               |
| `[o]`  | LU               | `src/decomp/LU.cpp`              | + `FullPivLU`                                        |
| `[o]`  | QR               | `src/decomp/QR.cpp`              | + `ColPivQR`, `GramSchmidt`                          |
| `[o]`  | Back-fill Matrix | `src/core/Matrix.cpp`            | `determinant`, `inverse`, `rank`, `RandomOrthogonal` |

### Phase 3 — Condensed forms

| Status | Component                             | File                        | Notes |
| ------ | ------------------------------------- | --------------------------- | ----- |
| `[ ]`  | Hessenberg / Tridiagonal / Bidiagonal | `src/decomp/Hessenberg.cpp` | —     |

### Phase 4 — Eigenvalues and SVD

| Status | Component                | File                   | Notes                                                                                 |
| ------ | ------------------------ | ---------------------- | ------------------------------------------------------------------------------------- |
| `[ ]`  | SymmetricEigen           | `src/decomp/Eigen.cpp` | —                                                                                     |
| `[ ]`  | SVD                      | `src/decomp/SVD.cpp`   | —                                                                                     |
| `[ ]`  | Schur, then GeneralEigen | `src/decomp/Eigen.cpp` | —                                                                                     |
| `[ ]`  | Back-fill                | —                      | `spectralNorm`, `conditionNumber`, `pseudoInverse`, `Norm::matrixTwo`/`matrixNuclear` |

### Phase 5 — High-level and iterative solvers

| Status | Component      | File                               | Notes                         |
| ------ | -------------- | ---------------------------------- | ----------------------------- |
| `[o]`  | LinearSolver   | `src/solve/LinearSolver.cpp`       | dispatch scaffolded           |
| `[o]`  | LeastSquares   | `src/solve/LeastSquares.cpp`       | —                             |
| `[ ]`  | Preconditioner | `src/iterative/Preconditioner.cpp` | Jacobi, SSOR, incomplete-Chol |
| `[ ]`  | KrylovSolver   | `src/iterative/KrylovSolver.cpp`   | CG, BiCGSTAB, GMRES, LSQR     |
| `[ ]`  | EigenSolver    | `src/iterative/EigenSolver.cpp`    | PowerIteration, Lanczos, IRAM |

### Phase 6 — Performance (optional, orthogonal)

| Status | Component              | Notes                                                               |
| ------ | ---------------------- | ------------------------------------------------------------------- |
| `[ ]`  | Blocked kernels        | `tunedBlockSizes` / `packPanel*` / `microKernel` + blocked `gemm`   |
| `[ ]`  | Blocked factorizations | LU / Cholesky / QR panel factorization + blocked `TriangularSolver` |
| `[ ]`  | Compact WY             | `HouseholderSequence::buildBlockRepresentation`                     |

## License

Not yet chosen — a license file still needs to be added before this is reusable.
