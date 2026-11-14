# linalg

A dense numerical linear algebra library in C++17, implemented from first principles rather than by wrapping
BLAS/LAPACK. It supports `float`, `double`, and their complex counterparts uniformly, through a single scalar
abstraction.

The goal is a correct, readable, and reasonably efficient library that works through the algorithms taught in a
numerical linear algebra course, and gets the hard parts right: stability-driven algorithm choices, floating-point edge
cases (overflow-safe norms, cancellation-free reflectors, deflation criteria), and verification against the mathematical
identities each component must satisfy. The routines follow the course outline in
[`docs/NumericalLinearAlgebra.pdf`](docs/NumericalLinearAlgebra.pdf).

## Status

Core types, the BLAS-shaped kernels, Householder and Givens primitives, triangular solves, and Cholesky are implemented
and tested; `LDL^H` is in progress. LU, QR, the condensed forms, eigen/SVD, and the iterative solvers are scaffolded:
their members throw `LinalgError` until implemented, so an unfinished call fails loudly instead of returning garbage.

Per-component status lives in [`ROADMAP.md`](ROADMAP.md). The remaining work per file:

```sh
grep -rc "not implemented" src/ | sort -t: -k2 -rn
```

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

## Build

Requires CMake 3.20+ and a C++17 compiler (CI builds with gcc and clang).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLINALG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Build tests as `Debug`: `Release` and `RelWithDebInfo` define `NDEBUG`, which compiles out the asserts the suite fails
through, so the tests would pass vacuously. Tests for unimplemented components report `SKIP` (exit 77) rather than
failing, so the suite stays green as the stubs are filled in.

| Option                    | Default | Effect                                                    |
| ------------------------- | ------- | --------------------------------------------------------- |
| `LINALG_BUILD_TESTS`      | `OFF`   | Build the `ctest` suite                                   |
| `LINALG_BUILD_BENCHMARKS` | `OFF`   | Build the benchmarks                                      |
| `LINALG_HEADER_SELF_TEST` | `ON`    | Compile every header on its own to catch missing includes |
| `LINALG_ENABLE_NATIVE`    | `OFF`   | Compile with `-march=native`                              |

The build exports `compile_commands.json` and links it into the source root for clangd.

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
core/        types & scalars           Matrix  Vector  MatrixView  NumericTraits  Exceptions
```

Headers in `include/linalg/` declare; the matching `src/*.cpp` defines and explicitly instantiates each class template
for the scalars listed in `LINALG_FOR_EACH_SCALAR` (`include/linalg/Instantiate.hpp`). This gives real separate
compilation instead of a header-only template dump: touching one body rebuilds one object file. A scalar outside the
list compiles but fails to link, and `tests/instantiate.cpp` fails to link if a class template is missing from the list.

```
include/linalg/    declarations only, mirroring the tiers above
src/               definitions + explicit instantiation, one .cpp per header
tests/             instantiate.cpp and the identity-based correctness suite
benchmarks/        optional benchmarks (LINALG_BUILD_BENCHMARKS)
docs/              design notes (Typst) and the course outline
```

Correctness is checked against identities rather than golden files: `PA = LU`, `Q^H Q = I`, residual and backward-error
bounds sized by machine epsilon and the condition number, and ill-conditioned inputs such as the Hilbert matrix.

## Documentation

- [`docs/main.typ`](docs/main.typ) is the design narrative: the algorithm choices and the numerical derivations the code
  cross-references.
- [`docs/documentation.typ`](docs/documentation.typ) documents the Typst template the notes are written with.

Build both PDFs with [Typst](https://typst.app) from inside `docs/`:

```sh
cd docs && ./compile.sh
```

## License

Not yet chosen. A license file still needs to be added before this is reusable.
