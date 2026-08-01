# linalg

A skeleton for a templated dense linear algebra library in C++17.

Headers hold declarations only: every declaration lives inside a class or
struct, with no free functions and no namespace-scope enums. Definitions
live in `src/` and are explicitly instantiated there, so this builds as an
ordinary compiled library rather than a header-only one. The intent is
that you fill in implementations one class at a time and watch the test
suite go from red to green.

## Layout

```
include/linalg/
  core/            Traits, Exceptions, Matrix, Vector, MatrixView
  ops/             Kernels (BLAS-shaped), Norm, Householder, Givens
  decomp/          LU, Cholesky, QR, Hessenberg, SVD, Eigen
  solve/           TriangularSolver, LinearSolver, LeastSquares
  iterative/       Preconditioner, KrylovSolver, EigenSolver
  Instantiate.hpp  the scalar list
  linalg.hpp       umbrella header
src/               definitions, one .cpp per header
tests/             instantiate.cpp
```

## Where the definitions live

In `src/`, compiled once, mirroring the header layout:

```
include/linalg/decomp/QR.hpp     declarations
src/decomp/QR.cpp                definitions + explicit instantiation
```

So `QR<T>::compute` is in `src/decomp/QR.cpp`. Replace the `throw` there
with the Householder loop and rebuild; nothing else moves.

This works despite templates needing visible definitions because each
source file ends with an explicit instantiation list:

```cpp
#define LINALG_INSTANTIATE(SCALAR) \
    template class QR<SCALAR>;     \
    template class ColPivQR<SCALAR>;

LINALG_FOR_EACH_SCALAR(LINALG_INSTANTIATE)
```

`LINALG_FOR_EACH_SCALAR` is defined once, in
`include/linalg/Instantiate.hpp`, and currently covers `float`, `double`,
`std::complex<float>`, and `std::complex<double>`. Adding a scalar is a
one-line change there. Anything outside the list compiles but fails to
link:

```
undefined reference to `linalg::Matrix<long double>::rows() const'
```

That is the tradeoff for real separate compilation. Touching one body
rebuilds one object file rather than every translation unit that includes
the header.

Each header also carries the matching `extern template` declarations, so
including it never triggers an implicit instantiation, and the supported
scalar set is visible where you read the interface.

## Stub bodies

Every member throws `LinalgError` naming itself, so an unimplemented call
fails loudly rather than returning garbage. It also doubles as a to-do
list:

```sh
grep -rc "not implemented" src/ | sort -t: -k2 -rn
```

The move constructors and move assignments are the exception. They are
declared `noexcept`, so a throwing stub would call `terminate` rather than
propagate; those bodies are empty with a TODO instead.

`src/core/Exceptions.cpp` is the one file already fully implemented, since
the exception hierarchy is the only non-template code in the tree.

## Adding a class

Three places, in order:

1. Declare it in the relevant header.
2. Define its members in the matching `src/*.cpp`.
3. Add `template class NewClass<SCALAR>;` to that file's
   `LINALG_INSTANTIATE` block, and `extern template` to the header's
   `LINALG_EXTERN` block.

Forget step 3 and `tests/instantiate.cpp` fails to link, naming the
missing symbol. That test odr-uses one member of every class template for
all four scalars, which is what keeps the lists honest.

## Build

```sh
cmake -B build -DLINALG_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

`LINALG_BUILD_SELF_CONTAINED` defaults to ON, so a bare `cmake --build build`
already compiles every header in isolation.

Note that the headers parse but do not link: every member is declared
and none is defined.
