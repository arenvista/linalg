#pragma once

#include <complex>

// The scalar types the library is compiled for.
//
// Definitions live in src/, not in the headers, so a type absent from this
// list will compile but fail to link. Add it here and rebuild to support it.
//
// Used two ways: headers declare `extern template` to suppress implicit
// instantiation, and each source file emits the matching definitions.

#define LINALG_FOR_EACH_SCALAR(MACRO) \
    MACRO(float)                      \
    MACRO(double)                     \
    MACRO(std::complex<float>)        \
    MACRO(std::complex<double>)
