cmake_minimum_required(VERSION 3.20)

# Must be set before project() to take effect on the first configure.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(linalg
    VERSION 0.1.0
    DESCRIPTION "A templated dense linear algebra library"
    LANGUAGES CXX)

# A normally compiled library. Templates are defined in src/ and explicitly
# instantiated there for the scalars listed in include/linalg/Instantiate.hpp,
# so headers carry declarations only.
add_library(linalg STATIC
    src/core/Exceptions.cpp
    src/core/Matrix.cpp
    src/core/MatrixView.cpp
    src/core/Traits.cpp
    src/core/Vector.cpp
    src/ops/Givens.cpp
    src/ops/Householder.cpp
    src/ops/Kernels.cpp
    src/ops/Norm.cpp
    src/decomp/Cholesky.cpp
    src/decomp/Eigen.cpp
    src/decomp/Hessenberg.cpp
    src/decomp/LU.cpp
    src/decomp/QR.cpp
    src/decomp/SVD.cpp
    src/solve/LeastSquares.cpp
    src/solve/LinearSolver.cpp
    src/solve/TriangularSolver.cpp
    src/iterative/EigenSolver.cpp
    src/iterative/KrylovSolver.cpp
    src/iterative/Preconditioner.cpp)

# Stub bodies name their parameters without using them. Drop this once
# real implementations replace the throws.
target_compile_options(linalg PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
add_library(linalg::linalg ALIAS linalg)

target_include_directories(linalg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(linalg PUBLIC cxx_std_17)

option(LINALG_BUILD_TESTS "Build the test suite" OFF)
option(LINALG_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(LINALG_HEADER_SELF_TEST "Compile each header on its own" ON)
option(LINALG_ENABLE_NATIVE "Compile with -march=native" OFF)

if(LINALG_ENABLE_NATIVE)
    target_compile_options(linalg PUBLIC -march=native)
endif()

# One generated translation unit per header. Two jobs: it catches headers
# that only compile because something else included their dependencies
# first, and it gives CMAKE_EXPORT_COMPILE_COMMANDS something real to
# write, since an INTERFACE target compiles nothing on its own.
if(LINALG_HEADER_SELF_TEST)
    file(GLOB_RECURSE LINALG_HEADERS
        CONFIGURE_DEPENDS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

    set(LINALG_SELF_TEST_SOURCES "")
    foreach(header IN LISTS LINALG_HEADERS)
        string(REGEX REPLACE "[/.]" "_" stem "${header}")
        set(generated "${CMAKE_CURRENT_BINARY_DIR}/header_self_test/${stem}.cpp")
        file(WRITE "${generated}" "#include \"${header}\"\n")
        list(APPEND LINALG_SELF_TEST_SOURCES "${generated}")
    endforeach()

    add_library(linalg_header_self_test OBJECT ${LINALG_SELF_TEST_SOURCES})
    target_link_libraries(linalg_header_self_test PRIVATE linalg::linalg)
    # Stub bodies name their parameters without using them. Drop the
    # suppression once real implementations replace the throws.
    target_compile_options(linalg_header_self_test PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
endif()

# Put the database where clangd looks by default, without a wrapper script.
if(CMAKE_EXPORT_COMPILE_COMMANDS AND NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC)
endif()

if(LINALG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LINALG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(GNUInstallDirs)
install(TARGETS linalg EXPORT linalgTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
