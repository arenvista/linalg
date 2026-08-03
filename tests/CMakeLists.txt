# Verifies every class template appears in an explicit instantiation list.
# A missing entry shows up here as a link error rather than in user code.
add_executable(test_instantiate instantiate.cpp)
target_link_libraries(test_instantiate PRIVATE linalg::linalg)
target_compile_options(test_instantiate PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME instantiate COMMAND test_instantiate)


# Behavioural tests for the exception hierarchy.
add_executable(test_exceptions test_exceptions.cpp)
target_link_libraries(test_exceptions PRIVATE linalg::linalg)
target_compile_options(test_exceptions PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME exceptions COMMAND test_exceptions)

# Behavioural tests for NumericTraits and IsComplex.
add_executable(test_traits test_traits.cpp)
target_link_libraries(test_traits PRIVATE linalg::linalg)
target_compile_options(test_traits PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME traits COMMAND test_traits)

# Behavioural tests for MatrixView / ConstMatrixView.
add_executable(test_matrixview test_matrixview.cpp)
target_link_libraries(test_matrixview PRIVATE linalg::linalg)
target_compile_options(test_matrixview PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME matrixview COMMAND test_matrixview)

# Behavioural tests for the Vector class.
add_executable(test_vector test_vector.cpp)
target_link_libraries(test_vector PRIVATE linalg::linalg)
target_compile_options(test_vector PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME vector COMMAND test_vector)
# The Matrix-bridging stragglers in this suite skip until Matrix lands.
set_tests_properties(vector PROPERTIES SKIP_RETURN_CODE 77)

# Behavioural tests for the BLAS-style compute kernels.
add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE linalg::linalg)
target_compile_options(test_kernels PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME kernels COMMAND test_kernels)


# Suites written ahead of the implementation (see harness.hpp): tests whose
# methods are still "not implemented" stubs report as Skipped via exit code
# 77, and flip to Passed/Failed as the methods land. Add later phases
# (LU, Cholesky, QR, ...) the same way when work on them starts.
function(add_skippable_test NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE linalg::linalg)
    target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    add_test(NAME ${NAME} COMMAND ${NAME})
    set_tests_properties(${NAME} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

add_skippable_test(matrix test_matrix.cpp)
add_skippable_test(norm test_norm.cpp)
add_skippable_test(householder test_householder.cpp)
add_skippable_test(givens test_givens.cpp)
add_skippable_test(triangular_solver test_triangular_solver.cpp)
add_skippable_test(lu test_lu.cpp)
add_skippable_test(cholesky test_cholesky.cpp)
add_skippable_test(qr test_qr.cpp)
add_skippable_test(hessenberg test_hessenberg.cpp)
add_skippable_test(eigen test_eigen.cpp)
add_skippable_test(svd test_svd.cpp)
add_skippable_test(linear_solver test_linear_solver.cpp)
add_skippable_test(least_squares test_least_squares.cpp)
add_skippable_test(preconditioner test_preconditioner.cpp)
add_skippable_test(krylov test_krylov.cpp)
add_skippable_test(eigen_solver test_eigen_solver.cpp)
