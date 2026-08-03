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

# Behavioural tests for the BLAS-style compute kernels.
add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE linalg::linalg)
target_compile_options(test_kernels PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME kernels COMMAND test_kernels)


# Add more behavioural tests alongside as classes get implemented, e.g.
#
#   add_executable(test_matrix test_matrix.cpp)
#   target_link_libraries(test_matrix PRIVATE linalg::linalg)
#   add_test(NAME matrix COMMAND test_matrix)
