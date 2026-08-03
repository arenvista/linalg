# Verifies every class template appears in an explicit instantiation list.
# A missing entry shows up here as a link error rather than in user code.
add_executable(test_instantiate instantiate.cpp)
target_link_libraries(test_instantiate PRIVATE linalg::linalg)
target_compile_options(test_instantiate PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
add_test(NAME instantiate COMMAND test_instantiate)

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
