// Behavioural tests for linalg::Householder / HouseholderSequence
// (PLAN.md step 6).
//
// Written ahead of the implementation: unimplemented methods make their test
// SKIP (see harness.hpp). buildBlockRepresentation (WY form) is Phase 6 and
// stays skipped until then.
//
// Identities from PLAN.md: H*H == I, H*x lands on +/- ||x|| e_1, and
// toMatrix agrees with applyLeft on the identity.

#include "linalg/core/Matrix.hpp"
#include "linalg/core/Traits.hpp"
#include "linalg/core/Vector.hpp"
#include "linalg/ops/Householder.hpp"

#include "harness.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>

using namespace linalg;
using testharness::finish;
using testharness::run;

using M = Matrix<double>;
using V = Vector<double>;
using C = std::complex<double>;
using H = Householder<double>;
using HC = Householder<C>;

namespace {

constexpr double kTol = 1e-12;

bool close(double a, double b) { return std::abs(a - b) < kTol; }

void test_identity_reflector() {
    H h;
    assert(h.isIdentity());
    // beta == 0: the dense form is a plain identity
    M d = h.toMatrix(3);
    assert(d.isApprox(M::Identity(3), kTol));
}

void test_from_vector_lands_on_e1() {
    V x{3, 4};
    H h = H::FromVector(x);
    assert(h.size() == 2);
    V y = x;
    h.apply(y);
    // documented sign choice: opposite x_1, so 3 maps to -5
    assert(close(y(0), -5.0));
    assert(close(y(1), 0.0));
    // norm preserved exactly up to roundoff
    assert(close(y.norm(), x.norm()));
}

void test_involutory() {
    // H is Hermitian and unitary, so H*H == I: applying twice restores x
    V x{1, -2, 3, 0.5};
    H h = H::FromVector(x);
    V y = x;
    h.apply(y);
    h.apply(y);
    assert(y.isApprox(x, kTol));
    // dense check of the same identity
    M d = h.toMatrix(4);
    assert((d * d).isApprox(M::Identity(4), kTol));
}

void test_to_matrix_properties() {
    V x{2, -1, 2}; // norm 3
    H h = H::FromVector(x);
    M d = h.toMatrix(3);
    // symmetric (Hermitian) and orthogonal
    assert(d.isSymmetric(1e-12));
    assert(d.isOrthogonal(1e-12));
    // determinant of a reflector is -1 (Phase 2 back-fill on determinant)
    assert(close(d.determinant(), -1.0));
}

void test_to_matrix_agrees_with_apply_left() {
    V x{1, 2, 2}; // norm 3
    H h = H::FromVector(x);
    M i = M::Identity(3);
    h.applyLeft(i.view()); // i <- H * I == H
    assert(i.isApprox(h.toMatrix(3), kTol));
}

void test_apply_left_right() {
    V x{3, 4};
    H h = H::FromVector(x);
    M a{{1, 2, 3}, {4, 5, 6}};
    M byMatrix = h.toMatrix(2) * a;
    M inPlace = a;
    h.applyLeft(inPlace.view());
    assert(inPlace.isApprox(byMatrix, kTol));

    M b{{1, 2}, {3, 4}, {5, 6}};
    M byMatrixR = b * h.toMatrix(2);
    M inPlaceR = b;
    h.applyRight(inPlaceR.view());
    assert(inPlaceR.isApprox(byMatrixR, kTol));
}

void test_from_column() {
    // FromColumn(a, col, startRow) must match FromVector on the column tail
    M a{{9, 1}, {8, 3}, {7, 4}};
    H fromCol = H::FromColumn(a.view(), 1, 1); // column 1 from row 1: (3, 4)
    H fromVec = H::FromVector(V{3, 4});
    assert(fromCol.size() == 2);
    V y{3, 4};
    fromCol.apply(y);
    assert(close(y(0), -5.0) && close(y(1), 0.0));
    assert(fromCol.essential().isApprox(fromVec.essential(), kTol));
}

void test_complex_reflector() {
    // the standing reminder: test with a complex scalar before moving on
    Vector<C> x{C(3, 4), C(1, -2), C(0, 1)};
    HC h = HC::FromVector(x);
    // beta is real-valued even for complex T
    assert(std::abs(NumericTraits<C>::imag(h.beta())) < 1e-12);
    Vector<C> y = x;
    h.apply(y);
    // lands on (phase) * ||x|| e_1: tail must vanish, head must carry ||x||
    double normx = x.norm();
    assert(std::abs(std::abs(y(0)) - normx) < 1e-12);
    assert(std::abs(y(1)) < 1e-12 && std::abs(y(2)) < 1e-12);
    // unitary and Hermitian in dense form; H * H == I
    Matrix<C> d = h.toMatrix(3);
    assert(d.isHermitian(1e-12));
    assert(d.isOrthogonal(1e-12));
    assert((d * d).isApprox(Matrix<C>::Identity(3), 1e-12));
    // applyLeftConjugate is applyLeft of H^H == H
    Matrix<C> i1 = Matrix<C>::Identity(3), i2 = Matrix<C>::Identity(3);
    h.applyLeft(i1.view());
    h.applyLeftConjugate(i2.view());
    assert(i1.isApprox(i2, 1e-12));
}

// ---- HouseholderSequence ---------------------------------------------------

void test_sequence_empty() {
    HouseholderSequence<double> seq;
    assert(seq.count() == 0);
    assert(seq.toMatrix(3).isApprox(M::Identity(3), kTol));
}

void test_sequence_product_order() {
    // Q = H_0 * H_1 in append order
    H h0 = H::FromVector(V{1, 2, 2});
    H h1 = H::FromVector(V{3, 4});
    HouseholderSequence<double> seq;
    seq.append(h0);
    seq.append(h1);
    assert(seq.count() == 2);
    // dense reference: h1 embedded bottom-right in a 3x3 identity
    M q = h0.toMatrix(3) * h1.toMatrix(3);
    assert(seq.toMatrix(3).isApprox(q, kTol));
    assert(seq.toMatrix(3).isOrthogonal(1e-12));
}

void test_sequence_apply() {
    H h0 = H::FromVector(V{1, 2, 2});
    H h1 = H::FromVector(V{3, 4});
    HouseholderSequence<double> seq;
    seq.append(h0);
    seq.append(h1);
    M q = seq.toMatrix(3);
    M a{{1, 0}, {2, 1}, {0, 3}};
    M inPlace = a;
    seq.applyLeft(inPlace.view());
    assert(inPlace.isApprox(q * a, kTol));
    // Q^H * (Q * a) == a
    seq.applyLeftTranspose(inPlace.view());
    assert(inPlace.isApprox(a, kTol));
    // right application
    M b{{1, 2, 3}, {4, 5, 6}};
    M inPlaceR = b;
    seq.applyRight(inPlaceR.view());
    assert(inPlaceR.isApprox(b * q, kTol));
}

void test_sequence_first_columns() {
    H h0 = H::FromVector(V{1, 2, 2});
    HouseholderSequence<double> seq;
    seq.append(h0);
    M thin = seq.firstColumns(3, 2);
    assert(thin.rows() == 3 && thin.cols() == 2);
    M full = seq.toMatrix(3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            assert(close(thin(i, j), full(i, j)));
}

void test_sequence_block_representation() { // Phase 6 (compact WY)
    H h0 = H::FromVector(V{1, 2, 2});
    H h1 = H::FromVector(V{3, 4});
    HouseholderSequence<double> seq;
    seq.append(h0);
    seq.append(h1);
    seq.buildBlockRepresentation(2);
    // Q == I - V * T * V^H
    M v = seq.blockV(), t = seq.blockT();
    M q = M::Identity(3) - v * t * v.transpose();
    assert(q.isApprox(seq.toMatrix(3), 1e-10));
}

} // namespace

int main() {
    run("identity_reflector", test_identity_reflector);
    run("from_vector_lands_on_e1", test_from_vector_lands_on_e1);
    run("involutory", test_involutory);
    run("to_matrix_properties", test_to_matrix_properties);
    run("to_matrix_agrees_with_apply_left", test_to_matrix_agrees_with_apply_left);
    run("apply_left_right", test_apply_left_right);
    run("from_column", test_from_column);
    run("complex_reflector", test_complex_reflector);
    run("sequence_empty", test_sequence_empty);
    run("sequence_product_order", test_sequence_product_order);
    run("sequence_apply", test_sequence_apply);
    run("sequence_first_columns", test_sequence_first_columns);
    run("sequence_block_representation", test_sequence_block_representation);
    return finish("Householder");
}
