// Behavioural tests for the linalg exception hierarchy.
//
// Convention (matches tests/): exits 0 on success, aborts via assert() on the
// first failure, which ctest reports as a failure.

#include "linalg/core/Exceptions.hpp"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace linalg;

namespace {

bool contains(const std::string &haystack, const std::string &needle) {
    return haystack.find(needle) != std::string::npos;
}

void test_linalg_error() {
    LinalgError e("something broke");
    assert(std::string(e.what()) == "something broke");
    // is a std::runtime_error
    bool caught = false;
    try {
        throw LinalgError("msg");
    } catch (const std::runtime_error &re) {
        caught = true;
        assert(std::string(re.what()) == "msg");
    }
    assert(caught);
}

void test_dimension_mismatch() {
    DimensionMismatch e(3, 4, 5, 6);
    assert(e.lhsRows() == 3 && e.lhsCols() == 4);
    assert(e.rhsRows() == 5 && e.rhsCols() == 6);
    std::string w = e.what();
    assert(contains(w, "3") && contains(w, "4") && contains(w, "5") &&
           contains(w, "6"));
    // accessors survive a catch by base reference
    try {
        throw DimensionMismatch(2, 2, 7, 1);
    } catch (const LinalgError &le) {
        auto *dm = dynamic_cast<const DimensionMismatch *>(&le);
        assert(dm != nullptr);
        assert(dm->rhsRows() == 7);
    }
}

void test_index_out_of_range() {
    IndexOutOfRange e(9, 4);
    assert(e.index() == 9 && e.bound() == 4);
    std::string w = e.what();
    assert(contains(w, "9") && contains(w, "4"));
}

void test_singular_matrix() {
    SingularMatrix e(2);
    assert(e.pivotIndex() == 2);
    assert(contains(e.what(), "singular"));
    assert(contains(e.what(), "2"));
}

void test_not_positive_definite() {
    NotPositiveDefinite e(1);
    assert(e.pivotIndex() == 1);
    assert(contains(e.what(), "positive definite"));
}

void test_convergence_failure() {
    ConvergenceFailure e("GMRES", 500);
    assert(e.algorithm() == "GMRES");
    assert(e.iterations() == 500);
    assert(contains(e.what(), "GMRES") && contains(e.what(), "500"));
}

void test_not_computed() {
    NotComputed e("QR");
    assert(contains(e.what(), "QR"));
    assert(contains(e.what(), "compute()"));
}

void test_hierarchy() {
    // every subclass is catchable as LinalgError
    bool caught = false;
    try {
        throw SingularMatrix(0);
    } catch (const LinalgError &) {
        caught = true;
    }
    assert(caught);
    caught = false;
    try {
        throw NotComputed("LU");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    assert(caught);
    // the specific type wins over the base in an ordered catch
    caught = false;
    try {
        throw IndexOutOfRange(1, 1);
    } catch (const IndexOutOfRange &ioor) {
        caught = true;
        assert(ioor.bound() == 1);
    } catch (const LinalgError &) {
        assert(false); // must not reach the base handler
    }
    assert(caught);
}

} // namespace

int main() {
    test_linalg_error();
    test_dimension_mismatch();
    test_index_out_of_range();
    test_singular_matrix();
    test_not_positive_definite();
    test_convergence_failure();
    test_not_computed();
    test_hierarchy();
    std::puts("all exception behavioural tests passed");
    return 0;
}
