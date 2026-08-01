#include "linalg/core/Exceptions.hpp"

#include <string>

namespace linalg {

// The only non-template classes in the library, so the only ones whose
// definitions can be compiled once instead of per instantiation.

LinalgError::LinalgError(const std::string &message)
    : std::runtime_error(message) {}

DimensionMismatch::DimensionMismatch(std::size_t lhsRows, std::size_t lhsCols,
                                     std::size_t rhsRows, std::size_t rhsCols)
    : LinalgError("dimension mismatch: (" + std::to_string(lhsRows) + " x " +
                  std::to_string(lhsCols) + ") against (" +
                  std::to_string(rhsRows) + " x " + std::to_string(rhsCols) +
                  ")"),
      lhsRows_(lhsRows), lhsCols_(lhsCols), rhsRows_(rhsRows),
      rhsCols_(rhsCols) {}

std::size_t DimensionMismatch::lhsRows() const { return lhsRows_; }
std::size_t DimensionMismatch::lhsCols() const { return lhsCols_; }
std::size_t DimensionMismatch::rhsRows() const { return rhsRows_; }
std::size_t DimensionMismatch::rhsCols() const { return rhsCols_; }

IndexOutOfRange::IndexOutOfRange(std::size_t index, std::size_t bound)
    : LinalgError("index " + std::to_string(index) + " outside bound " +
                  std::to_string(bound)),
      index_(index), bound_(bound) {}

std::size_t IndexOutOfRange::index() const { return index_; }
std::size_t IndexOutOfRange::bound() const { return bound_; }

SingularMatrix::SingularMatrix(std::size_t pivotIndex)
    : LinalgError("matrix is singular: zero pivot at " +
                  std::to_string(pivotIndex)),
      pivotIndex_(pivotIndex) {}

std::size_t SingularMatrix::pivotIndex() const { return pivotIndex_; }

NotPositiveDefinite::NotPositiveDefinite(std::size_t pivotIndex)
    : LinalgError("matrix is not positive definite: nonpositive pivot at " +
                  std::to_string(pivotIndex)),
      pivotIndex_(pivotIndex) {}

std::size_t NotPositiveDefinite::pivotIndex() const { return pivotIndex_; }

ConvergenceFailure::ConvergenceFailure(const std::string &algorithm,
                                       std::size_t iterations)
    : LinalgError(algorithm + " failed to converge in " +
                  std::to_string(iterations) + " iterations"),
      algorithm_(algorithm), iterations_(iterations) {}

const std::string &ConvergenceFailure::algorithm() const { return algorithm_; }
std::size_t ConvergenceFailure::iterations() const { return iterations_; }

NotComputed::NotComputed(const std::string &factorization)
    : LinalgError(factorization + " was queried before compute() was called") {}

} // namespace linalg
