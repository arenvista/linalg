#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace linalg {

// Root of the hierarchy. Also thrown directly by unimplemented stubs and
// for one-off contract violations that have no dedicated subclass.
class LinalgError : public std::runtime_error {
public:
  explicit LinalgError(const std::string &message);
};

// Operand shapes are incompatible for the requested operation. Vectors
// report their size as an n x 1 shape.
class DimensionMismatch : public LinalgError {
public:
  DimensionMismatch(std::size_t lhsRows, std::size_t lhsCols,
                    std::size_t rhsRows, std::size_t rhsCols);

  std::size_t lhsRows() const;
  std::size_t lhsCols() const;
  std::size_t rhsRows() const;
  std::size_t rhsCols() const;

private:
  std::size_t lhsRows_;
  std::size_t lhsCols_;
  std::size_t rhsRows_;
  std::size_t rhsCols_;
};

// Thrown by the checked accessors (at, checked element access); the
// unchecked operator() never throws this.
class IndexOutOfRange : public LinalgError {
public:
  IndexOutOfRange(std::size_t index, std::size_t bound);

  std::size_t index() const;
  std::size_t bound() const;

private:
  std::size_t index_;
  std::size_t bound_;
};

// A factorization or solve met an exactly (or numerically) zero pivot;
// pivotIndex is the elimination step that failed.
class SingularMatrix : public LinalgError {
public:
  explicit SingularMatrix(std::size_t pivotIndex);

  std::size_t pivotIndex() const;

private:
  std::size_t pivotIndex_;
};

// Cholesky-family failure: pivotIndex is the first non-positive pivot.
// Only raised when the caller opted into throwOnIndefinite.
class NotPositiveDefinite : public LinalgError {
public:
  explicit NotPositiveDefinite(std::size_t pivotIndex);

  std::size_t pivotIndex() const;

private:
  std::size_t pivotIndex_;
};

// An iterative algorithm exhausted its iteration budget. Solvers that
// report convergence through converged()/Report do not throw this.
class ConvergenceFailure : public LinalgError {
public:
  ConvergenceFailure(const std::string &algorithm, std::size_t iterations);

  const std::string &algorithm() const;
  std::size_t iterations() const;

private:
  std::string algorithm_;
  std::size_t iterations_;
};

// A factorization accessor (solve, factors, eigenvalues, ...) was called
// before compute(), or after a compute() that failed.
class NotComputed : public LinalgError {
public:
  explicit NotComputed(const std::string &factorization);
};

} // namespace linalg
