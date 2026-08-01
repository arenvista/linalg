#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace linalg {

// Root of the hierarchy. Also thrown directly by unimplemented stubs and
// for one-off contract violations that have no dedicated subclass.
class LinalgError : public std::runtime_error {
public:
  /// @brief Constructs the base library error with a message.
  /// @param message Human-readable description of the failure.
  explicit LinalgError(const std::string &message);
};

// Operand shapes are incompatible for the requested operation. Vectors
// report their size as an n x 1 shape.
class DimensionMismatch : public LinalgError {
public:
  /// @brief Constructs a shape-mismatch error from the two operand shapes.
  /// @param lhsRows Rows of the left operand.
  /// @param lhsCols Columns of the left operand.
  /// @param rhsRows Rows of the right operand.
  /// @param rhsCols Columns of the right operand.
  DimensionMismatch(std::size_t lhsRows, std::size_t lhsCols,
                    std::size_t rhsRows, std::size_t rhsCols);

  /// @brief Rows of the left operand.
  /// @return The recorded left-operand row count.
  std::size_t lhsRows() const;
  /// @brief Columns of the left operand.
  /// @return The recorded left-operand column count.
  std::size_t lhsCols() const;
  /// @brief Rows of the right operand.
  /// @return The recorded right-operand row count.
  std::size_t rhsRows() const;
  /// @brief Columns of the right operand.
  /// @return The recorded right-operand column count.
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
  /// @brief Constructs an out-of-range error from the offending index.
  /// @param index The index that was requested.
  /// @param bound The exclusive upper bound (valid indices are < bound).
  IndexOutOfRange(std::size_t index, std::size_t bound);

  /// @brief The offending index.
  /// @return The recorded index.
  std::size_t index() const;
  /// @brief The exclusive upper bound.
  /// @return The recorded bound.
  std::size_t bound() const;

private:
  std::size_t index_;
  std::size_t bound_;
};

// A factorization or solve met an exactly (or numerically) zero pivot;
// pivotIndex is the elimination step that failed.
class SingularMatrix : public LinalgError {
public:
  /// @brief Constructs a singular-matrix error.
  /// @param pivotIndex The elimination step whose pivot was (near) zero.
  explicit SingularMatrix(std::size_t pivotIndex);

  /// @brief The elimination step that failed.
  /// @return The recorded pivot index.
  std::size_t pivotIndex() const;

private:
  std::size_t pivotIndex_;
};

// Cholesky-family failure: pivotIndex is the first non-positive pivot.
// Only raised when the caller opted into throwOnIndefinite.
class NotPositiveDefinite : public LinalgError {
public:
  /// @brief Constructs a not-positive-definite error.
  /// @param pivotIndex Index of the first non-positive pivot.
  explicit NotPositiveDefinite(std::size_t pivotIndex);

  /// @brief The first non-positive pivot.
  /// @return The recorded pivot index.
  std::size_t pivotIndex() const;

private:
  std::size_t pivotIndex_;
};

// An iterative algorithm exhausted its iteration budget. Solvers that
// report convergence through converged()/Report do not throw this.
class ConvergenceFailure : public LinalgError {
public:
  /// @brief Constructs a convergence-failure error.
  /// @param algorithm Name of the algorithm that failed to converge.
  /// @param iterations Number of iterations performed before giving up.
  ConvergenceFailure(const std::string &algorithm, std::size_t iterations);

  /// @brief Name of the algorithm that failed to converge.
  /// @return The recorded algorithm name.
  const std::string &algorithm() const;
  /// @brief Number of iterations performed before failure.
  /// @return The recorded iteration count.
  std::size_t iterations() const;

private:
  std::string algorithm_;
  std::size_t iterations_;
};

// A factorization accessor (solve, factors, eigenvalues, ...) was called
// before compute(), or after a compute() that failed.
class NotComputed : public LinalgError {
public:
  /// @brief Constructs a not-computed error.
  /// @param factorization Name of the factorization queried before compute().
  explicit NotComputed(const std::string &factorization);
};

} // namespace linalg
