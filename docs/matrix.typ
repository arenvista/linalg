// Documentation for include/linalg/core/Matrix.hpp.
// Included from documentation.typ; imports the template for the `method` env.
#import "template.typ": *

= Matrix

`Matrix<T>` is an owning dense matrix, row-major and contiguous: element
$(i, j)$ lives at `data()[i * cols() + j]`. Every operator is a member, so
`matrix * scalar` exists but `scalar * matrix` does not — use `scaledBy` for
the reversed form. Shape mismatches throw `DimensionMismatch`; `operator==` is
exact equality (use `isApprox` for floating-point comparison). Several scalar
summaries (`determinant`, `spectralNorm`, `conditionNumber`, `rank`,
`inverse`, `pseudoInverse`) factorize internally and are $O(n^3)$
conveniences — hold a decomposition object when you need more than one query.

== Construction

#method(
	"Matrix()  /  Matrix(Index rows, Index cols)  /  Matrix(rows, cols, const T& fill)",
	example: ```cpp
	M e;               assert(e.rows() == 0 && e.isEmpty());
	M a(2, 3);         assert(a.size() == 6 && a(0, 0) == 0.0); // value-initialized
	M f(2, 2, 7.5);    assert(f(0, 0) == 7.5 && f(1, 1) == 7.5);
	```,
	description: [
		The default constructor makes an empty $0 times 0$ matrix; `Matrix(rows,
		cols)` makes a zero-initialized matrix of the given shape; the third
		overload fills every element with `fill`.
	],
	params: (
		("rows", [number of rows]),
		("cols", [number of columns]),
		("fill", [value assigned to every element (fill overload)]),
	),
	returns: [
		The constructed matrix.
	],
)

#method(
	"Matrix(rows, cols, const std::vector<T>&)  /  Matrix(nested initializer_list)",
	example: ```cpp
	M b(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6}); // flat, row-major
	assert(b(1, 0) == 4 && b(1, 2) == 6);
	M a{{1, 2, 3}, {4, 5, 6}};                        // braced rows
	assert(a.rows() == 2 && a(1, 2) == 6);
	```,
	description: [
		Build from a flat row-major element list (size must equal
		$"rows" times "cols"$), or from a braced list of rows (every inner list
		must have equal length).
	],
	params: (
		("rows / cols", [shape (flat overload)]),
		("rowMajorData", [elements in row-major order]),
	),
	returns: [
		The constructed matrix.
	],
)

#method(
	"Copy / move / from-view constructors and assignment",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	M b(a); b(0, 0) = 99; assert(a(0, 0) == 1 && b(0, 0) == 99); // deep copy
	M c(std::move(b));    assert(c(0, 0) == 99 && b.isEmpty());  // steals storage
	M fromView(someConstView);                                   // explicit
	```,
	description: [
		Copy operations deep-copy the elements; move operations take over the
		other matrix's storage and leave it empty. `explicit Matrix(const
		ConstMatrixView<T>&)` deep-copies a view's elements into a fresh
		contiguous matrix.
	],
	params: (
		("other", [matrix to copy from / move from]),
		("view", [read-only view to copy (from-view constructor)]),
	),
	returns: [
		A copy / reference to this matrix, per the operation.
	],
)

== Named constructors

#method(
	"static Matrix Zeros / Ones / Constant / Identity / Diagonal",
	example: ```cpp
	assert(M::Zeros(2, 3)(1, 2) == 0.0);
	assert(M::Ones(2, 2)(0, 1) == 1.0);
	assert(M::Constant(2, 2, 3.5)(1, 0) == 3.5);
	M i = M::Identity(3);       assert(i(0, 0) == 1 && i(0, 1) == 0);
	M d = M::Diagonal(V{5, 7}); assert(d(0, 0) == 5 && d(1, 1) == 7 && d(0, 1) == 0);
	```,
	description: [
		Factories for common matrices: all zeros, all ones, a constant fill, the
		square identity, and a square diagonal matrix built from a vector.
	],
	params: (
		("rows / cols", [shape (`Zeros`, `Ones`, `Constant`)]),
		("size", [side length (`Identity`)]),
		("value", [fill value (`Constant`)]),
		("values", [diagonal entries (`Diagonal`)]),
	),
	returns: [
		The requested matrix.
	],
)

#method(
	"static Matrix Random / RandomSymmetric / RandomOrthogonal(size, seed)",
	example: ```cpp
	M a = M::Random(3, 4, 42), b = M::Random(3, 4, 42);
	assert(a == b);                          // same seed -> same entries
	assert(M::RandomSymmetric(4, 1).isSymmetric(0.0));
	assert(M::RandomOrthogonal(4, 7).isOrthogonal(1e-12));
	```,
	description: [
		Deterministic random matrices (reproducible per seed). `Random` fills all
		entries; `RandomSymmetric` is symmetric (Hermitian for complex `T`);
		`RandomOrthogonal` is orthogonal/unitary — a perfectly conditioned $Q$
		factor.
	],
	params: (
		("rows / cols", [shape (`Random`)]),
		("size", [side length (`RandomSymmetric`, `RandomOrthogonal`)]),
		("seed", [seed determining the entries]),
	),
	returns: [
		The requested random matrix.
	],
)

#method(
	"static Matrix Hilbert(Index size)  /  Vandermonde(const Vector<T>& nodes, Index degree)",
	example: ```cpp
	M h = M::Hilbert(3);              // a(i,j) = 1/(i+j+1); ill-conditioned
	assert(std::abs(h(0, 1) - 0.5) < 1e-12 && h.isSymmetric(0.0));
	M v = M::Vandermonde(V{1, 2, 3}, 2); // a(i,j) = nodes[i]^j
	assert(v.cols() == 3 && v(1, 2) == 4 && v(2, 2) == 9);
	```,
	description: [
		Two structured test inputs: the Hilbert matrix $a(i,j) = 1/(i+j+1)$, the
		classic ill-conditioned stress case, and the Vandermonde matrix
		$a(i,j) = "nodes"[i]^j$ for $j$ from $0$ to `degree`.
	],
	params: (
		("size", [side length (`Hilbert`)]),
		("nodes", [node values, one per row (`Vandermonde`)]),
		("degree", [highest power; the result has $"degree" + 1$ columns]),
	),
	returns: [
		The requested matrix.
	],
)

#method(
	"static Matrix FromColumns / FromRows(const std::vector<Vector<T>>&)",
	example: ```cpp
	M a = M::FromColumns({V{1, 2}, V{3, 4}});
	assert(a(0, 0) == 1 && a(1, 0) == 2 && a(0, 1) == 3);
	M b = M::FromRows({V{1, 2}, V{3, 4}});
	assert(b(0, 0) == 1 && b(0, 1) == 2 && b(1, 0) == 3);
	```,
	description: [
		Assemble a matrix from column vectors (left to right) or row vectors (top
		to bottom); all vectors must have equal length.
	],
	params: (("columns / rows", [equal-length vectors forming the columns / rows]),),
	returns: [
		A matrix whose columns / rows are the given vectors.
	],
)

== Element access & shape

#method(
	"T& operator()(Index i, Index j)  /  T& at(Index i, Index j)  (+ const overloads)",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	a.at(0, 1) = 9; assert(a(0, 1) == 9);   // at() is bounds-checked
	try { a.at(2, 0); } catch (const IndexOutOfRange&) { /* ... */ }
	```,
	description: [
		`operator()` is unchecked element access; `at` throws `IndexOutOfRange`
		when $i >= "rows"()$ or $j >= "cols"()$. Each has a mutable and a
		read-only overload.
	],
	params: (
		("i", [row index (0-based)]),
		("j", [column index (0-based)]),
	),
	returns: [
		Reference to element $(i, j)$.
	],
)

#method(
	"T* data()  /  Index rows() / cols() / size()  /  bool isEmpty() / isSquare()",
	example: ```cpp
	M b(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
	assert(b.data()[1 * 3 + 2] == 6); // row-major: (i,j) at data()[i*cols + j]
	assert(b.rows() == 2 && b.cols() == 3 && b.size() == 6);
	assert(!b.isEmpty() && !b.isSquare());
	```,
	description: [
		`data` exposes the row-major buffer. `rows` / `cols` / `size` report the
		shape and element count; `isEmpty` is true when either dimension is zero;
		`isSquare` is true when $"rows"() = "cols"()$.
	],
	params: (),
	returns: [
		The pointer / count / predicate, respectively.
	],
)

== Views

#method(
	"MatrixView view() / block(...) / rowView(Index) / colView(Index)  (+ const)",
	example: ```cpp
	M a{{0, 1, 2}, {10, 11, 12}, {20, 21, 22}};
	auto v = a.view(); v(0, 0) = -5; assert(a(0, 0) == -5); // aliases storage
	auto blk = a.block(1, 1, 2, 2); blk.fill(0.0);
	assert(a(1, 1) == 0 && a(0, 1) == 1);                   // outside untouched
	assert(a.rowView(0).cols() == 3 && a.colView(2).rows() == 3);
	```,
	description: [
		Return `MatrixView` / `ConstMatrixView` windows that *alias* this matrix's
		storage: writes through them are visible here, and they dangle after a
		resize or destruction. `view` covers the whole matrix, `block` a
		rectangle, `rowView` / `colView` a single row / column.
	],
	params: (
		("i, j, numRows, numCols", [block position and shape (`block`)]),
		("i / j", [row / column index (`rowView` / `colView`)]),
	),
	returns: [
		A view aliasing this matrix's storage.
	],
)

#method(
	"Vector<T> row(Index i) / col(Index j) / diagonal()",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	V r = a.row(1); r(0) = 99; assert(a(1, 0) == 3); // owning COPY, not a view
	assert(a.col(0)(1) == 3);
	assert(a.diagonal() == V{1, 4});
	```,
	description: [
		Copy a row, column, or the main diagonal into an owning `Vector`. Unlike
		the `*View` members these return independent copies, so mutating the
		result does not touch the matrix.
	],
	params: (("i / j", [row / column index (0-based)]),),
	returns: [
		A vector holding the copied entries.
	],
)

#method(
	"void setRow(Index, const Vector&) / setCol(Index, const Vector&) / setBlock(i, j, const Matrix&)",
	example: ```cpp
	M a = M::Zeros(3, 3);
	a.setRow(0, V{1, 2, 3}); assert(a(0, 2) == 3);
	a.setCol(2, V{7, 8, 9}); assert(a(2, 2) == 9);
	a.setBlock(1, 0, M{{5, 6}}); assert(a(1, 0) == 5 && a(1, 1) == 6);
	```,
	description: [
		Overwrite a row (length must equal `cols()`), a column (length must equal
		`rows()`), or a rectangular block (the `source` matrix defines the block
		shape, placed with its top-left corner at $(i, j)$).
	],
	params: (
		("i / j", [target row / column, or block corner (0-based)]),
		("values", [replacement values (`setRow` / `setCol`)]),
		("source", [matrix copied into the block (`setBlock`)]),
	),
	returns: [
		Nothing.
	],
)

== Arithmetic

#method(
	"Matrix operator+ / operator- (binary and unary) / operator* (Matrix)",
	example: ```cpp
	M a{{1, 2, 3}, {4, 5, 6}};      // 2x3
	M b{{7, 8}, {9, 10}, {11, 12}}; // 3x2
	M p = a * b;                    // 2x2
	assert(p(0, 0) == 58 && p(1, 1) == 154);
	assert((a * M::Identity(3)).isApprox(a, 1e-10)); // identity is neutral
	EXPECT_LINALG_THROW(a * a);     // inner dims 3 vs 2
    ```,
	description: [
		Elementwise addition and subtraction (matching shapes required), unary
		negation, and matrix-matrix multiplication (the right operand's `rows()`
		must equal this matrix's `cols()`).
	],
	params: (("rhs", [right operand]),),
	returns: [
		The resulting matrix; throws `DimensionMismatch` on incompatible shapes.
	],
)

#method(
	"Vector<T> operator*(const Vector<T>& rhs)",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	V x{1, -1};
	V ax = a * x;              // length rows()
	assert(ax(0) == -1 && ax(1) == -1);
	```,
	description: [
		Matrix-vector multiplication; `rhs.size()` must equal this matrix's
		`cols()`.
	],
	params: (("rhs", [right operand; its size must equal `cols()`]),),
	returns: [
		The product vector of length `rows()`; throws `DimensionMismatch` on
		mismatch.
	],
)

#method(
	"Matrix operator* / operator/ (scalar)  /  scaledBy(const T&)",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	assert((a * 2.0)(1, 1) == 8);
	assert((a / 2.0)(0, 0) == 0.5);
	assert(a.scaledBy(3.0)(0, 1) == 6); // the scalar * A form
	```,
	description: [
		Scalar multiplication (matrix on the left) and division. `scaledBy` is the
		named form for scalar-on-the-left, which `operator*` cannot express as a
		member.
	],
	params: (("scalar", [scalar multiplier / divisor]),),
	returns: [
		The scaled matrix.
	],
)

#method(
	"operator+= / operator-= / operator*= (Matrix and scalar) / operator/=",
	example: ```cpp
	M a{{1, 2}, {3, 4}}, b{{10, 20}, {30, 40}};
	a += b; assert(a(1, 0) == 33);
	a -= b; assert(a(1, 0) == 3);
	M sq{{1, 1}, {0, 1}}, sq2 = sq; sq2 *= sq;
	assert(sq2.isApprox(sq * sq, 1e-10));
	a *= 10.0; a /= 10.0;
	```,
	description: [
		In-place compound assignment mirroring the binary operators (matrix
		add/subtract/multiply, and scalar multiply/divide); each returns a
		reference to this matrix.
	],
	params: (("rhs / scalar", [right operand or scalar]),),
	returns: [
		Reference to this matrix; matrix forms throw `DimensionMismatch` on
		mismatch.
	],
)

#method(
	"bool operator==(const Matrix&) / operator!=(const Matrix&)",
	example: ```cpp
	M a{{1, 2}, {3, 4}}, b = a;
	assert(a == b);
	b(0, 0) += 1e-15;
	assert(a != b);                       // exact comparison
	assert(!(a == M::Zeros(2, 3)));       // shape mismatch is inequality, not throw
	```,
	description: [
		Exact equality / inequality: equal means same shape and every element
		exactly equal. A shape mismatch is simply inequality (never an exception).
		Use `isApprox` for floating-point tolerance.
	],
	params: (("rhs", [matrix to compare against]),),
	returns: [
		`true` / `false` per the comparison.
	],
)

#method(
	"Matrix elementwiseProduct / elementwiseQuotient / kroneckerProduct / power",
	example: ```cpp
	M a{{1, 2}, {3, 4}}, b{{2, 2}, {2, 2}};
	assert(a.elementwiseProduct(b)(1, 1) == 8);
	assert(a.elementwiseQuotient(b)(1, 0) == 1.5);
	M k = M{{1, 2}}.kroneckerProduct(M{{0, 1}, {1, 0}}); // 2x4
	assert(k(0, 1) == 1 && k(0, 3) == 2);
	assert(a.power(0).isApprox(M::Identity(2), 1e-10) && a.power(3).isApprox(a*a*a, 1e-10));
	```,
	description: [
		`elementwiseProduct` / `elementwiseQuotient` are the Hadamard product and
		quotient (matching shapes). `kroneckerProduct` is the tensor product, of
		shape $("rows" dot "rhs.rows") times ("cols" dot "rhs.cols")$. `power`
		raises a square matrix to a non-negative integer power (exponent 0 yields
		the identity).
	],
	params: (
		("rhs", [right operand (elementwise / Kronecker)]),
		("exponent", [non-negative power (`power`)]),
	),
	returns: [
		The resulting matrix; throws `DimensionMismatch` on shape mismatch (or if
		`power` is called on a non-square matrix).
	],
)

== Shape manipulation

#method(
	"Matrix transpose()  /  conjugateTranspose()  /  void transposeInPlace()",
	example: ```cpp
	M a{{1, 2, 3}, {4, 5, 6}};
	M t = a.transpose(); assert(t.rows() == 3 && t(2, 0) == 3 && t.transpose() == a);
	MC c{{C(1, 2), C(3, -1)}}; // conjugateTranspose flips the sign of imag parts
	assert(c.conjugateTranspose()(0, 0) == C(1, -2));
	M sq{{1, 2}, {3, 4}}; sq.transposeInPlace(); assert(sq(0, 1) == 3);
	```,
	description: [
		`transpose` returns a new transposed matrix; `conjugateTranspose` returns
		the Hermitian transpose (equal to `transpose` for real `T`);
		`transposeInPlace` transposes this matrix in place.
	],
	params: (),
	returns: [
		The transposed matrix (`transpose` / `conjugateTranspose`) or nothing
		(`transposeInPlace`).
	],
)

#method(
	"Matrix reshaped(Index rows, Index cols)",
	example: ```cpp
	M a{{1, 2, 3}, {4, 5, 6}};
	M r = a.reshaped(3, 2);   // row-major order preserved
	assert(r(0, 0) == 1 && r(1, 0) == 3 && r(2, 1) == 6);
	EXPECT_LINALG_THROW(a.reshaped(4, 2)); // 8 != 6
	```,
	description: [
		Reinterprets the elements (in row-major order) with a new shape;
		$"rows" times "cols"$ must equal `size()`.
	],
	params: (
		("rows", [new row count]),
		("cols", [new column count]),
	),
	returns: [
		A matrix with the new shape and the same elements; throws
		`DimensionMismatch` if the element count changes.
	],
)

#method(
	"Matrix horizontalConcat / verticalConcat / withoutRow / withoutCol",
	example: ```cpp
	M h = M{{1}, {3}}.horizontalConcat(M{{2}, {4}}); // 2x2
	assert(h(0, 1) == 2 && h(1, 0) == 3);
	M v = M{{1, 2}}.verticalConcat(M{{3, 4}});       // 2x2
	assert(v(1, 0) == 3);
	M a{{1, 2, 3}, {4, 5, 6}};
	assert(a.withoutRow(0).rows() == 1 && a.withoutCol(1)(0, 1) == 3);
	```,
	description: [
		`horizontalConcat` stacks `rhs` to the right (matching row counts);
		`verticalConcat` stacks it below (matching column counts). `withoutRow` /
		`withoutCol` return a copy with one row / column removed.
	],
	params: (
		("rhs", [matrix to stack (concat)]),
		("i / j", [row / column index to drop (`withoutRow` / `withoutCol`)]),
	),
	returns: [
		The reshaped matrix; concat throws `DimensionMismatch` on mismatch,
		`without*` throws `IndexOutOfRange` if the index is out of range.
	],
)

#method(
	"void resize / conservativeResize / swapRows / swapCols / fill / setZero / setIdentity / swap",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	a.conservativeResize(3, 3); assert(a(1, 1) == 4 && a(2, 2) == 0); // keeps top-left
	a.resize(2, 2);             // discards contents
	a.setIdentity(); assert(a(0, 0) == 1 && a(0, 1) == 0);
	M c{{1, 2}, {3, 4}}; c.swapRows(0, 1); assert(c(0, 0) == 3);
	c.swapCols(0, 1); assert(c(0, 0) == 4);
	```,
	description: [
		In-place shape/content mutators: `resize` (discards contents),
		`conservativeResize` (keeps the overlapping top-left block, zero-fills
		growth), `swapRows` / `swapCols`, `fill` / `setZero`, `setIdentity`
		(square only), and `swap` (exchange contents with another matrix).
	],
	params: (
		("rows / cols", [new shape (`resize`, `conservativeResize`)]),
		("a, b", [indices to swap (`swapRows`, `swapCols`)]),
		("value", [fill value (`fill`)]),
		("other", [matrix to swap with (`swap`)]),
	),
	returns: [
		Nothing.
	],
)

== Scalar summaries

#method(
	"T trace() / sum() / determinant()",
	example: ```cpp
	M a{{1, -2}, {3, 4}};
	assert(a.trace() == 5.0 && a.sum() == 6.0);
	assert(std::abs(M{{1, 2}, {3, 4}}.determinant() - (-2.0)) < 1e-10);
	assert(std::abs(M{{1, 2}, {2, 4}}.determinant()) < 1e-10); // singular
	```,
	description: [
		`trace` sums the main diagonal; `sum` sums all elements; `determinant`
		computes the determinant of a square matrix via LU factorization.
	],
	params: (),
	returns: [
		The scalar result as `T`; `determinant` throws `DimensionMismatch` if the
		matrix is not square.
	],
)

#method(
	"Real oneNorm() / infinityNorm() / frobeniusNorm() / maxNorm()",
	example: ```cpp
	M a{{1, -2}, {3, 4}};
	assert(a.oneNorm() == 6.0);       // max abs column sum: |−2|+|4|
	assert(a.infinityNorm() == 7.0);  // max abs row sum: |3|+|4|
	assert(std::abs(a.frobeniusNorm() - std::sqrt(30.0)) < 1e-10);
	assert(a.maxNorm() == 4.0);
	// complex uses magnitudes: |3+4i| = 5
	assert(std::abs(MC{{C(3, 4)}}.frobeniusNorm() - 5.0) < 1e-10);
	```,
	description: [
		Matrix norms: `oneNorm` (maximum absolute column sum), `infinityNorm`
		(maximum absolute row sum), `frobeniusNorm` (root of the sum of squared
		magnitudes), and `maxNorm` (largest element magnitude).
	],
	params: (),
	returns: [
		The requested norm as `Real`.
	],
)

#method(
	"Real spectralNorm() / conditionNumber()  /  Index rank(Real tolerance)",
	example: ```cpp
	M d = M::Diagonal(V{3, -5});
	assert(std::abs(d.spectralNorm() - 5.0) < 1e-10);          // largest sigma
	assert(std::abs(d.conditionNumber() - 5.0 / 3.0) < 1e-10); // sigma_max/sigma_min
	assert(M{{1, 0}, {0, 2}}.rank(1e-12) == 2);
	assert(M{{1, 2}, {2, 4}}.rank(1e-12) == 1);                // rank-deficient
	```,
	description: [
		SVD-based summaries: `spectralNorm` is the largest singular value;
		`conditionNumber` is $sigma_max / sigma_min$; `rank` counts singular
		values above $"tolerance" dot sigma_max$. Each runs a full SVD, so prefer
		a held decomposition for repeated queries.
	],
	params: (("tolerance", [relative singular-value cutoff (`rank`)]),),
	returns: [
		The norm / condition number (`Real`) or the numerical rank (`Index`).
	],
)

== Predicates

#method(
	"bool isSymmetric / isHermitian / isDiagonal(Real tolerance)",
	example: ```cpp
	assert(M{{1, 2}, {2, 3}}.isSymmetric(0.0));
	assert(!M{{1, 2}, {3, 4}}.isSymmetric(0.0));
	MC h{{C(1, 0), C(2, 3)}, {C(2, -3), C(5, 0)}};
	assert(h.isHermitian(0.0));
	assert(M::Diagonal(V{1, 2}).isDiagonal(0.0));
	```,
	description: [
		Structural tests within an absolute per-element tolerance (pass 0 for
		exact): `isSymmetric` ($A = A^T$), `isHermitian` ($A = A^H$, the
		meaningful symmetry test for complex `T`), and `isDiagonal` (off-diagonal
		entries negligible).
	],
	params: (("tolerance", [absolute per-element tolerance; 0 for exact]),),
	returns: [
		`true` if the property holds within tolerance, otherwise `false`.
	],
)

#method(
	"bool isTriangular(Triangle::Kind which, Real tolerance) / isOrthogonal(Real tolerance)",
	example: ```cpp
	M up{{1, 2}, {0, 3}};
	assert(up.isTriangular(Triangle::Kind::Upper, 0.0));
	assert(!up.isTriangular(Triangle::Kind::Lower, 0.0));
	M rot{{0, -1}, {1, 0}};  // 90-degree rotation
	assert(rot.isOrthogonal(1e-14));
	assert(!M{{2, 0}, {0, 1}}.isOrthogonal(1e-6));
	```,
	description: [
		`isTriangular` tests whether the matrix is triangular of the given kind
		(the other triangle negligible); `isOrthogonal` tests $A^H A = I$. Both
		use an absolute tolerance (0 for exact).
	],
	params: (
		("which", [which triangle is expected nonzero (`isTriangular`)]),
		("tolerance", [absolute per-element tolerance; 0 for exact]),
	),
	returns: [
		`true` if the property holds within tolerance, otherwise `false`.
	],
)

#method(
	"bool isApprox(const Matrix& other, Real tolerance)  /  bool hasNaN()",
	example: ```cpp
	M a{{1, 2}, {3, 4}}, b = a; b(0, 0) += 1e-15;
	assert(a != b && a.isApprox(b, 1e-12));
	M nan{{1, 2}, {3, 4}}; nan(0, 1) = std::nan("");
	assert(nan.hasNaN());
	```,
	description: [
		`isApprox` tests same-shape, per-element agreement within an absolute
		`tolerance`; `hasNaN` reports whether any element is NaN.
	],
	params: (
		("other", [matrix to compare against (`isApprox`)]),
		("tolerance", [absolute per-element tolerance (`isApprox`)]),
	),
	returns: [
		`true` / `false` per the test.
	],
)

== Derived matrices

#method(
	"Matrix inverse()  /  Matrix pseudoInverse(Real tolerance)",
	example: ```cpp
	M a{{4, 7}, {2, 6}};
	assert((a * a.inverse()).isApprox(M::Identity(2), 1e-10));
	EXPECT_LINALG_THROW(M{{1, 2}, {2, 4}}.inverse()); // singular
	M tall{{1, 0}, {0, 2}, {0, 0}};
	M p = tall.pseudoInverse(1e-12);        // 2x3 left inverse
	assert((p * tall).isApprox(M::Identity(2), 1e-10));
	```,
	description: [
		`inverse` is the matrix inverse via LU (square only). `pseudoInverse` is
		the Moore-Penrose pseudoinverse via SVD, with `tolerance` the relative
		singular-value cutoff (as in `rank`).
	],
	params: (("tolerance", [relative singular-value cutoff (`pseudoInverse`)]),),
	returns: [
		The inverse / pseudoinverse; `inverse` throws `SingularMatrix` if
		singular and `DimensionMismatch` if not square.
	],
)

#method(
	"Matrix triangularPart(Triangle::Kind) / symmetricPart() / skewSymmetricPart()",
	example: ```cpp
	M a{{1, 2}, {3, 4}};
	M up = a.triangularPart(Triangle::Kind::Upper);
	assert(up(0, 1) == 2 && up(1, 0) == 0 && up(1, 1) == 4);
	M sym = a.symmetricPart(), skew = a.skewSymmetricPart();
	assert((sym + skew).isApprox(a, 1e-10)); // decomposition identity
	```,
	description: [
		`triangularPart` keeps one triangle (and the diagonal), zero-filling the
		other. `symmetricPart` is $(A + A^H) slash 2$ and `skewSymmetricPart` is
		$(A - A^H) slash 2$; the two sum back to $A$.
	],
	params: (("which", [which triangle to keep (`triangularPart`)]),),
	returns: [
		The requested derived matrix.
	],
)

== Serialization

#method(
	"std::string toString(int precision)  /  std::string toMatlabLiteral()",
	example: ```cpp
	M a{{1.5, 2}, {3, 4}};
	std::string s = a.toString(3);
	assert(s.find("1.5") != std::string::npos);
	std::string ml = a.toMatlabLiteral(); // e.g. "[a b; c d]"
	assert(ml.front() == '[' && ml.back() == ']' && ml.find(';') != std::string::npos);
	```,
	description: [
		`toString` formats the matrix as aligned rows of text for logging;
		`toMatlabLiteral` formats it as a MATLAB/Octave literal (rows separated by
		`;`).
	],
	params: (("precision", [digits shown per element (`toString`)]),),
	returns: [
		The formatted string.
	],
)

#method(
	"static Matrix FromCsv(const std::string& path)  /  void writeCsv(const std::string& path)",
	example: ```cpp
	M a{{1.25, -2}, {3, 4.5}};
	a.writeCsv("m.csv");
	M b = M::FromCsv("m.csv");
	assert(b.isApprox(a, 1e-12)); // round-trips
	```,
	description: [
		CSV I/O: `FromCsv` parses a matrix from a file, `writeCsv` writes one.
		Both throw `LinalgError` on I/O or parse failure.
	],
	params: (("path", [filesystem path to read / write]),),
	returns: [
		The parsed matrix (`FromCsv`) or nothing (`writeCsv`).
	],
)
