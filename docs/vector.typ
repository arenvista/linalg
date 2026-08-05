// Documentation for include/linalg/core/Vector.hpp.
// Included from documentation.typ; imports the template for the `method` env.
#import "template.typ": *

= Vector

`Vector<T>` is an owning dense vector, kept distinct from an $n times 1$
`Matrix` so that `dot`, `outer`, and `norm` have unambiguous meaning. All
arithmetic operators are members. Shape mismatches throw `DimensionMismatch`;
`operator==` is exact equality (use `isApprox` for floating-point comparison).
`Real` is `NumericTraits<T>::Real` — the magnitude type.

== Construction

#method(
	"Vector()  /  Vector(Index size)  /  Vector(Index size, const T& fill)",
	example: ```cpp
	Vector<double> empty;                 assert(empty.isEmpty());
	Vector<double> z(4);                  assert(z(0) == 0.0); // value-initialized
	Vector<double> filled(4, 7.0);        assert(filled(3) == 7.0);
	```,
	description: [
		The default constructor makes a length-zero vector; `Vector(size)` makes a
		zero-initialized vector of the given length; `Vector(size, fill)` sets
		every element to `fill`.
	],
	params: (
		("size", [number of elements]),
		("fill", [value assigned to every element (fill overload)]),
	),
	returns: [
		The constructed vector.
	],
)

#method(
	"Vector(std::initializer_list<T>)  /  Vector(const std::vector<T>&)",
	example: ```cpp
	Vector<double> a{1.0, 2.0, 3.0};
	assert(a.size() == 3 && a(0) == 1.0 && a(2) == 3.0);
	std::vector<double> raw{5.0, 6.0};
	Vector<double> fromStd(raw);
	assert(fromStd.size() == 2 && fromStd(1) == 6.0);
	```,
	description: [
		Constructs a vector from a braced list of values, or by copying a
		`std::vector`. The `std::vector` overload is `explicit`.
	],
	params: (("values", [elements, in order]),),
	returns: [
		The constructed vector.
	],
)

#method(
	"Copy / move constructors and assignment",
	example: ```cpp
	Vector<double> a{1.0, 2.0, 3.0};
	Vector<double> copy(a);              assert(copy == a);     // deep copy
	Vector<double> moved(std::move(copy)); assert(moved == a);  // steals storage
	Vector<double> assigned; assigned = a; assert(assigned == a);
	Vector<double> m; m = std::move(assigned); assert(m == a);
	```,
	description: [
		Copy operations deep-copy the elements; move operations take over the
		other vector's storage and leave it empty.
	],
	params: (("other", [vector to copy from / move from]),),
	returns: [
		A copy / reference to this vector, per the operation.
	],
)

== Named constructors

#method(
	"static Vector Zeros / Ones / Constant / Unit(Index size, ...)",
	example: ```cpp
	assert(Vector<double>::Zeros(3)    == Vector<double>({0, 0, 0}));
	assert(Vector<double>::Ones(2)     == Vector<double>({1, 1}));
	assert(Vector<double>::Constant(2, 9.0) == Vector<double>({9, 9}));
	assert(Vector<double>::Unit(3, 1)  == Vector<double>({0, 1, 0}));
	EXPECT_LINALG_THROW(Vector<double>::Unit(3, 3)); // axis out of range
	```,
	description: [
		Factory functions for common vectors: all zeros, all ones, a constant
		fill, and the standard basis vector $e_"axis"$ (1 at `axis`, 0 elsewhere).
	],
	params: (
		("size", [number of elements]),
		("value", [fill value (`Constant`)]),
		("axis", [index of the single nonzero entry (`Unit`, 0-based)]),
	),
	returns: [
		The requested length-`size` vector.
	],
)

#method(
	"static Vector Random(Index size, unsigned long seed)",
	example: ```cpp
	auto r1 = Vector<double>::Random(4, 123);
	auto r2 = Vector<double>::Random(4, 123);
	assert(r1 == r2);                       // reproducible per seed
	for (std::size_t i = 0; i < 4; ++i) assert(r1(i) >= -1.0 && r1(i) <= 1.0);
	```,
	description: [
		Creates a vector of pseudo-random entries in $[-1, 1]$, deterministic for
		a given seed so tests are reproducible.
	],
	params: (
		("size", [number of elements]),
		("seed", [seed determining the entries]),
	),
	returns: [
		A length-`size` random vector.
	],
)

#method(
	"static Vector LinSpace(Index size, const T& begin, const T& end)",
	example: ```cpp
	auto ls = Vector<double>::LinSpace(5, 0.0, 1.0);
	for (std::size_t i = 0; i < 5; ++i) assert(std::abs(ls(i) - 0.25 * i) < 1e-12);
	assert(ls(4) == 1.0);                             // endpoints inclusive
	assert(Vector<double>::LinSpace(1, 2.0, 9.0) == Vector<double>({2.0}));
	```,
	description: [
		Creates a vector of `size` equally spaced values from `begin` to `end`,
		with both endpoints included.
	],
	params: (
		("size", [number of elements]),
		("begin", [first value]),
		("end", [last value (inclusive)]),
	),
	returns: [
		The linearly spaced vector.
	],
)

== Element access

#method(
	"T& operator()(Index i)  /  T& operator[](Index i)  (and const overloads)",
	example: ```cpp
	Vector<double> v{10.0, 20.0, 30.0};
	v(1) = 99.0;
	assert(v[1] == 99.0);   // operator[] and operator() are equivalent, unchecked
	```,
	description: [
		Unchecked element access. `operator()` and `operator[]` are equivalent;
		each has a mutable and a read-only overload.
	],
	params: (("i", [element index (0-based)]),),
	returns: [
		Reference to element `i`.
	],
)

#method(
	"T& at(Index i)  /  const T& at(Index i) const",
	example: ```cpp
	Vector<double> v{10.0, 20.0, 30.0};
	v.at(2) = 42.0; assert(v(2) == 42.0);
	try { v.at(5); }
	catch (const IndexOutOfRange& e) { assert(e.index() == 5 && e.bound() == 3); }
	```,
	description: [
		Bounds-checked element access.
	],
	params: (("i", [element index (0-based)]),),
	returns: [
		Reference to element `i`; throws `IndexOutOfRange` if $i >= "size"()$.
	],
)

#method(
	"T* data()  /  Index size()  /  bool isEmpty()",
	example: ```cpp
	Vector<double> v{10.0, 20.0, 30.0};
	double* p = v.data();
	p[0] = -1.0; assert(v(0) == -1.0);   // contiguous, writes through
	assert(v.size() == 3 && !v.isEmpty());
	```,
	description: [
		`data` exposes the contiguous underlying buffer (mutable and read-only
		overloads); `size` is the length; `isEmpty` is true when the length is
		zero.
	],
	params: (),
	returns: [
		The pointer / length / emptiness, respectively.
	],
)

== Arithmetic

#method(
	"operator+ / operator- (binary and unary) / operator* / operator/ (scalar)",
	example: ```cpp
	Vector<double> a{1, 2, 3}, b{4, 5, 6};
	assert(a + b == Vector<double>({5, 7, 9}));
	assert(b - a == Vector<double>({3, 3, 3}));
	assert(a * 2.0 == Vector<double>({2, 4, 6}));
	assert(b / 2.0 == Vector<double>({2, 2.5, 3}));
	assert(-a == Vector<double>({-1, -2, -3}));
	EXPECT_LINALG_THROW(a + Vector<double>({1, 2})); // size mismatch
	```,
	description: [
		Elementwise addition and subtraction (matching sizes required), scalar
		multiplication and division, and unary negation. Each returns a new
		vector.
	],
	params: (
		("rhs", [vector of matching size (`+`, `-`)]),
		("scalar", [scalar multiplier / divisor (`*`, `/`)]),
	),
	returns: [
		The resulting vector; `+`/`-` throw `DimensionMismatch` on size mismatch.
	],
)

#method(
	"operator+= / operator-= / operator*= / operator/=",
	example: ```cpp
	Vector<double> a{1, 2, 3}, b{4, 5, 6};
	Vector<double> c = a;
	Vector<double>& ref = (c += b);
	assert(&ref == &c && c == Vector<double>({5, 7, 9})); // returns *this
	c -= b; assert(c == a);
	c *= 10.0; c /= 10.0; assert(c == a);
	```,
	description: [
		In-place compound assignment mirroring the binary operators; each mutates
		this vector and returns a reference to it.
	],
	params: (
		("rhs", [vector of matching size (`+=`, `-=`)]),
		("scalar", [scalar multiplier / divisor (`*=`, `/=`)]),
	),
	returns: [
		Reference to this vector; `+=`/`-=` throw `DimensionMismatch` on mismatch.
	],
)

#method(
	"bool operator==(const Vector&) / operator!=(const Vector&)",
	example: ```cpp
	Vector<double> a{1, 2, 3};
	assert(a == Vector<double>({1, 2, 3}));
	assert(a != Vector<double>({1, 2}));  // different size
	```,
	description: [
		Exact equality / inequality: equal means same size and every element
		exactly equal. For floating-point tolerance use `isApprox`.
	],
	params: (("rhs", [vector to compare against]),),
	returns: [
		`true` / `false` per the comparison.
	],
)

#method(
	"Vector scaledBy(const T&) / elementwiseProduct / elementwiseQuotient",
	example: ```cpp
	Vector<double> a{1, 2, 3}, b{4, 5, 6};
	assert(a.scaledBy(3.0) == Vector<double>({3, 6, 9}));
	assert(a.elementwiseProduct(b) == Vector<double>({4, 10, 18}));
	assert(b.elementwiseQuotient(a) == Vector<double>({4, 2.5, 2}));
	```,
	description: [
		`scaledBy` is a named mirror of `operator*` (scalar). `elementwiseProduct`
		and `elementwiseQuotient` are the Hadamard product and quotient.
	],
	params: (
		("scalar", [scalar multiplier (`scaledBy`)]),
		("rhs", [vector of matching size (elementwise ops)]),
	),
	returns: [
		The resulting vector; elementwise ops throw `DimensionMismatch` on
		mismatch.
	],
)

== Products

#method(
	"T dot(const Vector&)  /  T hermitianDot(const Vector&)",
	example: ```cpp
	Vector<double> a{1, 2, 3}, b{4, 5, 6};
	assert(a.dot(b) == 32.0);          // 4 + 10 + 18
	// hermitianDot conjugates the LEFT argument
	Vector<C> x{C(1, 2), C(3, -1)}, y{C(2, 0), C(1, 1)};
	assert(std::abs(x.hermitianDot(y) - C(4, 0)) < 1e-12);
	assert(std::abs(x.hermitianDot(x) - C(15, 0)) < 1e-12); // == ||x||^2
	```,
	description: [
		`dot` is the unconjugated bilinear form $sum x_i y_i$, even for complex
		`T`. `hermitianDot` conjugates *this* (the left argument):
		$sum overline(x_i) y_i$. The two coincide for real scalars; inner-product
		norms want `hermitianDot`.
	],
	params: (("rhs", [vector of matching size]),),
	returns: [
		The scalar product; throws `DimensionMismatch` on size mismatch.
	],
)

#method(
	"Matrix<T> outer(const Vector& rhs)",
	example: ```cpp
	Vector<double> x{1, 2}, y{10, 20, 30};
	Matrix<double> o = x.outer(y);   // 2x3, (i,j) = x_i * y_j
	assert(o(0, 0) == 10.0 && o(1, 1) == 40.0);
	// complex outer conjugates the right factor: i * conj(i) = 1
	Vector<C> cx{C(0, 1)}, cy{C(0, 1)};
	assert(cx.outer(cy)(0, 0) == C(1, 0));
	```,
	description: [
		Outer product $x y^H$ (the right factor is conjugated for complex `T`).
	],
	params: (("rhs", [right operand]),),
	returns: [
		A $"size"() times "rhs.size"()$ matrix.
	],
)

#method(
	"Vector cross(const Vector& rhs)",
	example: ```cpp
	Vector<double> e1{1, 0, 0}, e2{0, 1, 0};
	assert(e1.cross(e2) == Vector<double>({0, 0, 1}));
	assert(e2.cross(e1) == Vector<double>({0, 0, -1})); // anticommutative
	EXPECT_LINALG_THROW(e1.cross(Vector<double>({1, 2}))); // non-3 throws
	```,
	description: [
		Cross product, defined for 3-vectors only.
	],
	params: (("rhs", [right operand; must have size 3]),),
	returns: [
		The cross-product vector; throws `DimensionMismatch` if either vector is
		not size 3.
	],
)

#method(
	"Vector axpy(const T& alpha, const Vector& y)",
	example: ```cpp
	// fused scale-and-add: alpha * (*this) + y
	V lhs = (a + b) * x;
	V rhs = (a * x).axpy(1.0, b * x);   // == a*x + b*x
	assert(lhs.isApprox(rhs, 1e-10));
	```,
	description: [
		Fused scale-and-add, $alpha dot ("*this") + y$.
	],
	params: (
		("alpha", [scalar multiplier applied to this vector]),
		("y", [vector added to the scaled result; must match size]),
	),
	returns: [
		The vector $alpha dot ("*this") + y$; throws `DimensionMismatch` on
		mismatch.
	],
)

== Norms & reductions

#method(
	"Real norm() / squaredNorm() / oneNorm() / infinityNorm() / pNorm(Real p)",
	example: ```cpp
	Vector<double> v{3.0, -4.0};
	assert(std::abs(v.norm() - 5.0) < 1e-12);          // Euclidean
	assert(std::abs(v.squaredNorm() - 25.0) < 1e-12);
	assert(std::abs(v.oneNorm() - 7.0) < 1e-12);       // sum |x_i|
	assert(std::abs(v.infinityNorm() - 4.0) < 1e-12);  // max |x_i|
	assert(std::abs(v.pNorm(1.0) - 7.0) < 1e-12);
	// norm() scales to avoid overflow; squaredNorm() may overflow
	Vector<double> big{1e200, 1e200};
	assert(std::isfinite(big.norm()) && std::isinf(big.squaredNorm()));
	```,
	description: [
		`norm` is the Euclidean (2-)norm, computed with scaling so entries near
		the overflow threshold do not square to infinity. `squaredNorm` is the
		unscaled $sum abs(x_i)^2$ and may overflow where `norm` does not.
		`oneNorm`, `infinityNorm`, and `pNorm` are the 1-, $infinity$-, and
		general $p$-norms. An empty vector has norm 0.
	],
	params: (("p", [the norm order, $p >= 1$ (`pNorm`)]),),
	returns: [
		The requested norm as `Real`.
	],
)

#method(
	"T sum()  /  T product()",
	example: ```cpp
	Vector<double> a{1, 2, 3};
	assert(a.sum() == 6.0);
	assert(a.product() == 6.0);
	```,
	description: [
		Sum and product of all elements.
	],
	params: (),
	returns: [
		The total / product as `T`.
	],
)

#method(
	"Vector normalized()  /  void normalize()",
	example: ```cpp
	Vector<double> v{3.0, 4.0};
	Vector<double> u = v.normalized();  assert(std::abs(u.norm() - 1.0) < 1e-12);
	v.normalize();                      assert(std::abs(v.norm() - 1.0) < 1e-12);
	```,
	description: [
		Scales to unit Euclidean norm: `normalized` returns a new vector,
		`normalize` scales this one in place. Both throw `LinalgError` if the
		vector is zero.
	],
	params: (),
	returns: [
		The unit vector (`normalized`) / nothing (`normalize`).
	],
)

#method(
	"Index maxAbsIndex() / minAbsIndex()  /  T maxCoefficient() / minCoefficient()",
	example: ```cpp
	Vector<double> v{-5.0, 2.0, -1.0};
	assert(v.maxCoefficient() == 2.0 && v.minCoefficient() == -5.0); // by value
	assert(v.maxAbsIndex() == 0 && v.minAbsIndex() == 2);            // by magnitude
	EXPECT_LINALG_THROW(Vector<double>().maxCoefficient());          // empty throws
	```,
	description: [
		`maxAbsIndex` / `minAbsIndex` return the index of the largest / smallest
		*magnitude*. `maxCoefficient` / `minCoefficient` return the largest /
		smallest *coefficient* (by real part for complex `T`, since $CC$ has no
		natural order). Ties resolve to the lowest index; an empty vector throws
		`LinalgError`.
	],
	params: (),
	returns: [
		The index (`Index`) / coefficient (`T`).
	],
)

#method(
	"bool isApprox(const Vector& other, Real tolerance)  /  bool hasNaN()",
	example: ```cpp
	Vector<double> a{1, 2, 3}, b{1 + 1e-10, 2 - 1e-10, 3};
	assert(a.isApprox(b, 1e-9) && !a.isApprox(b, 1e-12));
	assert(!a.isApprox(Vector<double>({1, 2}), 1e9)); // size mismatch -> false
	assert(Vector<double>({1, std::nan(""), 3}).hasNaN());
	```,
	description: [
		`isApprox` tests same-size, per-element agreement within an absolute
		`tolerance` (a size mismatch is simply `false`, not an exception).
		`hasNaN` reports whether any element (real or imaginary part) is NaN.
	],
	params: (
		("other", [vector to compare against (`isApprox`)]),
		("tolerance", [absolute per-element tolerance (`isApprox`)]),
	),
	returns: [
		`true` / `false` per the test.
	],
)

== Shape manipulation

#method(
	"Vector segment(Index start, Index count) / head(Index) / tail(Index)",
	example: ```cpp
	Vector<double> v{1, 2, 3, 4, 5};
	assert(v.segment(1, 3) == Vector<double>({2, 3, 4}));
	assert(v.head(2) == Vector<double>({1, 2}));
	assert(v.tail(2) == Vector<double>({4, 5}));
	```,
	description: [
		Extract a contiguous sub-vector: `segment` from `start` for `count`
		elements, `head` the leading `count`, `tail` the trailing `count`.
	],
	params: (
		("start", [index of the first element (`segment`, 0-based)]),
		("count", [number of elements to take]),
	),
	returns: [
		The requested sub-vector; throws `IndexOutOfRange` if the range exceeds
		`size()`.
	],
)

#method(
	"Vector concat(const Vector& rhs)  /  Vector reversed()",
	example: ```cpp
	Vector<double> a{1, 2}, b{3, 4};
	assert(a.concat(b) == Vector<double>({1, 2, 3, 4}));
	assert(a.reversed() == Vector<double>({2, 1}));
	```,
	description: [
		`concat` appends `rhs` after this vector; `reversed` returns the elements
		in reverse order.
	],
	params: (("rhs", [vector to append (`concat`)]),),
	returns: [
		The concatenated / reversed vector.
	],
)

#method(
	"Matrix<T> asColumnMatrix() / asRowMatrix() / asDiagonalMatrix()",
	example: ```cpp
	Vector<double> x{1, 2};
	assert(x.asColumnMatrix().cols() == 1 && x.asColumnMatrix()(1, 0) == 2.0);
	assert(x.asRowMatrix().rows() == 1 && x.asRowMatrix()(0, 1) == 2.0);
	Matrix<double> d = x.asDiagonalMatrix();
	assert(d(0, 0) == 1.0 && d(1, 1) == 2.0 && d(0, 1) == 0.0);
	```,
	description: [
		Materialize the vector as a matrix: a $"size"() times 1$ column, a
		$1 times "size"()$ row, or a $"size"() times "size"()$ diagonal matrix.
		Each returns an owning copy.
	],
	params: (),
	returns: [
		The requested matrix.
	],
)

#method(
	"void resize(Index)  /  void conservativeResize(Index)",
	example: ```cpp
	Vector<double> r{1, 2, 3};
	r.resize(2);              assert(r == Vector<double>({0, 0})); // discards
	Vector<double> v{1, 2, 3};
	v.conservativeResize(5);  assert(v == Vector<double>({1, 2, 3, 0, 0})); // keeps
	```,
	description: [
		`resize` changes the length, discarding all existing contents.
		`conservativeResize` keeps overlapping elements and zero-fills any growth.
	],
	params: (("size", [new length]),),
	returns: [
		Nothing.
	],
)

#method(
	"void fill(const T&) / setZero() / setUnit(Index axis) / swap(Vector&)",
	example: ```cpp
	Vector<double> v(3);
	v.fill(5.0);      assert(v(0) == 5.0);
	v.setZero();      assert(v(0) == 0.0);
	v.setUnit(1);     assert(v == Vector<double>({0, 1, 0}));
	Vector<double> a{1, 2}, b{3, 4}; a.swap(b);
	assert(a(0) == 3 && b(0) == 1);
	```,
	description: [
		In-place mutators: set every element to a value / to zero / to the basis
		vector $e_"axis"$, or swap contents with another vector.
	],
	params: (
		("value", [value written to each element (`fill`)]),
		("axis", [index of the single nonzero entry (`setUnit`, 0-based)]),
		("other", [vector to swap with (`swap`)]),
	),
	returns: [
		Nothing.
	],
)

#method(
	"std::string toString(int precision) const",
	example: ```cpp
	assert(Vector<double>().toString(3) == "[]");
	assert(Vector<double>({1.0}).toString(3) == "[1]");
	```,
	description: [
		Formats the vector as a text string, for logging.
	],
	params: (("precision", [number of digits shown per element]),),
	returns: [
		A human-readable string.
	],
)
