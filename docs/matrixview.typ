// Documentation for include/linalg/core/MatrixView.hpp.
// Included from documentation.typ; imports the template for the `method` env.
#import "template.typ": *

= Views

A `MatrixView` is a non-owning, stride-aware window onto another matrix's
storage. Element $(i, j)$ lives at `data[i * rowStride + j * colStride]`, so
arbitrary strides make rows, columns, diagonals, and transposes all
expressible as views without moving any elements. Blocked algorithms operate
on views so panels are never copied. A view is invalidated when the owning
matrix resizes or dies.

Copy construction and copy assignment *rebind* the view (shallow: same
pointer, shape, and strides); they do not touch elements. Assigning from a
`Matrix`, or calling `copyFrom`, copies *elements* into the viewed storage and
requires matching shapes. `ConstMatrixView` is the read-only counterpart and
converts implicitly from `MatrixView`.

== Construction & assignment

#method(
	"MatrixView()",
	example: ```cpp
	MatrixView<double> v;
	assert(v.rows() == 0 && v.cols() == 0);
	assert(v.isEmpty() && v.data() == nullptr);
	```,
	description: [
		Constructs an empty view: null data, zero shape and strides.
	],
	params: (),
	returns: [
		An empty view.
	],
)

#method(
	"MatrixView(T* data, Index rows, Index cols, Index rowStride, Index colStride)",
	example: ```cpp
	double a[12];                      // a 3x4 row-major buffer
	MatrixView<double> v(a, 3, 4, 4, 1);
	assert(v.rows() == 3 && v.cols() == 4);
	assert(v.rowStride() == 4 && v.colStride() == 1);
	// a column-major window over the same kind of buffer uses strides (1, rows)
	double c[6] = {1, 2, 3, 4, 5, 6}; // columns (1,2,3) and (4,5,6)
	MatrixView<double> cm(c, 3, 2, 1, 3);
	assert(cm(2, 0) == 3 && cm(0, 1) == 4);
	```,
	description: [
		Constructs a view over caller-owned storage. `data` points at the view's
		element $(0, 0)$; `rowStride` / `colStride` are how many elements to
		advance to move down one row / right one column.
	],
	params: (
		("data", [pointer to the element at position $(0, 0)$ of the view]),
		("rows", [number of rows the view exposes]),
		("cols", [number of columns the view exposes]),
		("rowStride", [elements to advance in `data` to move down one row]),
		("colStride", [elements to advance in `data` to move right one column]),
	),
	returns: [
		A view over the given storage.
	],
)

#method(
	"MatrixView(const MatrixView& other) / MatrixView(MatrixView&& other)",
	example: ```cpp
	auto v = /* a 3x4 view over buffer b */;
	MatrixView<double> w(v);        // shallow: same storage
	assert(w.data() == v.data());
	w(0, 0) = 42.0;
	assert(v(0, 0) == 42.0);        // writes visible through the original
	MatrixView<double> m(std::move(v));
	assert(v.isEmpty() && v.data() == nullptr); // moved-from is empty
	```,
	description: [
		Copy construction rebinds this view to the same storage as `other`
		(shallow: shares pointer, shape, strides). Move construction takes over
		`other`'s binding and leaves it empty; since the view is non-owning it is
		equivalent to a shallow copy.
	],
	params: (("other", [view to copy the binding from / move from]),),
	returns: [
		A view bound to `other`'s storage.
	],
)

#method(
	"MatrixView& operator=(const MatrixView&) / operator=(MatrixView&&)",
	example: ```cpp
	double other[4] = {7, 7, 7, 7};
	MatrixView<double> o(other, 2, 2, 2, 1);
	o = v;                     // rebinds; does NOT copy elements
	assert(o.data() == v.data() && o.rows() == 3);
	assert(other[0] == 7.0);   // the old target storage is untouched
	```,
	description: [
		Rebinds this view to `other`'s storage (shallow); no elements are copied.
		Move assignment additionally empties `other` and is safe under
		self-assignment.
	],
	params: (("other", [view to rebind to / move from]),),
	returns: [
		Reference to this view.
	],
)

#method(
	"MatrixView& operator=(const Matrix<T>& source)",
	example: ```cpp
	auto blk = a.block(0, 0, 2, 2);   // view into matrix a
	blk = Matrix<double>{{-1, -2}, {-4, -5}};
	assert(a(0, 0) == -1 && a(1, 1) == -5); // elements copied into a
	assert(a(0, 2) == 3);                   // outside the block untouched
	```,
	description: [
		Copies the elements of `source` into the storage this view refers to; the
		binding is untouched. Shapes must match exactly.
	],
	params: (("source", [matrix whose element values are copied in]),),
	returns: [
		Reference to this view.
	],
)

== Element access & shape

#method(
	"T& operator()(Index i, Index j)  /  const T& operator()(...) const",
	example: ```cpp
	assert(v(0, 0) == 0.0 && v(1, 2) == 12.0 && v(2, 3) == 23.0);
	v(1, 2) = -5.0;
	assert(b.a[6] == -5.0); // writes land in the underlying buffer
	```,
	description: [
		Unchecked element access, honoring the strides. Indices are relative to
		the view, not to the owning matrix.
	],
	params: (
		("i", [row index, relative to this view (0-based)]),
		("j", [column index, relative to this view (0-based)]),
	),
	returns: [
		Reference to element $(i, j)$ in the viewed storage.
	],
)

#method(
	"T& at(Index i, Index j)  /  const T& at(...) const",
	example: ```cpp
	assert(v.at(2, 3) == 23.0);
	v.at(0, 1) = 99.0;
	bool threw = false;
	try { v.at(3, 0); } catch (const IndexOutOfRange&) { threw = true; }
	assert(threw);
	```,
	description: [
		Bounds-checked element access. Otherwise identical to `operator()`.
	],
	params: (
		("i", [row index, relative to this view (0-based)]),
		("j", [column index, relative to this view (0-based)]),
	),
	returns: [
		Reference to element $(i, j)$; throws `IndexOutOfRange` if
		$i >= "rows"()$ or $j >= "cols"()$.
	],
)

#method(
	"Index rows() / cols() / rowStride() / colStride() const",
	example: ```cpp
	MatrixView<double> v(a, 3, 4, 4, 1);
	assert(v.rows() == 3 && v.cols() == 4);
	assert(v.rowStride() == 4 && v.colStride() == 1);
	```,
	description: [
		The view's shape and the strides that map indices to storage offsets.
	],
	params: (),
	returns: [
		The requested count as `Index`.
	],
)

#method(
	"bool isContiguous() / isEmpty() const",
	example: ```cpp
	assert(b.view().isContiguous());              // dense, row-major, gap-free
	assert(!b.view().block(0, 0, 3, 2).isContiguous()); // skips columns
	assert(b.view().block(1, 0, 2, 4).isContiguous());  // full-width block
	assert(MatrixView<double>(&x, 0, 5, 5, 1).isEmpty());
	```,
	description: [
		`isContiguous` reports whether the view is a dense row-major block with no
		gaps ($"rowStride" = "cols"$ and $"colStride" = 1$), so the storage can be
		treated as a flat buffer. `isEmpty` reports whether `rows()` or `cols()`
		is zero.
	],
	params: (),
	returns: [
		`true` if the view is contiguous / empty, otherwise `false`.
	],
)

#method(
	"T* data()  /  const T* data() const",
	example: ```cpp
	assert(v.data() == b.a); // points at element (0, 0)
	// honor the strides unless isContiguous() is true
	```,
	description: [
		Direct access to the underlying buffer at position $(0, 0)$. Callers must
		honor the strides unless `isContiguous()` is true.
	],
	params: (),
	returns: [
		Pointer to the element at position $(0, 0)$.
	],
)

== Sub-views

#method(
	"MatrixView block(Index i, Index j, Index numRows, Index numCols)",
	example: ```cpp
	auto blk = v.block(1, 1, 2, 3);
	assert(blk.rows() == 2 && blk.cols() == 3);
	assert(blk(0, 0) == 11.0 && blk(1, 2) == 23.0);
	blk(0, 1) = -1.0;              // writes visible in the parent
	auto inner = blk.block(1, 1, 1, 2); // nested indices are block-relative
	assert(inner(0, 0) == 22.0);
	```,
	description: [
		Creates a sub-view over a rectangular block, sharing this view's storage.
		Indices are relative to this view, and nested blocks compose relative to
		their parent.
	],
	params: (
		("i", [row index of the block's top-left corner (0-based)]),
		("j", [column index of the block's top-left corner (0-based)]),
		("numRows", [number of rows in the block]),
		("numCols", [number of columns in the block]),
	),
	returns: [
		A view onto the block; throws `IndexOutOfRange` if the block does not fit
		within this view.
	],
)

#method(
	"MatrixView row(Index i)  /  MatrixView col(Index j)",
	example: ```cpp
	auto r = v.row(1);  assert(r.rows() == 1 && r.cols() == 4);
	auto c = v.col(2);  assert(c.rows() == 3 && c.cols() == 1);
	c(1, 0) = 77.0;     // shares storage
	assert(v(1, 2) == 77.0);
	```,
	description: [
		Creates a $1 times "cols"$ view onto a single row, or a
		$"rows" times 1$ view onto a single column, sharing this view's storage.
	],
	params: (("i / j", [row index / column index (0-based)]),),
	returns: [
		A one-row / one-column view; throws `IndexOutOfRange` if the index is out
		of range.
	],
)

#method(
	"MatrixView diagonal()",
	example: ```cpp
	auto d = v.diagonal(); // 1 x min(rows, cols)
	assert(d.rows() == 1 && d.cols() == 3);
	assert(d(0, 0) == 0.0 && d(0, 1) == 11.0 && d(0, 2) == 22.0);
	d(0, 1) = -3.0;
	assert(v(1, 1) == -3.0);
	```,
	description: [
		Creates a $1 times min("rows", "cols")$ view onto the main diagonal,
		sharing this view's storage.
	],
	params: (),
	returns: [
		A one-row view onto the diagonal.
	],
)

#method(
	"MatrixView transposed()",
	example: ```cpp
	auto t = v.transposed();
	assert(t.rows() == 4 && t.cols() == 3);
	assert(t.rowStride() == v.colStride() && t.colStride() == v.rowStride());
	t(3, 0) = -9.0;             // shares storage
	assert(v(0, 3) == -9.0);
	```,
	description: [
		Creates a transposed view by swapping shape and strides; no elements are
		moved. The round-trip `transposed().transposed()` recovers the original.
	],
	params: (),
	returns: [
		A $"cols" times "rows"$ view onto the same storage.
	],
)

#method(
	"Matrix<T> toMatrix() const",
	example: ```cpp
	M blockCopy = a.block(0, 1, 2, 2).toMatrix(); // deep copy
	blockCopy(0, 0) = 99;
	assert(a(0, 1) == 2);   // source unaffected
	M t = a.view().transposed().toMatrix(); // works through strided views
	```,
	description: [
		Deep-copies the viewed elements into a fresh contiguous `Matrix`, walking
		the strides so transposed and other strided views copy correctly.
	],
	params: (),
	returns: [
		A newly allocated `Matrix` holding a copy of the elements.
	],
)

== Element mutation

#method(
	"void fill(const T& value)  /  void setZero()",
	example: ```cpp
	auto blk = v.block(0, 0, 2, 2);
	blk.fill(5.0);
	assert(v(0, 0) == 5.0 && v(1, 1) == 5.0);
	assert(v(0, 2) == 2.0); // outside the block untouched
	blk.setZero();
	assert(v(0, 0) == 0.0);
	```,
	description: [
		Writes a constant (or zero) to every element the view refers to; storage
		outside the view is left alone.
	],
	params: (("value", [value written to each element (`fill` only)]),),
	returns: [
		Nothing.
	],
)

#method(
	"void scale(const T& factor)",
	example: ```cpp
	blk.scale(3.0);
	assert(v(0, 0) == 15.0); // 5.0 * 3.0
	// complex: scale by i rotates each element
	MatrixView<C> cv(a, 2, 2, 2, 1);
	cv.scale(C(0, 1));
	```,
	description: [
		Multiplies every element of the view in place by `factor`.
	],
	params: (("factor", [scalar multiplier applied to each element]),),
	returns: [
		Nothing.
	],
)

#method(
	"void swapWith(MatrixView& other)",
	example: ```cpp
	MatrixView<double> va(a, 2, 2, 2, 1), vb(b, 2, 2, 2, 1);
	va.swapWith(vb);
	assert(a[0] == 5 && b[0] == 1); // elements exchanged...
	assert(va.data() == a && vb.data() == b); // ...bindings stay put
	```,
	description: [
		Exchanges the elements of this view with those of `other`; the bindings
		are left unchanged. Shapes must match.
	],
	params: (("other", [view whose elements are exchanged with this view's]),),
	returns: [
		Nothing.
	],
)

#method(
	"void copyFrom(const MatrixView& source)",
	example: ```cpp
	vd.copyFrom(vs);
	assert(dst[0] == 1 && dst[3] == 4);
	assert(src[0] == 1);            // source untouched
	vd2.copyFrom(vs.transposed()); // copies across differing strides
	```,
	description: [
		Copies elements from `source` into this view. Shapes must match and the
		source must not alias this view's storage; differing strides (e.g. a
		transposed source) are handled correctly.
	],
	params: (("source", [view to copy element values from]),),
	returns: [
		Nothing.
	],
)

== ConstMatrixView

`ConstMatrixView` mirrors the read-only half of the API — construction, element
access (`operator()`, `at`), `rows`/`cols`/`rowStride`/`colStride`,
`isContiguous`/`isEmpty`, `data() const`, the sub-view factories (`block`,
`row`, `col`, `diagonal`, `transposed`), and `toMatrix` — with identical
semantics. It adds one constructor for the mutable-to-const conversion.

#method(
	"ConstMatrixView(const MatrixView<T>& view)",
	example: ```cpp
	auto mv = b.view();                      // mutable view
	ConstMatrixView<double> converted = mv;  // implicit conversion
	assert(converted.data() == b.a && converted.rows() == 3);
	// sub-views mirror the mutable API
	auto blk = converted.block(1, 1, 2, 3);
	assert(blk(0, 0) == 11.0 && blk(1, 2) == 23.0);
	```,
	description: [
		Implicitly constructs a read-only view from a mutable one, so a
		`MatrixView` can be passed wherever only reads are needed. (There is also
		the direct `ConstMatrixView(const T* data, rows, cols, rowStride,
		colStride)` constructor, mirroring the mutable one over `const` storage.)
	],
	params: (("view", [mutable view to expose as read-only]),),
	returns: [
		A read-only view sharing `view`'s storage.
	],
)
