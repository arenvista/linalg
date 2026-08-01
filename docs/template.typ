// Math notetaking template
// Usage in a note file:
//   #import "template.typ": *
//   #show: notes.with(title: "Topic", course: "MATH 302", date: "2026-07-30")

#let notes(
	title: "Notes",
	course: none,
	author: none,
	date: none,
	body,
) = {
	set document(title: title, author: if author == none { () } else { author })
	set page(
		paper: "us-letter",
		margin: (x: 1in, y: 1in),
		numbering: "1",
		number-align: center,
	)
	set text(font: "New Computer Modern", size: 11pt, lang: "en")
	set par(justify: true, leading: 0.65em)

	// Headings
	// set heading(numbering: "1.1")
	// show heading.where(level: 1): it => {
	//   set text(size: 15pt)
	//   block(above: 1.4em, below: 0.8em, it)
	// }
	// show heading.where(level: 2): it => {
	//   set text(size: 12.5pt)
	//   block(above: 1.1em, below: 0.6em, it)
	// }

	// Equations numbered by section
	set math.equation(numbering: "(1.1)")

	// Nice links and inline code
	show link: set text(fill: rgb("#1a5fb4"))
	show raw.where(block: false): box.with(
		fill: luma(240),
		inset: (x: 3pt, y: 0pt),
		outset: (y: 3pt),
		radius: 2pt,
	)

	// Title block
	block(
		width: 100%,
		{
			text(size: 20pt, weight: "bold", title)
			if course != none or date != none {
				v(0.3em)
				text(size: 10.5pt, fill: luma(90))[
					#if course != none [ #course ]
					#if author != none [ #author ]
					#if course != none and date != none [ #h(0.5em) · #h(0.5em) ]
				]
			}
			if date != none {
				v(0.2em)
				text(size: 10.5pt, fill: luma(90), date)
			}
		},
	)
	line(length: 100%, stroke: 0.5pt + luma(180))
	v(0.6em)

	body
}

// ---- Callout / theorem-like environments ----

#let _box(name, color, it) = block(
	width: 100%,
	fill: color.lighten(88%),
	stroke: (left: 2.5pt + color),
	inset: (x: 10pt, y: 8pt),
	radius: 2pt,
	above: 1em,
	below: 1em,
	it,
)

#let _labeled(kind, color, title, body) = _box(
	kind,
	color,
	{
		text(weight: "bold", fill: color.darken(15%), kind)
		if title != none [ #text(weight: "bold")[ (#title)] ]
		text(weight: "bold")[.]
		h(0.4em)
		body
	},
)

#let definition(title: none, body) = _labeled("Definition", rgb("#1a5fb4"), title, body)
#let theorem(title: none, body) = _labeled("Theorem", rgb("#a51d2d"), title, body)
#let lemma(title: none, body) = _labeled("Lemma", rgb("#a51d2d"), title, body)
#let corollary(title: none, body) = _labeled("Corollary", rgb("#a51d2d"), title, body)
#let proposition(title: none, body) = _labeled("Proposition", rgb("#a51d2d"), title, body)
#let question(title: none, body) = _labeled("Question", rgb("#a51d2d"), title, body)
#let example(title: none, body) = _labeled("Example", rgb("#2e7d32"), title, body)
#let remark(title: none, body) = _labeled("Remark", rgb("#6c5b00"), title, body)
#let case(title: none, body) = _labeled("Case", rgb("#6c5b00"), title, body)
#let proof(title: none, body) = _labeled("Proof", rgb("#2e7d32"), title, body)

// Proof environment with a QED tombstone
// #let proof(body) = block(
//   above: 1em,
//   below: 1em,
//   stroke: 0.5pt,
//   inset: 1em,
//   radius: 3pt,
//   width: 100%,
// )[
//   #emph[Proof.] #h(0.3em) #body #h(1fr) $square$
// ]

// ---- Handy math shortcuts ----
#let RR = math.bb("R")
#let NN = math.bb("N")
#let ZZ = math.bb("Z")
#let QQ = math.bb("Q")
#let CC = math.bb("C")
#let eps = math.epsilon
#let norm(x) = $lr(bar.v.double #x bar.v.double)$
#let abs(x) = $lr(bar.v #x bar.v)$
