## T2 — are any learned breakpoint features fingerprint bits?

**No. None of the five is a Morgan bit, and all five resolve uniquely.**

| Chemical class | Feature | Descriptor | Package | Morgan bit? | Breakpoint | Resolution |
|---|---|---|---|---|---|---|
| qm9 like small organic | `x15` | `HallKierAlpha` — Hall-Kier alpha shape correction | RDKit | **no** | -0.400 | unique (1 candidate) |
| near qm9 larger organic | `x13` | `TPSA` — Topological polar surface area (A^2) | RDKit | **no** | 12.530 | unique (1 candidate) |
| large neutral organic | `x29` | `VSA_EState6` — EState-weighted van der Waals surface area, bin 6 | RDKit | **no** | 0.000 | unique (1 candidate) |
| heteroatom rich non qm9 | `x260` | `Ipc` — Information content of the characteristic polynomial | RDKit | **no** | 1,186.695 | unique (1 candidate) |
| charged or radical | `x47` | `BCUT2D_MRLOW` — BCUT2D eigenvalue, molar-refractivity weighted (lowest) | RDKit | **no** | -0.451 | unique (1 candidate) |

None of the five falls among the 13 features whose names resolved only to a set
of value-identical candidates, so no caveat is needed on any of them.

### What this licenses

Of the 375 LASSO-selected descriptors, **176 (47%) are Morgan
fingerprint bits** — substructure indicators, not named chemical quantities. A broad
claim that the *feature set* is interpretable is not supportable.

But the architecture's interpretability claim does not rest on the feature set. It
rests on the five learned breakpoints, and **every one of them is a named
physicochemical descriptor**: topological polar surface area, Hall–Kier alpha shape
correction, a molar-refractivity-weighted BCUT2D eigenvalue, the information content
of the characteristic polynomial, and an EState-weighted surface-area bin.

### One-line verdict for the manuscript

> Although 176 of the 375 selected descriptors are Morgan fingerprint bits, none of
> the five learned breakpoints falls on one: each split is defined by a named
> physicochemical descriptor, so the partition the model learns is chemically
> interpretable even where the bulk feature block is not.

This is the difference the spec asked about, and it lands on the side of a chemical
result rather than a plot.
