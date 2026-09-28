## A3 — semiempirical method name

- Top-level PQR record keys containing 'pm': ['pm7']
- Keys inside the 'pm7' block: ['alpha', 'dipole', 'dipoleMoment', 'heatOfFormation', 'homo', 'lumo', 'orbitals', 'polarizability', 'rotational', 'surfaceArea', 'volume']
- Explicit PM6/PM7/MOPAC strings found in the first 5,000 records: NONE — the only evidence is the field name

**Finding.** The PQR download exposes the properties under a field literally named `pm7`, and no record in the first 5,000 carries any other method string. So the data as distributed is labelled PM7 by the source field name.

**However, the field name is not provenance.** The PQR publication and documentation should be checked directly, and two referees independently described the data as PM6. A field name is weaker evidence than the source publication.

**Recommendation:** do NOT run a global find-and-replace on this evidence alone. Confirm against the PQR paper/documentation, then use one term throughout. If the documentation says PM6, the `pm7` field name is a misnomer in the download and the manuscript should say PM6 and note the field name.
