# Round 3: does one calibration session help? (phase 8)

**Written after round 2, before any calibration model was run.**

Rounds 1–2 asked how well depth is decoded for a person the model has never
seen. Round 3 asks a different, practical question: if a person first does
one rated session, does that help decode their next session? Every
ds008836 participant has two sessions, so this can be tested directly.

## Setup
- For each person *p* and session *s*: train on all other people (both
  sessions) **plus p's other session**, with p's own reports weighted by *w*
  in the fit; predict session *s*. The model never sees session *s*.
- Reference: the same scheme with *w* = 0 (p's other session not used),
  i.e. plain `riemann` scored identically.
- Features and learner: `riemann` with the round-2 alpha selection
  (person-grouped CV). Alpha is chosen without p's reports, then the model
  is refit with them at that alpha.
- Score: within-person AUC with each recording's predictions z-scored first
  (session models differ in scale, and the re-centred features can't predict
  session-level offsets anyway). The same scoring for every *w*.

## Search and test
- Candidates: *w* ∈ {0, 1, 5, 20}. Chosen on the 26 search-set people
  (highest AUC; within 0.005, the smaller *w*).
- Final test once on the 13 held-out people: the chosen *w* vs *w* = 0,
  paired per-person difference; 1,000 circular shifts for p.
- Calibration counts as helping only if the held-out paired difference vs
  *w* = 0 has a 95% interval above 0.

## Fix before any result (22:44)
The first launch stopped at a check: sub-029 (held-out) has only one usable
session (its other recording failed the quality exclusion), so it can't be
calibrated. It sits this round out; both *w* = 0 and the chosen *w* are
scored on the same 12 held-out people.
