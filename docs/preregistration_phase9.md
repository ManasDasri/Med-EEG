# Pre-registration (phase 9): introspective fidelity and the lead-up horizon

**Committed before any analysis below was run.** Data: OpenNeuro ds008836
(39 experienced Vipassana meditators, two sessions, 1,707 depth reports
1–5), with the phase 7 features and predictions.

**What has been seen already (disclosed).** Phase 7's leave-one-person-out
`riemann_ridge` predictions, and each person's AUC pooled over both
sessions, were computed and looked at (range roughly 0.4–0.8; 13 randomly
chosen people average 0.544, the other 26 average 0.642). **Not** computed
or looked at: per-*session* scores, their agreement across sessions, their
relation to practice or rating behaviour, and any decoding from parts of the
report window. Participant traits (`retreat_days`, `meditation_years_experience`)
have not been opened in any analysis.

---

## Part A: Introspective fidelity

**Idea.** A person's reports can track their brain state closely or
loosely. We call the person's within-session AUC, from a model that never
saw them, their **fidelity**: how well their self-reports line up with
what their EEG shows. Question: is fidelity a stable trait, and does it grow
with practice?

**Measure.** `F[p, s]` = within-recording AUC (depth above vs below that
recording's mean) of phase 7's `riemann_ridge` LOSO predictions, for person
*p*, session *s*. Recordings whose ratings are all equal are unscorable and
dropped. People with two scorable sessions enter A1.

### A1 (primary): is fidelity stable across sessions?
- Spearman ρ between `F[p, 1]` and `F[p, 2]` across people.
- **One-sided test, ρ > 0**, p from 10,000 permutations of the pairing
  (session-2 scores shuffled across people).
- **Supported if p < 0.05.**
- Noise check, reported with it: the same ρ under 1,000 circular shifts of
  the ratings within each recording (the model's predictions fixed). It
  should centre on 0. If it doesn't, the measure has an artifact, and A1 is
  reported as uninterpretable.

### A1b: not just rating style
Partial Spearman ρ of the session pair, controlling for each session's
rating SD and its number of distinct ratings (the AUC's resolution depends
on both). Reported alongside A1. If A1 passes but A1b's ρ drops to near 0
(below half of A1's), fidelity is read as **rating style**, not
introspection.

### A2: does fidelity grow with practice?
- Each person's fidelity pooled over sessions (mean of the scorable
  sessions) vs log(1 + `retreat_days`), and vs `meditation_years_experience`.
- Spearman ρ, **one-sided ρ > 0**, permutation p (10,000), **Holm across the
  two**.
- Exploratory extras, labelled as such: partial ρ controlling for age and
  sex; `meditation_frequency` and `meditation_length`.

**Caveat, fixed in advance.** Every participant has ≥ 5 years of practice,
so the range is narrow and 39 people give low power. A null A2 is reported
as "not detected", not as "no relation".

### A3 (interpretation check): the group model vs the person's own model
Fidelity from a group model also depends on how typical a person's brain
signature is. Check: for each person, train on their *other* session only
(ridge on the tangent vectors, alpha fixed at the group value) and score
this session; average both directions. Report its Spearman ρ with the
group-model fidelity, and with A2's practice measures. Descriptive only.

## Part B: The lead-up horizon

**Idea.** How far before a report does the EEG already carry the depth that
will be reported? And does this differ between **spontaneous** reports
(the meditator chooses when to report, `emerge` blocks) and **prompted**
ones (`probe` blocks)?

**Bins.** Each report's window is split by how long before the report each
2-s segment starts: **B1 0–10 s, B2 10–30 s, B3 30–60 s, B4 60–120 s.** A
report enters a bin only if it has ≥ 3 clean segments there. Windows
already stop at the previous report, so bins never overlap the previous
report.

**Model.** Per bin: OAS covariance of that bin's segments → re-centred per
recording → tangent space (as phase 7) → ridge with alpha by
person-grouped CV (as phase 8). LOSO, within-person AUC, 200 circular-shift
permutations.

### B1 (primary): is depth decodable a minute or more before the report?
- Bin B4 (60–120 s) AUC > chance, permutation p.
- **Holm across the four bins.** Supported if B4's corrected p < 0.05.

### B2: does it fade with distance?
- Per person, the slope of AUC across B1→B4 (bin index 1–4).
- One-sample t-test of the slopes against 0, **one-sided, slope < 0**.

### B3: spontaneous vs prompted
- Separate models per report type (trained and scored within type), bins
  as above. Per person: the near-minus-far difference (B1 − B4) for each
  type.
- Paired test (Wilcoxon signed-rank) of spontaneous vs prompted. Prediction
  (one-sided): **spontaneous reports show a larger near-minus-far gain**,
  because the meditator reports when they notice a change.
- People without a scorable B1 and B4 in both types are dropped and
  counted.
- Reported as **secondary**: the per-type, per-bin report counts are
  smaller.

## What would change this plan
Only outcome-independent fixes (a crash, a wrong file, too few reports in a
bin to fit at all), each listed in the results with its time. No bin,
window, model, threshold or test changes after results are seen.
