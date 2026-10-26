# Results (phase 9): introspective fidelity and the lead-up horizon

Pre-registration: [`preregistration_phase9.md`](preregistration_phase9.md),
pushed 2026-10-26 15:44; code and implementation choices pushed before any
run. Runs: 15:53–16:01. Everything below the "Exploratory" headings was
decided after the pre-registered results were seen.

## Summary

| Test | Result | Verdict |
|---|---|---|
| **A1** fidelity stable across sessions | ρ = 0.15, p = 0.19 (38 people) | **Not supported**, and the measure can't support it (see below) |
| A1b same, rating style controlled | ρ = 0.10 | — |
| A2 fidelity vs retreat days / years | ρ = 0.17 / 0.13, Holm p = 0.31 / 0.31 | Not detected |
| **B1** depth decodable 60–120 s before the report | AUC 0.584, Holm p = 0.020 | **Supported** |
| B2 decodability fades with distance | slope +0.011 per bin (wrong direction), p = 0.90 | Not supported |
| B3 spontaneous reports build up more | median difference −0.011, p = 0.60 | Not supported |

**The two findings that carry weight:**
1. **The depth a meditator is about to report is already in their EEG a
   minute or two before they report it**, at a roughly constant level. Depth
   behaves like a slow state, not a moment-to-moment one.
2. **With ~21 reports per session, a person's fidelity can't be measured
   reliably.** About 73% of its session-to-session variation is sampling
   noise, so even a perfectly stable trait would correlate at most about
   0.27 across sessions. Measuring it properly needs roughly **6× more
   reports per session (~130)**. That's a concrete design requirement for
   any study of introspective accuracy against EEG.

## Part A: introspective fidelity

Fidelity = the within-session AUC of phase 7's leave-one-person-out
`riemann_ridge` predictions (from a model that never saw the person).

- **A1:** session 1 vs session 2, Spearman ρ = **0.149**, one-sided
  permutation p = 0.19, 38 people (one person has a single usable session).
  The rating-shift null centres on 0 (mean −0.003, 95% −0.32 to +0.34), so
  the measure isn't biased. It's noisy.
- **A1b:** controlling for each session's rating SD and number of distinct
  ratings: ρ = 0.098.
- **A2:** pooled fidelity vs log(1 + retreat days) ρ = 0.170 (p = 0.16) and
  vs years of practice ρ = 0.126 (p = 0.22). Holm p = 0.31 for both.
  Exploratory: partial ρ given age and sex 0.19 / 0.20; sessions per week
  −0.16; session length −0.31. Every participant has ≥ 5 years of
  practice, so the range is narrow (as the plan noted).
- **A3 (descriptive):** a model trained only on the person's *other*
  session averages AUC 0.593, close to the group model's 0.61. Its fidelity
  correlates with the group model's at ρ = 0.54, so the two mostly agree on
  who is well decoded. Its correlation with retreat days is ρ = 0.37, but
  that is exploratory and uncorrected, and it is the kind of number that
  needs a new sample before it means anything.

### Exploratory: how noisy is a per-session fidelity score?
For each session, the sampling variance of an AUC under no signal is
(n₁ + n₂ + 1) / (12 n₁ n₂) (Hanley–McNeil), where n₁ and n₂ are the
reports above and below that session's mean.
- **Observed SD of per-session fidelity:** 0.165.
- **Expected from noise alone:** 0.141.
- **True between-person share of the variance: ≈ 0.27.** That share is also
  the highest test–retest correlation possible, even for a perfectly stable
  trait. A1's 0.15 is consistent with that ceiling, but 38 people can't
  distinguish it from 0.
- **By Spearman–Brown,** a per-session reliability of 0.7 would need about
  6.3× the reports, roughly 130 per session instead of the median 21.

## Part B: the lead-up horizon

Covariance model per bin (re-centred per recording, tangent space, ridge
with person-grouped alpha), leave-one-person-out, 200 circular-shift
permutations, Holm across bins.

| Bin (s before report) | AUC | Chance | p | Holm p | Reports |
|---|---|---|---|---|---|
| B1 0–10 | 0.543 | 0.499 | 0.025 | 0.025 | 1,701 |
| B2 10–30 | 0.564 | 0.501 | 0.005 | 0.020 | 1,687 |
| B3 30–60 | 0.552 | 0.501 | 0.010 | 0.020 | 1,608 |
| **B4 60–120** | **0.584** | 0.500 | 0.005 | **0.020** | 1,414 |

- **B1 supported:** EEG from 60–120 s before a report decodes it.
- **B2 not supported:** there's no fade. The per-person slope across bins is
  +0.011 (39 people).
- **B3 not supported:** near-minus-far gain, spontaneous minus prompted:
  median −0.011, Wilcoxon p = 0.60 (39 people). Per type: spontaneous B1
  0.498, B4 0.527; prompted B1 0.559, B4 0.583.

### Exploratory: equal-length bins
The pre-registered bins grow longer with distance (10, 20, 30, 60 s), and a
longer bin gives a less noisy covariance, so B4's edge could just be more
data. With equal 10-s bins (`src/horizon_equal_bins.py`; one recording
dropped from the last bin because its Riemannian mean didn't converge):

| 10-s bin (s before report) | 0–10 | 20–30 | 50–60 | 80–90 | 110–120 |
|---|---|---|---|---|---|
| AUC | 0.543 | 0.560 | 0.496 | 0.542 | 0.542 |
| p (200 perms) | 0.025 | 0.005 | 0.63 | 0.025 | 0.040 |

Same amount of data, very different distances, similar AUC (apart from a dip
at 50–60 s). So B4's 0.584 is mostly bin length. The fair reading is that
**the signal is about as strong two minutes out as it is just before the
report.**

### Exploratory: spontaneous reports
In the last 10 s before a *spontaneous* report, the EEG carries no depth
information (0.498), while prompted reports do (0.559). One possibility:
deciding to report (the "I noticed something" moment) disturbs the EEG just
before a spontaneous report and masks the state. This wasn't predicted, and
it's a hypothesis for a new sample, not a finding.

## What this means for the project
- **Novel and positive:** reported meditative depth is set well before the
  report; the EEG signature holds steady over about two minutes. That is
  something the earlier decoding papers didn't examine.
- **Novel and methodological:** brain–report fidelity is a natural measure
  of introspective accuracy, but this design can't measure it per session.
  The numbers above say how many reports a study needs to do so.
- Caveats: one dataset; experienced meditators only; exploratory results
  need a new sample.

## Files
`audit-work/evaluation/ds008836-phase9/` (`fidelity/`, `horizon/`,
`equal_bins.npz`, logs).
