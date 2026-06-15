# Ablation candidates

**Status:** LOCKED with deferred final cut.

## Decision: defer the final cut

The actual ablation knob set is **locked after the easy/medium/hard prototype runs**, not now. Two reasons:

1. **Architecturally free.** The prototype generator already accepts a `{knob: level}` config. An ablation profile is `hard everywhere except K_x = medium` — same code path, different config dict. No new implementation work to add ablations after the prototype.
2. **Empirically better.** The qualitative ranking below *predicts* which knobs should produce the biggest marginal-contribution deltas. After the prototype runs, we'll have *measured* committee deltas per knob per stage. Picking from measurements strictly dominates picking from arguments.

**Final-cut timing:** between [../plan.md](../plan.md) Next Steps 5 ("Prototype Scenario 1") and 7 ("Difficulty validation"). Two checkpoint questions are also deferred to that point: (a) relax-levels — `hard→medium` only vs `hard→medium and hard→easy`, and (b) the Knob 1 vs Knob 2 trade-off.

## Selection criteria (locked now)

1. **Stage isolation.** If knob X is the only primary lever for stage Y, ablating it is mandatory for any stage-Y marginal-contribution story.
2. **Cross-domain baseline variance.** Knobs whose hard profile diverges sharply from baseline produce the biggest, most domain-specific deltas.
3. **Clean marginal story.** The knob's effect should be interpretable on its own, not just as an interaction.
4. **Monotone signal strength.** Committee-validated drop expected to be sharp, not noisy.

## Provisional shortlist (4 knobs + 1 alternate)

- **Knob 8 (schema naming)** — *mandatory* for any SM-isolation story; only SM-primary lever in S1. Biggest cross-domain spread of any knob (companies hard / games medium / music easy span the full Knob 8 scale at baseline).
- **Knob 5 (format/unit diversity)** — *mandatory* for any normalization-isolation story; only normalization-primary lever. Three completely different per-domain shapes (companies hard sub-domain on financials / games below-easy uniform / music hard on dates and durations).
- **Knob 2 (niche density)** — cleanest "count of hard pairs" lever for blocking + EM, independent of Knob 1's per-pair intensity. Each domain has a different natural niche signal (companies derived / games franchise / music label-collision).
- **Knob 3 (attribute drop)** — only knob that hits **three** stages (block + EM + fusion) with a single mechanism. Asymmetric per-domain baselines make the marginal contribution very domain-specific.
- **Knob 1 (surface augmentation) — alternate.** Promote into the shortlist if Knob 2's measured EM delta turns out small/noisy at the prototype phase, or if per-pair paraphrase intensity has a distinct signal from corner-case count.

## Soft constraint: one ablation per pipeline stage cluster

Whatever the final cut is, it must give every PyDI stage at least one ablation lever:

- **SM** → Knob 8 (mandatory in S1; Knob 9 added in S2)
- **Normalization** → Knob 5 (mandatory)
- **Blocking + EM** → Knob 2 (or Knob 1 as alternate)
- **Cross-stage / Fusion** → Knob 3 (or Knob 4)

## Hard requirement on the prototype implementation

Every knob must be independently togglable to any of {easy, medium, hard} via the difficulty config. **The provisional shortlist (or any future revision) must work without code changes — just config swaps.**

## S2 note — Knob 9 as the second SM ablation

Scenario 2 unlocks Knob 9 (schema completeness / distractors), the second primary SM lever. Natural S2 ablation set = S1 set + Knob 9, which decomposes "S2-only SM difficulty" into its naming-divergence (K8) and schema-completeness (K9) components. Out of scope for v1 prototype.

## Final-cut checklist (between plan.md Steps 5 and 7)

1. Run the easy/medium/hard prototype on at least one domain.
2. Read off committee deltas per knob per stage from the prototype output.
3. Verify the provisional shortlist against the measured deltas: each shortlist knob should produce a sharp, monotone `hard→medium` drop on its primary stage. If not, swap with the alternate (Knob 1) or with another knob from the criteria-ranked list.
4. Decide relax-levels (`hard→medium` only vs `hard→medium and hard→easy`) based on whether the measured deltas are large enough to need mid-point resolution.
5. Lock the final ablation set + relax-levels and proceed to plan.md Step 7.
