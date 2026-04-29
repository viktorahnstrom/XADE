# Hand spot-check log — grounded explanations

For each row, open the matching folder under this directory and the matching
entry in [`desktop/public/study-analyses.json`](../../desktop/public/study-analyses.json).
Read the explanation aloud while looking at the rendered ELA overlay,
forensic strip, and the original. Note any claim that does not match what
you can see or measure.

The minimum bar (Issue 1 acceptance criterion 4) is **4 hand-reviewed
explanations — one per provider**. Add more if a provider looks unstable.

## Spot-check rubric

For each cell, mark one of:

- **OK** — claims line up with the visible cues / metrics; ready to ship.
- **Soft** — mostly grounded, one or two filler sentences, but no
  confabulation. Ship if no better option.
- **Confab** — at least one claim cites a metric / region that does not
  exist in the evidence package, or describes something not in the image.
  Re-run precompute, change the provider, or open a follow-up bug.

## Per-provider sample

Pick one image per provider that has interesting forensic z-scores
(skim `forensic_strip.png`) — that's where confabulation is most likely
to surface, and where grounding is most informative when it works.

| Provider     | Image | Verdict       | Notes (what was cited, what was off, etc.) |
|--------------|-------|---------------|--------------------------------------------|
| `openai`     |       | OK / Soft / Confab |                                       |
| `google`     |       | OK / Soft / Confab |                                       |
| `anthropic`  |       | OK / Soft / Confab |                                       |
| `rule_based` |       | OK / Soft / Confab |                                       |

## Findings

_Free-form notes from the spot-check pass go here. Surprises, confabulation
patterns, provider-specific quirks, anything that should change before the
pilot. Keep it brief — 5–10 lines is plenty._

> Example: `anthropic` over-cites `sharpness_z` even when the value is
> within the real-face range. Worth changing the temperature to 0.2 if it
> keeps happening, or re-prompting once for soft-confab cells.

---

## Phase 1 ceiling/floor sanity check

Before recruiting, confirm Phase 1 isn't trivial. Two internal testers run
the 12-image classification quiz cold (no explanations, just real/fake +
confidence slider). Record the totals here:

| Tester       | Correct / 12 | At-ceiling images | At-floor images |
|--------------|--------------|-------------------|------------------|
| Viktor A     |              |                   |                  |
| Viktor C     |              |                   |                  |

**Pass condition:** at least one tester misclassifies at least one image
(so Phase 2 actually fires for them) AND no image is missed by both
testers (which would suggest the image is broken). If Phase 1 is trivial,
swap the easiest-to-spot fakes per Issue #94's curation flow before
running the full pilot.
