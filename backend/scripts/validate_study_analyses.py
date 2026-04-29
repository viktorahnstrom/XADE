"""
Validate desktop/public/study-analyses.json after a precompute run.

Checks every (image × provider) cell against the acceptance criteria from
the smoke-test issue:

- The provider returned ``structured_regions`` (i.e. the structured-output
  path engaged — we did not silently fall back to the legacy free-text
  parser).
- Every entry in ``structured_regions`` has a non-empty ``evidence_ref``
  so the frontend can highlight the cited cue and the VLM is grounded in
  *something*, not confabulating into a vacuum.

Exits non-zero on any failure so this can run in CI later. Prints a
per-cell pass/fail matrix and a concise failure summary so the researcher
can decide whether to re-run precompute, swap a provider's model, or open
a follow-up bug.

Usage::

    python backend/scripts/validate_study_analyses.py \\
        [--path desktop/public/study-analyses.json] \\
        [--allow-error provider_id]

``--allow-error`` may be repeated and skips structured-output checks for
listed providers — useful when a provider is genuinely unavailable
(e.g. no API key set during a partial run) but the rest of the matrix
should still be validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = REPO_ROOT / "desktop" / "public" / "study-analyses.json"

EXPECTED_PROVIDERS: tuple[str, ...] = ("openai", "google", "anthropic", "rule_based")


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"FAIL: {path} does not exist. Run /study/precompute first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"FAIL: {path} is not valid JSON: {exc}")


def _validate_explanation(
    img_id: str,
    provider_id: str,
    payload: dict | None,
    allow_error: set[str],
) -> list[str]:
    """Return a list of failure messages; empty list = pass."""
    failures: list[str] = []

    if payload is None:
        failures.append(f"image {img_id}/{provider_id}: explanation missing entirely")
        return failures

    error = payload.get("error")
    if error:
        if provider_id in allow_error:
            return []
        failures.append(f"image {img_id}/{provider_id}: provider returned error: {error}")
        return failures

    structured = payload.get("structured_regions")
    if structured is None:
        failures.append(
            f"image {img_id}/{provider_id}: structured_regions is None — fell back to free-text parser"
        )
        return failures

    if not isinstance(structured, list):
        failures.append(
            f"image {img_id}/{provider_id}: structured_regions is not a list ({type(structured).__name__})"
        )
        return failures

    if len(structured) == 0:
        failures.append(f"image {img_id}/{provider_id}: structured_regions is empty list")
        return failures

    for idx, region in enumerate(structured):
        if not isinstance(region, dict):
            failures.append(f"image {img_id}/{provider_id}: region #{idx} is not an object")
            continue

        evidence_ref = (region.get("evidence_ref") or "").strip()
        if not evidence_ref:
            label = region.get("region", "?")
            confidence = region.get("confidence")
            failures.append(
                f"image {img_id}/{provider_id}: region {label!r} (confidence={confidence}) "
                "has empty evidence_ref"
            )

    return failures


def _format_matrix(results: dict[str, dict[str, bool]]) -> str:
    """Render a small ASCII pass/fail matrix for the terminal."""
    image_ids = sorted(results.keys(), key=lambda k: (len(k), k))
    if not image_ids:
        return "(no images)"

    header = "image  | " + " | ".join(f"{p:<12}" for p in EXPECTED_PROVIDERS)
    sep = "-" * len(header)
    lines = [header, sep]
    for img_id in image_ids:
        row = results[img_id]
        cells = []
        for p in EXPECTED_PROVIDERS:
            ok = row.get(p)
            cells.append(f"{'PASS' if ok else 'FAIL':<12}")
        lines.append(f"{img_id:<6} | " + " | ".join(cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="Path to study-analyses.json (default: desktop/public/study-analyses.json)",
    )
    parser.add_argument(
        "--allow-error",
        action="append",
        default=[],
        metavar="PROVIDER",
        help="Skip structured checks for this provider when it returned an error. May repeat.",
    )
    args = parser.parse_args()

    allow_error: set[str] = set(args.allow_error)

    data = _load(args.path)
    analyses = data.get("analyses")
    if not isinstance(analyses, dict) or not analyses:
        sys.exit("FAIL: no 'analyses' object in JSON")

    all_failures: list[str] = []
    matrix: dict[str, dict[str, bool]] = {}

    for img_id, entry in analyses.items():
        if not isinstance(entry, dict):
            all_failures.append(f"image {img_id}: entry is not an object")
            continue
        if entry.get("error"):
            all_failures.append(f"image {img_id}: precompute failed: {entry['error']}")
            matrix[img_id] = dict.fromkeys(EXPECTED_PROVIDERS, False)
            continue

        explanations = entry.get("explanations") or {}
        row: dict[str, bool] = {}
        for provider_id in EXPECTED_PROVIDERS:
            payload = explanations.get(provider_id)
            failures = _validate_explanation(img_id, provider_id, payload, allow_error)
            row[provider_id] = not failures
            all_failures.extend(failures)
        matrix[img_id] = row

    print("Pass/fail matrix:")
    print(_format_matrix(matrix))
    print()

    total_cells = len(matrix) * len(EXPECTED_PROVIDERS)
    passed = sum(1 for row in matrix.values() for ok in row.values() if ok)
    print(f"Result: {passed}/{total_cells} cells passed")

    if all_failures:
        print("\nFailures:")
        for line in all_failures:
            print(f"  - {line}")
        return 1

    print("All cells passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
