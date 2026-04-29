"""
Render visual smoke-test artifacts for the 12 user-study images.

For each image under ``desktop/public/quiz-images/``:

1. Run the BiSeNet face parser to get the 6 UI region masks.
2. Compute the forensic features (Laplacian variance, FFT high-freq energy,
   ELA intensity) per region and z-score them.
3. Save:
   - ``original.jpg``           — copy of the input
   - ``ela_overlay.png``        — same overlay the VLM receives
   - ``forensic_strip.png``     — horizontal bar chart of per-region z-scores
   - ``metrics.json``           — raw + z-scored numbers for the spotcheck

Outputs land under ``docs/study_smoke_test/<img_id>_<filename>/``.

This script does NOT call any VLM provider — it only validates the
deterministic forensic-grounding pipeline so the researcher has visual
references when reading the spotcheck log.

Usage::

    python backend/scripts/render_study_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUIZ_IMAGES_DIR = REPO_ROOT / "desktop" / "public" / "quiz-images"
OUTPUT_ROOT = REPO_ROOT / "docs" / "study_smoke_test"

# Mirrors _STUDY_IMAGES in backend/app/routers/study.py — kept in sync by
# convention since both encode the same fixed list.
STUDY_IMAGES = [
    {"id": 1, "filename": "Fake1.jpg", "label": "fake"},
    {"id": 2, "filename": "Fake2.jpg", "label": "fake"},
    {"id": 3, "filename": "Fake3.jpg", "label": "fake"},
    {"id": 4, "filename": "Fake4.jpg", "label": "fake"},
    {"id": 5, "filename": "Fake5.jpg", "label": "fake"},
    {"id": 6, "filename": "Fake6.jpg", "label": "fake"},
    {"id": 7, "filename": "Real1.jpg", "label": "real"},
    {"id": 8, "filename": "Real2.jpg", "label": "real"},
    {"id": 9, "filename": "Real3.jpg", "label": "real"},
    {"id": 10, "filename": "Real4.jpg", "label": "real"},
    {"id": 11, "filename": "Real5.jpg", "label": "real"},
    {"id": 12, "filename": "Real6.jpg", "label": "real"},
]

METRIC_ORDER = ("laplacian_variance", "hf_energy", "ela_intensity")
METRIC_LABELS = {
    "laplacian_variance": "Sharpness",
    "hf_energy": "HF energy",
    "ela_intensity": "ELA intensity",
}


def _ensure_imports():
    """Lazy-import the backend services so the script fails fast with a
    clear hint when run from the wrong working directory."""
    backend_root = REPO_ROOT / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    try:
        from app.services.face_parser import FaceParser  # noqa: F401
        from app.services.forensics import extract, z_score  # noqa: F401
        from app.services.forensics.ela import compute_ela, create_ela_overlay  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"FAIL: cannot import backend services ({exc}). "
            "Activate the backend venv and run from the repo root."
        )


def _build_forensic_strip(
    z_table: dict[str, dict[str, float]],
    img_label: str,
) -> plt.Figure:
    """Horizontal bar chart of per-region z-scores, one row per region.

    Bars are colored red when |z| ≥ 1 (notable) and grey otherwise. The
    zero line is drawn so the reader can see direction (negative = below
    real-face mean = often suspicious for sharpness/HF).
    """
    regions = list(z_table.keys())
    if not regions:
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.text(0.5, 0.5, "(no regions)", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(
        len(regions),
        1,
        figsize=(7, max(1.2, 0.9 * len(regions))),
        sharex=True,
    )
    if len(regions) == 1:
        axes = [axes]

    fig.suptitle(f"Per-region forensic z-scores — {img_label}", fontsize=10)

    for ax, region_id in zip(axes, regions, strict=True):
        z_by_metric = z_table[region_id]
        values = [z_by_metric.get(m, 0.0) for m in METRIC_ORDER]
        labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
        colors = ["#cc4444" if abs(v) >= 1.0 else "#999999" for v in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlim(-5, 5)
        ax.set_ylabel(region_id, rotation=0, ha="right", va="center", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("z-score (real-face reference distribution)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _process_image(img_meta: dict, parser, extract_fn, ela_fns) -> bool:
    """Render artifacts for one image. Returns True on success."""
    compute_ela, create_ela_overlay = ela_fns
    from app.services.forensics import z_score

    img_path = QUIZ_IMAGES_DIR / img_meta["filename"]
    if not img_path.exists():
        print(f"  - missing input: {img_path}", file=sys.stderr)
        return False

    out_dir = OUTPUT_ROOT / f"{img_meta['id']:02d}_{Path(img_meta['filename']).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")

    # Original copy for side-by-side inspection
    image.save(out_dir / "original.jpg", quality=92)

    # ELA overlay (same one the VLM gets)
    ela_map = compute_ela(image, quality=95, scale=10)
    ela_overlay = create_ela_overlay(image, ela_map)
    ela_overlay.save(out_dir / "ela_overlay.png")

    # Per-region forensic features
    parsing = parser.parse(image)
    report = extract_fn(image, parsing.masks_ui)

    z_table: dict[str, dict[str, float]] = {}
    raw_table: dict[str, dict[str, float]] = {}
    for cat_id, region in report.regions.items():
        raw_table[cat_id] = {
            "laplacian_variance": region.laplacian_variance,
            "hf_energy": region.hf_energy,
            "ela_intensity": region.ela_intensity,
        }
        try:
            z_table[cat_id] = {
                metric: z_score(cat_id, metric, getattr(region, metric)) for metric in METRIC_ORDER
            }
        except RuntimeError as exc:
            print(f"  - z_score lookup failed: {exc}", file=sys.stderr)
            z_table[cat_id] = dict.fromkeys(METRIC_ORDER, 0.0)

    fig = _build_forensic_strip(z_table, f"{img_meta['filename']} ({img_meta['label']})")
    fig.savefig(out_dir / "forensic_strip.png", dpi=120)
    plt.close(fig)

    metrics_payload = {
        "image": img_meta,
        "image_size": list(report.image_size),
        "raw": raw_table,
        "z_scores": z_table,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=float),
        encoding="utf-8",
    )

    print(f"  + {out_dir.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    _ensure_imports()

    from app.services.face_parser import FaceParser
    from app.services.forensics import extract
    from app.services.forensics.ela import compute_ela, create_ela_overlay

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    parser = FaceParser()

    print(f"Rendering smoke-test artifacts to {OUTPUT_ROOT.relative_to(REPO_ROOT)}")
    successes = 0
    for img_meta in STUDY_IMAGES:
        try:
            if _process_image(img_meta, parser, extract, (compute_ela, create_ela_overlay)):
                successes += 1
        except Exception as exc:
            print(
                f"  ! image {img_meta['id']} ({img_meta['filename']}) failed: {exc}",
                file=sys.stderr,
            )

    print(f"Rendered {successes}/{len(STUDY_IMAGES)} images")
    return 0 if successes == len(STUDY_IMAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
