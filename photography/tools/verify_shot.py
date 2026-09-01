#!/usr/bin/env python3
"""Verify recorded shot intent against what the camera actually recorded.

This is the ground-truth feedback channel of the photography curriculum. It
answers only questions the file can settle: did the frame use the predicted
settings, did depth of field cover what it was meant to, was the shutter
adequate for the declared motion intent, and did the exposure land where it was
supposed to. It makes no aesthetic judgement whatsoever -- that belongs to
rubrics/image-critique-rubric.md and is never settled here.

Usage:
    python3 tools/verify_shot.py shoots/2026-08-30-riverside/intent.toml
    python3 tools/verify_shot.py shoots/<outing>/intent.toml --json

EXIF is read with exiftool when available (required for raw files) and falls
back to Pillow for JPEG/TIFF. Install exiftool for raw support:
    sudo pacman -S perl-image-exiftool
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# --- Domain constants -------------------------------------------------------

# Circle of confusion in mm by sensor format. Conventional values; they set the
# sharpness criterion for depth-of-field arithmetic and are not physical facts.
CIRCLE_OF_CONFUSION_MM = {
    "full-frame": 0.030,
    "aps-c": 0.020,
    "aps-c-canon": 0.019,
    "mft": 0.015,
    "1-inch": 0.011,
    "phone": 0.005,
}

CROP_FACTOR = {
    "full-frame": 1.0,
    "aps-c": 1.5,
    "aps-c-canon": 1.6,
    "mft": 2.0,
    "1-inch": 2.7,
    "phone": 7.0,
}

# Heuristic minimum shutter speeds to freeze a motion class, as 1/x seconds.
# Photographic convention, not measured physics. Declared here so the tutor can
# say which threshold a verdict came from.
MOTION_FREEZE_MIN_SHUTTER = {
    "static": None,
    "slow": 125,      # walking pace, foliage in light wind
    "moderate": 500,  # running, cycling, active children
    "fast": 1000,     # sport, birds in flight, vehicles
}

TOLERANCE_STOPS_DEFAULT = 0.34  # one third of a stop


# --- Data model -------------------------------------------------------------

@dataclass
class Check:
    name: str
    verdict: str  # PASS | FAIL | NA
    detail: str
    candidate_codes: list[str] = field(default_factory=list)


@dataclass
class FrameResult:
    frame: str
    subject: str
    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == "FAIL"]

    @property
    def met_intent(self) -> bool:
        return not self.failed and any(c.verdict == "PASS" for c in self.checks)


# --- EXIF extraction --------------------------------------------------------

def _exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def read_exif(path: Path) -> dict:
    """Return a normalised EXIF dict. Missing keys are simply absent."""
    if _exiftool_available():
        raw = _read_exif_exiftool(path)
    else:
        raw = _read_exif_pillow(path)
    return raw


def _read_exif_exiftool(path: Path) -> dict:
    proc = subprocess.run(
        ["exiftool", "-j", "-n", "-FNumber", "-ExposureTime", "-ISO",
         "-FocalLength", "-FocalLengthIn35mmFormat", "-Model", "-LensModel",
         "-SubjectDistance", "-ExposureCompensation", "-DateTimeOriginal", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    payload = json.loads(proc.stdout)[0]
    return {
        "aperture": _as_float(payload.get("FNumber")),
        "shutter": _as_float(payload.get("ExposureTime")),
        "iso": _as_float(payload.get("ISO")),
        "focal_length": _as_float(payload.get("FocalLength")),
        "focal_length_35mm": _as_float(payload.get("FocalLengthIn35mmFormat")),
        "subject_distance": _as_float(payload.get("SubjectDistance")),
        "model": payload.get("Model"),
        "lens": payload.get("LensModel"),
        "taken": payload.get("DateTimeOriginal"),
    }


def _read_exif_pillow(path: Path) -> dict:
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return {}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            ifd = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
    except Exception:
        return {}
    tag = {ExifTags.TAGS.get(k, k): v for k, v in (ifd or {}).items()}
    base = {ExifTags.TAGS.get(k, k): v for k, v in (exif or {}).items()}
    return {
        "aperture": _as_float(tag.get("FNumber")),
        "shutter": _as_float(tag.get("ExposureTime")),
        "iso": _as_float(tag.get("ISOSpeedRatings") or tag.get("PhotographicSensitivity")),
        "focal_length": _as_float(tag.get("FocalLength")),
        "focal_length_35mm": _as_float(tag.get("FocalLengthIn35mmFilm")),
        "subject_distance": _as_float(tag.get("SubjectDistance")),
        "model": base.get("Model"),
        "lens": tag.get("LensModel"),
        "taken": tag.get("DateTimeOriginal"),
    }


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- Optical arithmetic -----------------------------------------------------

def depth_of_field(focal_mm: float, f_number: float, distance_m: float,
                   coc_mm: float) -> tuple[float, float, float | None]:
    """Return (near_m, far_m, total_m). far is inf when at or beyond hyperfocal."""
    s_mm = distance_m * 1000.0
    hyperfocal_mm = (focal_mm ** 2) / (f_number * coc_mm) + focal_mm
    near_mm = (s_mm * (hyperfocal_mm - focal_mm)) / (hyperfocal_mm + s_mm - 2 * focal_mm)
    if s_mm >= hyperfocal_mm:
        return near_mm / 1000.0, math.inf, None
    far_mm = (s_mm * (hyperfocal_mm - focal_mm)) / (hyperfocal_mm - s_mm)
    near_m, far_m = near_mm / 1000.0, far_mm / 1000.0
    return near_m, far_m, far_m - near_m


def stops_between(a: float, b: float) -> float:
    """Difference in stops between two proportional quantities."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return abs(math.log2(a / b))


def aperture_stops_apart(a: float, b: float) -> float:
    """Apertures are stops apart by the square of their ratio."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return abs(2 * math.log2(a / b))


def fmt_shutter(seconds: float | None) -> str:
    if not seconds:
        return "?"
    if seconds >= 1:
        return f"{seconds:g}s"
    return f"1/{round(1 / seconds)}"


# --- Histogram --------------------------------------------------------------

def clipping(path: Path) -> tuple[float, float] | None:
    """Return (highlight_clipped_pct, shadow_clipped_pct) from the rendered image.

    Raw files need exiftool to extract an embedded preview; without it this
    check reports NA rather than guessing.
    """
    target = path
    tmp = None
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        if not _exiftool_available():
            return None
        tmp = path.with_suffix(".preview.jpg")
        proc = subprocess.run(
            ["exiftool", "-b", "-PreviewImage", str(path)],
            capture_output=True,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        tmp.write_bytes(proc.stdout)
        target = tmp
    try:
        from PIL import Image
        with Image.open(target) as img:
            grey = img.convert("L")
            grey.thumbnail((800, 800))
            hist = grey.histogram()
        total = sum(hist) or 1
        return 100.0 * sum(hist[254:]) / total, 100.0 * sum(hist[:2]) / total
    except Exception:
        return None
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


# --- Checks -----------------------------------------------------------------

def check_settings(intent: dict, exif: dict, tol: float) -> list[Check]:
    checks = []
    pairs = [
        ("aperture", "predicted_aperture", aperture_stops_apart, lambda v: f"f/{v:g}"),
        ("shutter", "predicted_shutter", stops_between, fmt_shutter),
        ("iso", "predicted_iso", stops_between, lambda v: f"ISO {v:g}"),
    ]
    for exif_key, intent_key, metric, render in pairs:
        want, got = intent.get(intent_key), exif.get(exif_key)
        if want is None:
            checks.append(Check(f"settings.{exif_key}", "NA", "no prediction recorded"))
            continue
        if got is None:
            checks.append(Check(f"settings.{exif_key}", "NA", "not present in EXIF"))
            continue
        delta = metric(float(want), float(got))
        if delta <= tol:
            checks.append(Check(f"settings.{exif_key}", "PASS",
                                f"predicted {render(want)}, shot {render(got)}"))
        else:
            checks.append(Check(
                f"settings.{exif_key}", "FAIL",
                f"predicted {render(want)}, shot {render(got)} "
                f"({delta:.2f} stops apart, tolerance {tol:.2f})",
                candidate_codes=["P", "C", "M"],
            ))
    return checks


def check_depth(intent: dict, exif: dict, sensor: str) -> Check:
    focal = exif.get("focal_length")
    aperture = exif.get("aperture")
    distance = intent.get("subject_distance_m") or exif.get("subject_distance")
    want = intent.get("depth_intent")
    if want is None:
        return Check("depth", "NA", "no depth intent recorded")
    if not (focal and aperture and distance):
        return Check("depth", "NA",
                     "needs focal length, aperture, and subject_distance_m "
                     "(record the distance in the intent file when the camera does not)")
    coc = CIRCLE_OF_CONFUSION_MM.get(sensor, CIRCLE_OF_CONFUSION_MM["full-frame"])
    near, far, total = depth_of_field(focal, aperture, float(distance), coc)
    span = "infinity" if far == math.inf else f"{far:.2f} m"
    measured = f"sharp {near:.2f}–{span}" + (f" (depth {total:.2f} m)" if total else "")

    if want == "shallow":
        declared_max_depth = intent.get("max_depth_m")
        target = declared_max_depth if declared_max_depth is not None else max(0.5, float(distance) * 0.25)
        source = "" if declared_max_depth is not None else " [tool default; declare max_depth_m]"
        ok = total is not None and total <= target
        return Check("depth", "PASS" if ok else "FAIL",
                     f"{measured}; wanted shallow (≤ {float(target):.2f} m){source}",
                     [] if ok else ["M", "P"])
    if want == "deep":
        need_far = intent.get("far_limit_m")
        ok = far == math.inf if need_far is None else far >= float(need_far)
        goal = "hyperfocal/infinity" if need_far is None else f"≥ {float(need_far):.2f} m"
        return Check("depth", "PASS" if ok else "FAIL",
                     f"{measured}; wanted deep ({goal})",
                     [] if ok else ["M", "P"])
    if want == "range":
        lo, hi = intent.get("near_limit_m"), intent.get("far_limit_m")
        if lo is None or hi is None:
            return Check("depth", "NA", "depth_intent='range' needs near_limit_m and far_limit_m")
        ok = near <= float(lo) and far >= float(hi)
        return Check("depth", "PASS" if ok else "FAIL",
                     f"{measured}; wanted {float(lo):.2f}–{float(hi):.2f} m covered",
                     [] if ok else ["M", "P"])
    return Check("depth", "NA", f"unknown depth_intent '{want}'")


def check_motion(intent: dict, exif: dict, sensor: str, stabilisation_stops: float) -> list[Check]:
    checks = []
    shutter = exif.get("shutter")
    if not shutter:
        return [Check("motion", "NA", "no shutter speed in EXIF")]

    # Camera shake, against the reciprocal-of-equivalent-focal-length convention.
    equiv = exif.get("focal_length_35mm")
    if equiv is None and exif.get("focal_length"):
        equiv = exif["focal_length"] * CROP_FACTOR.get(sensor, 1.0)
    if equiv and not intent.get("tripod", False):
        limit = (1.0 / equiv) * (2 ** stabilisation_stops)
        ok = shutter <= limit
        checks.append(Check(
            "motion.shake", "PASS" if ok else "FAIL",
            f"{fmt_shutter(shutter)} at {equiv:g}mm equiv; convention allows "
            f"{fmt_shutter(limit)} with {stabilisation_stops:g} stops stabilisation",
            [] if ok else ["M", "P", "C"],
        ))
    elif intent.get("tripod", False):
        checks.append(Check("motion.shake", "NA", "tripod declared"))

    want = intent.get("motion_intent")
    subject = intent.get("subject_motion", "static")
    if want == "freeze":
        need = MOTION_FREEZE_MIN_SHUTTER.get(subject)
        if need is None:
            checks.append(Check("motion.subject", "NA",
                                f"subject_motion='{subject}' imposes no freeze threshold"))
        else:
            ok = shutter <= 1.0 / need
            checks.append(Check(
                "motion.subject", "PASS" if ok else "FAIL",
                f"{fmt_shutter(shutter)} to freeze '{subject}'; convention wants "
                f"1/{need} or faster",
                [] if ok else ["M", "D", "P"],
            ))
    elif want == "blur":
        need = intent.get("blur_slower_than")
        if need is None:
            checks.append(Check("motion.subject", "NA",
                                "motion_intent='blur' needs blur_slower_than (as 1/x)"))
        else:
            ok = shutter >= 1.0 / float(need)
            checks.append(Check(
                "motion.subject", "PASS" if ok else "FAIL",
                f"{fmt_shutter(shutter)}; wanted slower than 1/{float(need):g}",
                [] if ok else ["M", "P"],
            ))
    else:
        checks.append(Check("motion.subject", "NA", "no motion intent recorded"))
    return checks


def check_exposure(intent: dict, path: Path) -> Check:
    want = intent.get("exposure_intent")
    if want is None:
        return Check("exposure", "NA", "no exposure intent recorded")
    result = clipping(path)
    if result is None:
        return Check("exposure", "NA",
                     "could not read a histogram (install exiftool for raw preview extraction)")
    high, low = result
    measured = f"{high:.2f}% clipped highlights, {low:.2f}% clipped shadows"
    if want == "protect-highlights":
        declared_limit = intent.get("max_clipped_highlight_pct")
        limit = float(declared_limit) if declared_limit is not None else 0.5
        source = "" if declared_limit is not None else " [tool default; declare max_clipped_highlight_pct]"
        ok = high <= limit
        return Check("exposure", "PASS" if ok else "FAIL",
                     f"{measured}; wanted ≤ {limit:g}% highlight clipping{source}",
                     [] if ok else ["M", "P", "C"])
    if want == "protect-shadows":
        declared_limit = intent.get("max_clipped_shadow_pct")
        limit = float(declared_limit) if declared_limit is not None else 1.0
        source = "" if declared_limit is not None else " [tool default; declare max_clipped_shadow_pct]"
        ok = low <= limit
        return Check("exposure", "PASS" if ok else "FAIL",
                     f"{measured}; wanted ≤ {limit:g}% shadow clipping{source}",
                     [] if ok else ["M", "P", "C"])
    if want == "full-range":
        ok = high <= 1.0 and low <= 1.0
        return Check("exposure", "PASS" if ok else "FAIL",
                     f"{measured}; wanted both ends ≤ 1% [tool default threshold]",
                     [] if ok else ["M", "P"])
    return Check("exposure", "NA", f"unknown exposure_intent '{want}'")


def check_iso_ceiling(intent: dict, exif: dict) -> Check:
    ceiling = intent.get("iso_ceiling")
    iso = exif.get("iso")
    if ceiling is None:
        return Check("iso_ceiling", "NA", "no ISO ceiling declared")
    if iso is None:
        return Check("iso_ceiling", "NA", "no ISO in EXIF")
    ok = iso <= float(ceiling)
    return Check("iso_ceiling", "PASS" if ok else "FAIL",
                 f"ISO {iso:g} against declared ceiling {float(ceiling):g}",
                 [] if ok else ["C", "D"])


# --- Driver -----------------------------------------------------------------

def verify_frame(frame: dict, outing: dict, root: Path) -> FrameResult:
    path = root / frame["file"]
    sensor = outing.get("sensor", "full-frame")
    tol = float(outing.get("tolerance_stops", TOLERANCE_STOPS_DEFAULT))
    stab = float(outing.get("stabilisation_stops", 0))

    if not path.exists():
        return FrameResult(frame["file"], frame.get("subject", ""),
                           [Check("file", "FAIL", f"missing file: {path}")])

    exif = read_exif(path)
    if not exif or exif.get("aperture") is None and exif.get("shutter") is None:
        note = "" if _exiftool_available() else " (exiftool not installed; raw files need it)"
        return FrameResult(frame["file"], frame.get("subject", ""),
                           [Check("exif", "FAIL", f"no usable EXIF read from {path.name}{note}")])

    checks: list[Check] = []
    checks += check_settings(frame, exif, tol)
    checks.append(check_depth(frame, exif, sensor))
    checks += check_motion(frame, exif, sensor, stab)
    checks.append(check_exposure(frame, path))
    checks.append(check_iso_ceiling({**outing, **frame}, exif))
    return FrameResult(frame["file"], frame.get("subject", ""), checks)


def render_text(outing: dict, results: list[FrameResult], predicted: float | None) -> str:
    lines = [
        f"Outing: {outing.get('name', 'unnamed')}   {outing.get('date', '')}",
        f"Blocked constraint: {outing.get('constraint', 'none declared')}",
        f"Sensor: {outing.get('sensor', 'full-frame')}   "
        f"EXIF source: {'exiftool' if _exiftool_available() else 'Pillow (no raw support)'}",
        "",
    ]
    for r in results:
        status = "MET INTENT" if r.met_intent else "MISSED INTENT"
        lines.append(f"── {r.frame}  [{status}]  {r.subject}")
        for c in r.checks:
            mark = {"PASS": "  ok ", "FAIL": " FAIL", "NA": "  -- "}[c.verdict]
            codes = f"   candidates: {'/'.join(c.candidate_codes)}" if c.candidate_codes else ""
            lines.append(f"{mark} {c.name:<18} {c.detail}{codes}")
        lines.append("")

    assessed = [r for r in results if any(c.verdict != "NA" for c in r.checks)]
    met = [r for r in assessed if r.met_intent]
    rate = 100.0 * len(met) / len(assessed) if assessed else 0.0
    lines.append(f"Frames assessed: {len(assessed)}   met intent: {len(met)}   "
                 f"keeper rate against intent: {rate:.0f}%")
    if predicted is not None:
        lines.append(f"Predicted keeper rate: {predicted:.0f}%   "
                     f"calibration gap: {predicted - rate:+.0f} points")

    tally: dict[str, int] = {}
    for r in results:
        for c in r.failed:
            for code in c.candidate_codes:
                tally[code] = tally.get(code, 0) + 1
    if tally:
        ordered = ", ".join(f"{k}×{v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
        lines.append(f"Candidate error codes across failures: {ordered}")
    lines.append("")
    lines.append("Candidates are hints from the checks that failed, not a diagnosis. "
                 "Classify with the tutor against the error-routing table.")
    lines.append("This tool settles technical questions only. Whether the picture "
                 "works is decided by the rubric, never here.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("intent", type=Path, help="path to an outing intent.toml")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    if not args.intent.exists():
        print(f"error: no intent file at {args.intent}", file=sys.stderr)
        return 1

    with args.intent.open("rb") as handle:
        doc = tomllib.load(handle)

    outing = doc.get("outing", {})
    frames = doc.get("frame", [])
    if not frames:
        print("error: intent file records no [[frame]] entries. Frames without a "
              "recorded pre-shutter intent are not evidence.", file=sys.stderr)
        return 1

    root = args.intent.parent
    results = [verify_frame(f, outing, root) for f in frames]
    predicted = outing.get("predicted_keeper_rate")

    if args.json:
        print(json.dumps({
            "outing": outing,
            "frames": [
                {"frame": r.frame, "subject": r.subject, "met_intent": r.met_intent,
                 "checks": [vars(c) for c in r.checks]}
                for r in results
            ],
        }, indent=2))
    else:
        print(render_text(outing, results, predicted))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
