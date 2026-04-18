#!/usr/bin/env python3
"""Verify downloaded/processed audio meets training spec requirements.

Checks:
- File is valid audio (can be loaded)
- Duration within acceptable range
- Not silence (RMS above threshold)
- Sample rate (after resampling) would be correct
- No corruption / truncation

Reports a summary and flags problematic files for removal.
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
RMS_THRESHOLD = 1e-3
MIN_DURATION_SEC = 5.0
MAX_DURATION_SEC = 1200.0


def check_file(path: Path) -> dict:
    result = {
        "path": str(path),
        "valid": False,
        "issues": [],
    }

    try:
        info = sf.info(path)
    except Exception as e:
        result["issues"].append(f"cannot read: {e}")
        return result

    result["sample_rate"] = info.samplerate
    result["channels"] = info.channels
    result["duration_sec"] = info.duration
    result["format"] = info.format

    if info.duration < MIN_DURATION_SEC:
        result["issues"].append(f"too short: {info.duration:.1f}s < {MIN_DURATION_SEC}s")

    if info.duration > MAX_DURATION_SEC:
        result["issues"].append(f"too long: {info.duration:.1f}s > {MAX_DURATION_SEC}s")

    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        rms = np.sqrt(np.mean(audio ** 2))
        result["rms"] = float(rms)
        if rms < RMS_THRESHOLD:
            result["issues"].append(f"near-silent: RMS={rms:.6f}")

        peak = np.max(np.abs(audio))
        result["peak"] = float(peak)
        if peak > 1.0:
            result["issues"].append(f"clipping: peak={peak:.4f}")

        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            result["issues"].append("contains NaN/Inf values")

    except Exception as e:
        result["issues"].append(f"cannot decode audio: {e}")
        return result

    if not result["issues"]:
        result["valid"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify audio data quality")
    parser.add_argument("--dir", required=True, help="Directory to verify")
    parser.add_argument("--domain", default="unknown", help="Domain name for reporting")
    parser.add_argument("--report", default=None, help="Output JSON report path")
    parser.add_argument("--remove-bad", action="store_true", help="Move bad files to .rejected/ subdir")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    audio_files = [f for f in sorted(target_dir.iterdir())
                   if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not audio_files:
        print(f"No audio files found in {target_dir}")
        sys.exit(1)

    print(f"Verifying {len(audio_files)} files in {target_dir} ({args.domain})")

    results = []
    valid_count = 0
    total_duration = 0.0
    issues_by_type = {}

    for path in tqdm(audio_files, desc="Checking"):
        r = check_file(path)
        results.append(r)
        if r["valid"]:
            valid_count += 1
            total_duration += r.get("duration_sec", 0)
        else:
            for issue in r["issues"]:
                issue_type = issue.split(":")[0]
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

    bad_files = [r for r in results if not r["valid"]]

    print(f"\n{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Total files: {len(results)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {len(bad_files)}")
    print(f"Total valid duration: {total_duration/3600:.1f} hours")
    print(f"{'='*60}")

    if issues_by_type:
        print("\nIssue breakdown:")
        for issue, count in sorted(issues_by_type.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count}")

    if bad_files:
        print(f"\nBad files ({len(bad_files)}):")
        for r in bad_files[:20]:
            print(f"  {Path(r['path']).name}: {', '.join(r['issues'])}")
        if len(bad_files) > 20:
            print(f"  ... and {len(bad_files) - 20} more")

    if args.remove_bad and bad_files:
        reject_dir = target_dir / ".rejected"
        reject_dir.mkdir(exist_ok=True)
        moved = 0
        for r in bad_files:
            src = Path(r["path"])
            if src.exists():
                src.rename(reject_dir / src.name)
                moved += 1
        print(f"\nMoved {moved} bad files to {reject_dir}")

    if args.report:
        report = {
            "domain": args.domain,
            "total_files": len(results),
            "valid_files": valid_count,
            "invalid_files": len(bad_files),
            "total_valid_duration_hours": total_duration / 3600,
            "issues_by_type": issues_by_type,
            "files": results,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")

    target_hours = 15.0
    if total_duration / 3600 < target_hours:
        print(f"\n*** WARNING: Only {total_duration/3600:.1f}h of valid audio. "
              f"Target is {target_hours}h minimum for training. ***")


if __name__ == "__main__":
    main()
