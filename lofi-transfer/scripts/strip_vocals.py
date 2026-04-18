#!/usr/bin/env python3
"""Strip vocals from audio files using Demucs (Meta's source separation).

Processes all audio in a source directory, extracts the instrumental stem
(everything minus vocals), and writes to the output directory.

For files that are already instrumental (most VGM/Pokemon), this is a no-op
quality-wise but ensures consistency.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def process_file(audio_path: Path, out_dir: Path, model: str, device: str) -> tuple[Path, bool, str]:
    """Run demucs on a single file and extract the no-vocals stem."""
    try:
        temp_dir = out_dir / ".demucs_tmp"
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", model,
            "-d", device,
            "-o", str(temp_dir),
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return audio_path, False, result.stderr.strip().split("\n")[-1]

        stem_name = audio_path.stem
        no_vocals_path = temp_dir / model / stem_name / "no_vocals.wav"

        if not no_vocals_path.exists():
            candidates = list((temp_dir / model / stem_name).glob("no_vocals*"))
            if candidates:
                no_vocals_path = candidates[0]
            else:
                return audio_path, False, "no_vocals stem not found in demucs output"

        import shutil
        dest = out_dir / f"{stem_name}.wav"
        shutil.move(str(no_vocals_path), str(dest))

        source_dir = temp_dir / model / stem_name
        for f in source_dir.glob("*"):
            f.unlink()
        source_dir.rmdir()

        return audio_path, True, str(dest)

    except subprocess.TimeoutExpired:
        return audio_path, False, "timeout"
    except Exception as e:
        return audio_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Strip vocals using Demucs")
    parser.add_argument("--src-dir", required=True, help="Directory with audio files")
    parser.add_argument("--out-dir", required=True, help="Output directory for instrumentals")
    parser.add_argument("--model", default="htdemucs", help="Demucs model (htdemucs recommended)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1 for GPU)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in sorted(src_dir.iterdir())
                   if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if args.skip_existing:
        existing = {f.stem for f in out_dir.glob("*.wav")}
        audio_files = [f for f in audio_files if f.stem not in existing]

    if not audio_files:
        print("No files to process.")
        return

    print(f"Processing {len(audio_files)} files with Demucs ({args.model}) on {args.device}")

    success = 0
    failed = 0

    if args.workers <= 1:
        for audio_path in tqdm(audio_files, desc="Stripping vocals"):
            path, ok, msg = process_file(audio_path, out_dir, args.model, args.device)
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  Failed: {path.name}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_file, f, out_dir, args.model, "cpu"): f
                for f in audio_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Stripping vocals"):
                path, ok, msg = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    print(f"  Failed: {path.name}: {msg}")

    print(f"\nDone: {success} succeeded, {failed} failed")
    print(f"Instrumentals saved to {out_dir}")


if __name__ == "__main__":
    main()
