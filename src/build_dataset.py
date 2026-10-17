"""Build matched human and transformed feature tables from two-bar GMD clips."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import (
    EXCLUDED_STYLE_KEYWORDS, FEATURES_DIR, HUMAN, NORMALIZED_VELOCITY,
    QUANTIZATION_CONDITIONS, RAW_DIR, VALID_BEAT_TYPE, VALID_TIME_SIGNATURE,
)
from src.common.midi import get_drum_notes, load_midi, note_grid
from src.data.clips import cut_clips
from src.data.gmd import load_info
from src.data.quantize import normalize_velocity_midi, quantize_midi, quantize_onsets
from src.features.magnitude import clip_features


class SkipFile(Exception):
    """Raised when a whole MIDI file cannot be used."""


def out_of_scope_reason(meta: dict) -> str | None:
    """
    Return why this performance is out of scope for the main experiment,
    or None if it's in scope.

    Scope is ordinary 4/4 beats/grooves: fills are excluded (`beat_type`),
    as are performances outside 4/4 (`time_signature`) and any whose
    style is explicitly marked as a swing or triplet feel. This is a
    dataset-level filter applied once per file in `main`, independent of
    per-clip processing in `process_file`, and only when GMD metadata was
    loaded (an empty `meta` from a missing info.csv is not filtered).
    """
    if meta.get("beat_type") != VALID_BEAT_TYPE:
        return "not a beat (fill)"
    if meta.get("time_signature") != VALID_TIME_SIGNATURE:
        return "not 4/4"
    style = str(meta.get("style", "")).lower()
    if any(keyword in style for keyword in EXCLUDED_STYLE_KEYWORDS):
        return "swing/triplet style"
    return None


def process_file(
    path: Path,
    source_file: str,
    meta: dict,
    include_conditions: bool = False,
    midi_output: Path | None = None,
) -> tuple[list[dict], int]:
    """
    Cut the original once, then measure the same notes in every condition.

    Returns (rows, number of clips skipped because they had no drum notes).
    Raises SkipFile if the file is unreadable or unusable.
    """
    try:
        midi = load_midi(path)
    except Exception as error:
        raise SkipFile("invalid MIDI") from error

    if not midi.time_signature_changes:
        raise SkipFile("no time signature")

    onsets, _, velocities = get_drum_notes(midi)

    if len(onsets) == 0:
        raise SkipFile("no drum notes")

    grid = note_grid(midi, midi.get_end_time(), 16)
    grid_32 = note_grid(midi, midi.get_end_time(), 32)
    clips = cut_clips(midi, onsets, grid)

    if not clips:
        raise SkipFile("shorter than one clip")

    rows = []
    empty_clips = 0
    variants = [(HUMAN, None, 0.0, None, onsets, velocities)]
    if include_conditions:
        for condition, subdivision, strength in QUANTIZATION_CONDITIONS:
            target_grid = grid if subdivision == 16 else grid_32
            variants.append((
                condition, subdivision, strength, None,
                quantize_onsets(onsets, target_grid, strength), velocities,
            ))
        variants.append((
            "velocity_normalized", None, 0.0, NORMALIZED_VELOCITY,
            onsets, np.full_like(velocities, NORMALIZED_VELOCITY),
        ))

    for clip in clips:
        if not clip.notes.any():
            empty_clips += 1
            continue

        source_clip_id = f"{Path(source_file).with_suffix('')}_clip{clip.index:03d}"
        for condition, subdivision, strength, velocity, starts, values in variants:
            rows.append({
                "clip_id": (
                    source_clip_id if condition == HUMAN
                    else f"{source_clip_id}__{condition}"
                ),
                "source_clip_id": source_clip_id,
                "source_file": source_file,
                "drummer": meta.get("drummer"),
                "style": meta.get("style"),
                "beat_type": meta.get("beat_type"),
                "time_signature": meta.get("time_signature"),
                "split": meta.get("split"),
                "tempo": clip.tempo,
                "clip_start": clip.start,
                "clip_end": clip.end,
                "condition": condition,
                "subdivision": subdivision,
                "strength": strength,
                "normalized_velocity": velocity,
                **clip_features(starts[clip.notes], values[clip.notes], grid, grid_32),
            })

    if include_conditions and midi_output is not None and rows:
        for condition, subdivision, strength in QUANTIZATION_CONDITIONS:
            destination = midi_output / condition / source_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            quantize_midi(midi, subdivision, strength).write(str(destination))
        destination = midi_output / "velocity_normalized" / source_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        normalize_velocity_midi(midi).write(str(destination))

    return rows, empty_clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build two-bar human features, optionally with matched transformations."
    )
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: data/features/human_features.csv (conditions_features.csv with --conditions)",
    )
    parser.add_argument(
        "--conditions", action="store_true",
        help="Include five timing-quantized conditions and independent velocity normalization.",
    )
    parser.add_argument(
        "--midi-output", type=Path,
        help="Optionally save transformed full performances here; requires --conditions.",
    )
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    if args.midi_output is not None and not args.conditions:
        parser.error("--midi-output requires --conditions")
    if args.output is None:
        filename = "conditions_features.csv" if args.conditions else "human_features.csv"
        args.output = FEATURES_DIR / filename

    files = sorted(
        path
        for path in args.input.rglob("*")
        if path.suffix.lower() in (".mid", ".midi")
    )
    print(f"Found {len(files)} MIDI files in {args.input}")
    if not files:
        parser.error(f"No MIDI files found in {args.input}")

    metadata = {}
    if (args.input / "info.csv").exists():
        info = load_info(args.input)
        metadata = info.set_index("midi_filename").to_dict("index")
    else:
        print("No info.csv found; drummer and style will be empty")

    rows = []
    skipped_files = Counter()
    skipped_records = []
    empty_clips = 0

    for index, path in enumerate(files, 1):
        source_file = path.relative_to(args.input).as_posix()
        meta = metadata.get(source_file, {})

        if metadata:
            reason = out_of_scope_reason(meta)
            if reason is not None:
                skipped_files[reason] += 1
                skipped_records.append({"source_file": source_file, "reason": reason})
                continue

        try:
            file_rows, file_empty_clips = process_file(
                path, source_file, meta,
                include_conditions=args.conditions, midi_output=args.midi_output,
            )
        except SkipFile as reason:
            skipped_files[str(reason)] += 1
            skipped_records.append({"source_file": source_file, "reason": str(reason)})
            continue

        rows.extend(file_rows)
        empty_clips += file_empty_clips
        if index % 100 == 0:
            print(f"Processed {index}/{len(files)} files", flush=True)

    if not rows:
        parser.error("No usable clips found; no feature table was written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    skipped_output = args.output.with_name(f"{args.output.stem}_skipped.csv")
    pd.DataFrame(skipped_records, columns=["source_file", "reason"]).to_csv(
        skipped_output, index=False
    )

    print(f"Files used:    {len(files) - sum(skipped_files.values())}")
    for reason, count in sorted(skipped_files.items()):
        print(f"Files skipped ({reason}): {count}")
    print(f"Clips skipped (no drum notes): {empty_clips}")
    print(f"Original clips: {sum(row['condition'] == HUMAN for row in rows)}")
    print(f"Feature rows:   {len(rows)}")
    print(f"Saved: {args.output}")
    print(f"Skip log: {skipped_output}")


if __name__ == "__main__":
    main()
