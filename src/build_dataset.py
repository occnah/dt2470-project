"""
Build the feature table: every in-scope GMD performance is cut into 2-bar
clips once, every condition in `src/data/conditions.py` is applied to the
whole performance, and features are measured on each clip's notes.

Output: one row per clip and condition in data/features/features.csv, a log
of skipped files, and a dataset summary in results/dataset/.

    env/bin/python -m src.build_dataset
    env/bin/python -m src.build_dataset --midi-output data/generated   # also save MIDI
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.common.config import FEATURES_TABLE, RAW_DIR, RESULTS_DIR
from src.common.midi import get_drum_notes, load_midi, note_grid, with_drum_notes
from src.data.clips import cut_clips
from src.data.conditions import CONDITIONS, Condition, Performance, apply_condition
from src.data.gmd import load_metadata, scope_exclusion
from src.features import extract_features
from src.reporting.dataset_summary import write_dataset_summary


class SkipFile(Exception):
    """Raised when a whole MIDI file cannot be used."""


def load_performance(path: Path, key: str):
    """
    Read one MIDI file and cut it into clips.

    Returns (midi, Performance). Raises SkipFile if the file is unusable.
    """
    try:
        midi = load_midi(path)
    except Exception as error:
        raise SkipFile("invalid MIDI") from error

    if not midi.time_signature_changes:
        raise SkipFile("no time signature")

    onsets, pitches, velocities = get_drum_notes(midi)
    if len(onsets) == 0:
        raise SkipFile("no drum notes")

    end = midi.get_end_time()
    grid_16 = note_grid(midi, end, 16)
    grid_32 = note_grid(midi, end, 32)
    clips = cut_clips(midi, onsets, grid_16)
    if not clips:
        raise SkipFile("shorter than one clip")

    return midi, Performance(key, onsets, velocities, pitches, grid_16, grid_32, clips)


def condition_columns(condition: Condition) -> dict:
    """Provenance columns describing a condition (not model inputs)."""
    return {
        "condition": condition.name,
        "family": condition.family,
        "subdivision": condition.subdivision,
        "strength": condition.strength,
        "jitter": condition.jitter,
        "jitter_sd_ms": condition.jitter_sd_ms,
        "matched": condition.matched,
    }


def process_file(
    path: Path,
    source_file: str,
    meta: dict,
    conditions: tuple[Condition, ...] = CONDITIONS,
    midi_output: Path | None = None,
) -> tuple[list[dict], int]:
    """
    Feature rows for every clip of one file in every condition.

    Returns (rows, number of clips skipped because they had no drum notes).
    Raises SkipFile if the file is unusable.
    """
    midi, performance = load_performance(path, source_file)
    clips = [clip for clip in performance.clips if clip.notes.any()]
    empty_clips = len(performance.clips) - len(clips)

    rows = []
    for condition in conditions:
        onsets, velocities = apply_condition(condition, performance)

        if midi_output is not None and condition.family != "human":
            destination = midi_output / condition.name / source_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            with_drum_notes(midi, onsets, velocities).write(str(destination))

        for clip in clips:
            source_clip_id = f"{Path(source_file).with_suffix('')}_clip{clip.index:03d}"
            notes = clip.notes
            rows.append({
                "clip_id": f"{source_clip_id}__{condition.name}",
                "source_clip_id": source_clip_id,
                "source_file": source_file,
                "drummer": meta.get("drummer"),
                "style": meta.get("style"),
                "gmd_split": meta.get("split"),
                "clip_start": clip.start,
                "clip_end": clip.end,
                "tempo": clip.tempo,
                "note_count": int(notes.sum()),
                **condition_columns(condition),
                **extract_features(
                    onsets[notes], velocities[notes], performance.pitches[notes],
                    performance.grid_16, performance.grid_32,
                ),
            })
    return rows, empty_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="GMD folder containing info.csv")
    parser.add_argument("--output", type=Path, default=FEATURES_TABLE)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "dataset")
    parser.add_argument("--midi-output", type=Path, help="Also save every edited performance as MIDI here")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    if not files:
        parser.error(f"No MIDI files found in {args.input}")
    metadata = load_metadata(args.input)
    print(f"Found {len(files)} MIDI files in {args.input}; {len(CONDITIONS)} conditions")
    if not metadata:
        print("No info.csv found: no scope filter, drummer and style will be empty")

    rows, skipped = [], []
    empty_clips = 0
    for index, path in enumerate(files, 1):
        source_file = path.relative_to(args.input).as_posix()
        meta = metadata.get(source_file, {})
        reason = scope_exclusion(meta) if metadata else None
        try:
            if reason is not None:
                raise SkipFile(reason)
            file_rows, file_empty = process_file(path, source_file, meta, midi_output=args.midi_output)
        except SkipFile as error:
            skipped.append({"source_file": source_file, "reason": str(error)})
            continue
        rows.extend(file_rows)
        empty_clips += file_empty
        if index % 100 == 0:
            print(f"  {index}/{len(files)} files", flush=True)

    if not rows:
        parser.error("No usable clips found; nothing was written")

    table = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped, columns=["source_file", "reason"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    skipped_path = args.output.with_name(f"{args.output.stem}_skipped.csv")
    skipped.to_csv(skipped_path, index=False)
    write_dataset_summary(table, skipped, len(files), empty_clips, args.summary)

    print(f"Files used: {table.source_file.nunique()} of {len(files)}")
    for reason, count in sorted(Counter(skipped.reason).items()):
        print(f"  skipped ({reason}): {count}")
    print(f"Clips: {table.source_clip_id.nunique()} ({empty_clips} empty clips skipped)")
    print(f"Rows:  {len(table)} ({len(CONDITIONS)} conditions)")
    print(f"Saved: {args.output}, {skipped_path}, {args.summary / 'DATASET.md'}")


if __name__ == "__main__":
    main()
