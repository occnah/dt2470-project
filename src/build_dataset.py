"""
Build the human-baseline feature table: one row per 2-bar clip of an
original (unmodified) GMD performance.

Usage:
    python -m src.build_dataset --input data/raw \
        --output data/features/human_features.csv
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pretty_midi

from src.common.config import FEATURES_DIR, HUMAN, RAW_DIR
from src.common.midi import get_drum_notes, sixteenth_grid
from src.data.clips import cut_clips
from src.data.gmd import load_info
from src.features.magnitude import clip_features


class SkipFile(Exception):
    """Raised when a whole MIDI file cannot be used."""


def process_file(
    path: Path,
    source_file: str,
    meta: dict,
) -> tuple[list[dict], int]:
    """
    Split one MIDI file into 2-bar clips and extract features.

    Returns (rows, number of clips skipped because they had no drum notes).
    Raises SkipFile if the file is unreadable or unusable.
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as error:
        raise SkipFile("invalid MIDI") from error

    if not midi.time_signature_changes:
        raise SkipFile("no time signature")

    onsets, _, velocities = get_drum_notes(midi)

    if len(onsets) == 0:
        raise SkipFile("no drum notes")

    grid = sixteenth_grid(midi, midi.get_end_time())
    clips = cut_clips(midi, onsets, grid)

    if not clips:
        raise SkipFile("shorter than one clip")

    rows = []
    empty_clips = 0

    for clip in clips:
        if not clip.notes.any():
            empty_clips += 1
            continue

        rows.append(
            {
                "clip_id": f"{Path(source_file).with_suffix('')}_clip{clip.index:03d}",
                "source_file": source_file,
                "drummer": meta.get("drummer"),
                "style": meta.get("style"),
                "tempo": clip.tempo,
                "clip_start": clip.start,
                "clip_end": clip.end,
                "condition": HUMAN,
                **clip_features(onsets[clip.notes], velocities[clip.notes], grid),
            }
        )

    return rows, empty_clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a feature table of 2-bar human drum clips."
    )
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURES_DIR / "human_features.csv",
    )
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.input.rglob("*")
        if path.suffix.lower() in (".mid", ".midi")
    )
    print(f"Found {len(files)} MIDI files in {args.input}")

    metadata = {}
    if (args.input / "info.csv").exists():
        info = load_info(args.input)
        metadata = info.set_index("midi_filename").to_dict("index")
    else:
        print("No info.csv found; drummer and style will be empty")

    rows = []
    skipped_files = Counter()
    empty_clips = 0

    for path in files:
        source_file = path.relative_to(args.input).as_posix()

        try:
            file_rows, file_empty_clips = process_file(
                path, source_file, metadata.get(source_file, {})
            )
        except SkipFile as reason:
            skipped_files[str(reason)] += 1
            continue

        rows.extend(file_rows)
        empty_clips += file_empty_clips

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    print(f"Files used:    {len(files) - sum(skipped_files.values())}")
    for reason, count in sorted(skipped_files.items()):
        print(f"Files skipped ({reason}): {count}")
    print(f"Clips skipped (no drum notes): {empty_clips}")
    print(f"Clips written: {len(rows)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
