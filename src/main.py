import argparse
from pathlib import Path

import pandas as pd
import pretty_midi

from dataset import DEFAULT_ROOT, filter_info, get_bpm, load_info
from features import extract_features
from quantize import quantize_midi

CONDITIONS = [
    ("q16_25", 16, 0.25),
    ("q16_50", 16, 0.50),
    ("q16_75", 16, 0.75),
    ("q16_100", 16, 1.00),
    ("q32_100", 32, 1.00),
]


def run_file(input_path: Path, output_dir: Path) -> None:
    """
    Quantize a single MIDI file, print features and save each condition.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(str(input_path))

    print(f"Loaded: {input_path}")
    print(f"Tempo: {get_bpm(midi):.2f} BPM")
    print()

    print("Human features:")
    for key, value in extract_features(midi).items():
        print(f"  {key}: {value}")

    for name, subdivision, strength in CONDITIONS:
        transformed = quantize_midi(
            midi,
            subdivision=subdivision,
            strength=strength,
        )

        output_path = output_dir / f"{input_path.stem}_{name}.mid"

        transformed.write(str(output_path))

        print()
        print(f"{name} features:")

        for key, value in extract_features(transformed).items():
            print(f"  {key}: {value}")

        print(f"Saved: {output_path}")


def run_dataset(
    root: Path,
    features_path: Path,
    split: str | None,
    beat_type: str | None,
    time_signature: str | None,
) -> None:
    """
    Extract features for the human performance and every quantization
    condition of each file in the dataset, and save them as one CSV.
    """
    info = filter_info(
        load_info(root),
        split=split,
        beat_type=beat_type,
        time_signature=time_signature,
    )

    print(f"Processing {len(info)} files from {root}")

    rows = []

    for i, item in enumerate(info.itertuples(), start=1):
        midi = pretty_midi.PrettyMIDI(str(item.midi_path))

        meta = {
            "id": item.id,
            "drummer": item.drummer,
            "style": item.style,
            "bpm": get_bpm(midi),
            "beat_type": item.beat_type,
            "time_signature": item.time_signature,
            "split": item.split,
        }

        rows.append({**meta, "condition": "human", **extract_features(midi)})

        for name, subdivision, strength in CONDITIONS:
            transformed = quantize_midi(
                midi,
                subdivision=subdivision,
                strength=strength,
            )
            rows.append({**meta, "condition": name, **extract_features(transformed)})

        if i % 100 == 0:
            print(f"  {i}/{len(info)}")

    features_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(features_path, index=False)

    print(f"Saved: {features_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test quantization on Groove MIDI files."
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to a single MIDI file. If omitted, the whole dataset is processed.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory for generated MIDI files (single-file mode)",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Groove MIDI Dataset root containing info.csv",
    )

    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/features/features.csv"),
        help="Output CSV for extracted features (dataset mode)",
    )

    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--beat-type", choices=["beat", "fill"])
    parser.add_argument("--time-signature", help="e.g. 4-4")

    args = parser.parse_args()

    if args.input is not None:
        run_file(args.input, args.output_dir)
    else:
        run_dataset(
            args.root,
            args.features_path,
            args.split,
            args.beat_type,
            args.time_signature,
        )


if __name__ == "__main__":
    main()
