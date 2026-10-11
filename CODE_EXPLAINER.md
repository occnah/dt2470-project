# Human baseline and quantization pipeline

## Run it

From the project root, with the dependencies installed in `env/`:

```bash
env/bin/python -m src.build_dataset
env/bin/python -m src.build_dataset --conditions
env/bin/python -m pytest -q
```

The input now defaults to the existing `data/groove/` folder. The commands write
`data/features/human_features.csv` and `data/features/conditions_features.csv`.
Each also writes a `_skipped.csv` listing unusable source files and reasons.
Use `--input` and `--output` to override the locations.

## What the code does

- `common/midi.py`: builds 1/16 and 1/32 grids in MIDI ticks, then converts them
  through the performance's tempo map. Both quantization and timing features use
  these grids. Exact halfway ties use the earlier grid point. The loader also
  restores raw drum note-ons omitted by the MIDI duration parser, with their
  original onset/velocity and zero duration, so every recorded drum hit is counted.
- `data/quantize.py`: moves each onset toward its nearest grid point:
  `new = original + strength * (target - original)`. It also provides a separate
  transformation setting every drum velocity to 80. MIDI helpers return copies,
  preserve pitches and note durations, and leave non-drum notes unchanged.
- `build_dataset.py`: cuts the original into non-overlapping two-bar clips once.
  Every version uses that original note membership, even if a transformed onset
  moves across the clip boundary. Empty clips and unusable files are counted.
- `features/magnitude.py`: extracts timing, inter-onset interval (IOI), and velocity
  summaries from those matched notes. Existing feature names are preserved;
  `timing_32_abs_mean` and `timing_32_std` add the finer-grid measurements.
- `tests/test_quantize.py`: checks partial strengths, tempo changes, boundary
  membership, independent velocity normalization, and optional MIDI exports.

## Conditions and identifiers

Each usable source clip has seven rows in the conditions table:

| `condition` | Timing | Velocity |
| --- | --- | --- |
| `human` | Original | Original |
| `quantized_16_25` | 25% toward 1/16 grid | Original |
| `quantized_16_50` | 50% toward 1/16 grid | Original |
| `quantized_16_75` | 75% toward 1/16 grid | Original |
| `quantized_16_100` | Fully on 1/16 grid | Original |
| `quantized_32_100` | Fully on 1/32 grid | Original |
| `velocity_normalized` | Original | Constant 80 |

`clip_id` uniquely identifies a row. `source_clip_id` links its seven versions.
`source_file`, `drummer`, and `style` retain the original metadata. `split` is the
existing GMD split, not a new drummer-disjoint split. `subdivision`, `strength`,
and `normalized_velocity` describe the transformation; inapplicable values are
blank. Human and velocity-only rows have strength 0.

Times and timing features are in seconds. `tempo` is quarter notes per minute.
`timing_abs_mean` and `timing_std` measure deviations from the 1/16 grid;
`timing_32_*` uses the 1/32 grid. IOIs are between distinct onset times, pooling
all drums. Exact simultaneous hits count as one onset event; a clip with only
one distinct onset gets missing IOI values. Velocity summaries pool all drum
notes and are in MIDI velocity units.

## Optional MIDI exports and scope

Add `--midi-output data/generated` to the conditions command to save six
transformed full performances per usable file, under condition/source paths.
These are full performances, not separately cut clip files. Original MIDI is
never rewritten. MIDI exports round fractional ticks to the native resolution;
the feature table is computed directly in memory before that rounding. Use
`load_midi` when reading files if zero-duration drum hits must be retained.

No style filtering, artificial humanization, new train/test split, or model
training is performed here. When training is added, group related rows by source
performance (preferably by drummer) and keep identifiers, conditions, and other
transformation metadata out of the classifier inputs.

## Full run on 11 October 2026

577 of 1,150 performances produced 10,804 original clips; 573 performances were
too short for a complete two-bar clip under the existing clipping rule. The
conditions table contains 75,628 rows, and 3,462 transformed MIDI files were
exported. The loader recovered 796 omitted drum hits across 204 usable files.
The 36 synthetic tests and full-data checks pass. The audit and feature plots are
saved in `data/features/DATASET_CHECKS.md` after checking every table row and MIDI
export. No classifier results are included.
