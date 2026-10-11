"""Check transformations, tempo changes, and original clip membership."""

import numpy as np
import mido
import pretty_midi
import pytest

from src.build_dataset import process_file
from src.common.config import QUANTIZATION_CONDITIONS
from src.common.midi import get_drum_notes, load_midi, nearest_grid_index, note_grid
from src.data.quantize import (
    normalize_velocity_midi, quantize_midi, quantize_onsets, quantize_time,
)
from src.features.magnitude import clip_features
from test_clips import BAR, groove, write_midi


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_partial_quantization_and_earlier_tie(strength):
    onsets = np.array([0.02, 0.115, 0.1875])
    targets = np.array([0.0, 0.125, 0.125])
    actual = quantize_onsets(onsets, np.array([0.0, 0.125, 0.25]), strength)
    np.testing.assert_allclose(actual, onsets + strength * (targets - onsets))
    np.testing.assert_array_equal(onsets, [0.02, 0.115, 0.1875])
    assert quantize_time(0.1875, 0.125, strength) == pytest.approx(actual[-1])


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        quantize_onsets(np.array([0.01]), np.array([0.0, 0.125]), strength)


@pytest.mark.parametrize("subdivision, target_tick", [(16, 120), (32, 180)])
def test_quantized_midi_follows_tempo_map(tmp_path, subdivision, target_tick):
    path = write_midi(
        tmp_path / "tempo.mid",
        [(0, 42, 70), (2 * BAR + 170, 38, 91), (4 * BAR - 240, 42, 70)],
        tempos=[(0, 120.0), (2 * BAR, 60.0)],
    )
    original = pretty_midi.PrettyMIDI(str(path))
    original_starts = [n.start for n in original.instruments[0].notes]
    quantized = quantize_midi(original, subdivision)
    snare = next(n for n in quantized.instruments[0].notes if n.pitch == 38)
    assert snare.start == pytest.approx(original.tick_to_time(2 * BAR + target_tick))
    for before, after in zip(original.instruments[0].notes, quantized.instruments[0].notes):
        assert after.end - after.start == pytest.approx(before.end - before.start)
        assert (after.pitch, after.velocity) == (before.pitch, before.velocity)
    assert [n.start for n in original.instruments[0].notes] == original_starts


def test_transformations_leave_non_drums_and_original_unchanged():
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    for is_drum in [True, False]:
        instrument = pretty_midi.Instrument(program=0, is_drum=is_drum)
        instrument.notes.append(pretty_midi.Note(velocity=63, pitch=38, start=0.02, end=0.1))
        midi.instruments.append(instrument)
    quantized = quantize_midi(midi)
    normalized = normalize_velocity_midi(midi)
    assert quantized.instruments[0].notes[0].start == 0.0
    assert normalized.instruments[0].notes[0].velocity == 80
    for changed in [quantized, normalized]:
        assert changed.instruments[1].notes[0].start == 0.02
        assert changed.instruments[1].notes[0].velocity == 63
    assert midi.instruments[0].notes[0].start == 0.02
    assert midi.instruments[0].notes[0].velocity == 63
    assert normalized.instruments[0].notes[0].end == 0.1


@pytest.mark.parametrize("subdivision", [8, 64])
def test_unsupported_grid(subdivision):
    with pytest.raises(ValueError, match="Unsupported subdivision"):
        note_grid(pretty_midi.PrettyMIDI(), 1.0, subdivision)


def test_matching_rows_and_original_boundary_assignment(tmp_path):
    notes = groove(4) + [(2 * BAR - 10, 49, 33)]
    path = write_midi(tmp_path / "a.mid", notes)
    meta = {"drummer": "d1", "style": "rock", "split": "train"}
    human, _ = process_file(path, "x/a.mid", meta)
    rows, empty = process_file(path, "x/a.mid", meta, include_conditions=True)
    assert empty == 0
    assert [r for r in rows if r["condition"] == "human"] == human
    assert len(rows) == 14
    assert len({r["clip_id"] for r in rows}) == 14
    for original in human:
        variants = [r for r in rows if r["source_clip_id"] == original["clip_id"]]
        assert len(variants) == 7
        for row in variants:
            for key in ["source_file", "drummer", "split", "clip_start", "clip_end", "note_count"]:
                assert row[key] == original[key]
        by_condition = {r["condition"]: r for r in variants}
        for condition, _, strength in QUANTIZATION_CONDITIONS[:4]:
            assert by_condition[condition]["timing_abs_mean"] == pytest.approx(
                original["timing_abs_mean"] * (1 - strength), abs=1e-12
            )
        normalized = by_condition["velocity_normalized"]
        assert normalized["timing_abs_mean"] == original["timing_abs_mean"]
        assert normalized["ioi_mean"] == original["ioi_mean"]
        assert normalized["velocity_mean"] == 80
        assert normalized["velocity_std"] == normalized["velocity_range"] == 0


def test_thirty_second_features_use_the_correct_grid():
    grid_16 = np.array([0.0, 0.125, 0.25])
    grid_32 = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25])
    features = clip_features(np.array([0.0625, 0.1875]), np.array([70, 90]), grid_16, grid_32)
    assert features["timing_abs_mean"] == 0.0625
    assert features["timing_32_abs_mean"] == features["timing_32_std"] == 0.0


def test_tick_to_seconds_rounding_keeps_earlier_tie(tmp_path):
    path = write_midi(tmp_path / "tie.mid", [(3330, 38, 90)], tempos=[(0, 102.0)])
    midi = pretty_midi.PrettyMIDI(str(path))
    onsets, _, _ = get_drum_notes(midi)
    grid = note_grid(midi, midi.get_end_time(), 32)
    assert nearest_grid_index(onsets, grid).tolist() == [55]
    quantized = quantize_midi(midi, 32)
    assert quantized.instruments[0].notes[0].start == midi.tick_to_time(3300)


def test_optional_midi_export(tmp_path):
    path = write_midi(tmp_path / "original.mid", groove(2))
    destination = tmp_path / "generated"
    rows, _ = process_file(path, "x/a.mid", {}, include_conditions=True, midi_output=destination)
    exports = sorted(destination.rglob("*.mid"))
    assert len(exports) == 6
    original = pretty_midi.PrettyMIDI(str(path))
    onsets, _, _ = get_drum_notes(original)
    normalized = pretty_midi.PrettyMIDI(str(destination / "velocity_normalized/x/a.mid"))
    starts, _, velocities = get_drum_notes(normalized)
    np.testing.assert_allclose(starts, onsets)
    np.testing.assert_array_equal(velocities, np.full(len(onsets), 80))
    assert len(rows) == 7


def test_zero_length_drum_hit_is_retained(tmp_path):
    path = write_midi(tmp_path / "zero.mid", groove(2))
    raw = mido.MidiFile(path)
    events, tick = [], 0
    for message in raw.tracks[0]:
        tick += message.time
        events.append((tick, message))
    events.extend([
        (600, mido.Message("note_on", channel=9, note=51, velocity=31)),
        (600, mido.Message("note_off", channel=9, note=51)),
    ])
    raw.tracks[0].clear()
    previous = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        raw.tracks[0].append(message.copy(time=tick - previous))
        previous = tick
    raw.save(path)
    midi = load_midi(path)
    onsets, pitches, velocities = get_drum_notes(midi)
    assert len(onsets) == 25
    assert onsets[pitches == 51].tolist() == [midi.tick_to_time(600)]
    assert velocities[pitches == 51].tolist() == [31]
    destination = tmp_path / "generated"
    rows, _ = process_file(path, "zero.mid", {}, True, destination)
    assert all(row["note_count"] == 25 for row in rows)
    for export in destination.rglob("*.mid"):
        assert sum(m.type == "note_on" and m.velocity > 0 for m in mido.MidiFile(export)) == 25
