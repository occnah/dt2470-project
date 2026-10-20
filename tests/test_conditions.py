"""Conditions: quantization, velocity normalization, humanization and the shuffled control."""

import numpy as np
import pretty_midi
import pytest

from src.build_dataset import load_performance, process_file
from src.common.midi import get_drum_notes, with_drum_notes
from src.data.conditions import (
    BY_NAME, CONDITIONS, EDITED_CONDITIONS, JITTER_SD_MS, VELOCITY_TARGET, apply_condition,
)
from src.data.humanize import JITTER_KINDS, jitter, pink_noise, rng_for
from src.data.quantize import normalize_velocities, quantize_onsets
from src.features import MAGNITUDE_COLUMNS, STRUCTURE_COLUMNS
from synthetic import groove, write_midi


@pytest.fixture
def performance(tmp_path):
    """Four bars with late off-beat hi-hats (systematic, human-like timing)."""
    _, performance = load_performance(write_midi(tmp_path / "p.mid", groove(4, swing_ticks=25)), "p.mid")
    return performance


# --- Quantization and velocity -------------------------------------------

@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_partial_quantization_moves_onsets_part_of_the_way(strength):
    onsets = np.array([0.02, 0.115, 0.1875])
    targets = np.array([0.0, 0.125, 0.125])  # 0.1875 is a tie: earlier point
    actual = quantize_onsets(onsets, np.array([0.0, 0.125, 0.25]), strength)
    np.testing.assert_allclose(actual, onsets + strength * (targets - onsets))
    np.testing.assert_array_equal(onsets, [0.02, 0.115, 0.1875])


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        quantize_onsets(np.array([0.01]), np.array([0.0, 0.125]), strength)
    with pytest.raises(ValueError, match="strength"):
        normalize_velocities(np.array([60.0]), 80, strength)


def test_velocity_normalization():
    velocities = np.array([40.0, 80.0, 120.0])
    np.testing.assert_allclose(normalize_velocities(velocities, 80, 0.5), [60, 80, 100])
    np.testing.assert_array_equal(normalize_velocities(velocities, 80, 1.0), [80, 80, 80])


# --- The condition list --------------------------------------------------

def test_condition_list():
    names = [c.name for c in CONDITIONS]
    assert len(names) == len(set(names))
    assert CONDITIONS[0].name == "human" and "human" not in [c.name for c in EDITED_CONDITIONS]
    assert {c.family for c in CONDITIONS} == {"human", "quantize", "velocity", "humanize", "shuffle"}
    humanized = [c for c in CONDITIONS if c.family == "humanize"]
    assert len(humanized) == len(JITTER_KINDS) * (len(JITTER_SD_MS) + 1)
    assert all(c.subdivision == 16 and c.strength == 1.0 for c in humanized)
    assert all(c.label for c in CONDITIONS)


# --- Jitter ----------------------------------------------------------------

@pytest.mark.parametrize("kind", JITTER_KINDS)
def test_jitter_is_standardized_and_reproducible(kind):
    values = jitter(kind, 500, rng_for("file", kind))
    assert values.mean() == pytest.approx(0.0, abs=1e-12)
    assert values.std() == pytest.approx(1.0)
    np.testing.assert_array_equal(values, jitter(kind, 500, rng_for("file", kind)))
    assert not np.array_equal(values, jitter(kind, 500, rng_for("other file", kind)))


def test_uniform_jitter_is_bounded_and_independent():
    values = jitter("uniform", 20000, rng_for("u"))
    assert np.abs(values).max() <= 1.01 * np.sqrt(3)  # standardized by the sample SD
    assert abs(np.corrcoef(values[:-1], values[1:])[0, 1]) < 0.03


def test_pink_noise_has_a_one_over_f_spectrum():
    rng = rng_for("pink")
    power = np.mean([np.abs(np.fft.rfft(pink_noise(4096, rng)))[1:] ** 2 for _ in range(30)], axis=0)
    frequencies = np.fft.rfftfreq(4096)[1:]
    slope = np.polyfit(np.log(frequencies), np.log(power), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)
    values = pink_noise(4096, rng)
    assert np.corrcoef(values[:-1], values[1:])[0, 1] > 0.3  # correlated over time


# --- Applying conditions ---------------------------------------------------

def test_every_condition_keeps_the_notes(performance):
    for condition in CONDITIONS:
        onsets, velocities = apply_condition(condition, performance)
        assert onsets.shape == performance.onsets.shape
        assert velocities.shape == performance.velocities.shape
    np.testing.assert_array_equal(apply_condition(BY_NAME["human"], performance)[0], performance.onsets)


def test_conditions_are_deterministic(performance):
    for condition in CONDITIONS:
        first = apply_condition(condition, performance)
        second = apply_condition(condition, performance)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


def test_timing_and_velocity_edits_are_independent(performance):
    for condition in EDITED_CONDITIONS:
        onsets, velocities = apply_condition(condition, performance)
        if condition.family == "velocity":
            np.testing.assert_array_equal(onsets, performance.onsets)
        else:
            np.testing.assert_array_equal(velocities, performance.velocities)


def test_humanized_jitter_has_the_nominal_size(performance):
    quantized = quantize_onsets(performance.onsets, performance.grid_16, 1.0)
    for kind in JITTER_KINDS:
        offsets = {sd: apply_condition(BY_NAME[f"humanize_{kind}_{sd}ms"], performance)[0] - quantized
                   for sd in JITTER_SD_MS}
        for sd, values in offsets.items():
            assert values.std() == pytest.approx(sd / 1000)
        # Same random sequence at every level, only the scale differs.
        np.testing.assert_allclose(offsets[10], 2 * offsets[5])


def test_matched_jitter_has_each_clips_human_mean_and_sd(performance):
    quantized = quantize_onsets(performance.onsets, performance.grid_16, 1.0)
    human = performance.onsets - quantized
    for kind in JITTER_KINDS:
        offsets = apply_condition(BY_NAME[f"humanize_{kind}_matched"], performance)[0] - quantized
        for clip in performance.clips:
            assert offsets[clip.notes].mean() == pytest.approx(human[clip.notes].mean())
            assert offsets[clip.notes].std() == pytest.approx(human[clip.notes].std())
            assert not np.allclose(offsets[clip.notes], human[clip.notes])


def test_shuffle_keeps_each_clips_deviations_but_not_their_order(performance):
    quantized = quantize_onsets(performance.onsets, performance.grid_16, 1.0)
    human = performance.onsets - quantized
    shuffled = apply_condition(BY_NAME["deviations_shuffled"], performance)[0] - quantized
    for clip in performance.clips:
        np.testing.assert_allclose(np.sort(shuffled[clip.notes]), np.sort(human[clip.notes]))
    assert not np.allclose(shuffled, human)


# --- What the features see -------------------------------------------------

def _rows_by_condition(tmp_path):
    path = write_midi(tmp_path / "f.mid", groove(4, swing_ticks=25))
    rows, _ = process_file(path, "f.mid", {})
    return {(row["source_clip_id"], row["condition"]): row for row in rows}


def test_partial_edits_scale_magnitude_and_leave_structure_unchanged(tmp_path):
    rows = _rows_by_condition(tmp_path)
    for (clip, name), row in rows.items():
        condition = BY_NAME[name]
        if condition.family not in ("quantize", "velocity") or condition.strength == 1.0:
            continue
        human = rows[(clip, "human")]
        for feature in STRUCTURE_COLUMNS:
            assert row[feature] == pytest.approx(human[feature], abs=1e-9, nan_ok=True)
        if condition.family == "quantize":
            assert row["mag_timing_std"] == pytest.approx(human["mag_timing_std"] * (1 - condition.strength))
        else:
            assert row["mag_timing_std"] == human["mag_timing_std"]
            assert row["mag_velocity_std"] == pytest.approx(human["mag_velocity_std"] * (1 - condition.strength))


def test_shuffle_keeps_timing_magnitude_exactly(tmp_path):
    rows = _rows_by_condition(tmp_path)
    timing = ("mag_timing_abs_mean", "mag_timing_std")
    for (clip, name), row in rows.items():
        if name == "deviations_shuffled":
            for feature in timing:
                assert row[feature] == pytest.approx(rows[(clip, "human")][feature])
    assert all(f.startswith("mag_") for f in MAGNITUDE_COLUMNS)


# --- MIDI export -------------------------------------------------------------

def test_midi_export_writes_every_edited_condition(tmp_path):
    path = write_midi(tmp_path / "original.mid", groove(2, swing_ticks=25))
    destination = tmp_path / "generated"
    process_file(path, "x/a.mid", {}, midi_output=destination)
    assert len(list(destination.rglob("*.mid"))) == len(EDITED_CONDITIONS)

    original = pretty_midi.PrettyMIDI(str(path))
    onsets, pitches, _ = get_drum_notes(original)
    quantized = pretty_midi.PrettyMIDI(str(destination / "quantize_16_100/x/a.mid"))
    starts, quantized_pitches, _ = get_drum_notes(quantized)
    ticks = np.array([quantized.time_to_tick(t) for t in starts])
    assert (ticks % 120 == 0).all()
    np.testing.assert_array_equal(np.sort(quantized_pitches), np.sort(pitches))
    normalized = pretty_midi.PrettyMIDI(str(destination / "velocity_100/x/a.mid"))
    assert set(get_drum_notes(normalized)[2]) == {VELOCITY_TARGET}
    assert len(get_drum_notes(normalized)[0]) == len(onsets)


def test_with_drum_notes_leaves_the_original_and_other_instruments_unchanged():
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    for is_drum in (True, False):
        instrument = pretty_midi.Instrument(program=0, is_drum=is_drum)
        instrument.notes.append(pretty_midi.Note(velocity=63, pitch=38, start=0.02, end=0.1))
        midi.instruments.append(instrument)
    edited = with_drum_notes(midi, np.array([0.0]), np.array([80.0]))
    assert (edited.instruments[0].notes[0].start, edited.instruments[0].notes[0].velocity) == (0.0, 80)
    assert edited.instruments[0].notes[0].end == pytest.approx(0.08)
    assert (edited.instruments[1].notes[0].start, edited.instruments[1].notes[0].velocity) == (0.02, 63)
    assert (midi.instruments[0].notes[0].start, midi.instruments[0].notes[0].velocity) == (0.02, 63)
