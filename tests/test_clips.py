"""Reading MIDI, cutting clips, and building feature rows for one file."""

import mido
import numpy as np
import pretty_midi
import pytest

from src.build_dataset import SkipFile, load_performance, process_file
from src.common.midi import get_drum_notes, load_midi, nearest_grid_index, note_grid
from src.data.conditions import CONDITIONS
from src.data.gmd import scope_exclusion
from src.features.magnitude import grid_deviations
from synthetic import BAR, RESOLUTION, SIXTEENTH, groove, write_midi


def test_get_drum_notes_keeps_timing_and_velocity(tmp_path):
    path = write_midi(tmp_path / "a.mid", [(0, 36, 101), (130, 38, 57)])
    midi = pretty_midi.PrettyMIDI(str(path))
    onsets, pitches, velocities = get_drum_notes(midi)
    np.testing.assert_allclose(onsets, [0.0, midi.tick_to_time(130)])
    assert pitches.tolist() == [36, 38]
    assert velocities.tolist() == [101, 57]


def test_grid_follows_tempo_change(tmp_path):
    # Tempo halves after 2 bars; a fixed-BPM grid would misplace bars 3-4.
    path = write_midi(tmp_path / "a.mid", groove(4), tempos=[(0, 120.0), (2 * BAR, 60.0)])
    _, performance = load_performance(path, "a")
    clips = performance.clips
    assert len(clips) == 2
    assert clips[0].tempo == pytest.approx(120.0)
    assert clips[1].tempo == pytest.approx(60.0)
    assert (clips[1].start, clips[1].end) == pytest.approx((4.0, 12.0))
    np.testing.assert_allclose(grid_deviations(performance.onsets, performance.grid_16), 0.0, atol=1e-9)


def test_trailing_partial_clip_is_dropped(tmp_path):
    _, performance = load_performance(write_midi(tmp_path / "a.mid", groove(5)), "a")
    assert len(performance.clips) == 2


def test_early_note_belongs_to_next_clip(tmp_path):
    notes = groove(4) + [(2 * BAR - 10, 49, 33)]  # crash just before clip 2
    _, performance = load_performance(write_midi(tmp_path / "a.mid", notes), "a")
    first, second = performance.clips
    assert 33 not in performance.velocities[first.notes]
    assert 33 in performance.velocities[second.notes]


def test_three_four_clip_length(tmp_path):
    notes = [(k * 2 * SIXTEENTH, 42, 80) for k in range(6 * 4)]  # 4 bars of 3/4
    path = write_midi(tmp_path / "a.mid", notes, meter=(3, 4))
    midi, performance = load_performance(path, "a")
    assert performance.clips[0].end == pytest.approx(midi.tick_to_time(2 * 3 * RESOLUTION))


def test_tick_rounding_keeps_the_earlier_grid_point_on_a_tie(tmp_path):
    path = write_midi(tmp_path / "tie.mid", [(3330, 38, 90)], tempos=[(0, 102.0)])
    midi = pretty_midi.PrettyMIDI(str(path))
    onsets, _, _ = get_drum_notes(midi)
    assert nearest_grid_index(onsets, note_grid(midi, midi.get_end_time(), 32)).tolist() == [55]


@pytest.mark.parametrize("subdivision", [8, 64])
def test_unsupported_grid(subdivision):
    with pytest.raises(ValueError, match="Unsupported subdivision"):
        note_grid(pretty_midi.PrettyMIDI(), 1.0, subdivision)


def test_process_file_rows_cover_every_clip_and_condition(tmp_path):
    path = write_midi(tmp_path / "a.mid", groove(4))
    rows, empty = process_file(path, "x/a.mid", {"drummer": "d1", "style": "rock", "split": "train"})
    assert empty == 0
    assert len(rows) == 2 * len(CONDITIONS)
    assert len({row["clip_id"] for row in rows}) == len(rows)
    human = [row for row in rows if row["condition"] == "human"]
    assert [row["source_clip_id"] for row in human] == ["x/a_clip000", "x/a_clip001"]
    assert human[0]["drummer"] == "d1" and human[0]["gmd_split"] == "train"
    assert human[0]["note_count"] == 2 * 12
    assert human[0]["mag_timing_abs_mean"] == pytest.approx(0.0)
    for source in ("x/a_clip000", "x/a_clip001"):
        versions = [row for row in rows if row["source_clip_id"] == source]
        assert {row["condition"] for row in versions} == {c.name for c in CONDITIONS}
        for key in ("source_file", "drummer", "clip_start", "clip_end", "note_count", "tempo"):
            assert len({row[key] for row in versions}) == 1


def test_empty_clip_is_counted(tmp_path):
    notes = groove(2) + [(4 * BAR + k * RESOLUTION, 36, 100) for k in range(8)]
    rows, empty = process_file(write_midi(tmp_path / "a.mid", notes), "a.mid", {})
    assert len(rows) == 2 * len(CONDITIONS)
    assert empty == 1


@pytest.mark.parametrize(
    "notes, meter, reason",
    [
        (groove(4), None, "no time signature"),
        ([], (4, 4), "no drum notes"),
        (groove(1), (4, 4), "shorter than one clip"),
    ],
)
def test_unusable_files_are_skipped(tmp_path, notes, meter, reason):
    path = write_midi(tmp_path / "a.mid", notes, meter=meter)
    with pytest.raises(SkipFile, match=reason):
        process_file(path, "a.mid", {})


def test_invalid_midi_is_skipped(tmp_path):
    path = tmp_path / "broken.mid"
    path.write_text("not a midi file")
    with pytest.raises(SkipFile, match="invalid MIDI"):
        process_file(path, "broken.mid", {})


def test_zero_length_drum_hit_is_retained(tmp_path):
    path = write_midi(tmp_path / "zero.mid", groove(2))
    raw = mido.MidiFile(path)
    events, tick = [], 0
    for message in raw.tracks[0]:
        tick += message.time
        events.append((tick, message))
    events += [
        (600, mido.Message("note_on", channel=9, note=51, velocity=31)),
        (600, mido.Message("note_off", channel=9, note=51)),
    ]
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


@pytest.mark.parametrize(
    "meta, reason",
    [
        ({"beat_type": "fill", "time_signature": "4-4", "style": "rock"}, "not a beat (fill)"),
        ({"beat_type": "beat", "time_signature": "3-4", "style": "rock"}, "not 4/4"),
        ({"beat_type": "beat", "time_signature": "4-4", "style": "jazz/swing"}, "swing/shuffle/triplet style"),
        ({"beat_type": "beat", "time_signature": "4-4", "style": "blues/shuffle"}, "swing/shuffle/triplet style"),
        ({"beat_type": "beat", "time_signature": "4-4", "style": "funk/triplet"}, "swing/shuffle/triplet style"),
    ],
)
def test_scope_exclusion(meta, reason):
    assert scope_exclusion(meta) == reason


def test_scope_keeps_ordinary_beats():
    assert scope_exclusion({"beat_type": "beat", "time_signature": "4-4", "style": "rock"}) is None
