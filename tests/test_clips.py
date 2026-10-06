"""Clip cutting and the end-to-end file processing, on synthetic MIDI files."""

from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest

from src.build_dataset import SkipFile, process_file
from src.common.midi import get_drum_notes, sixteenth_grid
from src.data.clips import cut_clips
from src.features.magnitude import grid_deviations

RESOLUTION = 480
SIXTEENTH = RESOLUTION // 4
BAR = 4 * RESOLUTION


def write_midi(path, notes, tempos=((0, 120.0),), meter=(4, 4)):
    """
    Write a one-track drum MIDI file.

    notes: (tick, pitch, velocity); tempos: (tick, bpm); meter=None writes
    no time signature.
    """
    events = [
        (tick, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        for tick, bpm in tempos
    ]
    if meter:
        events.append(
            (0, 0, mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1]))
        )
    for tick, pitch, velocity in notes:
        events.append((tick, 2, mido.Message("note_on", channel=9, note=pitch, velocity=velocity)))
        events.append((tick + 30, 1, mido.Message("note_off", channel=9, note=pitch, velocity=0)))
    events.sort(key=lambda event: (event[0], event[1]))

    track = mido.MidiTrack()
    previous = 0
    for tick, _, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick

    midi = mido.MidiFile(ticks_per_beat=RESOLUTION)
    midi.tracks.append(track)
    midi.save(path)
    return path


def groove(bars):
    """Straight 8th-note hi-hat with kick on 1 and 3, snare on 2 and 4."""
    notes = []
    for bar in range(bars):
        for eighth in range(8):
            notes.append((bar * BAR + eighth * 2 * SIXTEENTH, 42, 70))
        notes += [(bar * BAR, 36, 100), (bar * BAR + 2 * RESOLUTION, 36, 100)]
        notes += [(bar * BAR + RESOLUTION, 38, 90), (bar * BAR + 3 * RESOLUTION, 38, 90)]
    return notes


def clips_of(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    onsets, _, velocities = get_drum_notes(midi)
    grid = sixteenth_grid(midi, midi.get_end_time())
    return midi, onsets, velocities, grid, cut_clips(midi, onsets, grid)


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
    _, onsets, _, grid, clips = clips_of(path)

    assert len(clips) == 2
    assert clips[0].tempo == pytest.approx(120.0)
    assert clips[1].tempo == pytest.approx(60.0)
    assert (clips[1].start, clips[1].end) == pytest.approx((4.0, 12.0))

    np.testing.assert_allclose(grid_deviations(onsets, grid), 0.0, atol=1e-9)


def test_trailing_partial_clip_is_dropped(tmp_path):
    _, _, _, _, clips = clips_of(write_midi(tmp_path / "a.mid", groove(5)))
    assert len(clips) == 2


def test_early_note_belongs_to_next_clip(tmp_path):
    notes = groove(4) + [(2 * BAR - 10, 49, 33)]  # crash just before clip 2
    _, _, velocities, _, clips = clips_of(write_midi(tmp_path / "a.mid", notes))

    assert 33 not in velocities[clips[0].notes]
    assert 33 in velocities[clips[1].notes]


def test_three_four_clip_length(tmp_path):
    notes = [(k * 2 * SIXTEENTH, 42, 80) for k in range(6 * 4)]  # 4 bars of 3/4
    midi, _, _, _, clips = clips_of(write_midi(tmp_path / "a.mid", notes, meter=(3, 4)))
    assert clips[0].end == pytest.approx(midi.tick_to_time(2 * 3 * RESOLUTION))


def test_process_file_rows(tmp_path):
    path = write_midi(tmp_path / "a.mid", groove(4))
    rows, empty = process_file(path, "x/a.mid", {"drummer": "d1", "style": "rock"})

    assert [row["clip_id"] for row in rows] == ["x/a_clip000", "x/a_clip001"]
    assert all(row["condition"] == "human" for row in rows)
    assert rows[0]["drummer"] == "d1"
    assert rows[0]["note_count"] == 2 * 12
    assert rows[0]["timing_abs_mean"] == pytest.approx(0.0)
    assert empty == 0


def test_empty_clip_is_counted(tmp_path):
    notes = groove(2) + [(4 * BAR + k * RESOLUTION, 36, 100) for k in range(8)]
    rows, empty = process_file(write_midi(tmp_path / "a.mid", notes), "a.mid", {})
    assert len(rows) == 2
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
