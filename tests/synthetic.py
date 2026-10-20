"""Synthetic MIDI files for tests, so the GMD is not needed."""

from pathlib import Path

import mido

RESOLUTION = 480
SIXTEENTH = RESOLUTION // 4
BAR = 4 * RESOLUTION


def write_midi(path: Path, notes, tempos=((0, 120.0),), meter=(4, 4)) -> Path:
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


def groove(bars: int, swing_ticks: int = 0):
    """
    Straight 8th-note hi-hat with kick on 1 and 3, snare on 2 and 4.
    `swing_ticks` delays every off-beat hi-hat, giving systematic
    (structured) human-like timing.
    """
    notes = []
    for bar in range(bars):
        for eighth in range(8):
            delay = swing_ticks if eighth % 2 else 0
            notes.append((bar * BAR + eighth * 2 * SIXTEENTH + delay, 42, 70 + 5 * (eighth % 2)))
        notes += [(bar * BAR, 36, 100), (bar * BAR + 2 * RESOLUTION, 36, 100)]
        notes += [(bar * BAR + RESOLUTION, 38, 90), (bar * BAR + 3 * RESOLUTION, 38, 90)]
    return notes
