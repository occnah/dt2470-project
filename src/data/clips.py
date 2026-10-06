"""Cutting performances into fixed-length clips."""

from dataclasses import dataclass

import numpy as np
import pretty_midi

from src.common.config import BARS_PER_CLIP
from src.common.midi import downbeat_ticks, nearest_grid_index


@dataclass
class Clip:
    """
    One clip of a performance.

    `notes` is a boolean mask over the performance's drum notes selecting
    the notes inside the clip.
    """

    index: int
    start: float
    end: float
    tempo: float
    notes: np.ndarray


def cut_clips(
    midi: pretty_midi.PrettyMIDI,
    onsets: np.ndarray,
    grid: np.ndarray,
    bars_per_clip: int = BARS_PER_CLIP,
) -> list[Clip]:
    """
    Split a performance into consecutive, non-overlapping clips of
    `bars_per_clip` bars, starting at the first downbeat.

    Bars come from the MIDI's tempo and time signature. Each note belongs
    to the clip containing its nearest 1/16 grid point, so a hit played
    slightly ahead of a clip's first downbeat belongs to that clip.

    Clips that end more than one quarter note after the last drum onset
    (the trailing partial clip, or silence after the drummer stops) are
    dropped. Clips with no drum notes are returned with an empty mask.

    `onsets` must be sorted and non-empty; `grid` is the 1/16 grid from
    `sixteenth_grid`. `tempo` is the clip's average tempo in quarter notes
    per minute.
    """
    step = midi.resolution // 4
    note_steps = nearest_grid_index(onsets, grid)
    last_onset_tick = int(midi.time_to_tick(onsets[-1]))

    # A clip may end at most one quarter note after the last onset.
    bars = downbeat_ticks(midi, last_onset_tick + midi.resolution)
    clips = []

    for i in range(0, len(bars) - bars_per_clip, bars_per_clip):
        start_tick = bars[i]
        end_tick = bars[i + bars_per_clip]

        start = midi.tick_to_time(start_tick)
        end = midi.tick_to_time(end_tick)
        quarters = (end_tick - start_tick) / midi.resolution

        clips.append(
            Clip(
                index=i // bars_per_clip,
                start=start,
                end=end,
                tempo=60.0 * quarters / (end - start),
                notes=(note_steps >= start_tick // step) & (note_steps < end_tick // step),
            )
        )

    return clips
