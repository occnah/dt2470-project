"""
The experimental conditions: every version of a clip that is compared with
the original human performance, and how each is produced.

This file is the single place where conditions are defined. The dataset
builder, the experiments, the plots and the tests all read `CONDITIONS`.

Families (and the research question each serves):

- human     the original performance                         (reference)
- quantize  onsets moved toward the 1/16 or 1/32 grid         (RQ1)
- velocity  velocities moved toward a constant                (RQ1)
- humanize  full 1/16 quantization + random timing jitter     (RQ2, RQ3)
- shuffle   each clip's own human deviations, shuffled among  (RQ3)
            its notes: exactly the same sizes, structure removed

Humanization levels are jitter standard deviations of 5, 10 and 20 ms,
equal for all jitter kinds, plus a "matched" level: within each clip the
jitter is shifted and scaled to the mean and SD of that clip's human timing
deviations. At the matched level the size of the deviations is the same as
in the human clip, so only their structure can tell the two apart (RQ3).

Jitter is drawn once per performance and jitter kind (common random numbers):
the 5, 10 and 20 ms levels are the same sequence at different scales.
"""

from dataclasses import dataclass

import numpy as np

from src.common.config import HUMAN
from src.data.clips import Clip
from src.data.humanize import JITTER_KINDS, jitter, rng_for
from src.data.quantize import normalize_velocities, quantize_onsets

QUANTIZE_STRENGTHS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)
VELOCITY_TARGET = 80
VELOCITY_STRENGTHS = (0.25, 0.50, 0.75, 1.00)
JITTER_SD_MS = (5, 10, 20)


@dataclass(frozen=True)
class Condition:
    """One version of a clip. Unused parameters are None."""

    name: str
    family: str
    subdivision: int | None = None
    strength: float | None = None
    jitter: str | None = None
    jitter_sd_ms: float | None = None
    matched: bool = False

    @property
    def label(self) -> str:
        """Short human-readable description for tables and plots."""
        if self.family == "quantize":
            return f"1/{self.subdivision} at {self.strength:.0%}"
        if self.family == "velocity":
            return f"Velocity at {self.strength:.0%}"
        if self.family == "humanize":
            size = "matched" if self.matched else f"{self.jitter_sd_ms:g} ms"
            return f"{self.jitter.capitalize()} jitter, {size}"
        if self.family == "shuffle":
            return "Shuffled human deviations"
        return "Human"

    @property
    def series(self) -> str:
        """Which curve this condition belongs to in the detection plots."""
        if self.family == "quantize":
            return f"quantize_{self.subdivision}"
        if self.family == "humanize":
            return f"humanize_{self.jitter}"
        return self.family


def _build_conditions() -> tuple[Condition, ...]:
    conditions = [Condition(HUMAN, "human")]
    for strength in QUANTIZE_STRENGTHS:
        conditions.append(
            Condition(f"quantize_16_{round(strength * 100):02d}", "quantize", 16, strength)
        )
    conditions.append(Condition("quantize_32_100", "quantize", 32, 1.0))
    for strength in VELOCITY_STRENGTHS:
        conditions.append(Condition(f"velocity_{round(strength * 100)}", "velocity", strength=strength))
    for kind in JITTER_KINDS:
        for sd in JITTER_SD_MS:
            conditions.append(
                Condition(f"humanize_{kind}_{sd}ms", "humanize", 16, 1.0, kind, float(sd))
            )
        conditions.append(
            Condition(f"humanize_{kind}_matched", "humanize", 16, 1.0, kind, matched=True)
        )
    conditions.append(Condition("deviations_shuffled", "shuffle", 16, 1.0, matched=True))
    return tuple(conditions)


CONDITIONS = _build_conditions()
BY_NAME = {condition.name: condition for condition in CONDITIONS}
EDITED_CONDITIONS = tuple(c for c in CONDITIONS if c.family != "human")


@dataclass
class Performance:
    """
    One original performance with everything needed to produce conditions.

    Arrays have one entry per drum note, sorted by onset (as returned by
    `get_drum_notes`). `key` identifies the performance for seeding.
    """

    key: str
    onsets: np.ndarray
    velocities: np.ndarray
    pitches: np.ndarray
    grid_16: np.ndarray
    grid_32: np.ndarray
    clips: list[Clip]


def apply_condition(
    condition: Condition,
    performance: Performance,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (onsets, velocities) of the whole performance in `condition`.

    The note set, order and pitches never change, so each note keeps the
    clip it had in the original performance.
    """
    onsets, velocities = performance.onsets, performance.velocities

    if condition.family == "human":
        return onsets.copy(), velocities.copy()

    if condition.family == "quantize":
        grid = performance.grid_16 if condition.subdivision == 16 else performance.grid_32
        return quantize_onsets(onsets, grid, condition.strength), velocities.copy()

    if condition.family == "velocity":
        return onsets.copy(), normalize_velocities(velocities, VELOCITY_TARGET, condition.strength)

    if condition.family == "humanize":
        return _humanize(condition, performance), velocities.copy()

    if condition.family == "shuffle":
        return _shuffle_deviations(performance), velocities.copy()

    raise ValueError(f"Unknown condition family: {condition.family}")


def _humanize(condition: Condition, performance: Performance) -> np.ndarray:
    """Fully quantize to the 1/16 grid, then add jitter (see module docstring)."""
    onsets = performance.onsets
    quantized = quantize_onsets(onsets, performance.grid_16, 1.0)
    noise = jitter(condition.jitter, len(onsets), rng_for(performance.key, condition.jitter))

    if not condition.matched:
        return quantized + noise * condition.jitter_sd_ms / 1000.0

    # Match each clip's human deviations: original onsets minus their grid point.
    human_deviations = onsets - quantized
    offsets = np.zeros_like(onsets)
    for clip in performance.clips:
        mask = clip.notes
        if not mask.any():
            continue
        clip_noise = noise[mask]
        spread = clip_noise.std()
        scaled = (clip_noise - clip_noise.mean()) / spread if spread > 0 else np.zeros_like(clip_noise)
        target = human_deviations[mask]
        offsets[mask] = scaled * target.std() + target.mean()
    return quantized + offsets


def _shuffle_deviations(performance: Performance) -> np.ndarray:
    """
    Within each clip, give every note the grid deviation of a randomly
    chosen other note of the same clip (a permutation). The clip keeps
    exactly its human set of deviation sizes; which note has which, and so
    any timing structure, is random.
    """
    onsets = performance.onsets
    quantized = quantize_onsets(onsets, performance.grid_16, 1.0)
    deviations = onsets - quantized
    rng = rng_for(performance.key, "shuffle")
    shuffled = deviations.copy()
    for clip in performance.clips:
        indices = np.flatnonzero(clip.notes)
        shuffled[indices] = deviations[rng.permutation(indices)]
    return quantized + shuffled
