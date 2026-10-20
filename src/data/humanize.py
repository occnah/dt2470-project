"""
DAW-style humanization: random timing jitter added to quantized onsets.

Three kinds of jitter, each returned as a zero-mean, unit-SD sequence with
one value per note (in onset order), so that the kinds differ only in
distribution shape and correlation over time, never in size:

- "uniform":  independent, uniformly distributed
- "gaussian": independent, normally distributed
- "pink":     1/f ("pink") noise, correlated over time, i.e. slow drifts
              ahead of and behind the beat, as found in human timing

Callers scale the sequence to the wanted size (see `src/data/conditions.py`).
"""

import hashlib

import numpy as np

from src.common.config import SEED

JITTER_KINDS = ("uniform", "gaussian", "pink")


def rng_for(*keys: str) -> np.random.Generator:
    """
    A random generator seeded from the project seed and `keys`.

    The same keys always give the same numbers, independent of the order
    in which files or conditions are processed.
    """
    text = "|".join([str(SEED), *keys])
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def standardize(values: np.ndarray) -> np.ndarray:
    """Shift and scale to zero mean and unit SD (zeros if there is no spread)."""
    values = np.asarray(values, dtype=float)
    spread = values.std()
    if len(values) < 2 or spread == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    1/f noise of length `n`: white noise whose power spectrum is reshaped
    to fall off as 1/frequency. Returned standardized.
    """
    if n < 2:
        return np.zeros(n)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequencies = np.fft.rfftfreq(n)
    scale = np.zeros_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])  # amplitude ~ f^-1/2, power ~ 1/f
    return standardize(np.fft.irfft(spectrum * scale, n))


def jitter(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """A standardized jitter sequence of the given kind, one value per note."""
    if kind == "uniform":
        values = rng.uniform(-1.0, 1.0, n)
    elif kind == "gaussian":
        values = rng.standard_normal(n)
    elif kind == "pink":
        values = pink_noise(n, rng)
    else:
        raise ValueError(f"Unknown jitter kind: {kind}")
    return standardize(values)
