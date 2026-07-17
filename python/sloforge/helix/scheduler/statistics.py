"""Small-sample statistical helpers shared by Helix evaluation paths."""

from __future__ import annotations

import math
import statistics

# Two-sided 95% Student-t critical values for 1..31 degrees of freedom.  Helix
# evaluation matrices are deliberately bounded to at most 32 independent seeds.
_T_975 = (
    12.706204736,
    4.30265273,
    3.182446305,
    2.776445105,
    2.570581836,
    2.446911851,
    2.364624252,
    2.306004135,
    2.262157163,
    2.228138852,
    2.20098516,
    2.17881283,
    2.160368656,
    2.144786688,
    2.131449546,
    2.119905299,
    2.109815578,
    2.10092204,
    2.093024054,
    2.085963447,
    2.079613845,
    2.073873068,
    2.06865761,
    2.063898562,
    2.059538553,
    2.055529439,
    2.051830516,
    2.048407142,
    2.045229642,
    2.042272456,
    2.039513446,
)


def student_t_mean_interval_95(
    values: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Return mean, lower, upper, and sample deviation for 2..32 observations."""

    if not 2 <= len(values) <= 32:
        raise ValueError("Student-t interval requires two to 32 observations")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Student-t interval observations must be finite")
    mean = math.fsum(values) / len(values)
    deviation = statistics.stdev(values)
    critical = _T_975[len(values) - 2]
    half_width = critical * deviation / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width, deviation


__all__ = ["student_t_mean_interval_95"]
