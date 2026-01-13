"""
Discrete Amplitude Operations

Based on ESM Whitepaper v5.1 Sections 2.2-2.4 and Appendix A

Implements the 8-phase discrete amplitude system with:
- Cartesian conversion (to_cartesian, from_cartesian)
- Amplitude addition with interference
- Probability calculation
- Phase rotation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from esm.core.phase import (
    DiscretePhase,
    COS_TABLE,
    SIN_TABLE,
    quantize_phase_8,
    PRECISION,
    COS_TABLE_INT,
    SIN_TABLE_INT,
)


@dataclass
class DiscreteAmplitude:
    """
    Discrete amplitude with magnitude and 8-phase.

    Represents a quantum-like amplitude α = |α| × e^(iθ)
    where θ is quantized to 8 discrete phases (0°, 45°, ..., 315°).

    Attributes:
        magnitude: The absolute value |α| (always non-negative)
        phase: The discrete phase (one of 8 values)
    """
    magnitude: float
    phase: DiscretePhase

    def __post_init__(self):
        """Ensure magnitude is non-negative."""
        if self.magnitude < 0:
            # Flip sign by rotating phase 180°
            self.magnitude = -self.magnitude
            self.phase = self.phase.opposite()

    # =========================================================================
    # Cartesian Conversion
    # =========================================================================

    def to_cartesian(self) -> Tuple[float, float]:
        """
        Convert to Cartesian coordinates (real, imaginary).

        Returns:
            Tuple of (real, imaginary) components
        """
        real = self.magnitude * COS_TABLE[self.phase]
        imag = self.magnitude * SIN_TABLE[self.phase]
        return (real, imag)

    def to_cartesian_int(self) -> Tuple[int, int]:
        """
        Convert to integer Cartesian coordinates for deterministic verification.

        Uses the integer lookup tables from the whitepaper.

        Returns:
            Tuple of (real, imaginary) as integers
        """
        # Convert magnitude to fixed-point
        mag_int = int(self.magnitude * PRECISION)
        real = (mag_int * COS_TABLE_INT[self.phase]) // PRECISION
        imag = (mag_int * SIN_TABLE_INT[self.phase]) // PRECISION
        return (real, imag)

    @classmethod
    def from_cartesian(cls, real: float, imag: float) -> DiscreteAmplitude:
        """
        Create amplitude from Cartesian coordinates.

        The phase is quantized to the nearest 8-phase.

        Args:
            real: Real component
            imag: Imaginary component

        Returns:
            New DiscreteAmplitude instance
        """
        magnitude = np.sqrt(real**2 + imag**2)
        phase = quantize_phase_8(real, imag)
        return cls(magnitude, phase)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def add(self, other: DiscreteAmplitude) -> DiscreteAmplitude:
        """
        Add two amplitudes (with interference).

        This is the core operation for interference calculation.
        When phases differ, amplitudes can cancel or reinforce.

        Args:
            other: Another amplitude to add

        Returns:
            New amplitude representing the sum
        """
        r1, i1 = self.to_cartesian()
        r2, i2 = other.to_cartesian()
        return DiscreteAmplitude.from_cartesian(r1 + r2, i1 + i2)

    def __add__(self, other: DiscreteAmplitude) -> DiscreteAmplitude:
        """Operator overload for + (addition with interference)."""
        return self.add(other)

    def scale(self, factor: float) -> DiscreteAmplitude:
        """
        Scale the magnitude by a factor.

        Args:
            factor: Scaling factor (can be negative)

        Returns:
            New scaled amplitude
        """
        return DiscreteAmplitude(self.magnitude * abs(factor),
                                  self.phase if factor >= 0 else self.phase.opposite())

    def __mul__(self, factor: float) -> DiscreteAmplitude:
        """Operator overload for * (scaling)."""
        return self.scale(factor)

    def __rmul__(self, factor: float) -> DiscreteAmplitude:
        """Operator overload for * (scaling, reversed)."""
        return self.scale(factor)

    # =========================================================================
    # Probability and Measurement
    # =========================================================================

    def probability(self) -> float:
        """
        Calculate the probability |α|².

        Returns:
            Probability value (always non-negative)
        """
        return self.magnitude ** 2

    def normalized(self) -> DiscreteAmplitude:
        """
        Return a normalized version (magnitude = 1).

        Returns:
            New amplitude with magnitude 1.0
        """
        if self.magnitude == 0:
            return DiscreteAmplitude(0.0, DiscretePhase.P0)
        return DiscreteAmplitude(1.0, self.phase)

    # =========================================================================
    # Phase Operations
    # =========================================================================

    def rotate(self, steps: int) -> DiscreteAmplitude:
        """
        Rotate the phase by the given number of 45° steps.

        Args:
            steps: Number of 45° steps to rotate (can be negative)

        Returns:
            New amplitude with rotated phase
        """
        return DiscreteAmplitude(self.magnitude, self.phase.rotate(steps))

    def conjugate(self) -> DiscreteAmplitude:
        """
        Return the complex conjugate (negate imaginary part).

        Returns:
            New amplitude with conjugated phase
        """
        # Conjugate negates the phase angle
        conjugate_phase = DiscretePhase((8 - self.phase) % 8)
        return DiscreteAmplitude(self.magnitude, conjugate_phase)

    # =========================================================================
    # Comparison and Display
    # =========================================================================

    def is_zero(self, threshold: float = 1e-10) -> bool:
        """Check if amplitude is effectively zero."""
        return self.magnitude < threshold

    def __repr__(self) -> str:
        return f"DiscreteAmplitude(mag={self.magnitude:.4f}, phase={self.phase.name})"

    def __str__(self) -> str:
        r, i = self.to_cartesian()
        if i >= 0:
            return f"{r:.4f} + {i:.4f}i"
        else:
            return f"{r:.4f} - {-i:.4f}i"


# =============================================================================
# Factory Functions
# =============================================================================

def zero_amplitude() -> DiscreteAmplitude:
    """Create a zero amplitude."""
    return DiscreteAmplitude(0.0, DiscretePhase.P0)


def unit_amplitude(phase: DiscretePhase = DiscretePhase.P0) -> DiscreteAmplitude:
    """Create a unit amplitude with the given phase."""
    return DiscreteAmplitude(1.0, phase)


def from_probability(prob: float, phase: DiscretePhase = DiscretePhase.P0) -> DiscreteAmplitude:
    """
    Create an amplitude from a probability value.

    Args:
        prob: Probability value (0 to 1)
        phase: Phase to assign

    Returns:
        Amplitude with magnitude = sqrt(prob)
    """
    return DiscreteAmplitude(np.sqrt(prob), phase)


# =============================================================================
# Interference Analysis
# =============================================================================

def calculate_interference_result(
    amp1: DiscreteAmplitude,
    amp2: DiscreteAmplitude
) -> dict:
    """
    Analyze the interference between two amplitudes.

    Returns detailed information about the interference effect.

    Args:
        amp1: First amplitude
        amp2: Second amplitude

    Returns:
        Dictionary with interference analysis
    """
    # Calculate phase difference
    phase_diff = (amp2.phase - amp1.phase) % 8

    # Get interference type
    interference_types = {
        0: "constructive_full",
        1: "constructive_partial",
        2: "orthogonal",
        3: "destructive_partial",
        4: "destructive_full",
        5: "destructive_partial",
        6: "orthogonal",
        7: "constructive_partial",
    }

    # Calculate result
    result_amp = amp1.add(amp2)

    # Calculate probabilities
    prob_before = amp1.probability() + amp2.probability()
    prob_after = result_amp.probability()

    return {
        "phase_difference": phase_diff,
        "phase_difference_degrees": phase_diff * 45,
        "interference_type": interference_types[phase_diff],
        "interference_factor": COS_TABLE[phase_diff],
        "amplitude_before_1": amp1,
        "amplitude_before_2": amp2,
        "amplitude_after": result_amp,
        "probability_sum_before": prob_before,
        "probability_after": prob_after,
        "probability_change": prob_after - prob_before,
        "probability_change_percent": (prob_after - prob_before) / prob_before * 100 if prob_before > 0 else 0,
    }
