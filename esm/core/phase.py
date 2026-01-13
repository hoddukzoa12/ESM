"""
8-Phase Discrete Phase System

Based on ESM Whitepaper v5.2 Section 2.2-2.3

The 8-phase system uses 45° increments to represent phase angles,
enabling deterministic implementation with sufficient interference expressiveness.

v5.2 additions:
- Semantic aliases (InterferenceMode) for developer-friendly SDK
"""

from enum import IntEnum, Enum
import numpy as np
from typing import Tuple, Union


class DiscretePhase(IntEnum):
    """
    8-phase discrete phase representation.

    Each phase represents a 45° increment:
    - P0:   0° (positive real axis)
    - P45:  45°
    - P90:  90° (positive imaginary axis)
    - P135: 135°
    - P180: 180° (negative real axis)
    - P225: 225°
    - P270: 270° (negative imaginary axis)
    - P315: 315°
    """
    P0 = 0    # 0°
    P45 = 1   # 45°
    P90 = 2   # 90°
    P135 = 3  # 135°
    P180 = 4  # 180°
    P225 = 5  # 225°
    P270 = 6  # 270°
    P315 = 7  # 315°

    def to_radians(self) -> float:
        """Convert phase to radians."""
        return self.value * np.pi / 4

    def to_degrees(self) -> float:
        """Convert phase to degrees."""
        return self.value * 45.0

    def opposite(self) -> 'DiscretePhase':
        """Get the opposite phase (180° rotation)."""
        return DiscretePhase((self.value + 4) % 8)

    def rotate(self, steps: int) -> 'DiscretePhase':
        """Rotate phase by given number of 45° steps."""
        return DiscretePhase((self.value + steps) % 8)

    @classmethod
    def from_angle(cls, radians: float) -> 'DiscretePhase':
        """Quantize an angle (in radians) to the nearest 8-phase."""
        # Normalize to [0, 2π)
        normalized = radians % (2 * np.pi)
        # Convert to phase index (0-7)
        index = int(round(normalized / (np.pi / 4))) % 8
        return cls(index)


# =============================================================================
# Lookup Tables (Whitepaper Section 2.3)
# =============================================================================

# Integer precision for deterministic blockchain implementation
PRECISION: int = 1 << 64
SQRT2_HALF_INT: int = 13043817825332782212  # floor(0.7071067811865476 × 2^64)

# Integer lookup tables (for deterministic verification)
COS_TABLE_INT: Tuple[int, ...] = (
    PRECISION,       # cos(0°) = 1
    SQRT2_HALF_INT,  # cos(45°) ≈ 0.707
    0,               # cos(90°) = 0
    -SQRT2_HALF_INT, # cos(135°) ≈ -0.707
    -PRECISION,      # cos(180°) = -1
    -SQRT2_HALF_INT, # cos(225°) ≈ -0.707
    0,               # cos(270°) = 0
    SQRT2_HALF_INT,  # cos(315°) ≈ 0.707
)

SIN_TABLE_INT: Tuple[int, ...] = (
    0,               # sin(0°) = 0
    SQRT2_HALF_INT,  # sin(45°) ≈ 0.707
    PRECISION,       # sin(90°) = 1
    SQRT2_HALF_INT,  # sin(135°) ≈ 0.707
    0,               # sin(180°) = 0
    -SQRT2_HALF_INT, # sin(225°) ≈ -0.707
    -PRECISION,      # sin(270°) = -1
    -SQRT2_HALF_INT, # sin(315°) ≈ -0.707
)

# Float lookup tables (for simulation)
COS_TABLE: np.ndarray = np.cos(np.arange(8) * np.pi / 4)
SIN_TABLE: np.ndarray = np.sin(np.arange(8) * np.pi / 4)

# Precomputed values for convenience
SQRT2_HALF: float = np.sqrt(2) / 2  # ≈ 0.7071067811865476


def quantize_phase_8(real: float, imag: float) -> DiscretePhase:
    """
    Quantize a Cartesian coordinate to the nearest 8-phase.

    This implements the atan2-based quantization described in
    Whitepaper Section 6.6.

    Args:
        real: Real component of the amplitude
        imag: Imaginary component of the amplitude

    Returns:
        The nearest DiscretePhase
    """
    if real == 0 and imag == 0:
        return DiscretePhase.P0

    # Use atan2 for angle calculation
    angle = np.arctan2(imag, real)

    # Normalize to [0, 2π)
    if angle < 0:
        angle += 2 * np.pi

    # Quantize to nearest 45° (π/4)
    index = int(round(angle / (np.pi / 4))) % 8

    return DiscretePhase(index)


# =============================================================================
# Interference Type Classification
# =============================================================================

def get_interference_type(phase_diff: int) -> str:
    """
    Classify the interference type based on phase difference.

    Based on Whitepaper Section 2.4 Table.

    Args:
        phase_diff: Phase difference in 45° steps (0-7)

    Returns:
        String describing the interference type
    """
    phase_diff = phase_diff % 8

    interference_types = {
        0: "constructive_full",      # 0°: Complete constructive
        1: "constructive_partial",   # 45°: Partial constructive
        2: "orthogonal",             # 90°: Orthogonal (no interference)
        3: "destructive_partial",    # 135°: Partial destructive
        4: "destructive_full",       # 180°: Complete destructive
        5: "destructive_partial",    # 225°: Partial destructive
        6: "orthogonal",             # 270°: Orthogonal
        7: "constructive_partial",   # 315°: Partial constructive
    }

    return interference_types[phase_diff]


def get_interference_factor(phase_diff: int) -> float:
    """
    Get the interference factor based on phase difference.

    Returns cos(Δθ) which determines how amplitudes combine.

    Args:
        phase_diff: Phase difference in 45° steps (0-7)

    Returns:
        Interference factor from -1 (destructive) to +1 (constructive)
    """
    return COS_TABLE[phase_diff % 8]


# =============================================================================
# Semantic Aliases (v5.2 Section 2.3)
# =============================================================================

class InterferenceMode(Enum):
    """
    SDK-level semantic aliases for phases.

    Based on ESM Whitepaper v5.2 Section 2.3.
    Provides intuitive names for common interference patterns.

    Usage:
        branch = create_branch(data, mode="Normal")
        branch = create_branch(data, mode=InterferenceMode.Counter)
    """
    # Basic aliases (for general developers)
    Normal = DiscretePhase.P0          # Constructive interference
    Counter = DiscretePhase.P180       # Destructive interference (MEV defense)
    Independent = DiscretePhase.P90    # Orthogonal (no interference)

    # Detailed aliases (for advanced usage)
    Additive = DiscretePhase.P0        # Same as Normal
    Opposite = DiscretePhase.P180      # Same as Counter
    PartialAdd = DiscretePhase.P45     # Partial constructive
    PartialCounter = DiscretePhase.P135  # Partial destructive

    @classmethod
    def from_string(cls, name: str) -> 'InterferenceMode':
        """Get mode from string name (case-insensitive)."""
        name_lower = name.lower()
        for mode in cls:
            if mode.name.lower() == name_lower:
                return mode
        raise ValueError(f"Unknown interference mode: {name}")

    def to_phase(self) -> DiscretePhase:
        """Convert to protocol-level DiscretePhase."""
        return self.value


def phase_from_mode(mode: Union[str, InterferenceMode, DiscretePhase]) -> DiscretePhase:
    """
    Convert SDK mode to protocol-level phase.

    Args:
        mode: Can be a string ("Normal", "Counter"), InterferenceMode, or DiscretePhase

    Returns:
        DiscretePhase for protocol operations

    Examples:
        phase_from_mode("Normal")  # -> DiscretePhase.P0
        phase_from_mode(InterferenceMode.Counter)  # -> DiscretePhase.P180
        phase_from_mode(DiscretePhase.P45)  # -> DiscretePhase.P45
    """
    if isinstance(mode, DiscretePhase):
        return mode
    if isinstance(mode, InterferenceMode):
        return mode.value
    if isinstance(mode, str):
        return InterferenceMode.from_string(mode).value
    raise TypeError(f"Cannot convert {type(mode)} to DiscretePhase")


# Alias mapping table for documentation
SEMANTIC_ALIAS_TABLE = {
    "Normal": ("P0", "Constructive interference", "General transactions"),
    "Counter": ("P180", "Destructive interference", "MEV defense"),
    "Independent": ("P90", "Orthogonal (no interference)", "Independent states"),
    "Additive": ("P0", "Same as Normal", "Advanced usage"),
    "Opposite": ("P180", "Same as Counter", "Advanced usage"),
    "PartialAdd": ("P45", "Partial constructive", "Advanced usage"),
    "PartialCounter": ("P135", "Partial destructive", "Advanced usage"),
}
