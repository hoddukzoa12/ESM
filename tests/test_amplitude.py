"""
Tests for DiscreteAmplitude operations.

Based on ESM Whitepaper v5.1 examples.
"""

import pytest
import numpy as np

from esm.core.phase import DiscretePhase, quantize_phase_8
from esm.core.amplitude import (
    DiscreteAmplitude,
    zero_amplitude,
    unit_amplitude,
    calculate_interference_result,
)


class TestDiscretePhase:
    """Tests for DiscretePhase enum."""

    def test_phase_values(self):
        """Test phase enum values."""
        assert DiscretePhase.P0.value == 0
        assert DiscretePhase.P45.value == 1
        assert DiscretePhase.P90.value == 2
        assert DiscretePhase.P180.value == 4

    def test_to_radians(self):
        """Test conversion to radians."""
        assert DiscretePhase.P0.to_radians() == 0
        assert np.isclose(DiscretePhase.P90.to_radians(), np.pi / 2)
        assert np.isclose(DiscretePhase.P180.to_radians(), np.pi)

    def test_opposite(self):
        """Test opposite phase calculation."""
        assert DiscretePhase.P0.opposite() == DiscretePhase.P180
        assert DiscretePhase.P45.opposite() == DiscretePhase.P225
        assert DiscretePhase.P90.opposite() == DiscretePhase.P270

    def test_rotate(self):
        """Test phase rotation."""
        assert DiscretePhase.P0.rotate(2) == DiscretePhase.P90
        assert DiscretePhase.P315.rotate(1) == DiscretePhase.P0
        assert DiscretePhase.P0.rotate(-1) == DiscretePhase.P315


class TestQuantizePhase:
    """Tests for phase quantization."""

    def test_quantize_cardinal_directions(self):
        """Test quantization of cardinal directions."""
        assert quantize_phase_8(1, 0) == DiscretePhase.P0
        assert quantize_phase_8(0, 1) == DiscretePhase.P90
        assert quantize_phase_8(-1, 0) == DiscretePhase.P180
        assert quantize_phase_8(0, -1) == DiscretePhase.P270

    def test_quantize_diagonals(self):
        """Test quantization of diagonal directions."""
        sqrt2 = np.sqrt(2) / 2
        assert quantize_phase_8(sqrt2, sqrt2) == DiscretePhase.P45
        assert quantize_phase_8(-sqrt2, sqrt2) == DiscretePhase.P135
        assert quantize_phase_8(-sqrt2, -sqrt2) == DiscretePhase.P225
        assert quantize_phase_8(sqrt2, -sqrt2) == DiscretePhase.P315

    def test_quantize_zero(self):
        """Test quantization of zero vector."""
        assert quantize_phase_8(0, 0) == DiscretePhase.P0


class TestDiscreteAmplitude:
    """Tests for DiscreteAmplitude class."""

    def test_creation(self):
        """Test basic amplitude creation."""
        amp = DiscreteAmplitude(0.7, DiscretePhase.P0)
        assert amp.magnitude == 0.7
        assert amp.phase == DiscretePhase.P0

    def test_negative_magnitude_flip(self):
        """Test that negative magnitude flips phase."""
        amp = DiscreteAmplitude(-0.5, DiscretePhase.P0)
        assert amp.magnitude == 0.5
        assert amp.phase == DiscretePhase.P180

    def test_to_cartesian(self):
        """Test Cartesian conversion."""
        # P0: (magnitude, 0)
        amp = DiscreteAmplitude(1.0, DiscretePhase.P0)
        r, i = amp.to_cartesian()
        assert np.isclose(r, 1.0)
        assert np.isclose(i, 0.0)

        # P90: (0, magnitude)
        amp = DiscreteAmplitude(1.0, DiscretePhase.P90)
        r, i = amp.to_cartesian()
        assert np.isclose(r, 0.0)
        assert np.isclose(i, 1.0)

        # P45: (magnitude/√2, magnitude/√2)
        amp = DiscreteAmplitude(1.0, DiscretePhase.P45)
        r, i = amp.to_cartesian()
        sqrt2_half = np.sqrt(2) / 2
        assert np.isclose(r, sqrt2_half)
        assert np.isclose(i, sqrt2_half)

    def test_from_cartesian(self):
        """Test creation from Cartesian coordinates."""
        # From (1, 0) -> P0
        amp = DiscreteAmplitude.from_cartesian(1.0, 0.0)
        assert np.isclose(amp.magnitude, 1.0)
        assert amp.phase == DiscretePhase.P0

        # From (0, 1) -> P90
        amp = DiscreteAmplitude.from_cartesian(0.0, 1.0)
        assert np.isclose(amp.magnitude, 1.0)
        assert amp.phase == DiscretePhase.P90

    def test_probability(self):
        """Test probability calculation."""
        amp = DiscreteAmplitude(0.7, DiscretePhase.P0)
        assert np.isclose(amp.probability(), 0.49)

        amp = DiscreteAmplitude(1.0, DiscretePhase.P45)
        assert np.isclose(amp.probability(), 1.0)

    def test_scale(self):
        """Test amplitude scaling."""
        amp = DiscreteAmplitude(0.5, DiscretePhase.P0)
        scaled = amp.scale(2.0)
        assert np.isclose(scaled.magnitude, 1.0)
        assert scaled.phase == DiscretePhase.P0

    def test_rotate(self):
        """Test phase rotation."""
        amp = DiscreteAmplitude(1.0, DiscretePhase.P0)
        rotated = amp.rotate(2)
        assert rotated.phase == DiscretePhase.P90
        assert rotated.magnitude == 1.0


class TestInterference:
    """Tests for interference calculations.

    Based on Whitepaper Section 2.4 examples.
    """

    def test_complete_cancellation(self):
        """Test complete destructive interference (180° phase difference).

        Whitepaper example:
        α_v = (0.7, P0), α_f = (0.7, P180) → |result| ≈ 0
        """
        amp1 = DiscreteAmplitude(0.7, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(0.7, DiscretePhase.P180)
        result = amp1.add(amp2)

        assert result.magnitude < 0.01, f"Expected near-zero, got {result.magnitude}"

    def test_partial_cancellation(self):
        """Test partial destructive interference (135° phase difference).

        Whitepaper example:
        α_v = (0.7, P0), α_f = (0.7, P135)
        Result probability ≈ 0.29
        """
        amp1 = DiscreteAmplitude(0.7, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(0.7, DiscretePhase.P135)
        result = amp1.add(amp2)

        # P135 has cos=-0.707, sin=0.707
        # amp1: (0.7, 0)
        # amp2: (0.7*-0.707, 0.7*0.707) = (-0.495, 0.495)
        # sum: (0.205, 0.495)
        # magnitude = sqrt(0.205² + 0.495²) ≈ 0.536
        # probability ≈ 0.29

        assert 0.25 < result.probability() < 0.35, \
            f"Expected ~0.29, got {result.probability()}"

    def test_constructive_interference(self):
        """Test constructive interference (0° phase difference)."""
        amp1 = DiscreteAmplitude(0.5, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(0.5, DiscretePhase.P0)
        result = amp1.add(amp2)

        assert np.isclose(result.magnitude, 1.0), \
            f"Expected 1.0, got {result.magnitude}"
        assert result.phase == DiscretePhase.P0

    def test_orthogonal_no_interference(self):
        """Test orthogonal phases (90° difference) - no wave interference."""
        amp1 = DiscreteAmplitude(1.0, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(1.0, DiscretePhase.P90)
        result = amp1.add(amp2)

        # |1 + i| = √2
        assert np.isclose(result.magnitude, np.sqrt(2)), \
            f"Expected √2 ≈ 1.414, got {result.magnitude}"

        # Probabilities: 1 + 1 = 2 before, 2 after (no net change)
        prob_before = amp1.probability() + amp2.probability()
        prob_after = result.probability()
        assert np.isclose(prob_before, prob_after), \
            "Orthogonal phases should preserve total probability"

    def test_interference_result_analysis(self):
        """Test detailed interference analysis."""
        amp1 = DiscreteAmplitude(0.7, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(0.7, DiscretePhase.P180)

        analysis = calculate_interference_result(amp1, amp2)

        assert analysis['phase_difference'] == 4  # 180°
        assert analysis['interference_type'] == 'destructive_full'
        assert analysis['probability_change'] < 0  # Destructive


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_amplitude(self):
        """Test zero amplitude operations."""
        zero = zero_amplitude()
        assert zero.magnitude == 0
        assert zero.probability() == 0

        # Adding to zero
        amp = DiscreteAmplitude(1.0, DiscretePhase.P45)
        result = zero.add(amp)
        assert np.isclose(result.magnitude, 1.0)

    def test_unit_amplitude(self):
        """Test unit amplitude creation."""
        unit = unit_amplitude(DiscretePhase.P90)
        assert unit.magnitude == 1.0
        assert unit.phase == DiscretePhase.P90

    def test_very_small_amplitudes(self):
        """Test very small amplitude handling."""
        amp1 = DiscreteAmplitude(1e-10, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(1e-10, DiscretePhase.P180)
        result = amp1.add(amp2)

        assert result.is_zero()
