"""
Tests for MEV simulation.
"""

import pytest
import numpy as np

from esm.core.phase import DiscretePhase
from esm.simulation.mev_scenario import (
    assign_phase_by_delay,
    calculate_interference_factor,
    simulate_sandwich_attack,
    analyze_timing_sensitivity,
    MEV_WINDOW_FULL_CANCEL,
    MEV_WINDOW_PARTIAL_CANCEL,
    MEV_WINDOW_ORTHOGONAL,
)


class TestPhaseAssignment:
    """Tests for MEV defense phase assignment."""

    def test_immediate_attack_full_cancel(self):
        """Test that immediate attacks get full cancellation phase."""
        # < 100ms should get P180
        phase = assign_phase_by_delay(50)
        assert phase == DiscretePhase.P180

        phase = assign_phase_by_delay(99)
        assert phase == DiscretePhase.P180

    def test_fast_attack_partial_cancel(self):
        """Test that fast attacks get partial cancellation phase."""
        # 100-500ms should get P135
        phase = assign_phase_by_delay(200)
        assert phase == DiscretePhase.P135

        phase = assign_phase_by_delay(499)
        assert phase == DiscretePhase.P135

    def test_medium_attack_orthogonal(self):
        """Test that medium-timed attacks get orthogonal phase."""
        # 500-1000ms should get P90
        phase = assign_phase_by_delay(750)
        assert phase == DiscretePhase.P90

        phase = assign_phase_by_delay(999)
        assert phase == DiscretePhase.P90

    def test_slow_attack_normal(self):
        """Test that slow attacks get normal phase (no defense)."""
        # >1000ms should get P0
        phase = assign_phase_by_delay(1001)
        assert phase == DiscretePhase.P0

        phase = assign_phase_by_delay(5000)
        assert phase == DiscretePhase.P0

    def test_zero_delay(self):
        """Test zero delay edge case."""
        phase = assign_phase_by_delay(0)
        assert phase == DiscretePhase.P180


class TestInterferenceFactor:
    """Tests for interference factor calculation."""

    def test_full_cancel_factor(self):
        """Test P180 gives near-zero factor."""
        factor = calculate_interference_factor(DiscretePhase.P180)
        assert factor < 0.01

    def test_partial_cancel_factor(self):
        """Test P135 gives reduced factor."""
        factor = calculate_interference_factor(DiscretePhase.P135)
        # (1 + cos(135°)) / 2 = (1 - 0.707) / 2 ≈ 0.146
        assert 0.1 < factor < 0.2

    def test_orthogonal_factor(self):
        """Test P90 gives 0.5 factor."""
        factor = calculate_interference_factor(DiscretePhase.P90)
        # (1 + cos(90°)) / 2 = 0.5
        assert np.isclose(factor, 0.5)

    def test_no_defense_factor(self):
        """Test P0 gives full factor (1.0)."""
        factor = calculate_interference_factor(DiscretePhase.P0)
        assert np.isclose(factor, 1.0)


class TestSimulation:
    """Tests for MEV simulation."""

    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        result = simulate_sandwich_attack(n_rounds=100, seed=42)

        assert result.n_rounds == 100
        assert len(result.traditional_profits) == 100
        assert len(result.esm_profits) == 100

    def test_simulation_deterministic(self):
        """Test that simulation is deterministic with seed."""
        result1 = simulate_sandwich_attack(n_rounds=100, seed=42)
        result2 = simulate_sandwich_attack(n_rounds=100, seed=42)

        assert result1.traditional_profits == result2.traditional_profits
        assert result1.esm_profits == result2.esm_profits

    def test_esm_reduces_profit(self):
        """Test that ESM reduces attacker profit."""
        result = simulate_sandwich_attack(n_rounds=1000, seed=42)

        assert result.esm_total < result.traditional_total
        assert result.profit_reduction > 0

    def test_profit_reduction_significant(self):
        """Test that profit reduction is significant (>50%)."""
        # With default parameters (fast attacks), should see major reduction
        result = simulate_sandwich_attack(
            n_rounds=1000,
            attack_delay_mean=50,  # Fast attacks
            attack_delay_std=30,
            seed=42
        )

        assert result.profit_reduction > 50, \
            f"Expected >50% reduction, got {result.profit_reduction:.1f}%"

    def test_slow_attacks_less_reduction(self):
        """Test that slow attacks see less reduction."""
        # Slow attacks should bypass defense
        result = simulate_sandwich_attack(
            n_rounds=1000,
            attack_delay_mean=1500,  # Slow attacks
            attack_delay_std=200,
            seed=42
        )

        # Most attacks should have P0 phase, less reduction
        assert result.profit_reduction < 30, \
            f"Expected <30% reduction for slow attacks, got {result.profit_reduction:.1f}%"


class TestStatistics:
    """Tests for simulation statistics."""

    def test_stats_structure(self):
        """Test that stats contain expected fields."""
        result = simulate_sandwich_attack(n_rounds=100, seed=42)
        stats = result.get_stats()

        assert "n_rounds" in stats
        assert "traditional_total" in stats
        assert "esm_total" in stats
        assert "profit_reduction_percent" in stats
        assert "average_interference_factor" in stats
        assert "phase_distribution" in stats

    def test_phase_distribution(self):
        """Test phase distribution is recorded."""
        result = simulate_sandwich_attack(n_rounds=1000, seed=42)
        stats = result.get_stats()

        dist = stats["phase_distribution"]

        # Should have some phases recorded
        assert len(dist) > 0

        # Total should equal n_rounds
        total = sum(dist.values())
        assert total == 1000


class TestTimingSensitivity:
    """Tests for timing sensitivity analysis."""

    def test_analysis_runs(self):
        """Test that timing analysis completes."""
        analysis = analyze_timing_sensitivity(
            delay_range=(0, 2000),
            n_points=50
        )

        assert len(analysis["delays_ms"]) == 50
        assert len(analysis["phases"]) == 50
        assert len(analysis["interference_factors"]) == 50

    def test_factor_decreases_with_delay(self):
        """Test that interference factor generally increases with delay."""
        analysis = analyze_timing_sensitivity(
            delay_range=(0, 2000),
            n_points=100
        )

        factors = analysis["interference_factors"]

        # Factor at 0ms should be low (full cancel)
        assert factors[0] < 0.1

        # Factor at 2000ms should be high (no defense)
        assert factors[-1] > 0.9

    def test_window_boundaries(self):
        """Test specific window boundaries."""
        analysis = analyze_timing_sensitivity(
            delay_range=(0, 2000),
            n_points=200
        )

        delays = analysis["delays_ms"]
        phases = analysis["phases"]

        # Find indices near boundaries
        for i, delay in enumerate(delays):
            if delay < MEV_WINDOW_FULL_CANCEL:
                assert phases[i] == "P180"
            elif delay < MEV_WINDOW_PARTIAL_CANCEL:
                assert phases[i] == "P135"
            elif delay < MEV_WINDOW_ORTHOGONAL:
                assert phases[i] == "P90"
            else:
                assert phases[i] == "P0"


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_rounds(self):
        """Test simulation with zero rounds."""
        result = simulate_sandwich_attack(n_rounds=0, seed=42)
        assert result.n_rounds == 0
        assert result.traditional_total == 0
        assert result.esm_total == 0

    def test_single_round(self):
        """Test simulation with single round."""
        result = simulate_sandwich_attack(n_rounds=1, seed=42)
        assert result.n_rounds == 1
        assert len(result.traditional_profits) == 1

    def test_zero_extraction_rate(self):
        """Test with zero MEV extraction rate."""
        result = simulate_sandwich_attack(
            n_rounds=100,
            mev_extraction_rate=0.0,
            seed=42
        )

        assert result.traditional_total == 0
        assert result.esm_total == 0
