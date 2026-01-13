"""
Tests for PSC (Probabilistic State Container).
"""

import pytest
import numpy as np

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_branch
from esm.core.psc import PSC, create_psc, create_simple_psc


class TestPSCBasics:
    """Basic PSC functionality tests."""

    def test_create_empty_psc(self):
        """Test creating an empty PSC."""
        psc = create_psc("test-psc")
        assert psc.id == "test-psc"
        assert len(psc.branches) == 0

    def test_add_branch(self):
        """Test adding branches to PSC."""
        psc = create_psc()
        branch = create_branch({"value": 100}, magnitude=0.7)
        psc.add_branch(branch)

        assert len(psc.branches) == 1
        assert psc.branches[0].amplitude.magnitude == 0.7

    def test_remove_branch(self):
        """Test removing branches from PSC."""
        psc = create_psc()
        branch1 = create_branch({"value": 1})
        branch2 = create_branch({"value": 2})

        psc.add_branch(branch1)
        psc.add_branch(branch2)
        assert len(psc.branches) == 2

        removed = psc.remove_branch(0)
        assert len(psc.branches) == 1
        assert removed.state_data["value"] == 1

    def test_unique_state_ids(self):
        """Test getting unique state IDs."""
        psc = create_psc()

        # Add branches with same state_id
        branch1 = create_branch({"value": "A"}, state_id="state-1")
        branch2 = create_branch({"value": "A"}, state_id="state-1")
        branch3 = create_branch({"value": "B"}, state_id="state-2")

        psc.add_branch(branch1)
        psc.add_branch(branch2)
        psc.add_branch(branch3)

        unique_ids = psc.get_unique_state_ids()
        assert len(unique_ids) == 2
        assert "state-1" in unique_ids
        assert "state-2" in unique_ids


class TestInterference:
    """Tests for PSC interference calculation."""

    def test_no_interference_different_states(self):
        """Test that different state_ids don't interfere."""
        psc = create_psc()

        branch1 = create_branch({"value": "A"}, state_id="state-1", magnitude=1.0)
        branch2 = create_branch({"value": "B"}, state_id="state-2", magnitude=1.0)

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        interference = psc.calculate_interference()

        # Each state should have its original amplitude
        assert np.isclose(interference["state-1"].magnitude, 1.0)
        assert np.isclose(interference["state-2"].magnitude, 1.0)

    def test_interference_same_state(self):
        """Test interference between branches with same state_id."""
        psc = create_psc()

        # Two branches with same state but opposite phases
        branch1 = Branch(
            state_id="shared-state",
            state_data={"value": "X"},
            amplitude=DiscreteAmplitude(0.7, DiscretePhase.P0)
        )
        branch2 = Branch(
            state_id="shared-state",
            state_data={"value": "X"},
            amplitude=DiscreteAmplitude(0.7, DiscretePhase.P180)
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        interference = psc.calculate_interference()

        # Should cancel out
        assert interference["shared-state"].magnitude < 0.01

    def test_constructive_interference(self):
        """Test constructive interference (same phase)."""
        psc = create_psc()

        branch1 = Branch(
            state_id="shared",
            state_data={"v": 1},
            amplitude=DiscreteAmplitude(0.5, DiscretePhase.P0)
        )
        branch2 = Branch(
            state_id="shared",
            state_data={"v": 1},
            amplitude=DiscreteAmplitude(0.5, DiscretePhase.P0)
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        interference = psc.calculate_interference()

        # Should add up
        assert np.isclose(interference["shared"].magnitude, 1.0)

    def test_interference_impact(self):
        """Test interference impact calculation."""
        psc = create_psc()

        # Destructive interference case
        branch1 = Branch(
            state_id="s1",
            state_data={},
            amplitude=DiscreteAmplitude(1.0, DiscretePhase.P0)
        )
        branch2 = Branch(
            state_id="s1",
            state_data={},
            amplitude=DiscreteAmplitude(1.0, DiscretePhase.P180)
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        # Before: 1² + 1² = 2
        # After: 0 (cancelled)
        impact = psc.interference_impact()
        assert impact < 0.01  # Near zero


class TestProbabilities:
    """Tests for probability calculations."""

    def test_single_branch_probability(self):
        """Test probability with single branch."""
        psc = create_psc()
        branch = create_branch({"x": 1}, magnitude=1.0)
        psc.add_branch(branch)

        probs = psc.get_probabilities()
        assert len(probs) == 1
        assert np.isclose(list(probs.values())[0], 1.0)

    def test_equal_probability_distribution(self):
        """Test equal probability when amplitudes are equal."""
        psc = create_simple_psc([
            ("state-A", 1.0, DiscretePhase.P0),
            ("state-B", 1.0, DiscretePhase.P0),
        ])

        probs = psc.get_probabilities()

        # Each should have 50% probability
        assert np.isclose(probs[list(probs.keys())[0]], 0.5)
        assert np.isclose(probs[list(probs.keys())[1]], 0.5)

    def test_probability_normalization(self):
        """Test that probabilities sum to 1."""
        psc = create_simple_psc([
            ("A", 0.5, DiscretePhase.P0),
            ("B", 0.7, DiscretePhase.P45),
            ("C", 0.3, DiscretePhase.P90),
        ])

        probs = psc.get_probabilities()
        total = sum(probs.values())

        assert np.isclose(total, 1.0)


class TestCollapse:
    """Tests for PSC collapse."""

    def test_collapse_reduces_to_one(self):
        """Test that collapse reduces to one branch."""
        psc = create_simple_psc([
            ("A", 1.0, DiscretePhase.P0),
            ("B", 1.0, DiscretePhase.P0),
        ])

        assert len(psc.branches) == 2

        state_id, branch = psc.collapse(seed=42)

        assert len(psc.branches) == 1
        assert psc.branches[0].state_id == state_id

    def test_collapse_deterministic_with_seed(self):
        """Test that collapse is deterministic with same seed."""
        def make_psc():
            return create_simple_psc([
                ("A", 1.0, DiscretePhase.P0),
                ("B", 1.0, DiscretePhase.P0),
                ("C", 1.0, DiscretePhase.P0),
            ])

        results = []
        for _ in range(5):
            psc = make_psc()
            state_id, _ = psc.collapse(seed=12345)
            results.append(state_id)

        # All results should be the same
        assert len(set(results)) == 1

    def test_collapse_probability_weighted(self):
        """Test that collapse follows probability distribution."""
        # Run many collapses and check distribution
        n_trials = 1000
        counts = {"high": 0, "low": 0}

        for i in range(n_trials):
            psc = create_psc()

            # High probability state
            high_branch = Branch(
                state_id="high",
                state_data={},
                amplitude=DiscreteAmplitude(0.9, DiscretePhase.P0)
            )
            # Low probability state
            low_branch = Branch(
                state_id="low",
                state_data={},
                amplitude=DiscreteAmplitude(0.1, DiscretePhase.P0)
            )

            psc.add_branch(high_branch)
            psc.add_branch(low_branch)

            state_id, _ = psc.collapse(seed=i)
            counts[state_id] += 1

        # High should be selected much more often
        high_ratio = counts["high"] / n_trials
        assert high_ratio > 0.7, f"Expected >70% high, got {high_ratio*100:.1f}%"


class TestReport:
    """Tests for PSC reporting."""

    def test_interference_report(self):
        """Test interference report generation."""
        psc = create_psc()

        branch1 = Branch(
            state_id="s1",
            state_data={},
            amplitude=DiscreteAmplitude(0.7, DiscretePhase.P0),
            tx_type="victim"
        )
        branch2 = Branch(
            state_id="s1",
            state_data={},
            amplitude=DiscreteAmplitude(0.7, DiscretePhase.P180),
            tx_type="attacker"
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        report = psc.get_interference_report()

        assert report["total_branches"] == 2
        assert report["unique_states"] == 1

        state_key = list(report["states"].keys())[0]
        state_report = report["states"][state_key]

        assert state_report["branch_count"] == 2
        assert state_report["interference_type"] == "destructive"

    def test_summary(self):
        """Test PSC summary generation."""
        psc = create_simple_psc([
            ("A", 0.8, DiscretePhase.P0),
            ("B", 0.2, DiscretePhase.P0),
        ])

        summary = psc.summary()

        assert "PSC" in summary
        assert "2 branches" in summary
