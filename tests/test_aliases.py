"""
Tests for v5.2 Semantic Aliases and new fields.

Based on ESM Whitepaper v5.2 Section 2.3.
"""

import pytest
import numpy as np

from esm.core.phase import (
    DiscretePhase,
    InterferenceMode,
    phase_from_mode,
    SEMANTIC_ALIAS_TABLE,
)
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import (
    Branch,
    create_branch,
    create_branch_with_mode,
    compute_state_id,
)
from esm.core.psc import PSC, create_psc


class TestInterferenceMode:
    """Tests for InterferenceMode enum (v5.2)."""

    def test_basic_aliases(self):
        """Test basic semantic aliases map to correct phases."""
        assert InterferenceMode.Normal.value == DiscretePhase.P0
        assert InterferenceMode.Counter.value == DiscretePhase.P180
        assert InterferenceMode.Independent.value == DiscretePhase.P90

    def test_detailed_aliases(self):
        """Test detailed semantic aliases."""
        assert InterferenceMode.Additive.value == DiscretePhase.P0
        assert InterferenceMode.Opposite.value == DiscretePhase.P180
        assert InterferenceMode.PartialAdd.value == DiscretePhase.P45
        assert InterferenceMode.PartialCounter.value == DiscretePhase.P135

    def test_from_string_case_insensitive(self):
        """Test from_string is case-insensitive."""
        assert InterferenceMode.from_string("normal") == InterferenceMode.Normal
        assert InterferenceMode.from_string("NORMAL") == InterferenceMode.Normal
        assert InterferenceMode.from_string("Normal") == InterferenceMode.Normal
        assert InterferenceMode.from_string("counter") == InterferenceMode.Counter

    def test_from_string_invalid(self):
        """Test from_string raises on invalid input."""
        with pytest.raises(ValueError, match="Unknown interference mode"):
            InterferenceMode.from_string("invalid_mode")

    def test_to_phase(self):
        """Test to_phase returns correct DiscretePhase."""
        assert InterferenceMode.Normal.to_phase() == DiscretePhase.P0
        assert InterferenceMode.Counter.to_phase() == DiscretePhase.P180
        assert InterferenceMode.Independent.to_phase() == DiscretePhase.P90


class TestPhaseFromMode:
    """Tests for phase_from_mode conversion function."""

    def test_from_string(self):
        """Test conversion from string."""
        assert phase_from_mode("Normal") == DiscretePhase.P0
        assert phase_from_mode("Counter") == DiscretePhase.P180
        assert phase_from_mode("Independent") == DiscretePhase.P90
        assert phase_from_mode("PartialAdd") == DiscretePhase.P45

    def test_from_interference_mode(self):
        """Test conversion from InterferenceMode enum."""
        assert phase_from_mode(InterferenceMode.Normal) == DiscretePhase.P0
        assert phase_from_mode(InterferenceMode.Counter) == DiscretePhase.P180

    def test_from_discrete_phase(self):
        """Test pass-through of DiscretePhase."""
        assert phase_from_mode(DiscretePhase.P45) == DiscretePhase.P45
        assert phase_from_mode(DiscretePhase.P270) == DiscretePhase.P270

    def test_invalid_type(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            phase_from_mode(123)

        with pytest.raises(TypeError):
            phase_from_mode([1, 2, 3])


class TestSemanticAliasTable:
    """Tests for the semantic alias documentation table."""

    def test_table_completeness(self):
        """Test all modes are documented."""
        expected_modes = ["Normal", "Counter", "Independent",
                         "Additive", "Opposite", "PartialAdd", "PartialCounter"]
        for mode in expected_modes:
            assert mode in SEMANTIC_ALIAS_TABLE

    def test_table_structure(self):
        """Test table has correct structure."""
        for mode, info in SEMANTIC_ALIAS_TABLE.items():
            assert len(info) == 3  # (phase, description, use_case)
            assert info[0].startswith("P")  # Phase name


class TestCreateBranchWithMode:
    """Tests for create_branch_with_mode factory function."""

    def test_basic_creation_with_string(self):
        """Test creating branch with string mode."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode="Normal"
        )
        assert branch.amplitude.phase == DiscretePhase.P0
        assert branch.amplitude.magnitude == 1.0

    def test_counter_mode(self):
        """Test creating branch with Counter mode for MEV defense."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode="Counter"
        )
        assert branch.amplitude.phase == DiscretePhase.P180

    def test_independent_mode(self):
        """Test creating branch with Independent mode."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode="Independent"
        )
        assert branch.amplitude.phase == DiscretePhase.P90

    def test_with_interference_mode_enum(self):
        """Test creating branch with InterferenceMode enum."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode=InterferenceMode.PartialCounter
        )
        assert branch.amplitude.phase == DiscretePhase.P135

    def test_with_custom_magnitude(self):
        """Test creating branch with custom magnitude."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode="Normal",
            magnitude=0.5
        )
        assert branch.amplitude.magnitude == 0.5

    def test_v52_fields(self):
        """Test v5.2 fields are set correctly."""
        branch = create_branch_with_mode(
            state_data={"amount": 100},
            mode="Normal",
            interference_deposit=1000,
            stake_locked=5000
        )
        assert branch.interference_deposit == 1000
        assert branch.stake_locked == 5000


class TestBranchV52Fields:
    """Tests for v5.2 Branch fields."""

    def test_default_values(self):
        """Test v5.2 fields have correct defaults."""
        branch = create_branch(state_data={"test": 1})
        assert branch.interference_deposit == 0
        assert branch.stake_locked == 0

    def test_with_amplitude_preserves_v52_fields(self):
        """Test with_amplitude preserves v5.2 fields."""
        branch = create_branch(
            state_data={"test": 1},
            interference_deposit=1000,
            stake_locked=5000
        )
        new_branch = branch.with_amplitude(DiscreteAmplitude(0.5, DiscretePhase.P90))

        assert new_branch.interference_deposit == 1000
        assert new_branch.stake_locked == 5000

    def test_with_phase_preserves_v52_fields(self):
        """Test with_phase preserves v5.2 fields."""
        branch = create_branch(
            state_data={"test": 1},
            interference_deposit=1000,
            stake_locked=5000
        )
        new_branch = branch.with_phase(DiscretePhase.P180)

        assert new_branch.interference_deposit == 1000
        assert new_branch.stake_locked == 5000


class TestPSCV52Fields:
    """Tests for v5.2 PSC fields."""

    def test_default_values(self):
        """Test v5.2 fields have correct defaults."""
        psc = create_psc("test-psc")
        assert psc.collapse_deadline == 0
        assert psc.amplitude_fee_pool == 0
        assert psc.total_interference_deposit == 0

    def test_add_branch_updates_deposit(self):
        """Test add_branch updates total_interference_deposit."""
        psc = create_psc("test-psc")

        branch1 = create_branch(
            state_data={"id": 1},
            interference_deposit=100
        )
        branch2 = create_branch(
            state_data={"id": 2},
            interference_deposit=200
        )

        psc.add_branch(branch1)
        assert psc.total_interference_deposit == 100

        psc.add_branch(branch2)
        assert psc.total_interference_deposit == 300

    def test_remove_branch_updates_deposit(self):
        """Test remove_branch updates total_interference_deposit."""
        psc = create_psc("test-psc")

        branch1 = create_branch(
            state_data={"id": 1},
            interference_deposit=100
        )
        branch2 = create_branch(
            state_data={"id": 2},
            interference_deposit=200
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)
        assert psc.total_interference_deposit == 300

        psc.remove_branch(0)
        assert psc.total_interference_deposit == 200


class TestInterferenceWithModes:
    """Tests for interference using semantic modes."""

    def test_normal_counter_cancellation(self):
        """Test Normal + Counter causes destructive interference."""
        psc = create_psc("mev-test")

        # Victim transaction (Normal)
        victim = create_branch_with_mode(
            state_data={"type": "swap", "amount": 100},
            mode="Normal",
            magnitude=0.7
        )

        # Attacker transaction (Counter)
        attacker = create_branch_with_mode(
            state_data={"type": "swap", "amount": 100},
            mode="Counter",
            magnitude=0.7
        )
        # Same state_id to cause interference
        attacker = Branch(
            state_id=victim.state_id,
            state_data=attacker.state_data,
            amplitude=attacker.amplitude,
            tx_type="attacker"
        )

        psc.add_branch(victim)
        psc.add_branch(attacker)

        interference = psc.calculate_interference()
        result = interference[victim.state_id]

        # Should nearly cancel out
        assert result.magnitude < 0.01

    def test_independent_no_interference(self):
        """Test Independent mode causes orthogonal interference."""
        psc = create_psc("test")

        branch1 = create_branch_with_mode(
            state_data={"id": 1},
            mode="Normal",
            magnitude=1.0
        )
        branch2 = create_branch_with_mode(
            state_data={"id": 1},
            mode="Independent",
            magnitude=1.0
        )
        # Same state_id
        branch2 = Branch(
            state_id=branch1.state_id,
            state_data=branch2.state_data,
            amplitude=branch2.amplitude
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        interference = psc.calculate_interference()
        result = interference[branch1.state_id]

        # Should be sqrt(2) for orthogonal
        assert np.isclose(result.magnitude, np.sqrt(2), rtol=0.01)

    def test_partial_counter_partial_cancellation(self):
        """Test PartialCounter causes partial destructive interference."""
        psc = create_psc("test")

        branch1 = create_branch_with_mode(
            state_data={"id": 1},
            mode="Normal",
            magnitude=0.7
        )
        branch2 = create_branch_with_mode(
            state_data={"id": 1},
            mode="PartialCounter",
            magnitude=0.7
        )
        # Same state_id
        branch2 = Branch(
            state_id=branch1.state_id,
            state_data=branch2.state_data,
            amplitude=branch2.amplitude
        )

        psc.add_branch(branch1)
        psc.add_branch(branch2)

        interference = psc.calculate_interference()
        result = interference[branch1.state_id]

        # Should be reduced but not zero
        assert 0.2 < result.magnitude < 0.8
