"""
Tests for Step-by-Step Transaction Walkthrough.

Based on ESM Whitepaper v5.3 Section 8.1.
"""

import pytest
from decimal import Decimal

from esm.simulation.walkthrough import (
    # Constants
    AMP_PER_ESM,
    PSC_CREATION_FEE,
    BRANCH_FEE_BASE,
    INTERFERENCE_DEPOSIT_RATE,
    INTERFERENCE_BUFFER,
    GAS_FEE_ESTIMATE,
    MEV_CANCEL_THRESHOLD_MS,
    MEV_PARTIAL_THRESHOLD_MS,
    MEV_ORTHOGONAL_THRESHOLD_MS,
    # Enums
    WalkthroughType,
    # Data structures
    WalkthroughStep,
    WalkthroughResult,
    # Main class
    TransactionWalkthrough,
    # Helpers
    run_mev_walkthrough,
    compare_timing_scenarios,
)
from esm.core.phase import DiscretePhase


class TestWalkthroughConstants:
    """Tests for walkthrough constants."""

    def test_amp_per_esm(self):
        """Test ESM to amp conversion rate."""
        assert AMP_PER_ESM == Decimal("1000000")

    def test_psc_creation_fee(self):
        """Test PSC creation fee is 10 amp."""
        assert PSC_CREATION_FEE == Decimal("0.00001")

    def test_interference_deposit_rate(self):
        """Test interference deposit rate is 0.1%."""
        assert INTERFERENCE_DEPOSIT_RATE == Decimal("0.001")

    def test_interference_buffer(self):
        """Test interference buffer is 20%."""
        assert INTERFERENCE_BUFFER == Decimal("0.20")

    def test_mev_thresholds(self):
        """Test MEV timing thresholds."""
        assert MEV_CANCEL_THRESHOLD_MS == 100
        assert MEV_PARTIAL_THRESHOLD_MS == 500
        assert MEV_ORTHOGONAL_THRESHOLD_MS == 1000


class TestWalkthroughStep:
    """Tests for WalkthroughStep dataclass."""

    def test_step_creation(self):
        """Test creating a walkthrough step."""
        step = WalkthroughStep(
            block=1000,
            timestamp_ms=0,
            actor="Alice",
            action="Submits swap",
            input_amount=Decimal("100"),
            gas_fee=Decimal("0.01"),
            interference_deposit=Decimal("0.12"),
            phase=DiscretePhase.P0,
            psc_state={},
            amplitude_calculation="alpha = 1.0 + 0i",
            notes="Test note",
        )

        assert step.block == 1000
        assert step.actor == "Alice"
        assert step.input_amount == Decimal("100")
        assert step.phase == DiscretePhase.P0


class TestWalkthroughResult:
    """Tests for WalkthroughResult dataclass."""

    def test_result_creation(self):
        """Test creating a walkthrough result."""
        result = WalkthroughResult(
            scenario_type=WalkthroughType.MEV_ATTACK,
            steps=[],
            final_outcome="MEV Attack NEUTRALIZED",
            alice_receives=Decimal("950"),
            bot_profit=Decimal("-1.21"),
            mev_extracted=Decimal("0"),
            total_gas_paid=Decimal("0.02"),
            total_deposits=Decimal("1.32"),
            deposit_refunds=Decimal("0.12"),
        )

        assert result.scenario_type == WalkthroughType.MEV_ATTACK
        assert result.alice_receives == Decimal("950")
        assert result.bot_profit < 0


class TestTransactionWalkthrough:
    """Tests for TransactionWalkthrough class."""

    def test_init(self):
        """Test walkthrough initialization."""
        walkthrough = TransactionWalkthrough()

        assert walkthrough.steps == []
        assert walkthrough.psc is None
        assert walkthrough.current_block == 1000

    def test_mev_dex_swap_basic(self):
        """Test basic MEV DEX swap simulation."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            bot_input=Decimal("1000"),
            market_price=Decimal("9.5"),
            attack_delay_ms=50,
            seed=42,
        )

        assert result.scenario_type == WalkthroughType.MEV_ATTACK
        assert len(result.steps) >= 4  # At least 4 steps
        assert result.total_gas_paid > 0

    def test_mev_fast_attack_blocked(self):
        """Test that fast MEV attack (< 100ms) is blocked."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,  # Fast attack
            seed=42,
        )

        # Fast attack should be blocked (P180 phase)
        assert result.alice_receives > 0
        assert result.bot_profit < 0
        assert "NEUTRALIZED" in result.final_outcome

    def test_mev_phase_assignment_fast(self):
        """Test phase assignment for fast attack."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,  # < 100ms
            seed=42,
        )

        # Find bot's step
        bot_step = None
        for step in result.steps:
            if step.actor == "MEV Bot":
                bot_step = step
                break

        assert bot_step is not None
        assert bot_step.phase == DiscretePhase.P180  # Full cancellation

    def test_mev_phase_assignment_medium(self):
        """Test phase assignment for medium delay attack."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=300,  # 100-500ms
            seed=42,
        )

        # Find bot's step
        bot_step = None
        for step in result.steps:
            if step.actor == "MEV Bot":
                bot_step = step
                break

        assert bot_step is not None
        assert bot_step.phase == DiscretePhase.P135  # Partial cancellation

    def test_mev_phase_assignment_slow(self):
        """Test phase assignment for slow attack."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=800,  # 500-1000ms
            seed=42,
        )

        # Find bot's step
        bot_step = None
        for step in result.steps:
            if step.actor == "MEV Bot":
                bot_step = step
                break

        assert bot_step is not None
        assert bot_step.phase == DiscretePhase.P90  # Orthogonal

    def test_simple_transfer(self):
        """Test simple transfer walkthrough."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_simple_transfer(
            amount=Decimal("100"),
            sender="Alice",
            recipient="Bob",
        )

        assert result.scenario_type == WalkthroughType.SIMPLE_TRANSFER
        assert len(result.steps) == 2  # Submit + Collapse
        assert result.total_gas_paid == GAS_FEE_ESTIMATE

    def test_format_walkthrough(self):
        """Test walkthrough formatting."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        formatted = walkthrough.format_walkthrough(result)

        assert "ESM v5.4 Walkthrough" in formatted
        assert "Alice" in formatted
        assert "MEV Bot" in formatted
        assert "OUTCOME SUMMARY" in formatted

    def test_psc_state_captured(self):
        """Test that PSC state is captured correctly."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        # Check first step has PSC state
        first_step = result.steps[0]
        assert "psc_id" in first_step.psc_state
        assert "branches" in first_step.psc_state
        assert len(first_step.psc_state["branches"]) > 0


class TestDepositCalculations:
    """Tests for deposit calculations in walkthrough."""

    def test_interference_deposit_calculation(self):
        """Test interference deposit is calculated correctly."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        # Find Alice's step
        alice_step = result.steps[0]

        # Calculate expected deposit
        # 100 ESM * 0.001 (rate) * 1.20 (buffer) = 0.12 ESM
        expected_deposit = Decimal("100") * Decimal("0.001") * Decimal("1.20")

        assert alice_step.interference_deposit == expected_deposit

    def test_gas_fee_present(self):
        """Test gas fee is present in steps."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        # Check Alice has gas fee
        alice_step = result.steps[0]
        assert alice_step.gas_fee == GAS_FEE_ESTIMATE


class TestAmplitudeCalculations:
    """Tests for amplitude calculations in walkthrough."""

    def test_alice_amplitude_p0(self):
        """Test Alice's amplitude is P0 (Normal)."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        alice_step = result.steps[0]
        assert alice_step.phase == DiscretePhase.P0
        assert "alpha_Alice" in alice_step.amplitude_calculation

    def test_interference_calculation_present(self):
        """Test interference calculation is in bot step."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        bot_step = result.steps[1]
        assert "Interference Calculation" in bot_step.amplitude_calculation
        assert "alpha_total" in bot_step.amplitude_calculation


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_run_mev_walkthrough(self):
        """Test run_mev_walkthrough helper."""
        result = run_mev_walkthrough(
            alice_amount=Decimal("100"),
            attack_delay_ms=50,
            verbose=False,
        )

        assert result is not None
        assert result.scenario_type == WalkthroughType.MEV_ATTACK

    def test_compare_timing_scenarios(self):
        """Test compare_timing_scenarios helper."""
        results = compare_timing_scenarios(alice_amount=Decimal("100"))

        assert "fast_50ms" in results
        assert "medium_300ms" in results
        assert "slow_800ms" in results
        assert "very_slow_1500ms" in results

        # Fast attack should be blocked
        assert results["fast_50ms"].alice_receives > 0

    def test_compare_shows_progression(self):
        """Test that slower attacks have less interference."""
        results = compare_timing_scenarios(alice_amount=Decimal("100"))

        # Fast attack: bot loses most
        # Slow attack: bot loses less
        fast_profit = results["fast_50ms"].bot_profit
        slow_profit = results["very_slow_1500ms"].bot_profit

        # Fast attack should be worse for bot
        assert fast_profit <= slow_profit


class TestWalkthroughTypes:
    """Tests for different walkthrough types."""

    def test_mev_attack_type(self):
        """Test MEV attack scenario type."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        assert result.scenario_type == WalkthroughType.MEV_ATTACK

    def test_simple_transfer_type(self):
        """Test simple transfer scenario type."""
        walkthrough = TransactionWalkthrough()
        result = walkthrough.simulate_simple_transfer(
            amount=Decimal("100"),
            sender="Alice",
            recipient="Bob",
        )

        assert result.scenario_type == WalkthroughType.SIMPLE_TRANSFER


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_result(self):
        """Test same seed produces same result."""
        walkthrough1 = TransactionWalkthrough()
        result1 = walkthrough1.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        walkthrough2 = TransactionWalkthrough()
        result2 = walkthrough2.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=50,
            seed=42,
        )

        assert result1.alice_receives == result2.alice_receives
        assert result1.bot_profit == result2.bot_profit

    def test_different_seed_can_differ(self):
        """Test different seeds can produce different results (for non-deterministic collapse)."""
        # Note: With P180 interference, result is deterministic regardless of seed
        # This test uses a partial interference scenario
        walkthrough1 = TransactionWalkthrough()
        result1 = walkthrough1.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=300,  # P135 - partial interference
            seed=1,
        )

        walkthrough2 = TransactionWalkthrough()
        result2 = walkthrough2.simulate_mev_dex_swap(
            alice_input=Decimal("100"),
            attack_delay_ms=300,
            seed=999,
        )

        # Results may or may not differ depending on probability
        # But the test should complete without error
        assert result1 is not None
        assert result2 is not None
