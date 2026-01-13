"""
Tests for Deposit Buffer System.

Based on ESM Whitepaper v5.2 Section 6.2-6.4.
"""

import pytest

from esm.core.deposit import (
    DEFAULT_BUFFER_PERCENT,
    MAX_BUFFER_PERCENT,
    MIN_DEPOSIT,
    DepositStatus,
    DepositParams,
    DepositResult,
    DepositEstimate,
    calculate_deposit_with_buffer,
    estimate_deposit,
    process_deposit,
    validate_deposit,
    simulate_deposit_scenario,
    analyze_buffer_effectiveness,
)


class TestConstants:
    """Tests for deposit system constants."""

    def test_default_buffer(self):
        """Test default buffer is 20%."""
        assert DEFAULT_BUFFER_PERCENT == 20

    def test_max_buffer(self):
        """Test max buffer is 50%."""
        assert MAX_BUFFER_PERCENT == 50

    def test_min_deposit(self):
        """Test minimum deposit is 1."""
        assert MIN_DEPOSIT == 1


class TestCalculateDepositWithBuffer:
    """Tests for calculate_deposit_with_buffer function."""

    def test_default_buffer_20_percent(self):
        """Test default 20% buffer."""
        deposit = calculate_deposit_with_buffer(1000)
        assert deposit == 1200

    def test_custom_buffer(self):
        """Test custom buffer percentage."""
        assert calculate_deposit_with_buffer(1000, buffer_percent=10) == 1100
        assert calculate_deposit_with_buffer(1000, buffer_percent=30) == 1300
        assert calculate_deposit_with_buffer(1000, buffer_percent=50) == 1500

    def test_zero_buffer(self):
        """Test zero buffer."""
        deposit = calculate_deposit_with_buffer(1000, buffer_percent=0)
        assert deposit == 1000

    def test_max_buffer_clamped(self):
        """Test buffer is clamped to MAX_BUFFER_PERCENT."""
        deposit = calculate_deposit_with_buffer(1000, buffer_percent=100)
        assert deposit == 1500  # Clamped to 50%

    def test_negative_buffer_clamped(self):
        """Test negative buffer is clamped to 0."""
        deposit = calculate_deposit_with_buffer(1000, buffer_percent=-10)
        assert deposit == 1000  # Clamped to 0%

    def test_max_deposit_cap(self):
        """Test max_deposit caps the result."""
        deposit = calculate_deposit_with_buffer(1000, max_deposit=1100)
        assert deposit == 1100  # Capped at 1100

    def test_min_deposit_enforced(self):
        """Test minimum deposit is enforced."""
        deposit = calculate_deposit_with_buffer(0)
        assert deposit == MIN_DEPOSIT


class TestEstimateDeposit:
    """Tests for estimate_deposit function."""

    def test_estimate_breakdown(self):
        """Test estimate provides correct breakdown."""
        estimate = estimate_deposit(1000)

        assert estimate.estimated_cost == 1000
        assert estimate.buffer_amount == 200  # 20% of 1000
        assert estimate.total_deposit == 1200
        assert estimate.buffer_percent == 20

    def test_estimate_with_custom_buffer(self):
        """Test estimate with custom buffer."""
        estimate = estimate_deposit(1000, buffer_percent=30)

        assert estimate.buffer_amount == 300
        assert estimate.total_deposit == 1300
        assert estimate.buffer_percent == 30


class TestProcessDeposit:
    """Tests for process_deposit function."""

    def test_exact_deposit(self):
        """Test when deposit exactly matches cost."""
        result = process_deposit(deposit_paid=1000, actual_cost=1000)

        assert result.status == DepositStatus.SUCCESS
        assert result.deposit_paid == 1000
        assert result.actual_cost == 1000
        assert result.refund_amount == 0
        assert result.net_cost == 1000

    def test_excess_deposit_refunded(self):
        """Test excess deposit is refunded."""
        result = process_deposit(deposit_paid=1200, actual_cost=1000)

        assert result.status == DepositStatus.REFUNDED
        assert result.deposit_paid == 1200
        assert result.actual_cost == 1000
        assert result.refund_amount == 200
        assert result.net_cost == 1000

    def test_insufficient_deposit(self):
        """Test insufficient deposit is detected."""
        result = process_deposit(deposit_paid=800, actual_cost=1000)

        assert result.status == DepositStatus.INSUFFICIENT
        assert result.deposit_paid == 800
        assert result.actual_cost == 1000
        assert result.refund_amount == 0

    def test_effective_buffer_calculated(self):
        """Test effective buffer percentage is calculated."""
        result = process_deposit(deposit_paid=1200, actual_cost=1000)

        # Effective buffer: (1200 - 1000) / 1000 * 100 = 20%
        assert result.effective_buffer == 20.0

    def test_zero_actual_cost(self):
        """Test handling of zero actual cost."""
        result = process_deposit(deposit_paid=100, actual_cost=0)

        assert result.status == DepositStatus.REFUNDED
        assert result.refund_amount == 100
        assert result.effective_buffer == 0  # Avoid division by zero


class TestValidateDeposit:
    """Tests for validate_deposit function."""

    def test_valid_deposit(self):
        """Test valid deposit passes validation."""
        is_valid, message = validate_deposit(
            deposit=1200,
            estimated_cost=1000,
            required_buffer=20
        )

        assert is_valid is True
        assert "validated" in message.lower()

    def test_insufficient_deposit(self):
        """Test insufficient deposit fails validation."""
        is_valid, message = validate_deposit(
            deposit=1100,
            estimated_cost=1000,
            required_buffer=20
        )

        assert is_valid is False
        assert "insufficient" in message.lower()
        assert "shortfall" in message.lower()


class TestSimulateDepositScenario:
    """Tests for deposit simulation."""

    def test_simulation_runs(self):
        """Test simulation runs without error."""
        result = simulate_deposit_scenario(
            n_transactions=100,
            base_cost=1000,
            seed=42
        )

        assert result["n_transactions"] == 100
        assert result["successful"] + result["insufficient"] == 100

    def test_higher_buffer_reduces_reverts(self):
        """Test higher buffer reduces revert rate."""
        low_buffer = simulate_deposit_scenario(
            n_transactions=1000,
            base_cost=1000,
            buffer_percent=10,
            cost_variance=0.3,
            seed=42
        )

        high_buffer = simulate_deposit_scenario(
            n_transactions=1000,
            base_cost=1000,
            buffer_percent=40,
            cost_variance=0.3,
            seed=42
        )

        assert high_buffer["revert_rate"] <= low_buffer["revert_rate"]

    def test_zero_variance_no_reverts_with_buffer(self):
        """Test zero variance and buffer means no reverts."""
        result = simulate_deposit_scenario(
            n_transactions=100,
            base_cost=1000,
            cost_variance=0.0,  # No variance
            buffer_percent=20,
            seed=42
        )

        # With no variance, actual cost == estimated cost
        # So 20% buffer should always be enough
        assert result["revert_rate"] == 0.0


class TestAnalyzeBufferEffectiveness:
    """Tests for buffer effectiveness analysis."""

    def test_analysis_runs(self):
        """Test analysis runs without error."""
        result = analyze_buffer_effectiveness(
            base_cost=1000,
            cost_variance=0.3,
            n_simulations=100,
            seed=42
        )

        assert 0 in result
        assert 20 in result
        assert 50 in result

    def test_higher_buffer_higher_success(self):
        """Test higher buffer leads to higher success rate."""
        result = analyze_buffer_effectiveness(
            base_cost=1000,
            cost_variance=0.3,
            n_simulations=1000,
            seed=42
        )

        # Success rate should generally increase with buffer
        assert result[50]["success_rate"] >= result[0]["success_rate"]

    def test_higher_buffer_higher_refund(self):
        """Test higher buffer leads to higher refund rate."""
        result = analyze_buffer_effectiveness(
            base_cost=1000,
            cost_variance=0.3,
            n_simulations=1000,
            seed=42
        )

        # Refund rate should increase with buffer
        assert result[50]["avg_refund_rate"] >= result[0]["avg_refund_rate"]


class TestDepositDataClasses:
    """Tests for deposit data classes."""

    def test_deposit_params(self):
        """Test DepositParams dataclass."""
        params = DepositParams(estimated_cost=1000, buffer_percent=25)

        assert params.estimated_cost == 1000
        assert params.buffer_percent == 25
        assert params.max_deposit is None

    def test_deposit_result_net_cost(self):
        """Test DepositResult net_cost property."""
        result = DepositResult(
            status=DepositStatus.REFUNDED,
            deposit_paid=1200,
            actual_cost=1000,
            refund_amount=200,
            effective_buffer=20.0
        )

        assert result.net_cost == 1000

    def test_deposit_estimate(self):
        """Test DepositEstimate dataclass."""
        estimate = DepositEstimate(
            estimated_cost=1000,
            buffer_amount=200,
            total_deposit=1200,
            buffer_percent=20
        )

        assert estimate.estimated_cost == 1000
        assert estimate.buffer_amount == 200
        assert estimate.total_deposit == 1200
