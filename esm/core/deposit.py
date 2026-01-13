"""
Deposit Buffer System

Based on ESM Whitepaper v5.2 Section 6.2-6.4

The deposit buffer system ensures transaction cost predictability by:
1. Adding a buffer (default 20%) to estimated interference costs
2. Automatically refunding excess deposits after calculation
3. Preventing revert due to insufficient gas/deposit

Key features:
- DEFAULT_BUFFER_PERCENT: 20% buffer for cost variability
- MAX_BUFFER_PERCENT: 50% maximum allowed buffer
- Automatic refund of unused deposits
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


# =============================================================================
# Constants (from Whitepaper Section 6.3)
# =============================================================================

DEFAULT_BUFFER_PERCENT: int = 20
MAX_BUFFER_PERCENT: int = 50
MIN_DEPOSIT: int = 1  # Minimum deposit in smallest unit


class DepositStatus(Enum):
    """Status of deposit processing."""
    SUCCESS = "success"
    INSUFFICIENT = "insufficient"
    REFUNDED = "refunded"
    PENDING = "pending"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DepositParams:
    """
    Parameters for deposit calculation.

    Attributes:
        estimated_cost: Estimated interference calculation cost
        buffer_percent: Buffer percentage (default 20%)
        max_deposit: Optional maximum deposit limit
    """
    estimated_cost: int
    buffer_percent: int = DEFAULT_BUFFER_PERCENT
    max_deposit: Optional[int] = None


@dataclass
class DepositResult:
    """
    Result of deposit processing.

    Attributes:
        status: Processing status
        deposit_paid: Amount deposited
        actual_cost: Actual cost incurred
        refund_amount: Amount to be refunded
        effective_buffer: Actual buffer used (percent)
    """
    status: DepositStatus
    deposit_paid: int
    actual_cost: int
    refund_amount: int
    effective_buffer: float

    @property
    def net_cost(self) -> int:
        """Net cost after refund."""
        return self.deposit_paid - self.refund_amount


@dataclass
class DepositEstimate:
    """
    Deposit estimation before submission.

    Attributes:
        estimated_cost: Base estimated cost
        buffer_amount: Additional buffer amount
        total_deposit: Total deposit required
        buffer_percent: Buffer percentage used
    """
    estimated_cost: int
    buffer_amount: int
    total_deposit: int
    buffer_percent: int


# =============================================================================
# Core Functions
# =============================================================================

def calculate_deposit_with_buffer(
    estimated_cost: int,
    buffer_percent: int = DEFAULT_BUFFER_PERCENT,
    max_deposit: Optional[int] = None,
) -> int:
    """
    Calculate deposit amount including buffer.

    Based on Whitepaper Section 6.3 formula:
    deposit = estimated_cost * (100 + buffer_percent) / 100

    Args:
        estimated_cost: Estimated interference calculation cost
        buffer_percent: Buffer percentage (default 20%, max 50%)
        max_deposit: Optional maximum deposit cap

    Returns:
        Deposit amount with buffer applied

    Examples:
        >>> calculate_deposit_with_buffer(1000)  # 20% buffer
        1200
        >>> calculate_deposit_with_buffer(1000, buffer_percent=30)
        1300
        >>> calculate_deposit_with_buffer(1000, max_deposit=1100)
        1100
    """
    # Clamp buffer to valid range
    buffer_percent = max(0, min(buffer_percent, MAX_BUFFER_PERCENT))

    # Calculate buffered amount
    buffered = estimated_cost * (100 + buffer_percent) // 100

    # Ensure minimum deposit
    buffered = max(buffered, MIN_DEPOSIT)

    # Apply maximum cap if specified
    if max_deposit is not None:
        buffered = min(buffered, max_deposit)

    return buffered


def estimate_deposit(
    estimated_cost: int,
    buffer_percent: int = DEFAULT_BUFFER_PERCENT,
    max_deposit: Optional[int] = None,
) -> DepositEstimate:
    """
    Get detailed deposit estimation.

    Args:
        estimated_cost: Estimated interference calculation cost
        buffer_percent: Buffer percentage (default 20%)
        max_deposit: Optional maximum deposit cap

    Returns:
        DepositEstimate with breakdown of costs
    """
    buffer_percent = max(0, min(buffer_percent, MAX_BUFFER_PERCENT))
    buffer_amount = estimated_cost * buffer_percent // 100
    total = calculate_deposit_with_buffer(estimated_cost, buffer_percent, max_deposit)

    return DepositEstimate(
        estimated_cost=estimated_cost,
        buffer_amount=buffer_amount,
        total_deposit=total,
        buffer_percent=buffer_percent,
    )


def process_deposit(
    deposit_paid: int,
    actual_cost: int,
) -> DepositResult:
    """
    Process deposit after interference calculation.

    Calculates refund for excess deposit based on actual cost.

    Args:
        deposit_paid: Amount deposited before operation
        actual_cost: Actual cost of interference calculation

    Returns:
        DepositResult with refund information

    Raises:
        ValueError: If deposit is insufficient for actual cost
    """
    if deposit_paid < actual_cost:
        return DepositResult(
            status=DepositStatus.INSUFFICIENT,
            deposit_paid=deposit_paid,
            actual_cost=actual_cost,
            refund_amount=0,
            effective_buffer=0.0,
        )

    refund = deposit_paid - actual_cost
    effective_buffer = (deposit_paid - actual_cost) / actual_cost * 100 if actual_cost > 0 else 0

    return DepositResult(
        status=DepositStatus.REFUNDED if refund > 0 else DepositStatus.SUCCESS,
        deposit_paid=deposit_paid,
        actual_cost=actual_cost,
        refund_amount=refund,
        effective_buffer=effective_buffer,
    )


def validate_deposit(
    deposit: int,
    estimated_cost: int,
    required_buffer: int = DEFAULT_BUFFER_PERCENT,
) -> Tuple[bool, str]:
    """
    Validate if deposit meets minimum requirements.

    Args:
        deposit: Proposed deposit amount
        estimated_cost: Estimated cost
        required_buffer: Required buffer percentage

    Returns:
        Tuple of (is_valid, message)
    """
    min_required = calculate_deposit_with_buffer(estimated_cost, required_buffer)

    if deposit < min_required:
        shortfall = min_required - deposit
        return False, f"Insufficient deposit: {deposit} < {min_required} (shortfall: {shortfall})"

    return True, "Deposit validated"


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_deposit_scenario(
    n_transactions: int,
    base_cost: int,
    cost_variance: float = 0.3,
    buffer_percent: int = DEFAULT_BUFFER_PERCENT,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate deposit outcomes for multiple transactions.

    Useful for analyzing buffer effectiveness.

    Args:
        n_transactions: Number of transactions to simulate
        base_cost: Base estimated cost per transaction
        cost_variance: Variance in actual costs (0-1)
        buffer_percent: Buffer percentage to use
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation statistics
    """
    import random
    if seed is not None:
        random.seed(seed)

    results = {
        "n_transactions": n_transactions,
        "buffer_percent": buffer_percent,
        "successful": 0,
        "insufficient": 0,
        "total_deposited": 0,
        "total_actual_cost": 0,
        "total_refunded": 0,
        "revert_rate": 0.0,
        "avg_effective_buffer": 0.0,
    }

    effective_buffers = []

    for _ in range(n_transactions):
        # Calculate deposit with buffer
        deposit = calculate_deposit_with_buffer(base_cost, buffer_percent)

        # Simulate actual cost with variance
        variance_factor = 1 + (random.random() * 2 - 1) * cost_variance
        actual_cost = int(base_cost * variance_factor)
        actual_cost = max(1, actual_cost)  # Ensure positive

        # Process deposit
        result = process_deposit(deposit, actual_cost)

        results["total_deposited"] += deposit
        results["total_actual_cost"] += actual_cost

        if result.status == DepositStatus.INSUFFICIENT:
            results["insufficient"] += 1
        else:
            results["successful"] += 1
            results["total_refunded"] += result.refund_amount
            effective_buffers.append(result.effective_buffer)

    # Calculate statistics
    results["revert_rate"] = results["insufficient"] / n_transactions
    results["avg_effective_buffer"] = (
        sum(effective_buffers) / len(effective_buffers)
        if effective_buffers else 0
    )

    return results


def analyze_buffer_effectiveness(
    base_cost: int,
    cost_variance: float = 0.3,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Analyze effectiveness of different buffer percentages.

    Args:
        base_cost: Base estimated cost
        cost_variance: Variance in actual costs
        n_simulations: Number of simulations per buffer level
        seed: Random seed

    Returns:
        Dictionary mapping buffer_percent to success rate
    """
    buffer_levels = [0, 10, 20, 30, 40, 50]
    results = {}

    for buffer in buffer_levels:
        sim = simulate_deposit_scenario(
            n_transactions=n_simulations,
            base_cost=base_cost,
            cost_variance=cost_variance,
            buffer_percent=buffer,
            seed=seed,
        )
        results[buffer] = {
            "success_rate": 1 - sim["revert_rate"],
            "avg_refund_rate": sim["total_refunded"] / sim["total_deposited"]
            if sim["total_deposited"] > 0 else 0,
        }

    return results
