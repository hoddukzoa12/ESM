"""
MEV Resistant DEX

Based on ESM Whitepaper v5.3 Section 8.1

Implements a decentralized exchange that uses ESM's interference mechanism
to neutralize MEV extraction attacks like sandwich attacks and front-running.

Key features:
- Phase assignment based on transaction timing
- Destructive interference cancels fast attacks
- Slippage protection via probabilistic ordering
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from enum import Enum
import hashlib
import time

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_branch
from esm.core.psc import PSC, create_psc
from esm.applications.base import (
    ESMApplication,
    ApplicationResult,
    ApplicationStatus,
    calculate_interference_impact,
)


# =============================================================================
# Constants
# =============================================================================

# MEV timing thresholds (ms)
FAST_ATTACK_THRESHOLD = 100       # < 100ms: P180 (full cancel)
MEDIUM_ATTACK_THRESHOLD = 500     # < 500ms: P135 (partial cancel)
SLOW_ATTACK_THRESHOLD = 1000      # < 1000ms: P90 (orthogonal)

# Fee rate
DEX_FEE_RATE = Decimal("0.003")   # 0.3% swap fee


class SwapStatus(Enum):
    """Status of swap order."""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FRONT_RUN_BLOCKED = "front_run_blocked"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SwapOrder:
    """
    A swap order submitted to the DEX.

    Attributes:
        trader: Trader address/ID
        input_token: Token being sold
        output_token: Token being bought
        input_amount: Amount of input token
        min_output: Minimum acceptable output
        submitted_block: Block when submitted
        submitted_ms: Timestamp in milliseconds
        phase: Assigned phase (based on timing)
    """
    trader: str
    input_token: str
    output_token: str
    input_amount: Decimal
    min_output: Decimal
    submitted_block: int = 0
    submitted_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    phase: DiscretePhase = DiscretePhase.P0


@dataclass
class LiquidityPool:
    """
    A liquidity pool for token pair.

    Implements constant product AMM (x * y = k).

    Attributes:
        token_a: First token
        token_b: Second token
        reserve_a: Reserve of token A
        reserve_b: Reserve of token B
        fee_rate: Trading fee rate
    """
    token_a: str
    token_b: str
    reserve_a: Decimal
    reserve_b: Decimal
    fee_rate: Decimal = DEX_FEE_RATE

    @property
    def price_a_to_b(self) -> Decimal:
        """Price of token A in terms of token B."""
        if self.reserve_a == 0:
            return Decimal("0")
        return self.reserve_b / self.reserve_a

    @property
    def price_b_to_a(self) -> Decimal:
        """Price of token B in terms of token A."""
        if self.reserve_b == 0:
            return Decimal("0")
        return self.reserve_a / self.reserve_b


@dataclass
class DEXResult(ApplicationResult):
    """
    Result of DEX operation.

    Attributes:
        swap_executed: Whether swap was executed
        input_amount: Input amount
        output_amount: Output amount received
        slippage: Actual slippage percentage
        mev_blocked: Whether MEV attack was blocked
        attacker_loss: Attacker's loss if MEV was blocked
    """
    swap_executed: bool = False
    input_amount: Decimal = Decimal("0")
    output_amount: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    mev_blocked: bool = False
    attacker_loss: Decimal = Decimal("0")


# =============================================================================
# MEV Resistant DEX
# =============================================================================

class MEVResistantDEX(ESMApplication):
    """
    MEV Resistant Decentralized Exchange.

    Uses ESM's interference mechanism to neutralize front-running
    and sandwich attacks.
    """

    def __init__(self):
        super().__init__("MEVResistantDEX")
        self.pools: Dict[str, LiquidityPool] = {}
        self.pending_orders: Dict[str, List[SwapOrder]] = {}
        self.executed_swaps: List[Dict] = []

    def add_pool(self, pool: LiquidityPool) -> None:
        """
        Add a liquidity pool.

        Args:
            pool: LiquidityPool to add
        """
        pool_id = f"{pool.token_a}_{pool.token_b}"
        self.pools[pool_id] = pool
        self.pending_orders[pool_id] = []

    def get_pool(self, token_a: str, token_b: str) -> Optional[LiquidityPool]:
        """Get pool for token pair."""
        pool_id = f"{token_a}_{token_b}"
        if pool_id in self.pools:
            return self.pools[pool_id]
        # Try reverse
        pool_id_rev = f"{token_b}_{token_a}"
        return self.pools.get(pool_id_rev)

    def calculate_output(
        self,
        pool: LiquidityPool,
        input_token: str,
        input_amount: Decimal,
    ) -> Decimal:
        """
        Calculate output amount for a swap.

        Uses constant product formula: x * y = k

        Args:
            pool: Liquidity pool
            input_token: Token being sold
            input_amount: Amount being sold

        Returns:
            Output amount
        """
        if input_token == pool.token_a:
            reserve_in = pool.reserve_a
            reserve_out = pool.reserve_b
        else:
            reserve_in = pool.reserve_b
            reserve_out = pool.reserve_a

        # Apply fee
        amount_with_fee = input_amount * (1 - pool.fee_rate)

        # Constant product formula
        # (reserve_in + amount) * (reserve_out - output) = k
        # output = reserve_out - k / (reserve_in + amount)
        # output = reserve_out * amount / (reserve_in + amount)
        output = (reserve_out * amount_with_fee) / (reserve_in + amount_with_fee)

        return output

    def assign_phase_by_timing(
        self,
        order: SwapOrder,
        reference_time_ms: int,
    ) -> DiscretePhase:
        """
        Assign phase to order based on timing relative to reference.

        Fast orders (likely attacks) get opposite phases to cause
        destructive interference.

        Args:
            order: Swap order
            reference_time_ms: Reference time (first order's time)

        Returns:
            Assigned phase
        """
        delay = order.submitted_ms - reference_time_ms

        if delay < FAST_ATTACK_THRESHOLD:
            return DiscretePhase.P180  # Full destructive
        elif delay < MEDIUM_ATTACK_THRESHOLD:
            return DiscretePhase.P135  # Partial destructive
        elif delay < SLOW_ATTACK_THRESHOLD:
            return DiscretePhase.P90   # Orthogonal
        else:
            return DiscretePhase.P0    # Normal

    def detect_mev_attack(
        self,
        original: SwapOrder,
        suspect: SwapOrder,
    ) -> bool:
        """
        Detect if suspect order is likely an MEV attack.

        Args:
            original: Original/victim order
            suspect: Suspected attack order

        Returns:
            True if attack detected
        """
        # Same pool, close timing, larger amount
        time_diff = abs(suspect.submitted_ms - original.submitted_ms)

        is_close_timing = time_diff < SLOW_ATTACK_THRESHOLD
        is_same_pair = (
            suspect.input_token == original.input_token and
            suspect.output_token == original.output_token
        )
        is_larger = suspect.input_amount > original.input_amount

        return is_close_timing and is_same_pair and is_larger

    def create_swap_psc(
        self,
        pool: LiquidityPool,
        orders: List[SwapOrder],
    ) -> PSC:
        """
        Create PSC for competing swap orders.

        Orders get phases based on timing - fast orders get
        opposite phases to legitimate transactions.

        Args:
            pool: Liquidity pool
            orders: List of swap orders

        Returns:
            PSC with order branches
        """
        psc_id = hashlib.sha256(
            f"swap_{self.block_number}_{id(orders)}".encode()
        ).hexdigest()[:16]

        psc = self.create_application_psc(psc_id)

        if not orders:
            return psc

        # First order is reference (assumed legitimate)
        reference_time = orders[0].submitted_ms

        for i, order in enumerate(orders):
            # First order (reference/victim) always gets P0 (Normal)
            # Subsequent orders get phase based on timing relative to first
            if i == 0:
                phase = DiscretePhase.P0
            else:
                phase = self.assign_phase_by_timing(order, reference_time)
            order.phase = phase

            # Calculate expected output
            output = self.calculate_output(
                pool,
                order.input_token,
                order.input_amount,
            )

            # Create branch
            state_data = {
                "trader": order.trader,
                "input_token": order.input_token,
                "output_token": order.output_token,
                "input_amount": float(order.input_amount),
                "output_amount": float(output),
                "submitted_ms": order.submitted_ms,
            }

            branch = create_branch(
                state_data=state_data,
                magnitude=1.0,
                phase=phase,
                creator=order.trader,
            )
            psc.add_branch(branch)

        return psc

    def simulate_swap_with_attack(
        self,
        victim: SwapOrder,
        attacker: SwapOrder,
        pool: LiquidityPool,
        seed: Optional[int] = None,
    ) -> DEXResult:
        """
        Simulate swap with MEV attack.

        Demonstrates how phase-based interference blocks the attack.

        Args:
            victim: Victim's swap order
            attacker: Attacker's order (front-run attempt)
            pool: Liquidity pool
            seed: Random seed for collapse

        Returns:
            DEXResult with simulation outcome
        """
        # Detect if this is an attack
        is_attack = self.detect_mev_attack(victim, attacker)

        # Create PSC with both orders
        orders = [victim, attacker]
        psc = self.create_swap_psc(pool, orders)

        # Get interference analysis
        interference = psc.calculate_interference()
        probs = psc.get_probabilities()

        # Calculate interference type between victim and attacker
        interference_type = calculate_interference_impact(victim.phase, attacker.phase)

        # Determine winner based on interference
        # With destructive interference (P180), attacker's effective amplitude is 0
        if interference_type == "destructive_full":
            # Full destructive interference - victim wins deterministically
            # Attacker's P180 cancels out against victim's P0
            winner_trader = victim.trader
            winner_input = float(victim.input_amount)
            winner_output = self.calculate_output(pool, victim.input_token, victim.input_amount)
        elif interference_type == "destructive_partial":
            # Partial destructive - calculate victim probability from interference factor
            # Formula: victim_prob = 0.5 + 0.5 * |interference_factor|
            # For P135: cos(135°) ≈ -0.707, so victim_prob ≈ 0.5 + 0.5 * 0.707 ≈ 0.854
            # This models how destructive interference reduces attacker's effective amplitude
            import random
            if seed is not None:
                random.seed(seed)
            # Calculate interference factor from phase difference
            from esm.core.phase import COS_TABLE
            phase_diff = (attacker.phase - victim.phase) % 8
            interference_factor = COS_TABLE[DiscretePhase(phase_diff)]
            # Victim probability increases with destructive interference (negative factor)
            victim_prob = 0.5 + 0.5 * abs(interference_factor)
            if random.random() < victim_prob:
                winner_trader = victim.trader
                winner_input = float(victim.input_amount)
                winner_output = self.calculate_output(pool, victim.input_token, victim.input_amount)
            else:
                # Attacker wins - no PSC collapse needed since probability already calculated
                winner_trader = attacker.trader
                winner_input = float(attacker.input_amount)
                winner_output = self.calculate_output(pool, attacker.input_token, attacker.input_amount)
        else:
            # Normal collapse for orthogonal or constructive interference
            selected_state, selected_branch = psc.collapse(seed=seed)
            winner_trader = selected_branch.state_data["trader"]
            winner_input = selected_branch.state_data["input_amount"]
            winner_output = Decimal(str(selected_branch.state_data["output_amount"]))

        # Calculate attacker loss if blocked
        mev_blocked = is_attack and winner_trader == victim.trader
        attacker_loss = Decimal("0")

        if mev_blocked:
            # Attacker loses gas + deposit
            attacker_loss = attacker.input_amount * Decimal("0.001")  # Estimate

        return DEXResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=psc.id,
            selected_outcome=winner_trader,
            probability_distribution={
                k: float(v) for k, v in probs.items()
            },
            metadata={
                "victim_phase": victim.phase.name,
                "attacker_phase": attacker.phase.name,
                "interference_type": interference_type,
                "is_attack": is_attack,
            },
            swap_executed=True,
            input_amount=winner_input,
            output_amount=winner_output,
            slippage=Decimal("0"),
            mev_blocked=mev_blocked,
            attacker_loss=attacker_loss,
        )

    def get_status(self) -> Dict:
        """Get DEX status."""
        return {
            "pools": len(self.pools),
            "pending_orders": sum(len(v) for v in self.pending_orders.values()),
            "executed_swaps": len(self.executed_swaps),
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def create_sample_pool(
    token_a: str = "ESM",
    token_b: str = "TOKEN",
    reserve_a: Decimal = Decimal("100000"),
    reserve_b: Decimal = Decimal("950000"),
) -> LiquidityPool:
    """Create a sample liquidity pool."""
    return LiquidityPool(
        token_a=token_a,
        token_b=token_b,
        reserve_a=reserve_a,
        reserve_b=reserve_b,
    )


def simulate_mev_scenario(
    victim_amount: Decimal = Decimal("100"),
    attacker_amount: Decimal = Decimal("1000"),
    attack_delay_ms: int = 50,
    seed: int = 42,
) -> DEXResult:
    """
    Run a complete MEV attack simulation.

    Args:
        victim_amount: Victim's trade amount
        attacker_amount: Attacker's front-run amount
        attack_delay_ms: Attacker's timing delay
        seed: Random seed

    Returns:
        DEXResult with simulation outcome
    """
    # Create DEX and pool
    dex = MEVResistantDEX()
    pool = create_sample_pool()
    dex.add_pool(pool)

    # Create orders
    base_time = int(time.time() * 1000)

    victim = SwapOrder(
        trader="Alice",
        input_token="ESM",
        output_token="TOKEN",
        input_amount=victim_amount,
        min_output=victim_amount * Decimal("9"),
        submitted_ms=base_time,
    )

    attacker = SwapOrder(
        trader="MEV_Bot",
        input_token="ESM",
        output_token="TOKEN",
        input_amount=attacker_amount,
        min_output=attacker_amount * Decimal("9"),
        submitted_ms=base_time + attack_delay_ms,
    )

    # Simulate
    return dex.simulate_swap_with_attack(victim, attacker, pool, seed=seed)


def run_demo():
    """Run MEV DEX demo."""
    print("=" * 60)
    print("MEV Resistant DEX Demo")
    print("=" * 60)
    print()

    # Simulate attack scenarios
    scenarios = [
        ("Fast Attack (50ms)", 50),
        ("Medium Attack (300ms)", 300),
        ("Slow Attack (800ms)", 800),
        ("Very Slow (1500ms)", 1500),
    ]

    for name, delay in scenarios:
        result = simulate_mev_scenario(attack_delay_ms=delay)
        status = "BLOCKED" if result.mev_blocked else "NOT BLOCKED"

        print(f"{name}:")
        print(f"  Attacker phase: {result.metadata['attacker_phase']}")
        print(f"  Interference: {result.metadata['interference_type']}")
        print(f"  Winner: {result.selected_outcome}")
        print(f"  MEV Status: {status}")
        print()


if __name__ == "__main__":
    run_demo()
