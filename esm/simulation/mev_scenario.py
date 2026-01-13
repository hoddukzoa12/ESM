"""
MEV (Maximal Extractable Value) Simulation

Based on ESM Whitepaper v5.1 Section 7.1

Simulates sandwich attacks and demonstrates how ESM's phase-based
interference mechanism can neutralize MEV extraction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from esm.core.phase import DiscretePhase, COS_TABLE
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_victim_tx, create_attacker_tx
from esm.core.psc import PSC, create_psc


# =============================================================================
# MEV Timing Windows (Whitepaper Section 7.1)
# =============================================================================

# Time windows in milliseconds
MEV_WINDOW_FULL_CANCEL = 100    # <100ms: Complete cancellation (180°)
MEV_WINDOW_PARTIAL_CANCEL = 500  # <500ms: Partial cancellation (135°)
MEV_WINDOW_ORTHOGONAL = 1000     # <1s: Orthogonal (90°)
# >1s: Normal transaction (0°)


def assign_phase_by_delay(delay_ms: float) -> DiscretePhase:
    """
    Assign phase based on transaction timing delay.

    Implements the MEV defense mechanism from Whitepaper Section 7.1:
    - Transactions very close in time get opposite phases
    - This causes destructive interference, reducing attack profitability

    Args:
        delay_ms: Time delay in milliseconds after the target transaction

    Returns:
        DiscretePhase to assign to the transaction
    """
    if delay_ms < MEV_WINDOW_FULL_CANCEL:
        return DiscretePhase.P180  # Complete opposite phase
    elif delay_ms < MEV_WINDOW_PARTIAL_CANCEL:
        return DiscretePhase.P135  # Partial opposite
    elif delay_ms < MEV_WINDOW_ORTHOGONAL:
        return DiscretePhase.P90   # Orthogonal (no interference)
    else:
        return DiscretePhase.P0    # Normal transaction


def calculate_interference_factor(phase: DiscretePhase) -> float:
    """
    Calculate the interference factor for a given phase.

    Returns how much the attack amplitude is preserved after interference:
    - 0.0 = complete cancellation
    - 0.5 = 50% preserved
    - 1.0 = no interference

    Args:
        phase: The assigned phase

    Returns:
        Interference factor (0.0 to 1.0)
    """
    # Phase difference from P0 (victim phase)
    phase_diff = phase.value

    # Interference reduces based on cos(Δθ)
    # For attack: we want the probability preservation
    # cos(180°) = -1 → probability = (1 + (-1))² / 4 = 0
    # cos(135°) = -0.707 → probability ≈ 0.04
    # cos(90°) = 0 → probability = 0.25
    # cos(0°) = 1 → probability = 1.0

    cos_val = COS_TABLE[phase_diff]

    # When victim and attacker combine:
    # |α_v + α_a|² where α_v = 1 and α_a = e^(iθ)
    # = |1 + cos(θ) + i*sin(θ)|²
    # = (1 + cos(θ))² + sin²(θ)
    # = 1 + 2cos(θ) + cos²(θ) + sin²(θ)
    # = 2 + 2cos(θ)
    # = 2(1 + cos(θ))

    # Attacker's effective profit factor:
    # In destructive interference, attack profit is reduced
    interference_result = (1 + cos_val) / 2

    return max(0.0, interference_result)


# =============================================================================
# Simulation Results
# =============================================================================

@dataclass
class SimulationResult:
    """Results from MEV simulation."""
    n_rounds: int
    traditional_profits: List[float]
    esm_profits: List[float]
    phases_assigned: List[DiscretePhase]
    delays_ms: List[float]
    victim_amounts: List[float]

    @property
    def traditional_total(self) -> float:
        """Total profit in traditional chain."""
        return sum(self.traditional_profits)

    @property
    def esm_total(self) -> float:
        """Total profit in ESM chain."""
        return sum(self.esm_profits)

    @property
    def profit_reduction(self) -> float:
        """Percentage reduction in attacker profit."""
        if self.traditional_total == 0:
            return 0.0
        return (1 - self.esm_total / self.traditional_total) * 100

    @property
    def average_interference_factor(self) -> float:
        """Average interference factor across all attacks."""
        factors = [calculate_interference_factor(p) for p in self.phases_assigned]
        return np.mean(factors)

    def get_stats(self) -> dict:
        """Get summary statistics."""
        return {
            "n_rounds": self.n_rounds,
            "traditional_total": self.traditional_total,
            "esm_total": self.esm_total,
            "profit_reduction_percent": self.profit_reduction,
            "average_interference_factor": self.average_interference_factor,
            "phase_distribution": self._get_phase_distribution(),
        }

    def _get_phase_distribution(self) -> dict:
        """Get distribution of assigned phases."""
        dist = {p: 0 for p in DiscretePhase}
        for p in self.phases_assigned:
            dist[p] += 1
        return {p.name: count for p, count in dist.items() if count > 0}


# =============================================================================
# Simulation Functions
# =============================================================================

def simulate_sandwich_attack(
    n_rounds: int = 1000,
    mev_extraction_rate: float = 0.03,
    min_victim_amount: float = 1000,
    max_victim_amount: float = 10000,
    attack_delay_mean: float = 50,
    attack_delay_std: float = 30,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Simulate sandwich attacks comparing traditional chain vs ESM.

    Args:
        n_rounds: Number of attack rounds to simulate
        mev_extraction_rate: Fraction of victim amount extracted (default 3%)
        min_victim_amount: Minimum victim transaction amount
        max_victim_amount: Maximum victim transaction amount
        attack_delay_mean: Mean attack delay in ms
        attack_delay_std: Std dev of attack delay in ms
        seed: Random seed for reproducibility

    Returns:
        SimulationResult with detailed results
    """
    if seed is not None:
        np.random.seed(seed)

    traditional_profits = []
    esm_profits = []
    phases_assigned = []
    delays_ms = []
    victim_amounts = []

    for _ in range(n_rounds):
        # Generate victim transaction
        victim_amount = np.random.uniform(min_victim_amount, max_victim_amount)
        victim_amounts.append(victim_amount)

        # Generate attack timing (attacker tries to be fast)
        delay = max(0, np.random.normal(attack_delay_mean, attack_delay_std))
        delays_ms.append(delay)

        # Traditional chain comparison baseline
        # NOTE: This is a simplified assumption for comparison purposes.
        # On traditional chains without MEV protection, front-running attacks
        # are estimated to succeed ~90-95% of the time in favorable conditions.
        # Real-world success rates vary by: DEX design, network congestion,
        # private mempool usage, and Flashbots-style MEV protection adoption.
        # For this simulation, we assume 100% success to show maximum MEV exposure.
        traditional_profit = victim_amount * mev_extraction_rate
        traditional_profits.append(traditional_profit)

        # ESM chain: interference based on timing
        phase = assign_phase_by_delay(delay)
        phases_assigned.append(phase)

        interference_factor = calculate_interference_factor(phase)
        esm_profit = traditional_profit * interference_factor
        esm_profits.append(esm_profit)

    return SimulationResult(
        n_rounds=n_rounds,
        traditional_profits=traditional_profits,
        esm_profits=esm_profits,
        phases_assigned=phases_assigned,
        delays_ms=delays_ms,
        victim_amounts=victim_amounts,
    )


def simulate_with_psc(
    n_rounds: int = 100,
    seed: Optional[int] = None,
) -> List[dict]:
    """
    Simulate MEV attacks using actual PSC interference calculation.

    This provides a more detailed simulation using the full PSC mechanism.

    Args:
        n_rounds: Number of rounds
        seed: Random seed

    Returns:
        List of round results with detailed interference data
    """
    if seed is not None:
        np.random.seed(seed)

    results = []

    for round_num in range(n_rounds):
        # Create PSC for this transaction pair
        psc = create_psc(f"round_{round_num}")

        # Victim transaction (magnitude 0.7, phase 0°)
        victim_amount = np.random.uniform(1000, 10000)
        victim_branch = create_victim_tx(victim_amount, magnitude=0.7)
        psc.add_branch(victim_branch)

        # Attacker transaction
        delay = max(0, np.random.normal(50, 30))
        attack_phase = assign_phase_by_delay(delay)
        attacker_branch = create_attacker_tx(
            victim_amount * 0.03,
            phase=attack_phase,
            magnitude=0.7,
            delay_ms=int(delay)
        )

        # Use same state_id to cause interference
        attacker_branch.state_id = victim_branch.state_id
        psc.add_branch(attacker_branch)

        # Calculate interference
        interference = psc.calculate_interference()
        probs = psc.get_probabilities()

        # Get interference report
        report = psc.get_interference_report()

        results.append({
            "round": round_num,
            "victim_amount": victim_amount,
            "delay_ms": delay,
            "attack_phase": attack_phase.name,
            "interference_impact": psc.interference_impact(),
            "probabilities": probs,
            "report": report,
        })

    return results


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_timing_sensitivity(
    delay_range: Tuple[float, float] = (0, 2000),
    n_points: int = 100,
) -> dict:
    """
    Analyze how interference varies with attack timing.

    Args:
        delay_range: (min, max) delay in ms
        n_points: Number of points to sample

    Returns:
        Dictionary with delay values and corresponding factors
    """
    delays = np.linspace(delay_range[0], delay_range[1], n_points)
    phases = [assign_phase_by_delay(d) for d in delays]
    factors = [calculate_interference_factor(p) for p in phases]

    return {
        "delays_ms": delays.tolist(),
        "phases": [p.name for p in phases],
        "interference_factors": factors,
        "profit_retention": factors,  # Same as interference factor
    }


def compare_attack_strategies(
    strategies: dict,
    n_rounds: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare different attack timing strategies.

    Args:
        strategies: Dict of strategy_name -> (mean_delay, std_delay)
        n_rounds: Rounds per strategy
        seed: Random seed

    Returns:
        Comparison results
    """
    results = {}

    for name, (mean_delay, std_delay) in strategies.items():
        sim = simulate_sandwich_attack(
            n_rounds=n_rounds,
            attack_delay_mean=mean_delay,
            attack_delay_std=std_delay,
            seed=seed,
        )
        results[name] = sim.get_stats()

    return results


# =============================================================================
# Quick Demo
# =============================================================================

def run_demo():
    """Run a quick demonstration of the MEV simulation."""
    print("=" * 60)
    print("ESM MEV Resistance Simulation Demo")
    print("=" * 60)

    # Run simulation
    result = simulate_sandwich_attack(n_rounds=1000, seed=42)
    stats = result.get_stats()

    print(f"\nSimulation: {stats['n_rounds']} attack rounds")
    print("-" * 40)
    print(f"Traditional Chain Total Profit: ${stats['traditional_total']:,.2f}")
    print(f"ESM Chain Total Profit:         ${stats['esm_total']:,.2f}")
    print(f"Profit Reduction:               {stats['profit_reduction_percent']:.1f}%")
    print(f"Average Interference Factor:    {stats['average_interference_factor']:.3f}")

    print("\nPhase Distribution:")
    for phase, count in stats['phase_distribution'].items():
        pct = count / stats['n_rounds'] * 100
        print(f"  {phase}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)

    return result


if __name__ == "__main__":
    run_demo()
