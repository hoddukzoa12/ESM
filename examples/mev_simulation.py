#!/usr/bin/env python3
"""
ESM MEV Resistance Simulation Example

This script demonstrates the MEV (Maximal Extractable Value) resistance
mechanism in ESM using 8-phase interference.

Run:
    python examples/mev_simulation.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude, calculate_interference_result
from esm.core.branch import Branch, create_victim_tx, create_attacker_tx
from esm.core.psc import PSC, create_psc
from esm.simulation.mev_scenario import (
    simulate_sandwich_attack,
    analyze_timing_sensitivity,
    compare_attack_strategies,
    assign_phase_by_delay,
)
from esm.visualization.amplitude_plot import (
    plot_amplitude_polar,
    plot_interference_demo,
    plot_all_interference_patterns,
)
from esm.visualization.mev_plot import (
    plot_mev_comparison,
    plot_mev_cumulative,
    plot_timing_sensitivity,
    create_dashboard,
)


def demo_basic_interference():
    """Demonstrate basic interference between two amplitudes."""
    print("\n" + "=" * 60)
    print("1. Basic Interference Demo")
    print("=" * 60)

    # Create two amplitudes with opposite phases
    victim = DiscreteAmplitude(0.7, DiscretePhase.P0)
    attacker = DiscreteAmplitude(0.7, DiscretePhase.P180)

    print(f"\nVictim amplitude:   {victim}")
    print(f"Attacker amplitude: {attacker}")

    # Calculate interference
    result = victim.add(attacker)

    print(f"\nAfter interference: {result}")
    print(f"Result magnitude:   {result.magnitude:.4f}")
    print(f"Result probability: {result.probability():.4f}")

    # Compare probabilities
    prob_before = victim.probability() + attacker.probability()
    prob_after = result.probability()

    print(f"\nProbability before: {prob_before:.4f}")
    print(f"Probability after:  {prob_after:.4f}")
    print(f"Reduction:          {(1 - prob_after/prob_before) * 100:.1f}%")


def demo_psc_interference():
    """Demonstrate PSC with multiple interfering branches."""
    print("\n" + "=" * 60)
    print("2. PSC Interference Demo")
    print("=" * 60)

    psc = create_psc("mev-demo")

    # Add victim transaction
    victim = create_victim_tx(amount=10000, magnitude=0.7)
    psc.add_branch(victim)
    print(f"\nAdded victim tx: amount=$10,000, phase=P0")

    # Add attacker transaction (simulating front-run)
    # Attacker tries to be fast (<100ms) so gets P180
    delay_ms = 50  # Very fast attack
    attack_phase = assign_phase_by_delay(delay_ms)

    attacker = create_attacker_tx(
        amount=300,  # 3% extraction
        phase=attack_phase,
        magnitude=0.7,
        delay_ms=delay_ms
    )
    # Same state_id to cause interference
    attacker.state_id = victim.state_id
    psc.add_branch(attacker)

    print(f"Added attacker tx: delay={delay_ms}ms, phase={attack_phase.name}")

    # Calculate interference
    print("\n--- Interference Results ---")
    report = psc.get_interference_report()

    for state_id, data in report['states'].items():
        print(f"\nState: {state_id}")
        print(f"  Branches: {data['branch_count']}")
        print(f"  Result magnitude: {data['result_amplitude'].magnitude:.4f}")
        print(f"  Interference type: {data['interference_type']}")
        print(f"  Interference effect: {data['interference_effect']:.2%}")


def demo_mev_simulation():
    """Run full MEV simulation and show results."""
    print("\n" + "=" * 60)
    print("3. MEV Attack Simulation (1000 rounds)")
    print("=" * 60)

    result = simulate_sandwich_attack(
        n_rounds=1000,
        mev_extraction_rate=0.03,
        attack_delay_mean=50,
        attack_delay_std=30,
        seed=42
    )

    stats = result.get_stats()

    print(f"\nSimulation completed: {stats['n_rounds']} rounds")
    print("-" * 40)
    print(f"Traditional Chain Profit: ${stats['traditional_total']:,.2f}")
    print(f"ESM Chain Profit:         ${stats['esm_total']:,.2f}")
    print(f"Attacker Profit Reduction: {stats['profit_reduction_percent']:.1f}%")
    print(f"Average Interference:      {stats['average_interference_factor']:.3f}")

    print("\nPhase Distribution:")
    for phase, count in sorted(stats['phase_distribution'].items()):
        pct = count / stats['n_rounds'] * 100
        bar = "█" * int(pct / 2)
        print(f"  {phase:5s}: {bar} {pct:.1f}%")

    return result


def demo_timing_analysis():
    """Analyze how attack timing affects profitability."""
    print("\n" + "=" * 60)
    print("4. Timing Sensitivity Analysis")
    print("=" * 60)

    analysis = analyze_timing_sensitivity(delay_range=(0, 2000), n_points=10)

    print("\nDelay (ms)  |  Phase  |  Profit Factor")
    print("-" * 40)

    for delay, phase, factor in zip(
        analysis['delays_ms'],
        analysis['phases'],
        analysis['interference_factors']
    ):
        bar = "█" * int(factor * 20)
        print(f"  {delay:7.0f}  |  {phase:5s}  |  {bar} {factor:.2f}")


def demo_strategy_comparison():
    """Compare different attack timing strategies."""
    print("\n" + "=" * 60)
    print("5. Attack Strategy Comparison")
    print("=" * 60)

    strategies = {
        "Ultra Fast (10ms)": (10, 5),
        "Fast (50ms)":       (50, 20),
        "Normal (200ms)":    (200, 50),
        "Slow (1500ms)":     (1500, 300),
    }

    results = compare_attack_strategies(strategies, n_rounds=1000, seed=42)

    print("\nStrategy           | Profit Reduction | Avg Interference")
    print("-" * 60)

    for name, stats in results.items():
        print(f"{name:18s} | {stats['profit_reduction_percent']:14.1f}% | "
              f"{stats['average_interference_factor']:.3f}")


def generate_visualizations(result, output_dir: str = "output"):
    """Generate and save all visualizations."""
    print("\n" + "=" * 60)
    print("6. Generating Visualizations")
    print("=" * 60)

    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. All interference patterns
    print("  - Generating interference patterns...")
    fig = plot_all_interference_patterns(magnitude=0.7)
    fig.savefig(output_path / "interference_patterns.png", dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path / 'interference_patterns.png'}")

    # 2. MEV comparison
    print("  - Generating MEV comparison...")
    fig = plot_mev_comparison(result)
    fig.savefig(output_path / "mev_comparison.png", dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path / 'mev_comparison.png'}")

    # 3. Cumulative profit
    print("  - Generating cumulative profit chart...")
    fig = plot_mev_cumulative(result)
    fig.savefig(output_path / "mev_cumulative.png", dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path / 'mev_cumulative.png'}")

    # 4. Timing sensitivity
    print("  - Generating timing sensitivity chart...")
    fig = plot_timing_sensitivity()
    fig.savefig(output_path / "timing_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path / 'timing_sensitivity.png'}")

    # 5. Full dashboard
    print("  - Generating dashboard...")
    fig = create_dashboard(result)
    fig.savefig(output_path / "dashboard.png", dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path / 'dashboard.png'}")

    print(f"\nAll visualizations saved to: {output_path.absolute()}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("  ESM MEV Resistance Simulation")
    print("  Based on Whitepaper v5.1 Section 7.1")
    print("=" * 60)

    # Run demos
    demo_basic_interference()
    demo_psc_interference()
    result = demo_mev_simulation()
    demo_timing_analysis()
    demo_strategy_comparison()

    # Generate visualizations
    try:
        generate_visualizations(result)
    except Exception as e:
        print(f"\nVisualization generation failed: {e}")
        print("(This may happen if matplotlib backend is not configured)")

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
