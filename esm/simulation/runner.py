"""
Simulation Runner

CLI entry point for running ESM simulations.
"""

import argparse
import json
from pathlib import Path

from esm.simulation.mev_scenario import (
    simulate_sandwich_attack,
    analyze_timing_sensitivity,
    compare_attack_strategies,
)


def main():
    """Main entry point for simulation runner."""
    parser = argparse.ArgumentParser(
        description="ESM Simulator - MEV Resistance Analysis"
    )

    parser.add_argument(
        "-n", "--rounds",
        type=int,
        default=1000,
        help="Number of simulation rounds (default: 1000)"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )

    parser.add_argument(
        "--mev-rate",
        type=float,
        default=0.03,
        help="MEV extraction rate (default: 0.03 = 3%%)"
    )

    parser.add_argument(
        "--delay-mean",
        type=float,
        default=50,
        help="Mean attack delay in ms (default: 50)"
    )

    parser.add_argument(
        "--delay-std",
        type=float,
        default=30,
        help="Std dev of attack delay in ms (default: 30)"
    )

    parser.add_argument(
        "--analyze-timing",
        action="store_true",
        help="Run timing sensitivity analysis"
    )

    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare different attack strategies"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ESM Simulator - MEV Resistance Analysis")
    print("=" * 60)

    if args.analyze_timing:
        print("\nRunning timing sensitivity analysis...")
        results = analyze_timing_sensitivity()
        print(f"Analyzed {len(results['delays_ms'])} delay points")

    elif args.compare_strategies:
        print("\nComparing attack strategies...")
        strategies = {
            "ultra_fast": (10, 5),
            "fast": (50, 20),
            "normal": (200, 50),
            "slow": (500, 100),
            "very_slow": (1500, 300),
        }
        results = compare_attack_strategies(
            strategies,
            n_rounds=args.rounds,
            seed=args.seed
        )

        print("\nStrategy Comparison:")
        print("-" * 60)
        for name, stats in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Profit Reduction: {stats['profit_reduction_percent']:.1f}%")
            print(f"  Avg Interference: {stats['average_interference_factor']:.3f}")

    else:
        print(f"\nRunning {args.rounds} simulation rounds...")
        result = simulate_sandwich_attack(
            n_rounds=args.rounds,
            mev_extraction_rate=args.mev_rate,
            attack_delay_mean=args.delay_mean,
            attack_delay_std=args.delay_std,
            seed=args.seed,
        )
        stats = result.get_stats()
        results = stats

        print("\nResults:")
        print("-" * 40)
        print(f"Traditional Profit: ${stats['traditional_total']:,.2f}")
        print(f"ESM Profit:         ${stats['esm_total']:,.2f}")
        print(f"Reduction:          {stats['profit_reduction_percent']:.1f}%")

        print("\nPhase Distribution:")
        for phase, count in stats['phase_distribution'].items():
            pct = count / args.rounds * 100
            print(f"  {phase}: {pct:.1f}%")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
