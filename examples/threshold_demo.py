#!/usr/bin/env python3
"""
ESM Threshold Reveal and Backup Validator Demo

This script demonstrates the v5.2 Threshold Reveal mechanism
and backup validator system.

Run:
    python examples/threshold_demo.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from esm.simulation.threshold_reveal import (
    REVEAL_THRESHOLD,
    NON_REVEAL_SLASH_RATE,
    BACKUP_NON_REVEAL_SLASH_RATE,
    analyze_threshold_sensitivity,
    simulate_collapse_protocol,
    simulate_adversarial_scenario,
)
from esm.simulation.backup_validator import (
    BACKUP_VALIDATOR_COUNT,
    BACKUP_REWARD_SHARE,
    simulate_backup_scenario,
    analyze_backup_reliability,
)


def demo_threshold_basics():
    """Demonstrate basic threshold reveal mechanics."""
    print("\n" + "=" * 60)
    print("1. Threshold Reveal Basics")
    print("=" * 60)

    print(f"\nProtocol Parameters:")
    print(f"  - Reveal Threshold: {REVEAL_THRESHOLD * 100:.0f}%")
    print(f"  - Non-Reveal Slash Rate: {NON_REVEAL_SLASH_RATE * 100:.0f}%")
    print(f"  - Backup Slash Rate: {BACKUP_NON_REVEAL_SLASH_RATE * 100:.0f}%")
    print(f"  - Backup Validator Count: {BACKUP_VALIDATOR_COUNT}")
    print(f"  - Backup Reward Share: {BACKUP_REWARD_SHARE * 100:.0f}%")

    print("\n--- High Reveal Rate Scenario (90%) ---")
    result = simulate_collapse_protocol(
        n_validators=20,
        reveal_rate=0.9,
        seed=42,
    )

    print(f"  Validators: {result['n_validators']}")
    print(f"  Target Reveal Rate: {result['reveal_rate_target'] * 100:.0f}%")
    print(f"  Actual Reveal Rate: {result['actual_reveal_rate'] * 100:.1f}%")
    print(f"  Status: {result['result'].status.value}")
    print(f"  Phase Completed: {result['phase']}")
    print(f"  Slashed Validators: {len(result['result'].slashed_validators)}")
    print(f"  Total Slashed: {result['result'].total_slashed:,}")

    print("\n--- Low Reveal Rate Scenario (50%) ---")
    result = simulate_collapse_protocol(
        n_validators=20,
        reveal_rate=0.5,
        seed=42,
    )

    print(f"  Actual Reveal Rate: {result['actual_reveal_rate'] * 100:.1f}%")
    print(f"  Status: {result['result'].status.value}")
    print(f"  Extension Needed: {result['extension_needed']}")
    print(f"  Backup Activated: {result['backup_activated']}")
    print(f"  Slashed Validators: {len(result['result'].slashed_validators)}")
    print(f"  Total Slashed: {result['result'].total_slashed:,}")


def demo_threshold_sensitivity():
    """Analyze how reveal rate affects success probability."""
    print("\n" + "=" * 60)
    print("2. Threshold Sensitivity Analysis")
    print("=" * 60)

    print("\nRunning 1000 simulations per reveal rate...")
    analysis = analyze_threshold_sensitivity(
        n_validators=20,
        n_simulations=1000,
        seed=42,
    )

    print("\nReveal Rate | Success Rate | Backup Rate | Avg Slashed")
    print("-" * 55)

    for rate, stats in sorted(analysis.items()):
        bar = "█" * int(stats["success_rate"] * 20)
        print(f"    {rate * 100:5.0f}%  |  {bar:20s} {stats['success_rate'] * 100:5.1f}%  |  "
              f"{stats['backup_rate'] * 100:4.1f}%  |  {stats['avg_slashed']:,.0f}")

    print("\n📊 Insight: Success rate increases sharply above 67% threshold")


def demo_adversarial_scenarios():
    """Demonstrate system resilience to adversarial validators."""
    print("\n" + "=" * 60)
    print("3. Adversarial Validator Scenarios")
    print("=" * 60)

    scenarios = [
        {"n_honest": 17, "n_adversarial": 3, "name": "15% adversarial"},
        {"n_honest": 15, "n_adversarial": 5, "name": "25% adversarial"},
        {"n_honest": 12, "n_adversarial": 8, "name": "40% adversarial"},
        {"n_honest": 10, "n_adversarial": 10, "name": "50% adversarial"},
    ]

    print("\nScenario         | Reveal Ratio | Status         | Slashed")
    print("-" * 65)

    for scenario in scenarios:
        result = simulate_adversarial_scenario(
            n_honest=scenario["n_honest"],
            n_adversarial=scenario["n_adversarial"],
            honest_reveal_rate=0.95,
            adversarial_reveal_rate=0.0,
            seed=42,
        )

        status_icon = "✅" if result["is_successful"] else "❌"
        print(f"  {scenario['name']:14s}  |    {result['revealed_ratio'] * 100:5.1f}%   | "
              f"{status_icon} {result['status']:14s} | {result['total_slashed']:,}")

    print("\n📊 Insight: System remains secure up to ~33% adversarial validators")


def demo_backup_system():
    """Demonstrate backup validator activation."""
    print("\n" + "=" * 60)
    print("4. Backup Validator System")
    print("=" * 60)

    print("\nSimulating primary validator failure (50% reveal rate)...")
    result = simulate_backup_scenario(
        n_primary=20,
        n_backup_candidates=30,
        primary_reveal_rate=0.5,
        backup_reveal_rate=0.9,
        seed=42,
    )

    print("\nPrimary Validators:")
    print(f"  - Total: {result['primary']['n_validators']}")
    print(f"  - Revealed: {result['primary']['revealed_count']}")
    print(f"  - Failed: {result['primary']['failed_count']}")
    print(f"  - Reveal Ratio: {result['primary']['revealed_ratio'] * 100:.1f}%")

    print("\nBackup Validators:")
    print(f"  - Candidates: {result['backup']['n_candidates']}")
    print(f"  - Selected: {result['backup']['selected_count']}")
    print(f"  - Participated: {result['backup']['participating_count']}")
    print(f"  - Participation Rate: {result['backup']['participation_rate'] * 100:.1f}%")

    print("\nEconomics:")
    print(f"  - Slashed from Primary: {result['economics']['slashed_amount']:,}")
    print(f"  - Backup Reward Pool: {result['economics']['reward_pool']:,}")
    print(f"  - Rewards Distributed: {result['economics']['rewards_distributed']:,}")

    print(f"\nOutcome: {result['outcome']['status']}")
    print(f"Success: {'✅ Yes' if result['outcome']['is_successful'] else '❌ No'}")


def demo_backup_reliability():
    """Analyze backup system reliability across scenarios."""
    print("\n" + "=" * 60)
    print("5. Backup System Reliability Analysis")
    print("=" * 60)

    print("\nRunning 500 simulations per scenario...")
    analysis = analyze_backup_reliability(n_simulations=500, seed=42)

    print("\nScenario                          | Success | Avg Slashed | Avg Rewards")
    print("-" * 75)

    for key, stats in sorted(analysis.items()):
        print(f"  {key:32s} | {stats['success_rate'] * 100:5.1f}%  |   "
              f"{stats['avg_slashed']:7,.0f}  |   {stats['avg_rewards']:7,.0f}")


def demo_litepaper_data():
    """Generate data for Litepaper inclusion."""
    print("\n" + "=" * 60)
    print("6. Litepaper Data Summary")
    print("=" * 60)

    # Threshold analysis
    threshold_analysis = analyze_threshold_sensitivity(
        n_validators=20,
        n_simulations=2000,
        seed=42,
    )

    # At exactly 67% threshold
    at_threshold = threshold_analysis.get(0.67, {})

    # At 80% reveal rate
    at_80 = threshold_analysis.get(0.8, {})

    # Backup analysis
    backup_analysis = analyze_backup_reliability(n_simulations=1000, seed=42)

    print("\n📊 Key Metrics for Litepaper:")
    print("-" * 40)
    print(f"\n1. Threshold Reveal (67%):")
    print(f"   - Success Rate at Threshold: {at_threshold.get('success_rate', 0) * 100:.1f}%")
    print(f"   - Success Rate at 80%: {at_80.get('success_rate', 0) * 100:.1f}%")
    print(f"   - Backup Activation Rate at 50%: {threshold_analysis.get(0.5, {}).get('backup_rate', 0) * 100:.1f}%")

    print(f"\n2. Slashing Economics:")
    print(f"   - Non-Reveal Slash: {NON_REVEAL_SLASH_RATE * 100:.0f}%")
    print(f"   - Backup Activation Slash: {BACKUP_NON_REVEAL_SLASH_RATE * 100:.0f}%")
    print(f"   - Backup Reward Share: {BACKUP_REWARD_SHARE * 100:.0f}%")

    print(f"\n3. Backup System Reliability:")
    key_50_90 = "primary_0.5_backup_0.9"
    if key_50_90 in backup_analysis:
        print(f"   - 50% Primary + 90% Backup: {backup_analysis[key_50_90]['success_rate'] * 100:.1f}% success")

    print(f"\n4. Adversarial Resilience:")
    adv_result = simulate_adversarial_scenario(
        n_honest=14,
        n_adversarial=6,  # 30% adversarial
        honest_reveal_rate=0.95,
        adversarial_reveal_rate=0.0,
        seed=42,
    )
    print(f"   - 30% Adversarial: {'✅ Secure' if adv_result['is_successful'] else '❌ Vulnerable'}")


def generate_visualizations(output_dir: str = "output"):
    """Generate and save all v5.2 visualizations."""
    print("\n" + "=" * 60)
    print("7. Generating Visualizations")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available, skipping visualizations")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Threshold Sensitivity Chart
    print("  - Generating threshold_sensitivity.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    analysis = analyze_threshold_sensitivity(n_validators=20, n_simulations=500, seed=42)
    rates = sorted(analysis.keys())
    success_rates = [analysis[r]["success_rate"] * 100 for r in rates]
    backup_rates = [analysis[r]["backup_rate"] * 100 for r in rates]

    x = [r * 100 for r in rates]
    ax.bar([xi - 2 for xi in x], success_rates, width=4, label="Success Rate", color="#2ecc71", alpha=0.8)
    ax.bar([xi + 2 for xi in x], backup_rates, width=4, label="Backup Activation", color="#e74c3c", alpha=0.8)
    ax.axvline(x=67, color="#3498db", linestyle="--", linewidth=2, label="67% Threshold")

    ax.set_xlabel("Reveal Rate (%)", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("ESM v5.2: Threshold Reveal Sensitivity", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(output_path / "threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path / 'threshold_sensitivity.png'}")

    # 2. Adversarial Resilience Chart
    print("  - Generating adversarial_resilience.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    adversarial_pcts = [10, 20, 30, 40, 50]
    reveal_ratios = []
    statuses = []

    for adv_pct in adversarial_pcts:
        n_adv = adv_pct
        n_honest = 100 - adv_pct
        result = simulate_adversarial_scenario(
            n_honest=n_honest,
            n_adversarial=n_adv,
            honest_reveal_rate=0.95,
            adversarial_reveal_rate=0.0,
            seed=42,
        )
        reveal_ratios.append(result["revealed_ratio"] * 100)
        statuses.append(result["is_successful"])

    colors = ["#2ecc71" if s else "#e74c3c" for s in statuses]
    bars = ax.bar(adversarial_pcts, reveal_ratios, color=colors, alpha=0.8, edgecolor="black")

    ax.axhline(y=67, color="#3498db", linestyle="--", linewidth=2, label="67% Threshold")
    ax.set_xlabel("Adversarial Validators (%)", fontsize=12)
    ax.set_ylabel("Effective Reveal Ratio (%)", fontsize=12)
    ax.set_title("ESM v5.2: Adversarial Resilience", fontsize=14, fontweight="bold")
    ax.legend()

    # Add labels
    for bar, ratio, success in zip(bars, reveal_ratios, statuses):
        label = "✓" if success else "✗"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha="center", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path / "adversarial_resilience.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path / 'adversarial_resilience.png'}")

    # 3. Backup Activation Chart
    print("  - Generating backup_activation.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Primary reveal rate effect
    primary_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
    backup_success = []
    slashed_amounts = []

    for rate in primary_rates:
        result = simulate_backup_scenario(
            n_primary=20,
            primary_reveal_rate=rate,
            backup_reveal_rate=0.9,
            seed=42,
        )
        backup_success.append(100 if result["outcome"]["is_successful"] else 0)
        slashed_amounts.append(result["economics"]["slashed_amount"])

    x = [r * 100 for r in primary_rates]
    ax1.bar(x, backup_success, color="#2ecc71", alpha=0.8, label="Success")
    ax1.set_xlabel("Primary Reveal Rate (%)", fontsize=12)
    ax1.set_ylabel("Collapse Success Rate (%)", fontsize=12)
    ax1.set_title("Backup System Success", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 110)

    # Right: Slashing economics
    ax2.bar(x, slashed_amounts, color="#e74c3c", alpha=0.8)
    ax2.set_xlabel("Primary Reveal Rate (%)", fontsize=12)
    ax2.set_ylabel("Total Slashed (units)", fontsize=12)
    ax2.set_title("Slashing by Primary Reveal Rate", fontsize=12, fontweight="bold")

    fig.suptitle("ESM v5.2: Backup Validator System", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path / "backup_activation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path / 'backup_activation.png'}")

    # 4. Slashing Economics Chart
    print("  - Generating slashing_economics.png...")
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Normal\nNon-Reveal", "Backup\nActivated"]
    slash_rates = [NON_REVEAL_SLASH_RATE * 100, BACKUP_NON_REVEAL_SLASH_RATE * 100]
    colors = ["#f39c12", "#c0392b"]

    bars = ax.bar(categories, slash_rates, color=colors, alpha=0.8, edgecolor="black", width=0.5)

    # Add value labels
    for bar, rate in zip(bars, slash_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel("Slash Rate (%)", fontsize=12)
    ax.set_title("ESM v5.2: Slashing Economics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 60)

    # Add annotation
    ax.annotate(f"Backup Reward: {BACKUP_REWARD_SHARE * 100:.0f}% of slashed",
                xy=(1, 50), fontsize=11, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.3))

    fig.tight_layout()
    fig.savefig(output_path / "slashing_economics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path / 'slashing_economics.png'}")

    print(f"\n  All v5.2 visualizations saved to: {output_path.absolute()}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("  ESM v5.2 Threshold Reveal Demo")
    print("  Based on Whitepaper Section 5.2-5.6")
    print("=" * 60)

    demo_threshold_basics()
    demo_threshold_sensitivity()
    demo_adversarial_scenarios()
    demo_backup_system()
    demo_backup_reliability()
    demo_litepaper_data()

    # Generate visualizations
    try:
        generate_visualizations()
    except Exception as e:
        print(f"\n  Visualization generation failed: {e}")
        print("  (This may happen if matplotlib backend is not configured)")

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
