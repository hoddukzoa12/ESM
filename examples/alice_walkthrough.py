#!/usr/bin/env python3
"""
Alice Walkthrough Demo

Complete step-by-step demonstration of "Alice sends 100 ESM" with real numbers.
Based on ESM Whitepaper v5.3 Section 8.1.

This script demonstrates:
1. Simple transfer walkthrough
2. MEV DEX swap walkthrough (fast attack - blocked)
3. MEV DEX swap walkthrough (slow attack - may succeed)
4. Timing sensitivity comparison

Usage:
    python examples/alice_walkthrough.py
"""

import sys
import os
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esm.simulation.walkthrough import (
    TransactionWalkthrough,
    run_mev_walkthrough,
    compare_timing_scenarios,
    WalkthroughType,
)


def print_section(title: str, char: str = "="):
    """Print a section header."""
    width = 75
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print()


def demo_simple_transfer():
    """Demonstrate simple transfer walkthrough."""
    print_section("DEMO 1: Simple Transfer")

    walkthrough = TransactionWalkthrough()
    result = walkthrough.simulate_simple_transfer(
        amount=Decimal("100"),
        sender="Alice",
        recipient="Bob",
    )

    print(walkthrough.format_walkthrough(result))

    print("\nKey Observations:")
    print("  - Single branch PSC (no interference)")
    print("  - Immediate deterministic collapse")
    print("  - Full deposit refund after success")
    print(f"  - Net cost: {result.total_gas_paid} ESM (gas only)")


def demo_mev_attack_blocked():
    """Demonstrate MEV attack being blocked."""
    print_section("DEMO 2: MEV Attack (Fast Attack - BLOCKED)")

    print("Scenario: Alice swaps 100 ESM -> TOKEN")
    print("          Bot attempts front-run with 50ms delay")
    print()

    result = run_mev_walkthrough(
        alice_amount=Decimal("100"),
        attack_delay_ms=50,  # Fast attack
        market_price=Decimal("9.5"),
        verbose=True,
    )

    print("\nKey Observations:")
    print("  - Bot's fast attack (50ms) triggers P180 phase assignment")
    print("  - P180 causes complete DESTRUCTIVE interference with Alice's P0")
    print("  - Bot's probability drops to 0%, Alice wins with 100%")
    print(f"  - Bot LOSES: gas ({result.total_gas_paid / 2} ESM) + deposit")
    print("  - MEV extraction: $0 (vs ~$3 on traditional chain)")


def demo_mev_attack_slow():
    """Demonstrate MEV attack with slow timing."""
    print_section("DEMO 3: MEV Attack (Slow Attack - May Succeed)")

    print("Scenario: Alice swaps 100 ESM -> TOKEN")
    print("          Bot attempts front-run with 1500ms delay")
    print()

    walkthrough = TransactionWalkthrough()
    result = walkthrough.simulate_mev_dex_swap(
        alice_input=Decimal("100"),
        bot_input=Decimal("1000"),
        market_price=Decimal("9.5"),
        attack_delay_ms=1500,  # Slow attack
        seed=42,
    )

    print(walkthrough.format_walkthrough(result))

    print("\nKey Observations:")
    print("  - Bot's slow attack (1500ms) gets P0 phase (same as Alice)")
    print("  - No interference effect (both constructive)")
    print("  - Attack becomes regular competition")
    print("  - ESM's protection requires timely detection")


def demo_timing_comparison():
    """Compare different attack timing scenarios."""
    print_section("DEMO 4: Attack Timing Sensitivity Analysis")

    print("Comparing bot profit across different attack timings:")
    print()

    results = compare_timing_scenarios(alice_amount=Decimal("100"))

    # Table header
    print("  {:<20} {:<10} {:<15} {:<15} {:<15}".format(
        "Timing", "Delay", "Phase", "Bot Profit", "Alice Wins?"
    ))
    print("  " + "-" * 70)

    for name, result in results.items():
        # Extract delay from name
        delay = name.split("_")[1]

        # Find bot's phase
        bot_phase = "P0"
        for step in result.steps:
            if step.actor == "MEV Bot":
                bot_phase = step.phase.name
                break

        alice_wins = "YES" if result.alice_receives > 0 else "NO"

        print("  {:<20} {:<10} {:<15} {:<15.4f} {:<15}".format(
            name, delay, bot_phase, float(result.bot_profit), alice_wins
        ))

    print()
    print("Phase Assignment Rules:")
    print("  - < 100ms:   P180 (Counter)         -> Full destructive interference")
    print("  - < 500ms:   P135 (PartialCounter)  -> Partial destructive")
    print("  - < 1000ms:  P90  (Independent)     -> Orthogonal (no interference)")
    print("  - >= 1000ms: P0   (Normal)          -> Constructive (no protection)")


def demo_fee_breakdown():
    """Show detailed fee breakdown."""
    print_section("DEMO 5: Fee Structure Breakdown")

    print("ESM v5.3 Fee Structure (Section 7.7):")
    print()
    print("  Token Units:")
    print("    1 ESM = 1,000,000 amp (smallest unit)")
    print()
    print("  Fee Types:")
    print("    - PSC Creation:      0.00001 ESM (10 amp)")
    print("    - Branch Fee Base:   0.0001 ESM (100 amp)")
    print("    - Branch Fee/KB:     0.00001 ESM per KB of data")
    print("    - Read Fee:          0.000001 ESM (1 amp)")
    print("    - Gas Fee (est):     0.01 ESM (10,000 amp)")
    print()
    print("  Interference Deposit:")
    print("    - Rate: 0.1% of transaction value")
    print("    - Buffer: +20% for cost variability")
    print("    - Formula: value * 0.001 * 1.20")
    print()

    # Calculate example
    tx_value = Decimal("100")
    deposit_rate = Decimal("0.001")
    buffer = Decimal("1.20")
    deposit = tx_value * deposit_rate * buffer

    print("  Example (100 ESM transaction):")
    print(f"    Base deposit:    100 * 0.001 = 0.1 ESM")
    print(f"    With buffer:     0.1 * 1.20 = {deposit} ESM")
    print(f"    Total upfront:   {deposit} ESM + 0.01 ESM gas = {deposit + Decimal('0.01')} ESM")


def demo_interference_math():
    """Show the interference math in detail."""
    print_section("DEMO 6: Interference Math Deep Dive")

    print("8-Phase Discrete Amplitude System:")
    print()
    print("  Phase Table:")
    print("  {:<8} {:<10} {:<10} {:<10}".format("Phase", "Degrees", "cos(theta)", "sin(theta)"))
    print("  " + "-" * 40)

    phases = [
        ("P0", 0, 1.0, 0.0),
        ("P45", 45, 0.7071, 0.7071),
        ("P90", 90, 0.0, 1.0),
        ("P135", 135, -0.7071, 0.7071),
        ("P180", 180, -1.0, 0.0),
        ("P225", 225, -0.7071, -0.7071),
        ("P270", 270, 0.0, -1.0),
        ("P315", 315, 0.7071, -0.7071),
    ]

    for name, deg, cos_v, sin_v in phases:
        print(f"  {name:<8} {deg:<10} {cos_v:<10.4f} {sin_v:<10.4f}")

    print()
    print("  Interference Example (Alice P0, Bot P180):")
    print()
    print("    Alice: alpha_A = 1.0 * (cos(0) + i*sin(0)) = 1.0 + 0i")
    print("    Bot:   alpha_B = 1.0 * (cos(180) + i*sin(180)) = -1.0 + 0i")
    print()
    print("    Combined: alpha_total = alpha_A + alpha_B")
    print("                        = (1.0 + 0i) + (-1.0 + 0i)")
    print("                        = 0 + 0i")
    print()
    print("    Probability: |alpha_total|^2 = |0|^2 = 0")
    print()
    print("  Result: Bot's transaction has 0% probability of being selected!")


def main():
    """Run all demos."""
    print("\n" + "=" * 75)
    print("         ESM v5.4 - Alice Walkthrough Complete Demo")
    print("         Based on Whitepaper v5.3 Section 8.1")
    print("=" * 75)

    # Run demos
    demo_simple_transfer()
    demo_mev_attack_blocked()
    demo_mev_attack_slow()
    demo_timing_comparison()
    demo_fee_breakdown()
    demo_interference_math()

    print_section("DEMO COMPLETE")
    print("All walkthrough demos completed successfully.")
    print()
    print("Summary:")
    print("  - ESM uses 8-phase discrete amplitudes for deterministic interference")
    print("  - Fast MEV attacks (< 100ms) receive opposite phase (P180)")
    print("  - P180 causes complete destructive interference -> attack blocked")
    print("  - Attackers lose gas + deposits when blocked")
    print("  - Fee structure ensures predictable costs with 20% buffer")
    print()


if __name__ == "__main__":
    main()
