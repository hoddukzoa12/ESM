#!/usr/bin/env python3
"""
ESM Applications Demo

Demonstrates all 6 ESM applications from Whitepaper v5.3 Section 8.

Usage:
    python examples/applications_demo.py
"""

import sys
import os
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esm.applications.mev_dex import (
    MEVResistantDEX,
    SwapOrder,
    create_sample_pool,
    simulate_mev_scenario,
)
from esm.applications.privacy_transfer import (
    PrivacyTransferSystem,
    TransferMode,
)
from esm.applications.prediction_market import (
    PredictionMarket,
    Bet,
)
from esm.applications.insurance import (
    DecentralizedInsurance,
    PREMIUM_RATE,
)
from esm.applications.auction import (
    SealedBidAuction,
)
from esm.applications.quantum_nft import (
    QuantumNFTCollection,
    create_sample_states,
    RARITY_WEIGHTS,
)


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def demo_mev_dex():
    """Demo: MEV Resistant DEX."""
    print_header("Application 1: MEV Resistant DEX")

    print("ESM's phase-based interference neutralizes front-running attacks.")
    print("Fast transactions receive opposite phases, causing destructive interference.")
    print()

    # Compare different attack timings
    scenarios = [
        ("Fast attack (50ms)", 50),
        ("Medium attack (300ms)", 300),
        ("Slow attack (1500ms)", 1500),
    ]

    print("Simulating MEV attack scenarios:")
    print("-" * 50)
    print(f"{'Scenario':<25} {'Phase':<10} {'Winner':<10} {'MEV Blocked':<12}")
    print("-" * 50)

    for name, delay in scenarios:
        result = simulate_mev_scenario(
            victim_amount=Decimal("100"),
            attack_delay_ms=delay,
            seed=42,
        )
        phase = result.metadata["attacker_phase"]
        winner = result.selected_outcome
        blocked = "YES" if result.mev_blocked else "NO"

        print(f"{name:<25} {phase:<10} {winner:<10} {blocked:<12}")

    print()
    print("Key insight: Fast attacks (< 100ms) get P180 phase,")
    print("causing complete destructive interference with victim's P0 phase.")


def demo_privacy_transfer():
    """Demo: Privacy Transfer."""
    print_header("Application 2: Privacy Transfer")

    print("Transfers exist in superposition until collapse.")
    print("Observers cannot determine the actual recipient beforehand.")
    print()

    system = PrivacyTransferSystem()

    # Equal distribution
    print("Scenario 1: Equal Distribution (5 recipients)")
    print("-" * 50)

    transfer = system.create_private_transfer(
        sender="Alice",
        recipients=["Bob", "Charlie", "Diana", "Eve", "Frank"],
        amount=Decimal("100"),
        mode=TransferMode.EQUAL,
    )

    probs = system.get_recipient_probabilities(transfer)
    print("Probabilities before collapse:")
    for r, p in sorted(probs.items(), key=lambda x: x[0]):
        print(f"  {r}: {float(p)*100:.1f}%")

    result = system.collapse_transfer(transfer, seed=42)
    print(f"\nCollapse result: {result.recipient} receives 100 ESM")

    # Decoy mode
    print()
    print("Scenario 2: Decoy Mode (real=Bob, 3 decoys)")
    print("-" * 50)

    transfer2 = system.create_decoy_transfer(
        sender="Alice",
        real_recipient="Bob",
        decoy_recipients=["Charlie", "Diana", "Eve"],
        amount=Decimal("100"),
    )

    probs2 = system.get_recipient_probabilities(transfer2)
    print("Probabilities:")
    for r, p in sorted(probs2.items(), key=lambda x: -float(x[1])):
        print(f"  {r}: {float(p)*100:.1f}%")


def demo_prediction_market():
    """Demo: Prediction Market."""
    print_header("Application 3: Quantum Prediction Market")

    print("Market odds are represented as amplitudes.")
    print("Betting affects probabilities through constructive interference.")
    print()

    pm = PredictionMarket()

    market = pm.create_market(
        question="Will BTC reach $100k by end of year?",
        outcomes=["Yes", "No"],
        deadline_blocks=100,
    )

    print(f"Question: {market.question}")
    print(f"Outcomes: {market.outcomes}")

    initial_odds = pm.get_current_odds(market)
    print("\nInitial odds:")
    for outcome, prob in initial_odds.items():
        print(f"  {outcome}: {float(prob)*100:.1f}%")

    # Place bets
    bets = [
        ("Alice", "Yes", Decimal("200")),
        ("Bob", "No", Decimal("100")),
        ("Charlie", "Yes", Decimal("300")),
        ("Diana", "Yes", Decimal("150")),
    ]

    print("\nPlacing bets:")
    for bettor, outcome, amount in bets:
        bet = Bet(bettor=bettor, outcome=outcome, amount=amount)
        pm.place_bet(market, bet)
        print(f"  {bettor} bets {amount} ESM on '{outcome}'")

    new_odds = pm.get_current_odds(market)
    print("\nUpdated odds after betting:")
    for outcome, prob in sorted(new_odds.items(), key=lambda x: -float(x[1])):
        print(f"  {outcome}: {float(prob)*100:.1f}%")

    # Resolve
    pm.oracle_report(market, "Yes")
    result = pm.resolve_market(market)

    print(f"\nOracle reports: {result.resolved_outcome}")
    print(f"Total pool: {result.total_pool} ESM")
    print(f"Payout per winning share: {float(result.payout_per_share):.2f}x")


def demo_insurance():
    """Demo: Decentralized Insurance."""
    print_header("Application 4: Decentralized Insurance")

    print("Insurance payouts exist in superposition.")
    print("Oracle conditions adjust amplitudes, affecting payout probability.")
    print()

    insurance = DecentralizedInsurance()

    policy = insurance.create_policy(
        policyholder="Alice",
        coverage_amount=Decimal("10000"),
        conditions=[
            "Flight delayed > 4 hours",
            "Flight cancelled",
            "Lost baggage",
        ],
    )

    print(f"Policy: {policy.policy_id}")
    print(f"Coverage: {policy.coverage_amount} ESM")
    print(f"Conditions: {len(policy.conditions)}")

    prob = insurance.get_payout_probability(policy)
    print(f"\nInitial payout probability: {float(prob)*100:.1f}%")

    # Pay premium
    premium = policy.coverage_amount * PREMIUM_RATE
    insurance.pay_premium(policy, premium)
    print(f"Premium paid: {premium} ESM")

    prob = insurance.get_payout_probability(policy)
    print(f"Probability after premium: {float(prob)*100:.1f}%")

    # Oracle reports conditions
    print("\nOracle reports: Flight delayed > 4 hours")
    insurance.oracle_condition_met(policy, "cond_0")

    prob = insurance.get_payout_probability(policy)
    print(f"Probability after condition 1: {float(prob)*100:.1f}%")

    print("Oracle reports: Lost baggage")
    insurance.oracle_condition_met(policy, "cond_2")

    prob = insurance.get_payout_probability(policy)
    print(f"Probability after condition 2: {float(prob)*100:.1f}%")

    # Collapse
    result = insurance.trigger_collapse(policy, seed=42)
    print(f"\nClaim result: {'PAYOUT' if result.payout_amount > 0 else 'NO PAYOUT'}")
    if result.payout_amount > 0:
        print(f"Payout amount: {result.payout_amount} ESM")


def demo_auction():
    """Demo: Sealed-Bid Auction."""
    print_header("Application 5: Sealed-Bid Auction")

    print("Bids are cryptographically committed, then revealed.")
    print("Prevents bid manipulation and front-running.")
    print()

    auction_sys = SealedBidAuction()

    auction = auction_sys.create_auction(
        item_description="Rare Digital Artwork #42",
        seller="ArtGallery",
    )

    print(f"Auction: {auction.auction_id}")
    print(f"Item: {auction.item_description}")

    # Submit bids
    bidders = [
        ("Alice", Decimal("500")),
        ("Bob", Decimal("750")),
        ("Charlie", Decimal("600")),
        ("Diana", Decimal("900")),
    ]

    print("\nCommitment phase - submitting sealed bids:")
    secrets = {}
    for bidder, amount in bidders:
        encrypted, nonce = auction_sys.submit_bid(auction, bidder, amount)
        secrets[bidder] = (amount, nonce)
        print(f"  {bidder}: committed (amount hidden)")

    # Reveal phase
    auction_sys.block_number = auction.end_commitment_block + 1
    print("\nReveal phase - revealing bids:")
    for bidder, (amount, nonce) in secrets.items():
        revealed = auction_sys.reveal_bid(auction, bidder, amount, nonce)
        status = "valid" if revealed.valid else "INVALID"
        print(f"  {bidder}: {amount} ESM ({status})")

    # Finalize
    auction_sys.block_number = auction.end_reveal_block + 1

    print("\nFirst-price auction:")
    result = auction_sys.finalize_auction(auction, use_second_price=False)
    print(f"  Winner: {result.winner}")
    print(f"  Pays: {result.winning_amount} ESM")

    # Show what second-price would be
    print(f"\n(Second-price would pay: {result.second_highest} ESM)")


def demo_quantum_nft():
    """Demo: Quantum NFT."""
    print_header("Application 6: Quantum NFT")

    print("NFTs exist in superposition of multiple states until observed.")
    print("Properties are not determined until 'measurement'.")
    print()

    collection = QuantumNFTCollection("QuantumCreatures")
    states = create_sample_states("CryptoCreature")

    print(f"Collection: {collection.collection_id}")
    print(f"Possible states: {len(states)}")

    print("\nRarity weights:")
    for rarity, weight in RARITY_WEIGHTS.items():
        print(f"  {rarity}: {weight*100:.0f}%")

    # Mint
    print("\nMinting quantum NFT...")
    nft = collection.mint(states, owner="Alice")
    print(f"Token ID: {nft.token_id}")
    print(f"Owner: {nft.owner}")
    print(f"Observed: {nft.observed}")

    # Superposition
    print("\nSuperposition state (before observation):")
    probs = collection.get_superposition(nft)
    for name, prob in sorted(probs.items(), key=lambda x: -float(x[1])):
        print(f"  {name}: {float(prob)*100:.1f}%")

    # Transfer
    print("\nTransferring to Bob (while in superposition)...")
    collection.transfer(nft, "Bob")
    print(f"New owner: {nft.owner}")

    # Observe
    print("\nObserving NFT (collapsing wavefunction)...")
    result = collection.observe(nft, seed=42)

    print(f"\nFinal state:")
    print(f"  Name: {result.final_state.name}")
    print(f"  Rarity: {result.final_state.rarity}")
    print(f"  Attributes: {result.final_state.attributes}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("         ESM v5.4 Applications Demo")
    print("         Based on Whitepaper v5.3 Section 8")
    print("=" * 70)

    demo_mev_dex()
    demo_privacy_transfer()
    demo_prediction_market()
    demo_insurance()
    demo_auction()
    demo_quantum_nft()

    print_header("DEMO COMPLETE")

    print("Summary of ESM Applications:")
    print()
    print("  1. MEV Resistant DEX")
    print("     - Phase-based interference neutralizes front-running")
    print("     - Fast attacks get opposite phase (P180)")
    print()
    print("  2. Privacy Transfer")
    print("     - Recipient in superposition until collapse")
    print("     - Decoy mode provides plausible deniability")
    print()
    print("  3. Prediction Market")
    print("     - Odds as amplitudes with betting interference")
    print("     - Oracle-triggered resolution")
    print()
    print("  4. Decentralized Insurance")
    print("     - Conditional payouts as superposition states")
    print("     - Oracle conditions adjust probabilities")
    print()
    print("  5. Sealed-Bid Auction")
    print("     - Commit-reveal with cryptographic verification")
    print("     - Supports first-price and second-price")
    print()
    print("  6. Quantum NFT")
    print("     - Properties undetermined until observation")
    print("     - Transferable while in superposition")
    print()


if __name__ == "__main__":
    main()
