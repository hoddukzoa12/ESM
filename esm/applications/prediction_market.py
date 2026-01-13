"""
Quantum Prediction Market

Based on ESM Whitepaper v5.3 Section 8.3

Implements a prediction market using ESM's amplitude-based probabilities.
Market odds are represented as amplitudes, and betting affects the
probability distribution through constructive/destructive interference.

Key features:
- Amplitude-based odds representation
- Betting with phase-aware interference
- Oracle-triggered resolution
- Proportional payout distribution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from enum import Enum
import hashlib
import random

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_branch
from esm.core.psc import PSC, create_psc
from esm.applications.base import (
    ESMApplication,
    ApplicationResult,
    ApplicationStatus,
)


# =============================================================================
# Data Structures
# =============================================================================

class MarketStatus(Enum):
    """Prediction market status."""
    OPEN = "open"             # Accepting bets
    CLOSED = "closed"         # No more bets, awaiting oracle
    RESOLVED = "resolved"     # Oracle reported, payouts available
    CANCELLED = "cancelled"   # Market cancelled


@dataclass
class Bet:
    """
    A bet placed on a market.

    Attributes:
        bettor: Bettor address
        outcome: Predicted outcome
        amount: Bet amount
        timestamp: When bet was placed
        phase: Phase assigned to this bet
    """
    bettor: str
    outcome: str
    amount: Decimal
    timestamp: int = 0
    phase: DiscretePhase = DiscretePhase.P0


@dataclass
class Market:
    """
    A prediction market.

    Attributes:
        market_id: Unique identifier
        question: Market question
        outcomes: List of possible outcomes
        deadline_block: Block deadline for betting
        psc: Associated PSC
        total_pool: Total amount in pool
        bets: List of bets placed
        status: Market status
        resolved_outcome: Final outcome (after oracle)
    """
    market_id: str
    question: str
    outcomes: List[str]
    deadline_block: int
    psc: PSC
    total_pool: Decimal = Decimal("0")
    bets: List[Bet] = field(default_factory=list)
    status: MarketStatus = MarketStatus.OPEN
    resolved_outcome: Optional[str] = None


@dataclass
class MarketResult(ApplicationResult):
    """
    Result of market resolution.

    Attributes:
        market_id: Market identifier
        question: Market question
        resolved_outcome: Oracle-reported outcome
        total_pool: Total betting pool
        n_bets: Number of bets
        winning_bets: Number of winning bets
        payout_per_share: Payout per unit bet on winning outcome
    """
    market_id: str = ""
    question: str = ""
    resolved_outcome: str = ""
    total_pool: Decimal = Decimal("0")
    n_bets: int = 0
    winning_bets: int = 0
    payout_per_share: Decimal = Decimal("0")


# =============================================================================
# Prediction Market
# =============================================================================

class PredictionMarket(ESMApplication):
    """
    Quantum prediction market using ESM amplitudes.

    Odds are represented as amplitudes, allowing for
    interference effects when bets are placed.
    """

    def __init__(self):
        super().__init__("PredictionMarket")
        self.markets: Dict[str, Market] = {}
        self.resolved_markets: List[MarketResult] = []

    def create_market(
        self,
        question: str,
        outcomes: List[str],
        deadline_blocks: int = 100,
        initial_odds: Optional[Dict[str, float]] = None,
    ) -> Market:
        """
        Create a new prediction market.

        Args:
            question: Market question
            outcomes: List of possible outcomes
            deadline_blocks: Blocks until betting closes
            initial_odds: Initial odds for each outcome

        Returns:
            New Market instance
        """
        market_id = hashlib.sha256(
            f"market_{question}_{self.block_number}".encode()
        ).hexdigest()[:16]

        # Create PSC
        psc = self.create_application_psc(f"market_{market_id}")

        # Initialize branches with equal or specified odds
        if initial_odds:
            total_odds = sum(initial_odds.values())
            weights = {k: v / total_odds for k, v in initial_odds.items()}
        else:
            n = len(outcomes)
            weights = {o: 1.0 / n for o in outcomes}

        # Create outcome branches
        for outcome in outcomes:
            weight = weights.get(outcome, 1.0 / len(outcomes))
            magnitude = weight ** 0.5  # sqrt for amplitude

            state_data = {
                "outcome": outcome,
                "initial_weight": weight,
            }

            branch = create_branch(
                state_data=state_data,
                magnitude=magnitude,
                phase=DiscretePhase.P0,
                creator="market",
            )
            psc.add_branch(branch)

        market = Market(
            market_id=market_id,
            question=question,
            outcomes=outcomes,
            deadline_block=self.block_number + deadline_blocks,
            psc=psc,
            status=MarketStatus.OPEN,
        )

        self.markets[market_id] = market
        return market

    def place_bet(
        self,
        market: Market,
        bet: Bet,
    ) -> bool:
        """
        Place a bet on a market.

        The bet affects the amplitude of the chosen outcome,
        increasing its probability through constructive interference.

        Args:
            market: Target market
            bet: Bet to place

        Returns:
            True if bet was placed successfully
        """
        if market.status != MarketStatus.OPEN:
            return False

        if self.block_number > market.deadline_block:
            market.status = MarketStatus.CLOSED
            return False

        if bet.outcome not in market.outcomes:
            return False

        # Find the outcome branch and boost it
        for branch in market.psc.branches:
            if branch.state_data.get("outcome") == bet.outcome:
                # Add constructive contribution
                # Magnitude increase proportional to sqrt(bet_amount)
                bet_contribution = float(bet.amount) ** 0.5 / 10.0

                # Create new amplitude with boosted magnitude
                new_mag = branch.amplitude.magnitude + bet_contribution
                branch.amplitude = DiscreteAmplitude(new_mag, branch.amplitude.phase)
                break

        # Record bet
        bet.timestamp = self.block_number
        bet.phase = DiscretePhase.P0  # Constructive
        market.bets.append(bet)
        market.total_pool += bet.amount

        # Invalidate PSC cache
        market.psc._interference_dirty = True

        return True

    def get_current_odds(
        self,
        market: Market,
    ) -> Dict[str, Decimal]:
        """
        Get current odds for each outcome.

        Returns probabilities derived from current amplitudes.

        Args:
            market: Target market

        Returns:
            Dictionary mapping outcome to probability
        """
        probs = market.psc.get_probabilities()

        result = {}
        for branch in market.psc.branches:
            outcome = branch.state_data.get("outcome")
            if outcome:
                prob = probs.get(branch.state_id, 0)
                result[outcome] = Decimal(str(prob))

        return result

    def oracle_report(
        self,
        market: Market,
        result_outcome: str,
    ) -> bool:
        """
        Report oracle result for market.

        Args:
            market: Target market
            result_outcome: Actual outcome

        Returns:
            True if report was accepted
        """
        if market.status == MarketStatus.RESOLVED:
            return False

        if result_outcome not in market.outcomes:
            return False

        market.resolved_outcome = result_outcome
        market.status = MarketStatus.RESOLVED

        return True

    def resolve_market(
        self,
        market: Market,
    ) -> MarketResult:
        """
        Resolve market and calculate payouts.

        Args:
            market: Market to resolve

        Returns:
            MarketResult with payout information
        """
        if market.status != MarketStatus.RESOLVED:
            return MarketResult(
                status=ApplicationStatus.FAILED,
                psc_id=market.psc.id,
                metadata={"error": "Market not resolved"},
            )

        # Calculate winning bets
        winning_bets = [
            b for b in market.bets
            if b.outcome == market.resolved_outcome
        ]

        total_winning = sum(b.amount for b in winning_bets)

        # Calculate payout per share
        if total_winning > 0:
            payout_per_share = market.total_pool / total_winning
        else:
            payout_per_share = Decimal("0")

        result = MarketResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=market.psc.id,
            selected_outcome=market.resolved_outcome,
            probability_distribution={
                k: float(v) for k, v in self.get_current_odds(market).items()
            },
            metadata={
                "question": market.question,
                "deadline_block": market.deadline_block,
            },
            market_id=market.market_id,
            question=market.question,
            resolved_outcome=market.resolved_outcome,
            total_pool=market.total_pool,
            n_bets=len(market.bets),
            winning_bets=len(winning_bets),
            payout_per_share=payout_per_share,
        )

        self.resolved_markets.append(result)
        return result

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "open_markets": len([m for m in self.markets.values() if m.status == MarketStatus.OPEN]),
            "closed_markets": len([m for m in self.markets.values() if m.status == MarketStatus.CLOSED]),
            "resolved_markets": len(self.resolved_markets),
            "total_markets": len(self.markets),
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_prediction_market(
    question: str = "Will ETH reach $5000 by end of year?",
    outcomes: List[str] = None,
    bets: List[Tuple[str, str, Decimal]] = None,
    oracle_result: str = "Yes",
    seed: int = 42,
) -> MarketResult:
    """
    Run a prediction market simulation.

    Args:
        question: Market question
        outcomes: Possible outcomes
        bets: List of (bettor, outcome, amount) tuples
        oracle_result: Oracle's reported outcome
        seed: Random seed

    Returns:
        MarketResult
    """
    if outcomes is None:
        outcomes = ["Yes", "No"]

    if bets is None:
        bets = [
            ("Alice", "Yes", Decimal("100")),
            ("Bob", "No", Decimal("50")),
            ("Charlie", "Yes", Decimal("75")),
            ("Diana", "Yes", Decimal("25")),
        ]

    pm = PredictionMarket()

    # Create market
    market = pm.create_market(
        question=question,
        outcomes=outcomes,
        deadline_blocks=100,
    )

    print(f"Market created: {market.market_id}")
    print(f"Question: {question}")
    print(f"Outcomes: {outcomes}")

    # Initial odds
    print("\nInitial odds:")
    odds = pm.get_current_odds(market)
    for outcome, prob in odds.items():
        print(f"  {outcome}: {float(prob)*100:.1f}%")

    # Place bets
    print("\nPlacing bets:")
    for bettor, outcome, amount in bets:
        bet = Bet(bettor=bettor, outcome=outcome, amount=amount)
        pm.place_bet(market, bet)
        print(f"  {bettor} bets {amount} on '{outcome}'")

    # Updated odds
    print("\nUpdated odds:")
    odds = pm.get_current_odds(market)
    for outcome, prob in sorted(odds.items(), key=lambda x: -float(x[1])):
        print(f"  {outcome}: {float(prob)*100:.1f}%")

    # Oracle reports result
    print(f"\nOracle reports: {oracle_result}")
    pm.oracle_report(market, oracle_result)

    # Resolve
    result = pm.resolve_market(market)

    print(f"\nResolution:")
    print(f"  Total pool: {result.total_pool}")
    print(f"  Winning bets: {result.winning_bets}")
    print(f"  Payout per share: {result.payout_per_share:.2f}x")

    return result


def run_demo():
    """Run prediction market demo."""
    print("=" * 60)
    print("Quantum Prediction Market Demo")
    print("=" * 60)
    print()

    simulate_prediction_market()


if __name__ == "__main__":
    run_demo()
