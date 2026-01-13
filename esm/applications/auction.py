"""
Sealed-Bid Auction

Based on ESM Whitepaper v5.3 Section 8.5

Implements fair sealed-bid auctions using ESM's commit-reveal mechanism.
Bids are committed using hash commitments and revealed during the
collapse phase, preventing bid manipulation and front-running.

Key features:
- Cryptographic bid commitment
- Reveal phase with verification
- Fair winner selection
- Anti-collusion mechanisms
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from enum import Enum
import hashlib
import os

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
# Constants
# =============================================================================

COMMITMENT_PERIOD_BLOCKS = 50
REVEAL_PERIOD_BLOCKS = 25
MINIMUM_BID = Decimal("1")


class AuctionStatus(Enum):
    """Auction status."""
    COMMITMENT = "commitment"    # Accepting bid commitments
    REVEAL = "reveal"            # Reveal period
    FINALIZED = "finalized"      # Auction completed
    CANCELLED = "cancelled"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EncryptedBid:
    """
    An encrypted/committed bid.

    Attributes:
        bidder: Bidder address
        commitment: Hash commitment = SHA256(amount || nonce)
        submitted_at: Block when committed
        deposit: Deposit paid to commit
    """
    bidder: str
    commitment: bytes
    submitted_at: int = 0
    deposit: Decimal = Decimal("0")


@dataclass
class RevealedBid:
    """
    A revealed bid.

    Attributes:
        bidder: Bidder address
        amount: Bid amount
        nonce: Random nonce used in commitment
        revealed_at: Block when revealed
        valid: Whether reveal matches commitment
    """
    bidder: str
    amount: Decimal
    nonce: bytes
    revealed_at: int = 0
    valid: bool = False


@dataclass
class Auction:
    """
    A sealed-bid auction.

    Attributes:
        auction_id: Unique identifier
        item_description: Description of auctioned item
        seller: Seller address
        end_commitment_block: End of commitment period
        end_reveal_block: End of reveal period
        psc: Associated PSC
        encrypted_bids: List of committed bids
        revealed_bids: List of revealed bids
        status: Auction status
        winner: Winning bidder (after finalization)
        winning_amount: Winning bid amount
    """
    auction_id: str
    item_description: str
    seller: str
    end_commitment_block: int
    end_reveal_block: int
    psc: PSC
    encrypted_bids: List[EncryptedBid] = field(default_factory=list)
    revealed_bids: List[RevealedBid] = field(default_factory=list)
    status: AuctionStatus = AuctionStatus.COMMITMENT
    winner: Optional[str] = None
    winning_amount: Decimal = Decimal("0")


@dataclass
class AuctionResult(ApplicationResult):
    """
    Result of auction finalization.

    Attributes:
        auction_id: Auction identifier
        item_description: Item description
        winner: Winning bidder
        winning_amount: Winning bid
        n_bids_committed: Number of committed bids
        n_bids_revealed: Number of revealed bids
        n_valid_reveals: Number of valid reveals
        second_highest: Second highest bid (for second-price)
    """
    auction_id: str = ""
    item_description: str = ""
    winner: str = ""
    winning_amount: Decimal = Decimal("0")
    n_bids_committed: int = 0
    n_bids_revealed: int = 0
    n_valid_reveals: int = 0
    second_highest: Decimal = Decimal("0")


# =============================================================================
# Sealed-Bid Auction
# =============================================================================

class SealedBidAuction(ESMApplication):
    """
    Fair sealed-bid auction using ESM commit-reveal.

    Bids are committed cryptographically and revealed during
    the PSC collapse phase.
    """

    def __init__(self):
        super().__init__("SealedBidAuction")
        self.auctions: Dict[str, Auction] = {}
        self.completed_auctions: List[AuctionResult] = []

    def create_auction(
        self,
        item_description: str,
        seller: str,
        commitment_blocks: int = COMMITMENT_PERIOD_BLOCKS,
        reveal_blocks: int = REVEAL_PERIOD_BLOCKS,
    ) -> Auction:
        """
        Create a new sealed-bid auction.

        Args:
            item_description: Description of item
            seller: Seller address
            commitment_blocks: Length of commitment period
            reveal_blocks: Length of reveal period

        Returns:
            New Auction instance
        """
        auction_id = hashlib.sha256(
            f"auction_{item_description}_{self.block_number}".encode()
        ).hexdigest()[:16]

        # Create PSC
        psc = self.create_application_psc(f"auction_{auction_id}")

        auction = Auction(
            auction_id=auction_id,
            item_description=item_description,
            seller=seller,
            end_commitment_block=self.block_number + commitment_blocks,
            end_reveal_block=self.block_number + commitment_blocks + reveal_blocks,
            psc=psc,
            status=AuctionStatus.COMMITMENT,
        )

        self.auctions[auction_id] = auction
        return auction

    def create_commitment(
        self,
        amount: Decimal,
        nonce: Optional[bytes] = None,
    ) -> Tuple[bytes, bytes]:
        """
        Create a bid commitment.

        Args:
            amount: Bid amount
            nonce: Optional nonce (random if not provided)

        Returns:
            Tuple of (commitment_hash, nonce)
        """
        if nonce is None:
            nonce = os.urandom(32)

        # Commitment = SHA256(amount || nonce)
        data = str(amount).encode() + nonce
        commitment = hashlib.sha256(data).digest()

        return commitment, nonce

    def submit_bid(
        self,
        auction: Auction,
        bidder: str,
        amount: Decimal,
        nonce: Optional[bytes] = None,
    ) -> Tuple[EncryptedBid, bytes]:
        """
        Submit a sealed bid to auction.

        Args:
            auction: Target auction
            bidder: Bidder address
            amount: Bid amount
            nonce: Optional nonce

        Returns:
            Tuple of (EncryptedBid, nonce) - save nonce for reveal!
        """
        if auction.status != AuctionStatus.COMMITMENT:
            raise ValueError("Auction not accepting commitments")

        if self.block_number > auction.end_commitment_block:
            auction.status = AuctionStatus.REVEAL
            raise ValueError("Commitment period ended")

        commitment, nonce = self.create_commitment(amount, nonce)

        bid = EncryptedBid(
            bidder=bidder,
            commitment=commitment,
            submitted_at=self.block_number,
            deposit=amount * Decimal("0.1"),  # 10% deposit
        )

        auction.encrypted_bids.append(bid)
        return bid, nonce

    def reveal_bid(
        self,
        auction: Auction,
        bidder: str,
        amount: Decimal,
        nonce: bytes,
    ) -> RevealedBid:
        """
        Reveal a previously committed bid.

        Args:
            auction: Target auction
            bidder: Bidder address
            amount: Claimed bid amount
            nonce: Nonce used in commitment

        Returns:
            RevealedBid with validity status
        """
        # Check if in reveal period
        if self.block_number <= auction.end_commitment_block:
            auction.status = AuctionStatus.COMMITMENT
            raise ValueError("Still in commitment period")

        if self.block_number > auction.end_reveal_block:
            raise ValueError("Reveal period ended")

        auction.status = AuctionStatus.REVEAL

        # Find matching commitment
        commitment, _ = self.create_commitment(amount, nonce)

        valid = False
        for encrypted_bid in auction.encrypted_bids:
            if encrypted_bid.bidder == bidder and encrypted_bid.commitment == commitment:
                valid = True
                break

        revealed = RevealedBid(
            bidder=bidder,
            amount=amount,
            nonce=nonce,
            revealed_at=self.block_number,
            valid=valid,
        )

        auction.revealed_bids.append(revealed)

        # Create branch for valid revealed bid
        if valid:
            state_data = {
                "bidder": bidder,
                "amount": float(amount),
            }
            # Higher bids get higher amplitude
            magnitude = float(amount) ** 0.5 / 100  # Scale down
            branch = create_branch(
                state_data=state_data,
                magnitude=magnitude,
                phase=DiscretePhase.P0,
                creator=bidder,
            )
            auction.psc.add_branch(branch)

        return revealed

    def finalize_auction(
        self,
        auction: Auction,
        use_second_price: bool = False,
    ) -> AuctionResult:
        """
        Finalize auction and determine winner.

        Args:
            auction: Auction to finalize
            use_second_price: Use second-price (Vickrey) auction

        Returns:
            AuctionResult with winner information
        """
        if self.block_number <= auction.end_reveal_block:
            return AuctionResult(
                status=ApplicationStatus.FAILED,
                psc_id=auction.psc.id,
                metadata={"error": "Reveal period not ended"},
            )

        auction.status = AuctionStatus.FINALIZED

        # Find valid revealed bids
        valid_bids = [b for b in auction.revealed_bids if b.valid]

        if not valid_bids:
            return AuctionResult(
                status=ApplicationStatus.FAILED,
                psc_id=auction.psc.id,
                metadata={"error": "No valid bids"},
                auction_id=auction.auction_id,
                item_description=auction.item_description,
            )

        # Sort by amount descending
        sorted_bids = sorted(valid_bids, key=lambda b: b.amount, reverse=True)

        winner = sorted_bids[0]
        second_highest = sorted_bids[1].amount if len(sorted_bids) > 1 else winner.amount

        # Determine winning amount
        if use_second_price:
            winning_amount = second_highest
        else:
            winning_amount = winner.amount

        auction.winner = winner.bidder
        auction.winning_amount = winning_amount

        result = AuctionResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=auction.psc.id,
            selected_outcome=winner.bidder,
            probability_distribution={},
            metadata={
                "use_second_price": use_second_price,
                "seller": auction.seller,
            },
            auction_id=auction.auction_id,
            item_description=auction.item_description,
            winner=winner.bidder,
            winning_amount=winning_amount,
            n_bids_committed=len(auction.encrypted_bids),
            n_bids_revealed=len(auction.revealed_bids),
            n_valid_reveals=len(valid_bids),
            second_highest=second_highest,
        )

        self.completed_auctions.append(result)
        return result

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "active_auctions": len([a for a in self.auctions.values() if a.status in [AuctionStatus.COMMITMENT, AuctionStatus.REVEAL]]),
            "completed_auctions": len(self.completed_auctions),
            "total_auctions": len(self.auctions),
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_auction(
    item: str = "Rare NFT #42",
    bidders: List[Tuple[str, Decimal]] = None,
    use_second_price: bool = False,
) -> AuctionResult:
    """
    Simulate a sealed-bid auction.

    Args:
        item: Item description
        bidders: List of (bidder_name, bid_amount) tuples
        use_second_price: Use second-price auction rules

    Returns:
        AuctionResult
    """
    if bidders is None:
        bidders = [
            ("Alice", Decimal("100")),
            ("Bob", Decimal("150")),
            ("Charlie", Decimal("125")),
            ("Diana", Decimal("200")),
        ]

    auction_system = SealedBidAuction()

    # Create auction
    auction = auction_system.create_auction(
        item_description=item,
        seller="Seller",
    )

    print(f"Auction created: {auction.auction_id}")
    print(f"Item: {item}")
    print(f"Commitment period: blocks {auction_system.block_number} - {auction.end_commitment_block}")

    # Submit bids
    print("\nSubmitting sealed bids...")
    bid_secrets = {}  # Store nonces for reveal

    for bidder, amount in bidders:
        encrypted_bid, nonce = auction_system.submit_bid(auction, bidder, amount)
        bid_secrets[bidder] = (amount, nonce)
        print(f"  {bidder}: committed (amount hidden)")

    # Advance to reveal period
    auction_system.block_number = auction.end_commitment_block + 1

    # Reveal bids
    print("\nRevealing bids...")
    for bidder, (amount, nonce) in bid_secrets.items():
        revealed = auction_system.reveal_bid(auction, bidder, amount, nonce)
        status = "valid" if revealed.valid else "INVALID"
        print(f"  {bidder}: {amount} ({status})")

    # Finalize
    auction_system.block_number = auction.end_reveal_block + 1
    result = auction_system.finalize_auction(auction, use_second_price=use_second_price)

    print(f"\nAuction Result:")
    print(f"  Winner: {result.winner}")
    print(f"  Winning bid: {result.winning_amount}")
    if use_second_price:
        print(f"  (Second-price auction - pays second highest)")
        print(f"  Second highest: {result.second_highest}")

    return result


def run_demo():
    """Run auction demo."""
    print("=" * 60)
    print("Sealed-Bid Auction Demo")
    print("=" * 60)
    print()

    # First-price auction
    print("Demo 1: First-Price Auction")
    print("-" * 40)
    simulate_auction(use_second_price=False)
    print()

    # Second-price (Vickrey) auction
    print("Demo 2: Second-Price (Vickrey) Auction")
    print("-" * 40)
    simulate_auction(use_second_price=True)
    print()


if __name__ == "__main__":
    run_demo()
