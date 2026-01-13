"""
Tests for ESM Applications.

Based on ESM Whitepaper v5.3 Section 8.
"""

import pytest
from decimal import Decimal
import time

from esm.applications.base import (
    ESMApplication,
    ApplicationResult,
    ApplicationStatus,
    calculate_interference_impact,
    distribute_amplitudes,
)

from esm.applications.mev_dex import (
    MEVResistantDEX,
    SwapOrder,
    LiquidityPool,
    DEXResult,
    create_sample_pool,
    simulate_mev_scenario,
    FAST_ATTACK_THRESHOLD,
)

from esm.applications.privacy_transfer import (
    PrivacyTransferSystem,
    PrivateTransfer,
    TransferResult,
    TransferMode,
)

from esm.applications.prediction_market import (
    PredictionMarket,
    Market,
    Bet,
    MarketResult,
    MarketStatus,
)

from esm.applications.insurance import (
    DecentralizedInsurance,
    InsurancePolicy,
    InsuranceResult,
    PolicyStatus,
)

from esm.applications.auction import (
    SealedBidAuction,
    Auction,
    EncryptedBid,
    RevealedBid,
    AuctionResult,
    AuctionStatus,
)

from esm.applications.quantum_nft import (
    QuantumNFTCollection,
    QuantumNFT,
    NFTState,
    NFTResult,
    create_sample_states,
)

from esm.core.phase import DiscretePhase


# =============================================================================
# Base Application Tests
# =============================================================================

class TestApplicationBase:
    """Tests for base application functionality."""

    def test_application_status_enum(self):
        """Test ApplicationStatus enum values."""
        assert ApplicationStatus.PENDING.value == "pending"
        assert ApplicationStatus.COMPLETED.value == "completed"
        assert ApplicationStatus.FAILED.value == "failed"

    def test_application_result_is_successful(self):
        """Test ApplicationResult.is_successful property."""
        success = ApplicationResult(
            status=ApplicationStatus.COMPLETED,
            psc_id="test",
        )
        assert success.is_successful is True

        failure = ApplicationResult(
            status=ApplicationStatus.FAILED,
            psc_id="test",
        )
        assert failure.is_successful is False

    def test_calculate_interference_impact(self):
        """Test interference impact calculation."""
        # Same phase - constructive
        assert calculate_interference_impact(DiscretePhase.P0, DiscretePhase.P0) == "constructive_full"

        # Opposite phase - destructive
        assert calculate_interference_impact(DiscretePhase.P0, DiscretePhase.P180) == "destructive_full"

        # Orthogonal
        assert calculate_interference_impact(DiscretePhase.P0, DiscretePhase.P90) == "orthogonal"

    def test_distribute_amplitudes_equal(self):
        """Test equal amplitude distribution."""
        amps = distribute_amplitudes(4, "equal")

        assert len(amps) == 4
        # All should have same magnitude
        mags = [m for m, p in amps]
        assert all(abs(m - mags[0]) < 0.001 for m in mags)


# =============================================================================
# MEV DEX Tests
# =============================================================================

class TestMEVResistantDEX:
    """Tests for MEV Resistant DEX."""

    def test_create_dex(self):
        """Test DEX creation."""
        dex = MEVResistantDEX()
        assert dex.app_name == "MEVResistantDEX"
        assert len(dex.pools) == 0

    def test_add_pool(self):
        """Test adding liquidity pool."""
        dex = MEVResistantDEX()
        pool = create_sample_pool()
        dex.add_pool(pool)

        assert len(dex.pools) == 1

    def test_calculate_output(self):
        """Test swap output calculation."""
        dex = MEVResistantDEX()
        pool = create_sample_pool(
            reserve_a=Decimal("100000"),
            reserve_b=Decimal("950000"),
        )

        output = dex.calculate_output(pool, "ESM", Decimal("100"))

        # Should get approximately 9.5 TOKEN per ESM
        assert output > Decimal("900")
        assert output < Decimal("1000")

    def test_fast_attack_blocked(self):
        """Test that fast MEV attack is blocked."""
        result = simulate_mev_scenario(
            victim_amount=Decimal("100"),
            attack_delay_ms=50,  # Fast attack
            seed=42,
        )

        assert result.mev_blocked is True
        assert result.metadata["attacker_phase"] == "P180"

    def test_slow_attack_not_blocked(self):
        """Test that slow attack is not blocked by timing."""
        result = simulate_mev_scenario(
            victim_amount=Decimal("100"),
            attack_delay_ms=1500,  # Slow attack
            seed=42,
        )

        assert result.metadata["attacker_phase"] == "P0"

    def test_phase_assignment(self):
        """Test phase assignment by timing."""
        dex = MEVResistantDEX()
        base_time = int(time.time() * 1000)

        order_fast = SwapOrder(
            trader="Bot",
            input_token="ESM",
            output_token="TOKEN",
            input_amount=Decimal("100"),
            min_output=Decimal("900"),
            submitted_ms=base_time + 50,
        )

        phase = dex.assign_phase_by_timing(order_fast, base_time)
        assert phase == DiscretePhase.P180


# =============================================================================
# Privacy Transfer Tests
# =============================================================================

class TestPrivacyTransfer:
    """Tests for Privacy Transfer System."""

    def test_create_system(self):
        """Test system creation."""
        system = PrivacyTransferSystem()
        assert system.app_name == "PrivacyTransfer"

    def test_create_private_transfer(self):
        """Test creating private transfer."""
        system = PrivacyTransferSystem()

        transfer = system.create_private_transfer(
            sender="Alice",
            recipients=["Bob", "Charlie", "Diana"],
            amount=Decimal("100"),
            mode=TransferMode.EQUAL,
        )

        assert transfer.sender == "Alice"
        assert transfer.amount == Decimal("100")
        assert len(transfer.possible_recipients) == 3

    def test_equal_probability_distribution(self):
        """Test equal probability distribution."""
        system = PrivacyTransferSystem()

        transfer = system.create_private_transfer(
            sender="Alice",
            recipients=["Bob", "Charlie", "Diana"],
            amount=Decimal("100"),
            mode=TransferMode.EQUAL,
        )

        probs = system.get_recipient_probabilities(transfer)

        # Should be roughly equal
        for recipient, prob in probs.items():
            assert abs(float(prob) - 1/3) < 0.1

    def test_decoy_transfer(self):
        """Test decoy transfer mode."""
        system = PrivacyTransferSystem()

        transfer = system.create_decoy_transfer(
            sender="Alice",
            real_recipient="Bob",
            decoy_recipients=["Charlie", "Diana"],
            amount=Decimal("100"),
        )

        probs = system.get_recipient_probabilities(transfer)

        # Real recipient should have ~99% probability
        assert float(probs.get("Bob", 0)) > 0.9

    def test_collapse_transfer(self):
        """Test transfer collapse."""
        system = PrivacyTransferSystem()

        transfer = system.create_private_transfer(
            sender="Alice",
            recipients=["Bob", "Charlie"],
            amount=Decimal("100"),
        )

        result = system.collapse_transfer(transfer, seed=42)

        assert result.status == ApplicationStatus.COMPLETED
        assert result.recipient in ["Bob", "Charlie"]
        assert transfer.collapsed is True


# =============================================================================
# Prediction Market Tests
# =============================================================================

class TestPredictionMarket:
    """Tests for Prediction Market."""

    def test_create_market(self):
        """Test market creation."""
        pm = PredictionMarket()

        market = pm.create_market(
            question="Will it rain tomorrow?",
            outcomes=["Yes", "No"],
        )

        assert market.question == "Will it rain tomorrow?"
        assert len(market.outcomes) == 2
        assert market.status == MarketStatus.OPEN

    def test_place_bet(self):
        """Test placing bet."""
        pm = PredictionMarket()

        market = pm.create_market(
            question="Test?",
            outcomes=["Yes", "No"],
        )

        bet = Bet(bettor="Alice", outcome="Yes", amount=Decimal("100"))
        success = pm.place_bet(market, bet)

        assert success is True
        assert len(market.bets) == 1
        assert market.total_pool == Decimal("100")

    def test_betting_changes_odds(self):
        """Test that betting changes odds."""
        pm = PredictionMarket()

        market = pm.create_market(
            question="Test?",
            outcomes=["Yes", "No"],
        )

        initial_odds = pm.get_current_odds(market)

        # Place large bet on Yes
        bet = Bet(bettor="Alice", outcome="Yes", amount=Decimal("1000"))
        pm.place_bet(market, bet)

        new_odds = pm.get_current_odds(market)

        # Yes probability should increase
        assert float(new_odds.get("Yes", 0)) > float(initial_odds.get("Yes", 0))

    def test_oracle_report_and_resolve(self):
        """Test oracle reporting and market resolution."""
        pm = PredictionMarket()

        market = pm.create_market(
            question="Test?",
            outcomes=["Yes", "No"],
        )

        bet = Bet(bettor="Alice", outcome="Yes", amount=Decimal("100"))
        pm.place_bet(market, bet)

        pm.oracle_report(market, "Yes")
        result = pm.resolve_market(market)

        assert result.status == ApplicationStatus.COMPLETED
        assert result.resolved_outcome == "Yes"
        assert result.winning_bets == 1


# =============================================================================
# Decentralized Insurance Tests
# =============================================================================

class TestDecentralizedInsurance:
    """Tests for Decentralized Insurance."""

    def test_create_policy(self):
        """Test policy creation."""
        insurance = DecentralizedInsurance()

        policy = insurance.create_policy(
            policyholder="Alice",
            coverage_amount=Decimal("10000"),
            conditions=["Condition A", "Condition B"],
        )

        assert policy.policyholder == "Alice"
        assert policy.coverage_amount == Decimal("10000")
        assert len(policy.conditions) == 2

    def test_pay_premium(self):
        """Test premium payment."""
        insurance = DecentralizedInsurance()

        policy = insurance.create_policy(
            policyholder="Alice",
            coverage_amount=Decimal("10000"),
            conditions=["Test"],
        )

        initial_prob = insurance.get_payout_probability(policy)
        insurance.pay_premium(policy, Decimal("500"))
        new_prob = insurance.get_payout_probability(policy)

        # Probability should increase after premium
        assert float(new_prob) >= float(initial_prob)

    def test_condition_met_increases_probability(self):
        """Test that meeting conditions increases payout probability."""
        insurance = DecentralizedInsurance()

        policy = insurance.create_policy(
            policyholder="Alice",
            coverage_amount=Decimal("10000"),
            conditions=["Condition A"],
        )

        initial_prob = insurance.get_payout_probability(policy)
        insurance.oracle_condition_met(policy, "cond_0")
        new_prob = insurance.get_payout_probability(policy)

        assert float(new_prob) > float(initial_prob)

    def test_trigger_collapse(self):
        """Test policy collapse."""
        insurance = DecentralizedInsurance()

        policy = insurance.create_policy(
            policyholder="Alice",
            coverage_amount=Decimal("10000"),
            conditions=["Test"],
        )

        result = insurance.trigger_collapse(policy, seed=42)

        assert result.status == ApplicationStatus.COMPLETED
        assert policy.status == PolicyStatus.CLAIMED


# =============================================================================
# Sealed-Bid Auction Tests
# =============================================================================

class TestSealedBidAuction:
    """Tests for Sealed-Bid Auction."""

    def test_create_auction(self):
        """Test auction creation."""
        auction_system = SealedBidAuction()

        auction = auction_system.create_auction(
            item_description="Test Item",
            seller="Seller",
        )

        assert auction.item_description == "Test Item"
        assert auction.status == AuctionStatus.COMMITMENT

    def test_submit_and_reveal_bid(self):
        """Test bid submission and reveal."""
        auction_system = SealedBidAuction()

        auction = auction_system.create_auction(
            item_description="Test Item",
            seller="Seller",
        )

        # Submit bid
        encrypted_bid, nonce = auction_system.submit_bid(
            auction, "Alice", Decimal("100")
        )

        assert len(auction.encrypted_bids) == 1

        # Move to reveal period
        auction_system.block_number = auction.end_commitment_block + 1

        # Reveal bid
        revealed = auction_system.reveal_bid(
            auction, "Alice", Decimal("100"), nonce
        )

        assert revealed.valid is True

    def test_invalid_reveal(self):
        """Test invalid bid reveal."""
        auction_system = SealedBidAuction()

        auction = auction_system.create_auction(
            item_description="Test Item",
            seller="Seller",
        )

        # Submit bid
        _, nonce = auction_system.submit_bid(
            auction, "Alice", Decimal("100")
        )

        auction_system.block_number = auction.end_commitment_block + 1

        # Try to reveal with wrong amount
        revealed = auction_system.reveal_bid(
            auction, "Alice", Decimal("200"), nonce  # Wrong amount
        )

        assert revealed.valid is False

    def test_finalize_auction(self):
        """Test auction finalization."""
        auction_system = SealedBidAuction()

        auction = auction_system.create_auction(
            item_description="Test Item",
            seller="Seller",
        )

        # Submit bids
        _, nonce1 = auction_system.submit_bid(auction, "Alice", Decimal("100"))
        _, nonce2 = auction_system.submit_bid(auction, "Bob", Decimal("150"))

        # Reveal
        auction_system.block_number = auction.end_commitment_block + 1
        auction_system.reveal_bid(auction, "Alice", Decimal("100"), nonce1)
        auction_system.reveal_bid(auction, "Bob", Decimal("150"), nonce2)

        # Finalize
        auction_system.block_number = auction.end_reveal_block + 1
        result = auction_system.finalize_auction(auction)

        assert result.status == ApplicationStatus.COMPLETED
        assert result.winner == "Bob"
        assert result.winning_amount == Decimal("150")


# =============================================================================
# Quantum NFT Tests
# =============================================================================

class TestQuantumNFT:
    """Tests for Quantum NFT."""

    def test_create_collection(self):
        """Test collection creation."""
        collection = QuantumNFTCollection("TestCollection")
        assert collection.collection_id == "TestCollection"

    def test_mint_nft(self):
        """Test NFT minting."""
        collection = QuantumNFTCollection()
        states = create_sample_states()

        nft = collection.mint(states, owner="Alice")

        assert nft.owner == "Alice"
        assert not nft.observed
        assert len(nft.possible_states) == len(states)

    def test_superposition_probabilities(self):
        """Test superposition probability retrieval."""
        collection = QuantumNFTCollection()
        states = create_sample_states()

        nft = collection.mint(states, owner="Alice")
        probs = collection.get_superposition(nft)

        # Should have probabilities for all states
        assert len(probs) == len(states)
        # Sum should be approximately 1
        total = sum(float(p) for p in probs.values())
        assert abs(total - 1.0) < 0.1

    def test_observe_nft(self):
        """Test NFT observation."""
        collection = QuantumNFTCollection()
        states = create_sample_states()

        nft = collection.mint(states, owner="Alice")
        result = collection.observe(nft, seed=42)

        assert result.status == ApplicationStatus.COMPLETED
        assert nft.observed is True
        assert nft.final_state is not None

    def test_transfer_before_observation(self):
        """Test NFT transfer while in superposition."""
        collection = QuantumNFTCollection()
        states = create_sample_states()

        nft = collection.mint(states, owner="Alice")
        collection.transfer(nft, "Bob")

        assert nft.owner == "Bob"
        assert not nft.observed  # Still in superposition

    def test_double_observation_fails(self):
        """Test that double observation fails."""
        collection = QuantumNFTCollection()
        states = create_sample_states()

        nft = collection.mint(states, owner="Alice")
        collection.observe(nft, seed=42)

        # Second observation should fail
        result = collection.observe(nft, seed=43)
        assert result.status == ApplicationStatus.FAILED


# =============================================================================
# Integration Tests
# =============================================================================

class TestApplicationIntegration:
    """Integration tests across applications."""

    def test_all_applications_initialize(self):
        """Test all applications can be initialized."""
        apps = [
            MEVResistantDEX(),
            PrivacyTransferSystem(),
            PredictionMarket(),
            DecentralizedInsurance(),
            SealedBidAuction(),
            QuantumNFTCollection(),
        ]

        for app in apps:
            status = app.get_status()
            assert isinstance(status, dict)
            assert "block_number" in status

    def test_applications_use_psc(self):
        """Test all applications properly use PSC mechanism."""
        # MEV DEX
        dex = MEVResistantDEX()
        pool = create_sample_pool()
        dex.add_pool(pool)
        result = simulate_mev_scenario(seed=42)
        assert result.psc_id is not None

        # Privacy Transfer
        pts = PrivacyTransferSystem()
        transfer = pts.create_private_transfer(
            "Alice", ["Bob", "Charlie"], Decimal("100")
        )
        assert transfer.psc is not None

        # Prediction Market
        pm = PredictionMarket()
        market = pm.create_market("Test?", ["Yes", "No"])
        assert market.psc is not None

        # Insurance
        insurance = DecentralizedInsurance()
        policy = insurance.create_policy("Alice", Decimal("1000"), ["Test"])
        assert policy.psc is not None

        # Auction
        auction_sys = SealedBidAuction()
        auction = auction_sys.create_auction("Item", "Seller")
        assert auction.psc is not None

        # NFT
        nft_collection = QuantumNFTCollection()
        states = create_sample_states()
        nft = nft_collection.mint(states, "Alice")
        assert nft.psc is not None
