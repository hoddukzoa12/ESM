"""
ESM Application Examples Package

Implements the 6 applications from ESM Whitepaper v5.3 Section 8:
1. MEV Resistant DEX - Front-running cancellation
2. Privacy Transfer - Multi-recipient obfuscation
3. Prediction Market - Amplitude-based odds
4. Decentralized Insurance - Conditional payouts
5. Sealed-Bid Auction - Commit-reveal bidding
6. Quantum NFT - Superposition states
"""

from esm.applications.base import (
    ESMApplication,
    ApplicationResult,
    ApplicationStatus,
)

from esm.applications.mev_dex import (
    MEVResistantDEX,
    SwapOrder,
    LiquidityPool,
    DEXResult,
)

from esm.applications.privacy_transfer import (
    PrivacyTransferSystem,
    PrivateTransfer,
    TransferResult,
)

from esm.applications.prediction_market import (
    PredictionMarket,
    Market,
    Bet,
    MarketResult,
)

from esm.applications.insurance import (
    DecentralizedInsurance,
    InsurancePolicy,
    InsuranceResult,
)

from esm.applications.auction import (
    SealedBidAuction,
    Auction,
    EncryptedBid,
    RevealedBid,
    AuctionResult,
)

from esm.applications.quantum_nft import (
    QuantumNFTCollection,
    QuantumNFT,
    NFTState,
    NFTResult,
)

__all__ = [
    # Base
    "ESMApplication",
    "ApplicationResult",
    "ApplicationStatus",
    # MEV DEX
    "MEVResistantDEX",
    "SwapOrder",
    "LiquidityPool",
    "DEXResult",
    # Privacy Transfer
    "PrivacyTransferSystem",
    "PrivateTransfer",
    "TransferResult",
    # Prediction Market
    "PredictionMarket",
    "Market",
    "Bet",
    "MarketResult",
    # Insurance
    "DecentralizedInsurance",
    "InsurancePolicy",
    "InsuranceResult",
    # Auction
    "SealedBidAuction",
    "Auction",
    "EncryptedBid",
    "RevealedBid",
    "AuctionResult",
    # Quantum NFT
    "QuantumNFTCollection",
    "QuantumNFT",
    "NFTState",
    "NFTResult",
]
