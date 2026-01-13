"""
Quantum NFT Collection

Based on ESM Whitepaper v5.3 Section 8.6

Implements dynamic NFTs that exist in superposition states until
observed. NFT properties are not determined until the token is
"measured" through observation.

Key features:
- Superposition of multiple possible states
- Observation triggers collapse
- Transferable while in superposition
- Rarity determined by amplitudes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
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
    distribute_amplitudes,
)


# =============================================================================
# Constants
# =============================================================================

# Rarity levels with target probabilities
RARITY_WEIGHTS = {
    "Legendary": 0.01,    # 1%
    "Epic": 0.05,         # 5%
    "Rare": 0.15,         # 15%
    "Uncommon": 0.30,     # 30%
    "Common": 0.49,       # 49%
}


class NFTStateType(Enum):
    """NFT state type."""
    SUPERPOSITION = "superposition"
    OBSERVED = "observed"
    BURNED = "burned"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NFTState:
    """
    A possible NFT state.

    Attributes:
        name: State name
        image_uri: Image URI for this state
        attributes: Dictionary of attributes
        rarity: Rarity level
        probability: Probability of this state
    """
    name: str
    image_uri: str
    attributes: Dict[str, Any]
    rarity: str
    probability: float = 0.0


@dataclass
class QuantumNFT:
    """
    A quantum NFT in superposition.

    Attributes:
        token_id: Unique token identifier
        collection_id: Collection this NFT belongs to
        possible_states: List of possible states
        psc: Associated PSC
        owner: Current owner
        observed: Whether NFT has been observed
        final_state: Final state after observation
        created_at: Block when created
        observed_at: Block when observed
    """
    token_id: str
    collection_id: str
    possible_states: List[NFTState]
    psc: PSC
    owner: str
    observed: bool = False
    final_state: Optional[NFTState] = None
    created_at: int = 0
    observed_at: Optional[int] = None


@dataclass
class NFTResult(ApplicationResult):
    """
    Result of NFT operation.

    Attributes:
        token_id: Token identifier
        collection_id: Collection identifier
        owner: Current owner
        final_state: Final state (if observed)
        rarity: Rarity of final state
        n_possible_states: Number of possible states
    """
    token_id: str = ""
    collection_id: str = ""
    owner: str = ""
    final_state: Optional[NFTState] = None
    rarity: str = ""
    n_possible_states: int = 0


# =============================================================================
# Quantum NFT Collection
# =============================================================================

class QuantumNFTCollection(ESMApplication):
    """
    Quantum NFT collection using ESM superposition.

    NFTs exist in multiple possible states until observed,
    at which point they collapse to a definite state.
    """

    def __init__(self, collection_id: str = "quantum_collection"):
        super().__init__("QuantumNFT")
        self.collection_id = collection_id
        self.nfts: Dict[str, QuantumNFT] = {}
        self.observation_history: List[NFTResult] = []
        self.next_token_id = 1

    def mint(
        self,
        possible_states: List[NFTState],
        owner: str,
        custom_probabilities: Optional[Dict[str, float]] = None,
    ) -> QuantumNFT:
        """
        Mint a new quantum NFT.

        The NFT is created in superposition of all possible states.

        Args:
            possible_states: List of possible NFT states
            owner: Initial owner address
            custom_probabilities: Optional custom state probabilities

        Returns:
            New QuantumNFT
        """
        token_id = f"QTN_{self.next_token_id:06d}"
        self.next_token_id += 1

        # Create PSC
        psc = self.create_application_psc(f"nft_{token_id}")

        # Determine probabilities
        if custom_probabilities:
            probs = custom_probabilities
        else:
            # Use rarity-based probabilities
            probs = {}
            for state in possible_states:
                probs[state.name] = RARITY_WEIGHTS.get(state.rarity, 0.1)

            # Normalize
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}

        # Create branches for each state
        for state in possible_states:
            prob = probs.get(state.name, 1.0 / len(possible_states))
            state.probability = prob
            magnitude = prob ** 0.5  # sqrt for amplitude

            state_data = {
                "name": state.name,
                "image_uri": state.image_uri,
                "attributes": state.attributes,
                "rarity": state.rarity,
            }

            branch = create_branch(
                state_data=state_data,
                magnitude=magnitude,
                phase=DiscretePhase.P0,
                creator=owner,
            )
            psc.add_branch(branch)

        nft = QuantumNFT(
            token_id=token_id,
            collection_id=self.collection_id,
            possible_states=possible_states,
            psc=psc,
            owner=owner,
            created_at=self.block_number,
        )

        self.nfts[token_id] = nft
        return nft

    def get_superposition(
        self,
        nft: QuantumNFT,
    ) -> Dict[str, Decimal]:
        """
        Get current superposition state probabilities.

        Args:
            nft: Target NFT

        Returns:
            Dictionary mapping state name to probability
        """
        if nft.observed:
            return {nft.final_state.name: Decimal("1.0")} if nft.final_state else {}

        probs = nft.psc.get_probabilities()

        result = {}
        for branch in nft.psc.branches:
            name = branch.state_data.get("name")
            if name:
                prob = probs.get(branch.state_id, 0)
                result[name] = Decimal(str(prob))

        return result

    def observe(
        self,
        nft: QuantumNFT,
        seed: Optional[int] = None,
    ) -> NFTResult:
        """
        Observe the NFT, collapsing it to a definite state.

        Args:
            nft: NFT to observe
            seed: Optional random seed

        Returns:
            NFTResult with final state
        """
        if nft.observed:
            return NFTResult(
                status=ApplicationStatus.FAILED,
                psc_id=nft.psc.id,
                metadata={"error": "NFT already observed"},
                token_id=nft.token_id,
                final_state=nft.final_state,
            )

        # Get probabilities before collapse
        probs = self.get_superposition(nft)

        # Collapse PSC
        selected_state, selected_branch = nft.psc.collapse(seed=seed)

        # Find matching state
        final_state = None
        for state in nft.possible_states:
            if state.name == selected_branch.state_data.get("name"):
                final_state = state
                break

        nft.observed = True
        nft.final_state = final_state
        nft.observed_at = self.block_number

        result = NFTResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=nft.psc.id,
            selected_outcome=final_state.name if final_state else "unknown",
            probability_distribution={k: float(v) for k, v in probs.items()},
            metadata={
                "owner": nft.owner,
                "observed_at": nft.observed_at,
            },
            token_id=nft.token_id,
            collection_id=nft.collection_id,
            owner=nft.owner,
            final_state=final_state,
            rarity=final_state.rarity if final_state else "",
            n_possible_states=len(nft.possible_states),
        )

        self.observation_history.append(result)
        return result

    def transfer(
        self,
        nft: QuantumNFT,
        to: str,
    ) -> bool:
        """
        Transfer NFT to new owner.

        NFT can be transferred while still in superposition.

        Args:
            nft: NFT to transfer
            to: New owner address

        Returns:
            True if transfer successful
        """
        nft.owner = to
        return True

    def is_observed(
        self,
        nft: QuantumNFT,
    ) -> bool:
        """Check if NFT has been observed."""
        return nft.observed

    def get_rarity_stats(self) -> Dict[str, int]:
        """Get count of observed NFTs by rarity."""
        stats = {rarity: 0 for rarity in RARITY_WEIGHTS.keys()}

        for result in self.observation_history:
            if result.rarity:
                stats[result.rarity] = stats.get(result.rarity, 0) + 1

        return stats

    def get_status(self) -> Dict:
        """Get collection status."""
        return {
            "collection_id": self.collection_id,
            "total_minted": len(self.nfts),
            "observed": len([n for n in self.nfts.values() if n.observed]),
            "in_superposition": len([n for n in self.nfts.values() if not n.observed]),
            "rarity_stats": self.get_rarity_stats(),
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def create_sample_states(
    base_name: str = "CryptoCreature",
    n_states: int = 5,
) -> List[NFTState]:
    """
    Create sample NFT states with different rarities.

    Args:
        base_name: Base name for states
        n_states: Number of states to create

    Returns:
        List of NFTState instances
    """
    rarities = ["Legendary", "Epic", "Rare", "Uncommon", "Common"]

    states = []
    for i in range(n_states):
        rarity = rarities[i] if i < len(rarities) else "Common"

        state = NFTState(
            name=f"{base_name} #{i+1} ({rarity})",
            image_uri=f"ipfs://Qm.../{base_name.lower()}_{i+1}.png",
            attributes={
                "power": (5 - i) * 20,  # Higher for rarer
                "element": ["Fire", "Water", "Earth", "Air", "Void"][i % 5],
                "level": 1,
            },
            rarity=rarity,
        )
        states.append(state)

    return states


def simulate_quantum_nft(
    n_mints: int = 10,
    observe_all: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate quantum NFT minting and observation.

    Args:
        n_mints: Number of NFTs to mint
        observe_all: Whether to observe all minted NFTs
        seed: Random seed

    Returns:
        Simulation results
    """
    random.seed(seed)

    collection = QuantumNFTCollection("QuantumCreatures")
    states = create_sample_states()

    print(f"Collection: {collection.collection_id}")
    print(f"Possible states: {len(states)}")
    print()

    # Mint NFTs
    print(f"Minting {n_mints} quantum NFTs...")
    minted = []
    for i in range(n_mints):
        nft = collection.mint(states, owner=f"User_{i % 5}")
        minted.append(nft)

    # Show superposition
    if minted:
        sample = minted[0]
        print(f"\nSample NFT {sample.token_id} superposition:")
        probs = collection.get_superposition(sample)
        for name, prob in sorted(probs.items(), key=lambda x: -float(x[1])):
            print(f"  {name}: {float(prob)*100:.1f}%")

    # Observe NFTs
    if observe_all:
        print(f"\nObserving all NFTs...")
        results = []
        for i, nft in enumerate(minted):
            result = collection.observe(nft, seed=seed + i)
            results.append(result)

        # Show results
        print("\nObservation results:")
        rarity_counts = {}
        for result in results:
            rarity = result.rarity
            rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
            print(f"  {result.token_id}: {result.selected_outcome}")

        print("\nRarity distribution:")
        for rarity in ["Legendary", "Epic", "Rare", "Uncommon", "Common"]:
            count = rarity_counts.get(rarity, 0)
            expected = RARITY_WEIGHTS.get(rarity, 0) * 100
            print(f"  {rarity}: {count} ({count/n_mints*100:.0f}% - expected ~{expected:.0f}%)")

    return {
        "collection": collection.collection_id,
        "minted": len(minted),
        "observed": len([n for n in minted if n.observed]),
        "status": collection.get_status(),
    }


def run_demo():
    """Run quantum NFT demo."""
    print("=" * 60)
    print("Quantum NFT Collection Demo")
    print("=" * 60)
    print()

    simulate_quantum_nft(n_mints=20, seed=42)


if __name__ == "__main__":
    run_demo()
