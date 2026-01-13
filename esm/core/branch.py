"""
Branch Data Structure

Based on ESM Whitepaper v5.2 Section 3.1

A Branch represents one possible state outcome within a PSC (Probabilistic State Container).
Each branch has a discrete amplitude that determines its probability of being selected
during collapse.

v5.2 additions:
- interference_deposit: Prepaid deposit for interference costs
- stake_locked: Locked stake for branch validation
- mode parameter support using semantic aliases
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from hashlib import sha256
import time

from esm.core.amplitude import DiscreteAmplitude
from esm.core.phase import DiscretePhase, InterferenceMode, phase_from_mode


@dataclass
class Branch:
    """
    A branch within a PSC representing one possible state.

    Attributes:
        state_id: Hash identifying the state (branches with same state_id interfere)
        state_data: Arbitrary data representing the state
        amplitude: Discrete amplitude determining probability
        creator: Address/ID of the branch creator
        created_at: Block number when created
        tx_type: Transaction type for MEV simulation ("normal", "victim", "attacker")
        tx_timestamp: Timestamp in milliseconds for MEV timing analysis

        # v5.2 additions:
        interference_deposit: Prepaid deposit for interference calculation costs
        stake_locked: Locked stake for branch validation
    """
    state_id: str
    state_data: Dict[str, Any]
    amplitude: DiscreteAmplitude
    creator: str = "system"
    created_at: int = 0
    tx_type: str = "normal"  # "normal", "victim", "attacker"
    tx_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    # v5.2 fields
    interference_deposit: int = 0
    stake_locked: int = 0

    def probability(self) -> float:
        """Get the probability of this branch (|α|²)."""
        return self.amplitude.probability()

    def with_amplitude(self, new_amplitude: DiscreteAmplitude) -> Branch:
        """Create a copy with a different amplitude."""
        return Branch(
            state_id=self.state_id,
            state_data=self.state_data.copy(),
            amplitude=new_amplitude,
            creator=self.creator,
            created_at=self.created_at,
            tx_type=self.tx_type,
            tx_timestamp=self.tx_timestamp,
            interference_deposit=self.interference_deposit,
            stake_locked=self.stake_locked,
        )

    def with_phase(self, new_phase: DiscretePhase) -> Branch:
        """Create a copy with a different phase (same magnitude)."""
        return self.with_amplitude(
            DiscreteAmplitude(self.amplitude.magnitude, new_phase)
        )

    def scale(self, factor: float) -> Branch:
        """Create a copy with scaled amplitude."""
        return self.with_amplitude(self.amplitude.scale(factor))

    def __repr__(self) -> str:
        return (
            f"Branch(state_id={self.state_id[:8]}..., "
            f"amplitude={self.amplitude}, "
            f"tx_type={self.tx_type})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_branch(
    state_data: Dict[str, Any],
    magnitude: float = 1.0,
    phase: DiscretePhase = DiscretePhase.P0,
    state_id: Optional[str] = None,
    creator: str = "system",
    tx_type: str = "normal",
    interference_deposit: int = 0,
    stake_locked: int = 0,
) -> Branch:
    """
    Create a new branch with the given parameters.

    Args:
        state_data: Data representing the state
        magnitude: Amplitude magnitude (default 1.0)
        phase: Amplitude phase (default P0)
        state_id: Optional state ID (computed from state_data if not provided)
        creator: Creator identifier
        tx_type: Transaction type for MEV simulation
        interference_deposit: v5.2 prepaid deposit (default 0)
        stake_locked: v5.2 locked stake (default 0)

    Returns:
        New Branch instance
    """
    if state_id is None:
        state_id = compute_state_id(state_data)

    return Branch(
        state_id=state_id,
        state_data=state_data,
        amplitude=DiscreteAmplitude(magnitude, phase),
        creator=creator,
        tx_type=tx_type,
        interference_deposit=interference_deposit,
        stake_locked=stake_locked,
    )


def create_branch_with_mode(
    state_data: Dict[str, Any],
    mode: Union[str, InterferenceMode] = "Normal",
    magnitude: float = 1.0,
    state_id: Optional[str] = None,
    creator: str = "system",
    tx_type: str = "normal",
    interference_deposit: int = 0,
    stake_locked: int = 0,
) -> Branch:
    """
    Create a new branch using semantic alias mode (v5.2).

    This is the SDK-friendly way to create branches using intuitive
    mode names instead of raw phase values.

    Args:
        state_data: Data representing the state
        mode: Interference mode ("Normal", "Counter", "Independent", etc.)
        magnitude: Amplitude magnitude (default 1.0)
        state_id: Optional state ID (computed from state_data if not provided)
        creator: Creator identifier
        tx_type: Transaction type for MEV simulation
        interference_deposit: v5.2 prepaid deposit (default 0)
        stake_locked: v5.2 locked stake (default 0)

    Returns:
        New Branch instance

    Examples:
        # General transaction
        branch = create_branch_with_mode({"amount": 100}, mode="Normal")

        # MEV defense (opposite phase)
        branch = create_branch_with_mode({"amount": 100}, mode="Counter")

        # Independent state
        branch = create_branch_with_mode({"amount": 100}, mode="Independent")
    """
    phase = phase_from_mode(mode)
    return create_branch(
        state_data=state_data,
        magnitude=magnitude,
        phase=phase,
        state_id=state_id,
        creator=creator,
        tx_type=tx_type,
        interference_deposit=interference_deposit,
        stake_locked=stake_locked,
    )


def compute_state_id(state_data: Dict[str, Any]) -> str:
    """
    Compute a deterministic state ID from state data.

    Args:
        state_data: Dictionary of state data

    Returns:
        SHA256 hash as hex string
    """
    # Sort keys for deterministic ordering
    sorted_items = sorted(state_data.items())
    data_str = str(sorted_items)
    return sha256(data_str.encode()).hexdigest()


# =============================================================================
# MEV Simulation Helpers
# =============================================================================

def create_victim_tx(
    amount: float,
    target: str = "DEX",
    magnitude: float = 1.0,
) -> Branch:
    """
    Create a victim transaction branch for MEV simulation.

    Args:
        amount: Transaction amount
        target: Target contract/DEX
        magnitude: Amplitude magnitude

    Returns:
        Branch representing victim transaction
    """
    return create_branch(
        state_data={
            "type": "swap",
            "amount": amount,
            "target": target,
        },
        magnitude=magnitude,
        phase=DiscretePhase.P0,  # Victim always starts at 0°
        tx_type="victim",
    )


def create_attacker_tx(
    amount: float,
    target: str = "DEX",
    phase: DiscretePhase = DiscretePhase.P180,
    magnitude: float = 1.0,
    delay_ms: int = 0,
) -> Branch:
    """
    Create an attacker transaction branch for MEV simulation.

    Args:
        amount: Transaction amount
        target: Target contract/DEX
        phase: Phase assigned based on timing (from MEV defense mechanism)
        magnitude: Amplitude magnitude
        delay_ms: Delay after victim transaction in milliseconds

    Returns:
        Branch representing attacker transaction
    """
    return create_branch(
        state_data={
            "type": "frontrun",
            "amount": amount,
            "target": target,
            "delay_ms": delay_ms,
        },
        magnitude=magnitude,
        phase=phase,
        tx_type="attacker",
    )
