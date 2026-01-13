"""
Base Application Interface

Common interfaces and utilities for ESM applications.
Based on ESM Whitepaper v5.3 Section 8.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from decimal import Decimal

from esm.core.psc import PSC, create_psc
from esm.core.branch import Branch, create_branch
from esm.core.amplitude import DiscreteAmplitude
from esm.core.phase import DiscretePhase


class ApplicationStatus(Enum):
    """Status of application operation."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ApplicationResult:
    """
    Base result class for all applications.

    Attributes:
        status: Operation status
        psc_id: Associated PSC identifier
        selected_outcome: Final selected outcome (if collapsed)
        probability_distribution: Pre-collapse probabilities
        metadata: Additional application-specific data
    """
    status: ApplicationStatus
    psc_id: str
    selected_outcome: Optional[str] = None
    probability_distribution: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if operation completed successfully."""
        return self.status == ApplicationStatus.COMPLETED


class ESMApplication(ABC):
    """
    Abstract base class for ESM applications.

    Provides common infrastructure for PSC management
    and interference-based state handling.
    """

    def __init__(self, app_name: str):
        """
        Initialize application.

        Args:
            app_name: Name of this application instance
        """
        self.app_name = app_name
        self.pscs: Dict[str, PSC] = {}
        self.block_number: int = 0

    def create_application_psc(self, psc_id: Optional[str] = None) -> PSC:
        """
        Create a new PSC for this application.

        Args:
            psc_id: Optional custom PSC ID

        Returns:
            New PSC instance
        """
        psc = create_psc(psc_id)
        psc.created_at = self.block_number
        self.pscs[psc.id] = psc
        return psc

    def get_psc(self, psc_id: str) -> Optional[PSC]:
        """Get PSC by ID."""
        return self.pscs.get(psc_id)

    def advance_block(self, blocks: int = 1) -> None:
        """Advance simulation block number."""
        self.block_number += blocks

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current application status."""
        pass

    def add_branch_to_psc(
        self,
        psc: PSC,
        state_data: Dict[str, Any],
        magnitude: float = 1.0,
        phase: DiscretePhase = DiscretePhase.P0,
        creator: str = "system",
    ) -> Branch:
        """
        Add a branch to a PSC.

        Args:
            psc: Target PSC
            state_data: State data for the branch
            magnitude: Amplitude magnitude
            phase: Amplitude phase
            creator: Creator identifier

        Returns:
            Created branch
        """
        branch = create_branch(
            state_data=state_data,
            magnitude=magnitude,
            phase=phase,
            creator=creator,
        )
        psc.add_branch(branch)
        return branch

    def collapse_psc(
        self,
        psc: PSC,
        seed: Optional[int] = None,
    ) -> Tuple[str, Branch]:
        """
        Collapse a PSC to a definite state.

        Args:
            psc: PSC to collapse
            seed: Optional random seed

        Returns:
            Tuple of (selected_state_id, selected_branch)
        """
        return psc.collapse(seed=seed)

    def get_probabilities(self, psc: PSC) -> Dict[str, float]:
        """Get current probability distribution for a PSC."""
        return psc.get_probabilities()


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_interference_impact(
    phase1: DiscretePhase,
    phase2: DiscretePhase,
) -> str:
    """
    Calculate and describe the interference between two phases.

    Args:
        phase1: First phase
        phase2: Second phase

    Returns:
        Description of interference type
    """
    diff = (phase2.value - phase1.value) % 8

    interference_types = {
        0: "constructive_full",      # 0 degrees
        1: "constructive_partial",   # 45 degrees
        2: "orthogonal",             # 90 degrees
        3: "destructive_partial",    # 135 degrees
        4: "destructive_full",       # 180 degrees
        5: "destructive_partial",    # 225 degrees
        6: "orthogonal",             # 270 degrees
        7: "constructive_partial",   # 315 degrees
    }

    return interference_types[diff]


def distribute_amplitudes(
    n_outcomes: int,
    distribution: str = "equal",
    weights: Optional[List[float]] = None,
) -> List[Tuple[float, DiscretePhase]]:
    """
    Distribute amplitudes across multiple outcomes.

    Args:
        n_outcomes: Number of outcomes
        distribution: Distribution type ("equal", "weighted", "superposition")
        weights: Custom weights (for "weighted" distribution)

    Returns:
        List of (magnitude, phase) tuples for each outcome
    """
    if distribution == "equal":
        # Equal probability distribution
        # |alpha|^2 = 1/n for each, so |alpha| = 1/sqrt(n)
        magnitude = 1.0 / (n_outcomes ** 0.5)
        return [(magnitude, DiscretePhase.P0) for _ in range(n_outcomes)]

    elif distribution == "weighted" and weights:
        # Custom weighted distribution
        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]
        # Magnitude = sqrt(probability)
        magnitudes = [w ** 0.5 for w in normalized]
        return [(m, DiscretePhase.P0) for m in magnitudes]

    elif distribution == "superposition":
        # Create superposition with different phases
        # Each outcome gets a different phase for maximum interference potential
        magnitude = 1.0 / (n_outcomes ** 0.5)
        phases = [DiscretePhase(i % 8) for i in range(n_outcomes)]
        return list(zip([magnitude] * n_outcomes, phases))

    else:
        # Default to equal
        magnitude = 1.0 / (n_outcomes ** 0.5)
        return [(magnitude, DiscretePhase.P0) for _ in range(n_outcomes)]


def normalize_amplitudes(branches: List[Branch]) -> None:
    """
    Normalize branch amplitudes so probabilities sum to 1.

    Modifies branches in place.

    Args:
        branches: List of branches to normalize
    """
    total_prob = sum(b.probability() for b in branches)

    if total_prob > 0:
        factor = 1.0 / (total_prob ** 0.5)
        for branch in branches:
            new_amp = DiscreteAmplitude(
                branch.amplitude.magnitude * factor,
                branch.amplitude.phase,
            )
            branch.amplitude = new_amp
