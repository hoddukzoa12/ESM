"""
Probabilistic State Container (PSC)

Based on ESM Whitepaper v5.2 Section 3.1

PSC is the fundamental data structure for probabilistic state in ESM.
It contains multiple branches, each representing a possible state with
a discrete amplitude. Branches with the same state_id interfere with each other.

v5.2 additions:
- collapse_deadline: Block deadline for collapse operations
- amplitude_fee_pool: Accumulated fees from amplitude operations
- total_interference_deposit: Sum of all branch interference deposits
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from hashlib import sha256
import random
import numpy as np

from esm.core.amplitude import DiscreteAmplitude, zero_amplitude
from esm.core.branch import Branch
from esm.core.phase import DiscretePhase


@dataclass
class PSC:
    """
    Probabilistic State Container.

    A PSC holds multiple branches representing possible states.
    Branches with the same state_id undergo interference during calculation.

    Attributes:
        id: Unique identifier for this PSC
        created_at: Block number when created
        branches: List of branches in this PSC
        collapse_after: Block number after which collapse can occur

        # v5.2 additions:
        collapse_deadline: Block deadline for collapse (0 = no deadline)
        amplitude_fee_pool: Accumulated fees from amplitude operations
        total_interference_deposit: Sum of all branch interference deposits
    """
    id: str
    created_at: int = 0
    branches: List[Branch] = field(default_factory=list)
    collapse_after: int = 0

    # v5.2 fields
    collapse_deadline: int = 0
    amplitude_fee_pool: int = 0
    total_interference_deposit: int = 0

    # Caching for efficiency
    _interference_dirty: bool = True
    _cached_interference: Dict[str, DiscreteAmplitude] = field(default_factory=dict)
    _cached_probabilities: Dict[str, float] = field(default_factory=dict)

    # =========================================================================
    # Branch Management
    # =========================================================================

    def add_branch(self, branch: Branch) -> None:
        """
        Add a branch to this PSC.

        Args:
            branch: Branch to add
        """
        self.branches.append(branch)
        self._interference_dirty = True
        # v5.2: Track total interference deposit
        self.total_interference_deposit += branch.interference_deposit

    def remove_branch(self, index: int) -> Branch:
        """
        Remove a branch by index.

        Args:
            index: Index of branch to remove

        Returns:
            The removed branch
        """
        branch = self.branches.pop(index)
        self._interference_dirty = True
        # v5.2: Update total interference deposit
        self.total_interference_deposit -= branch.interference_deposit
        return branch

    def get_branches_by_state(self, state_id: str) -> List[Branch]:
        """
        Get all branches with a specific state_id.

        Args:
            state_id: State ID to filter by

        Returns:
            List of branches with matching state_id
        """
        return [b for b in self.branches if b.state_id == state_id]

    def get_unique_state_ids(self) -> List[str]:
        """Get list of unique state IDs in this PSC."""
        return list(set(b.state_id for b in self.branches))

    # =========================================================================
    # Interference Calculation (Whitepaper Section 6.1)
    # =========================================================================

    def calculate_interference(self) -> Dict[str, DiscreteAmplitude]:
        """
        Calculate interference for all branches.

        Implements local interference: branches with the same state_id
        have their amplitudes summed (with phase interference).

        This is the core ESM mechanism - amplitudes can cancel or reinforce.

        Returns:
            Dictionary mapping state_id to resulting amplitude
        """
        if not self._interference_dirty and self._cached_interference:
            return self._cached_interference

        # Group branches by state_id
        groups: Dict[str, List[Branch]] = defaultdict(list)
        for branch in self.branches:
            groups[branch.state_id].append(branch)

        # Calculate interference within each group
        result: Dict[str, DiscreteAmplitude] = {}
        for state_id, branches in groups.items():
            if len(branches) == 1:
                result[state_id] = branches[0].amplitude
            else:
                # Sum all amplitudes (interference)
                total = branches[0].amplitude
                for branch in branches[1:]:
                    total = total.add(branch.amplitude)
                result[state_id] = total

        self._cached_interference = result
        self._interference_dirty = False
        return result

    def get_probabilities(self) -> Dict[str, float]:
        """
        Get normalized probabilities for each state.

        Returns:
            Dictionary mapping state_id to probability (summing to 1.0)
        """
        if not self._interference_dirty and self._cached_probabilities:
            return self._cached_probabilities

        interference = self.calculate_interference()

        # Calculate raw probabilities (|α|²)
        raw_probs = {sid: amp.probability() for sid, amp in interference.items()}

        # Normalize
        total = sum(raw_probs.values())
        if total == 0:
            # All cancelled out - equal distribution
            n = len(raw_probs)
            self._cached_probabilities = {sid: 1.0/n for sid in raw_probs}
        else:
            self._cached_probabilities = {sid: p/total for sid, p in raw_probs.items()}

        return self._cached_probabilities

    def get_probability(self, state_id: str) -> float:
        """
        Get probability for a specific state.

        Args:
            state_id: State ID to query

        Returns:
            Probability value (0 to 1)
        """
        probs = self.get_probabilities()
        return probs.get(state_id, 0.0)

    # =========================================================================
    # Collapse (Whitepaper Section 2.6)
    # =========================================================================

    def collapse(self, seed: Optional[int] = None) -> Tuple[str, Branch]:
        """
        Collapse the PSC to a single definite state.

        Selects a state probabilistically based on interference-adjusted
        probabilities, then removes all other branches.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (selected_state_id, selected_branch)
        """
        if seed is not None:
            random.seed(seed)

        probs = self.get_probabilities()

        # Weighted random selection
        state_ids = list(probs.keys())
        weights = [probs[sid] for sid in state_ids]

        selected_state_id = random.choices(state_ids, weights=weights, k=1)[0]

        # Find a representative branch for the selected state
        selected_branches = self.get_branches_by_state(selected_state_id)
        selected_branch = selected_branches[0]  # Take first

        # Collapse: keep only selected state
        self.branches = [
            Branch(
                state_id=selected_state_id,
                state_data=selected_branch.state_data,
                amplitude=DiscreteAmplitude(1.0, DiscretePhase.P0),
                creator=selected_branch.creator,
                created_at=selected_branch.created_at,
                tx_type=selected_branch.tx_type,
                tx_timestamp=selected_branch.tx_timestamp,
                # v5.2 fields preserved
                interference_deposit=selected_branch.interference_deposit,
                stake_locked=selected_branch.stake_locked,
            )
        ]
        self._interference_dirty = True
        # v5.2: Reset total deposit to collapsed branch's deposit only
        self.total_interference_deposit = selected_branch.interference_deposit

        return selected_state_id, selected_branch

    # =========================================================================
    # Analysis and Reporting
    # =========================================================================

    def get_interference_report(self) -> Dict:
        """
        Generate a detailed interference report.

        Returns:
            Dictionary with detailed interference analysis
        """
        # Group by state_id
        groups: Dict[str, List[Branch]] = defaultdict(list)
        for branch in self.branches:
            groups[branch.state_id].append(branch)

        interference = self.calculate_interference()
        probs = self.get_probabilities()

        report = {
            "total_branches": len(self.branches),
            "unique_states": len(groups),
            "states": {}
        }

        for state_id, branches in groups.items():
            state_report = {
                "branch_count": len(branches),
                "branches": [],
                "result_amplitude": interference[state_id],
                "probability": probs[state_id],
            }

            for branch in branches:
                state_report["branches"].append({
                    "magnitude": branch.amplitude.magnitude,
                    "phase": branch.amplitude.phase.name,
                    "tx_type": branch.tx_type,
                    "probability_contribution": branch.probability(),
                })

            # Calculate interference effect
            sum_individual_probs = sum(b.probability() for b in branches)
            result_prob = interference[state_id].probability()

            if sum_individual_probs > 0:
                interference_effect = (result_prob - sum_individual_probs) / sum_individual_probs
            else:
                interference_effect = 0

            state_report["interference_effect"] = interference_effect
            state_report["interference_type"] = (
                "constructive" if interference_effect > 0.01
                else "destructive" if interference_effect < -0.01
                else "neutral"
            )

            report["states"][state_id[:16]] = state_report

        return report

    def total_probability_before_interference(self) -> float:
        """Calculate sum of individual branch probabilities (before interference)."""
        return sum(b.probability() for b in self.branches)

    def total_probability_after_interference(self) -> float:
        """Calculate sum of probabilities after interference."""
        interference = self.calculate_interference()
        return sum(amp.probability() for amp in interference.values())

    def interference_impact(self) -> float:
        """
        Calculate overall interference impact.

        Returns:
            Ratio of (after/before) probability sums.
            < 1.0 means net destructive interference
            > 1.0 means net constructive interference
        """
        before = self.total_probability_before_interference()
        after = self.total_probability_after_interference()
        return after / before if before > 0 else 1.0

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        return f"PSC(id={self.id[:8]}..., branches={len(self.branches)})"

    def summary(self) -> str:
        """Get a text summary of the PSC state."""
        probs = self.get_probabilities()
        lines = [f"PSC {self.id[:8]}... ({len(self.branches)} branches)"]
        lines.append("-" * 40)
        for state_id, prob in sorted(probs.items(), key=lambda x: -x[1]):
            lines.append(f"  {state_id[:16]}...: {prob*100:.1f}%")
        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================

def create_psc(psc_id: Optional[str] = None) -> PSC:
    """
    Create a new empty PSC.

    Args:
        psc_id: Optional ID (generated if not provided)

    Returns:
        New PSC instance
    """
    if psc_id is None:
        psc_id = sha256(str(random.random()).encode()).hexdigest()
    return PSC(id=psc_id)


def create_simple_psc(states: List[Tuple[str, float, DiscretePhase]]) -> PSC:
    """
    Create a PSC with simple state definitions.

    Args:
        states: List of (state_data_str, magnitude, phase) tuples

    Returns:
        PSC with the specified branches
    """
    psc = create_psc()
    for state_data_str, magnitude, phase in states:
        branch = Branch(
            state_id=sha256(state_data_str.encode()).hexdigest(),
            state_data={"value": state_data_str},
            amplitude=DiscreteAmplitude(magnitude, phase),
        )
        psc.add_branch(branch)
    return psc
