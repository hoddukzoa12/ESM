"""
Backup Validator System

Based on ESM Whitepaper v5.2 Section 5.5-5.6

When the primary validators fail to meet the 67% reveal threshold even after
extension, backup validators are activated to ensure liveness.

Key features:
- 5 backup validators selected based on stake/reputation
- Backup validators receive 50% of slashed stake as reward
- Higher slashing (50%) for non-revealers when backup activates
- Deterministic backup selection for reproducibility
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import random
import hashlib

from esm.simulation.threshold_reveal import (
    BACKUP_VALIDATOR_COUNT,
    BACKUP_REWARD_SHARE,
    BACKUP_NON_REVEAL_SLASH_RATE,
    VdfCommitment,
    VdfReveal,
    CollapseStatus,
)


# =============================================================================
# Backup Validator Constants
# =============================================================================

BACKUP_SELECTION_SEED: str = "esm_backup_selection_v52"
MIN_BACKUP_STAKE: int = 100
BACKUP_VDF_TIMEOUT_BLOCKS: int = 3


class BackupStatus(Enum):
    """Status of backup validator activation."""
    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BackupValidator:
    """
    A backup validator candidate.

    Attributes:
        validator_id: Unique identifier
        stake: Staked amount
        reputation_score: Historical reliability (0-1)
        selection_weight: Combined weight for selection
    """
    validator_id: str
    stake: int
    reputation_score: float = 1.0
    selection_weight: float = 0.0

    def __post_init__(self):
        self.selection_weight = self.stake * self.reputation_score


@dataclass
class BackupActivation:
    """
    Record of backup validator activation.

    Attributes:
        activated_at: Block number of activation
        backup_validators: List of selected backup validator IDs
        trigger_reason: Why backup was needed
        primary_revealed_ratio: Ratio from primary reveal phase
        slashed_from_primary: Amount slashed from primary validators
        reward_pool: Reward pool for backup validators
    """
    activated_at: int
    backup_validators: List[str]
    trigger_reason: str
    primary_revealed_ratio: float
    slashed_from_primary: int
    reward_pool: int

    @property
    def reward_per_validator(self) -> int:
        """Calculate reward per backup validator."""
        if not self.backup_validators:
            return 0
        return self.reward_pool // len(self.backup_validators)


@dataclass
class BackupCollapseResult:
    """
    Result of backup validator collapse.

    Attributes:
        status: Backup status
        selected_validators: Validators selected for backup
        participating_validators: Validators that participated
        collapsed_state_seed: Seed for collapsed state
        rewards_distributed: Total rewards distributed
    """
    status: BackupStatus
    selected_validators: List[str]
    participating_validators: List[str]
    collapsed_state_seed: Optional[bytes] = None
    rewards_distributed: int = 0

    @property
    def participation_rate(self) -> float:
        """Calculate backup validator participation rate."""
        if not self.selected_validators:
            return 0.0
        return len(self.participating_validators) / len(self.selected_validators)


# =============================================================================
# Core Functions
# =============================================================================

def select_backup_validators(
    candidates: List[BackupValidator],
    n_backups: int = BACKUP_VALIDATOR_COUNT,
    exclude_validators: Optional[List[str]] = None,
    selection_seed: Optional[bytes] = None,
) -> List[BackupValidator]:
    """
    Select backup validators using weighted random selection.

    Selection is deterministic given the same seed and candidates.

    Args:
        candidates: List of potential backup validators
        n_backups: Number of backup validators to select
        exclude_validators: Validators to exclude (e.g., failed primary validators)
        selection_seed: Seed for deterministic selection

    Returns:
        List of selected backup validators
    """
    if exclude_validators is None:
        exclude_validators = []

    # Filter eligible candidates
    eligible = [
        c for c in candidates
        if c.validator_id not in exclude_validators
        and c.stake >= MIN_BACKUP_STAKE
    ]

    if len(eligible) <= n_backups:
        return eligible

    # Deterministic random selection based on seed
    if selection_seed:
        random.seed(int.from_bytes(selection_seed[:8], 'big'))

    # Weighted selection based on stake * reputation
    weights = [c.selection_weight for c in eligible]
    total_weight = sum(weights)

    if total_weight == 0:
        return eligible[:n_backups]

    selected = []
    remaining = list(zip(eligible, weights))

    for _ in range(n_backups):
        if not remaining:
            break

        candidates_left, weights_left = zip(*remaining)
        total = sum(weights_left)

        r = random.random() * total
        cumulative = 0
        for i, (candidate, weight) in enumerate(remaining):
            cumulative += weight
            if r <= cumulative:
                selected.append(candidate)
                remaining.pop(i)
                break

    return selected


def calculate_backup_rewards(
    slashed_amount: int,
    n_backup_validators: int,
    reward_share: float = BACKUP_REWARD_SHARE,
) -> Tuple[int, int]:
    """
    Calculate rewards for backup validators.

    Args:
        slashed_amount: Total amount slashed from primary validators
        n_backup_validators: Number of backup validators
        reward_share: Share of slashed amount for backup (default 50%)

    Returns:
        Tuple of (total_reward, reward_per_validator)
    """
    total_reward = int(slashed_amount * reward_share)
    if n_backup_validators == 0:
        return total_reward, 0
    return total_reward, total_reward // n_backup_validators


def activate_backup_validators(
    failed_commitments: List[VdfCommitment],
    candidates: List[BackupValidator],
    activation_block: int,
    selection_seed: Optional[bytes] = None,
) -> BackupActivation:
    """
    Activate backup validators after primary reveal failure.

    Args:
        failed_commitments: Commitments that failed to reveal
        candidates: Potential backup validators
        activation_block: Block number of activation
        selection_seed: Seed for selection

    Returns:
        BackupActivation record
    """
    # Calculate slashing
    slashed_amount = int(sum(
        c.stake * BACKUP_NON_REVEAL_SLASH_RATE
        for c in failed_commitments
    ))

    # Exclude failed validators from backup selection
    failed_ids = {c.validator_id for c in failed_commitments}

    # Select backup validators
    selected = select_backup_validators(
        candidates=candidates,
        exclude_validators=list(failed_ids),
        selection_seed=selection_seed,
    )

    reward_pool = int(slashed_amount * BACKUP_REWARD_SHARE)

    return BackupActivation(
        activated_at=activation_block,
        backup_validators=[v.validator_id for v in selected],
        trigger_reason="threshold_not_met",
        primary_revealed_ratio=0.0,  # Will be set by caller
        slashed_from_primary=slashed_amount,
        reward_pool=reward_pool,
    )


def simulate_backup_collapse(
    activation: BackupActivation,
    backup_reveal_rate: float = 0.9,
    seed: Optional[int] = None,
) -> BackupCollapseResult:
    """
    Simulate backup validator collapse.

    Args:
        activation: Backup activation record
        backup_reveal_rate: Probability each backup validator participates
        seed: Random seed

    Returns:
        BackupCollapseResult
    """
    if seed is not None:
        random.seed(seed)

    participating = [
        v for v in activation.backup_validators
        if random.random() < backup_reveal_rate
    ]

    if not participating:
        return BackupCollapseResult(
            status=BackupStatus.FAILED,
            selected_validators=activation.backup_validators,
            participating_validators=[],
            collapsed_state_seed=None,
            rewards_distributed=0,
        )

    # Generate collapsed state from backup VDFs
    seed_data = b"".join(v.encode() for v in sorted(participating))
    collapsed_state_seed = hashlib.sha256(seed_data).digest()

    # Distribute rewards to participating validators
    rewards = activation.reward_pool // len(participating) if participating else 0
    total_rewards = rewards * len(participating)

    return BackupCollapseResult(
        status=BackupStatus.COMPLETED,
        selected_validators=activation.backup_validators,
        participating_validators=participating,
        collapsed_state_seed=collapsed_state_seed,
        rewards_distributed=total_rewards,
    )


# =============================================================================
# Simulation Helpers
# =============================================================================

def create_validator_pool(
    n_validators: int,
    stake_range: Tuple[int, int] = (100, 5000),
    reputation_range: Tuple[float, float] = (0.5, 1.0),
    seed: Optional[int] = None,
) -> List[BackupValidator]:
    """
    Create a pool of potential backup validators.

    Args:
        n_validators: Number of validators
        stake_range: Range of stake values
        reputation_range: Range of reputation scores
        seed: Random seed

    Returns:
        List of BackupValidator instances
    """
    if seed is not None:
        random.seed(seed)

    validators = []
    for i in range(n_validators):
        validators.append(BackupValidator(
            validator_id=f"validator_{i}",
            stake=random.randint(*stake_range),
            reputation_score=random.uniform(*reputation_range),
        ))

    return validators


def simulate_backup_scenario(
    n_primary: int = 20,
    n_backup_candidates: int = 30,
    primary_reveal_rate: float = 0.5,
    backup_reveal_rate: float = 0.9,
    seed: int = 42,
) -> Dict:
    """
    Simulate a full backup scenario.

    Args:
        n_primary: Number of primary validators
        n_backup_candidates: Number of backup candidates
        primary_reveal_rate: Primary validator reveal rate
        backup_reveal_rate: Backup validator reveal rate
        seed: Random seed

    Returns:
        Simulation results
    """
    random.seed(seed)

    # Create primary validators
    primary_commitments = []
    for i in range(n_primary):
        commitment = VdfCommitment(
            validator_id=f"primary_{i}",
            commitment_hash=hashlib.sha256(f"hash_{i}".encode()).hexdigest(),
            stake=1000,
            committed_at=0,
        )
        primary_commitments.append(commitment)

    # Simulate primary reveal failure
    revealed = [c for c in primary_commitments if random.random() < primary_reveal_rate]
    failed = [c for c in primary_commitments if c not in revealed]

    revealed_ratio = len(revealed) / len(primary_commitments)

    # Create backup candidates
    backup_candidates = create_validator_pool(n_backup_candidates, seed=seed + 1)

    # Activate backup
    selection_seed = hashlib.sha256(f"backup_seed_{seed}".encode()).digest()
    activation = activate_backup_validators(
        failed_commitments=failed,
        candidates=backup_candidates,
        activation_block=100,
        selection_seed=selection_seed,
    )
    activation.primary_revealed_ratio = revealed_ratio

    # Simulate backup collapse
    backup_result = simulate_backup_collapse(
        activation=activation,
        backup_reveal_rate=backup_reveal_rate,
        seed=seed + 2,
    )

    return {
        "primary": {
            "n_validators": n_primary,
            "revealed_count": len(revealed),
            "failed_count": len(failed),
            "revealed_ratio": revealed_ratio,
        },
        "backup": {
            "n_candidates": n_backup_candidates,
            "selected_count": len(activation.backup_validators),
            "participating_count": len(backup_result.participating_validators),
            "participation_rate": backup_result.participation_rate,
        },
        "economics": {
            "slashed_amount": activation.slashed_from_primary,
            "reward_pool": activation.reward_pool,
            "rewards_distributed": backup_result.rewards_distributed,
        },
        "outcome": {
            "status": backup_result.status.value,
            "is_successful": backup_result.status == BackupStatus.COMPLETED,
        },
    }


def analyze_backup_reliability(
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Analyze backup system reliability across scenarios.

    Args:
        n_simulations: Number of simulations
        seed: Random seed

    Returns:
        Analysis results
    """
    scenarios = [
        {"primary_reveal_rate": 0.3, "backup_reveal_rate": 0.9},
        {"primary_reveal_rate": 0.5, "backup_reveal_rate": 0.9},
        {"primary_reveal_rate": 0.6, "backup_reveal_rate": 0.9},
        {"primary_reveal_rate": 0.5, "backup_reveal_rate": 0.7},
        {"primary_reveal_rate": 0.5, "backup_reveal_rate": 0.5},
    ]

    results = {}

    for scenario in scenarios:
        key = f"primary_{scenario['primary_reveal_rate']}_backup_{scenario['backup_reveal_rate']}"
        successes = 0
        total_slashed = 0
        total_rewards = 0

        for i in range(n_simulations):
            sim = simulate_backup_scenario(
                primary_reveal_rate=scenario["primary_reveal_rate"],
                backup_reveal_rate=scenario["backup_reveal_rate"],
                seed=seed + i,
            )

            if sim["outcome"]["is_successful"]:
                successes += 1
            total_slashed += sim["economics"]["slashed_amount"]
            total_rewards += sim["economics"]["rewards_distributed"]

        results[key] = {
            "success_rate": successes / n_simulations,
            "avg_slashed": total_slashed / n_simulations,
            "avg_rewards": total_rewards / n_simulations,
            **scenario,
        }

    return results
