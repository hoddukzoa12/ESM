"""
Threshold Reveal Simulation

Based on ESM Whitepaper v5.2 Section 5.2-5.6

The Threshold Reveal mechanism ensures PSC collapse can proceed even when
some validators fail to reveal their VDF outputs. Key features:

- 67% stake threshold for valid collapse
- 5-block extension period for additional reveals
- 10% slashing for non-revealers (50% if backup activates)
- Backup validator activation when threshold not met

Protocol Flow:
1. Validators commit VDF output hash during commitment phase
2. After deadline, validators reveal VDF output + proof
3. If 67% stake reveals, collapse proceeds normally
4. If below threshold, 5-block extension activates
5. If still below after extension, backup validators take over
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import random
import hashlib


# =============================================================================
# Protocol Constants (from Whitepaper Section 5.3-5.6)
# =============================================================================

REVEAL_THRESHOLD: float = 0.67  # 67% (2/3) stake threshold
REVEAL_EXTENSION_BLOCKS: int = 5  # Extension period in blocks
NON_REVEAL_SLASH_RATE: float = 0.10  # 10% slashing for non-revealers
BACKUP_NON_REVEAL_SLASH_RATE: float = 0.50  # 50% slashing when backup activates
BACKUP_REWARD_SHARE: float = 0.50  # 50% of slashed stake goes to backup
BACKUP_VALIDATOR_COUNT: int = 5  # Number of backup validators


class CollapseStatus(Enum):
    """Status of collapse operation."""
    SUCCESS = "success"
    BELOW_THRESHOLD = "below_threshold"
    EXTENSION_ACTIVE = "extension_active"
    BACKUP_ACTIVATED = "backup_activated"
    FAILED = "failed"


class RevealStatus(Enum):
    """Status of individual reveal."""
    REVEALED = "revealed"
    PENDING = "pending"
    MISSED = "missed"
    SLASHED = "slashed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VdfCommitment:
    """
    VDF commitment from a validator.

    Attributes:
        validator_id: Unique validator identifier
        commitment_hash: Hash of VDF output + salt
        stake: Validator's staked amount
        committed_at: Block number of commitment
    """
    validator_id: str
    commitment_hash: str
    stake: int
    committed_at: int


@dataclass
class VdfReveal:
    """
    VDF reveal from a validator.

    Attributes:
        validator_id: Validator identifier (must match commitment)
        vdf_output: VDF computation output (y)
        proof: VDF correctness proof
        salt: Random salt used in commitment
        revealed_at: Block number of reveal
    """
    validator_id: str
    vdf_output: bytes
    proof: bytes
    salt: bytes
    revealed_at: int = 0


@dataclass
class ValidatorState:
    """
    Complete state of a validator in the reveal process.

    Attributes:
        commitment: VDF commitment
        reveal: Optional VDF reveal
        status: Current reveal status
        slash_amount: Amount to be slashed (if any)
    """
    commitment: VdfCommitment
    reveal: Optional[VdfReveal] = None
    status: RevealStatus = RevealStatus.PENDING
    slash_amount: int = 0


@dataclass
class CollapseResult:
    """
    Result of collapse operation.

    Attributes:
        status: Collapse status
        revealed_stake: Total stake that revealed
        total_stake: Total committed stake
        revealed_ratio: Ratio of revealed to total stake
        slashed_validators: List of slashed validator IDs
        total_slashed: Total amount slashed
        backup_activated: Whether backup validators were needed
        collapsed_state_seed: Seed for state selection (if successful)
    """
    status: CollapseStatus
    revealed_stake: int
    total_stake: int
    revealed_ratio: float
    slashed_validators: List[str] = field(default_factory=list)
    total_slashed: int = 0
    backup_activated: bool = False
    collapsed_state_seed: Optional[bytes] = None

    @property
    def is_successful(self) -> bool:
        """Check if collapse was successful."""
        return self.status in [CollapseStatus.SUCCESS, CollapseStatus.BACKUP_ACTIVATED]


# =============================================================================
# Core Functions
# =============================================================================

def calculate_reveal_ratio(
    commitments: List[VdfCommitment],
    reveals: List[VdfReveal],
) -> Tuple[int, int, float]:
    """
    Calculate reveal ratio based on stake.

    Args:
        commitments: List of VDF commitments
        reveals: List of VDF reveals

    Returns:
        Tuple of (revealed_stake, total_stake, ratio)
    """
    total_stake = sum(c.stake for c in commitments)
    if total_stake == 0:
        return 0, 0, 0.0

    revealed_validators = {r.validator_id for r in reveals}
    revealed_stake = sum(
        c.stake for c in commitments
        if c.validator_id in revealed_validators
    )

    return revealed_stake, total_stake, revealed_stake / total_stake


def verify_reveal(commitment: VdfCommitment, reveal: VdfReveal) -> bool:
    """
    Verify that a reveal matches its commitment.

    Args:
        commitment: Original VDF commitment
        reveal: Claimed VDF reveal

    Returns:
        True if reveal is valid for commitment
    """
    if commitment.validator_id != reveal.validator_id:
        return False

    # Recompute commitment hash
    data = reveal.vdf_output + reveal.salt
    expected_hash = hashlib.sha256(data).hexdigest()

    return expected_hash == commitment.commitment_hash


def process_threshold_reveal(
    commitments: List[VdfCommitment],
    reveals: List[VdfReveal],
    is_extension_phase: bool = False,
) -> CollapseResult:
    """
    Process threshold reveal and determine collapse outcome.

    Args:
        commitments: List of VDF commitments
        reveals: List of VDF reveals
        is_extension_phase: Whether we're in the extension phase

    Returns:
        CollapseResult with outcome details
    """
    revealed_stake, total_stake, ratio = calculate_reveal_ratio(commitments, reveals)
    revealed_validators = {r.validator_id for r in reveals}

    # Find non-revealers
    non_revealers = [
        c.validator_id for c in commitments
        if c.validator_id not in revealed_validators
    ]

    if ratio >= REVEAL_THRESHOLD:
        # Success: threshold met
        slash_rate = NON_REVEAL_SLASH_RATE
        total_slashed = int(sum(
            c.stake * slash_rate
            for c in commitments
            if c.validator_id in non_revealers
        ))

        # Generate collapsed state seed from reveals
        seed_data = b"".join(r.vdf_output for r in sorted(reveals, key=lambda x: x.validator_id))
        collapsed_state_seed = hashlib.sha256(seed_data).digest()

        return CollapseResult(
            status=CollapseStatus.SUCCESS,
            revealed_stake=revealed_stake,
            total_stake=total_stake,
            revealed_ratio=ratio,
            slashed_validators=non_revealers,
            total_slashed=total_slashed,
            backup_activated=False,
            collapsed_state_seed=collapsed_state_seed,
        )

    elif not is_extension_phase:
        # Below threshold, extension needed
        return CollapseResult(
            status=CollapseStatus.BELOW_THRESHOLD,
            revealed_stake=revealed_stake,
            total_stake=total_stake,
            revealed_ratio=ratio,
            slashed_validators=[],
            total_slashed=0,
            backup_activated=False,
        )

    else:
        # Extension phase complete, still below threshold
        # Activate backup validators, apply higher slash
        slash_rate = BACKUP_NON_REVEAL_SLASH_RATE
        total_slashed = int(sum(
            c.stake * slash_rate
            for c in commitments
            if c.validator_id in non_revealers
        ))

        return CollapseResult(
            status=CollapseStatus.BACKUP_ACTIVATED,
            revealed_stake=revealed_stake,
            total_stake=total_stake,
            revealed_ratio=ratio,
            slashed_validators=non_revealers,
            total_slashed=total_slashed,
            backup_activated=True,
        )


# =============================================================================
# Simulation Functions
# =============================================================================

def create_mock_commitment(
    validator_id: str,
    stake: int,
    block: int = 0,
) -> Tuple[VdfCommitment, VdfReveal]:
    """
    Create a mock commitment and matching reveal for simulation.

    Args:
        validator_id: Validator identifier
        stake: Stake amount
        block: Block number

    Returns:
        Tuple of (commitment, reveal)
    """
    vdf_output = hashlib.sha256(f"{validator_id}_vdf".encode()).digest()
    salt = hashlib.sha256(f"{validator_id}_salt".encode()).digest()
    proof = b"mock_proof"

    commitment_hash = hashlib.sha256(vdf_output + salt).hexdigest()

    commitment = VdfCommitment(
        validator_id=validator_id,
        commitment_hash=commitment_hash,
        stake=stake,
        committed_at=block,
    )

    reveal = VdfReveal(
        validator_id=validator_id,
        vdf_output=vdf_output,
        proof=proof,
        salt=salt,
        revealed_at=block + 10,
    )

    return commitment, reveal


def simulate_collapse_protocol(
    n_validators: int = 10,
    reveal_rate: float = 0.8,
    stake_distribution: str = "uniform",
    seed: Optional[int] = None,
) -> Dict:
    """
    Simulate the collapse protocol with multiple validators.

    Args:
        n_validators: Number of validators
        reveal_rate: Probability each validator reveals
        stake_distribution: "uniform" or "weighted"
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    if seed is not None:
        random.seed(seed)

    # Generate validators
    commitments = []
    all_reveals = []

    for i in range(n_validators):
        if stake_distribution == "uniform":
            stake = 1000
        else:
            stake = random.randint(100, 5000)

        commitment, reveal = create_mock_commitment(
            validator_id=f"validator_{i}",
            stake=stake,
        )
        commitments.append(commitment)
        all_reveals.append(reveal)

    # Simulate reveals based on reveal_rate
    actual_reveals = [
        reveal for reveal in all_reveals
        if random.random() < reveal_rate
    ]

    # Phase 1: Initial reveal period
    result1 = process_threshold_reveal(commitments, actual_reveals, is_extension_phase=False)

    if result1.status == CollapseStatus.SUCCESS:
        return {
            "phase": 1,
            "n_validators": n_validators,
            "reveal_rate_target": reveal_rate,
            "actual_reveal_rate": len(actual_reveals) / n_validators,
            "result": result1,
            "extension_needed": False,
            "backup_activated": False,
        }

    # Phase 2: Extension period (some more validators might reveal)
    extension_reveals = [
        reveal for reveal in all_reveals
        if reveal not in actual_reveals and random.random() < 0.3  # 30% chance during extension
    ]
    actual_reveals.extend(extension_reveals)

    result2 = process_threshold_reveal(commitments, actual_reveals, is_extension_phase=True)

    return {
        "phase": 2,
        "n_validators": n_validators,
        "reveal_rate_target": reveal_rate,
        "actual_reveal_rate": len(actual_reveals) / n_validators,
        "result": result2,
        "extension_needed": True,
        "extension_reveals": len(extension_reveals),
        "backup_activated": result2.backup_activated,
    }


def analyze_threshold_sensitivity(
    n_validators: int = 20,
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Analyze how reveal rate affects collapse success.

    Args:
        n_validators: Number of validators per simulation
        n_simulations: Number of simulations per reveal rate
        seed: Random seed

    Returns:
        Dictionary mapping reveal_rate to success metrics
    """
    reveal_rates = [0.5, 0.6, 0.67, 0.7, 0.75, 0.8, 0.9, 1.0]
    results = {}

    for rate in reveal_rates:
        successes = 0
        backup_activations = 0
        total_slashed = 0

        for i in range(n_simulations):
            sim = simulate_collapse_protocol(
                n_validators=n_validators,
                reveal_rate=rate,
                seed=seed + i,
            )

            if sim["result"].is_successful:
                successes += 1
            if sim["backup_activated"]:
                backup_activations += 1
            total_slashed += sim["result"].total_slashed

        results[rate] = {
            "success_rate": successes / n_simulations,
            "backup_rate": backup_activations / n_simulations,
            "avg_slashed": total_slashed / n_simulations,
        }

    return results


def simulate_adversarial_scenario(
    n_honest: int = 15,
    n_adversarial: int = 5,
    honest_reveal_rate: float = 0.95,
    adversarial_reveal_rate: float = 0.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Simulate scenario with adversarial validators.

    Args:
        n_honest: Number of honest validators
        n_adversarial: Number of adversarial (non-revealing) validators
        honest_reveal_rate: Reveal rate for honest validators
        adversarial_reveal_rate: Reveal rate for adversarial validators
        seed: Random seed

    Returns:
        Simulation results
    """
    if seed is not None:
        random.seed(seed)

    commitments = []
    reveals = []

    # Honest validators
    for i in range(n_honest):
        commitment, reveal = create_mock_commitment(
            validator_id=f"honest_{i}",
            stake=1000,
        )
        commitments.append(commitment)
        if random.random() < honest_reveal_rate:
            reveals.append(reveal)

    # Adversarial validators
    for i in range(n_adversarial):
        commitment, reveal = create_mock_commitment(
            validator_id=f"adversarial_{i}",
            stake=1000,
        )
        commitments.append(commitment)
        if random.random() < adversarial_reveal_rate:
            reveals.append(reveal)

    # Process
    result = process_threshold_reveal(commitments, reveals, is_extension_phase=True)

    return {
        "n_honest": n_honest,
        "n_adversarial": n_adversarial,
        "honest_reveal_rate": honest_reveal_rate,
        "adversarial_reveal_rate": adversarial_reveal_rate,
        "revealed_ratio": result.revealed_ratio,
        "status": result.status.value,
        "is_successful": result.is_successful,
        "slashed_count": len(result.slashed_validators),
        "total_slashed": result.total_slashed,
    }
