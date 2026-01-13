"""
Tests for Threshold Reveal and Backup Validator Systems.

Based on ESM Whitepaper v5.2 Section 5.2-5.6.
"""

import pytest
import hashlib

from esm.simulation.threshold_reveal import (
    REVEAL_THRESHOLD,
    REVEAL_EXTENSION_BLOCKS,
    NON_REVEAL_SLASH_RATE,
    BACKUP_NON_REVEAL_SLASH_RATE,
    CollapseStatus,
    RevealStatus,
    VdfCommitment,
    VdfReveal,
    ValidatorState,
    CollapseResult,
    calculate_reveal_ratio,
    verify_reveal,
    process_threshold_reveal,
    create_mock_commitment,
    simulate_collapse_protocol,
    analyze_threshold_sensitivity,
    simulate_adversarial_scenario,
)

from esm.simulation.backup_validator import (
    BACKUP_VALIDATOR_COUNT,
    BACKUP_REWARD_SHARE,
    BackupStatus,
    BackupValidator,
    BackupActivation,
    BackupCollapseResult,
    select_backup_validators,
    calculate_backup_rewards,
    activate_backup_validators,
    simulate_backup_collapse,
    create_validator_pool,
    simulate_backup_scenario,
    analyze_backup_reliability,
)


class TestThresholdConstants:
    """Tests for threshold reveal constants."""

    def test_reveal_threshold(self):
        """Test reveal threshold is 67%."""
        assert REVEAL_THRESHOLD == 0.67

    def test_extension_blocks(self):
        """Test extension period is 5 blocks."""
        assert REVEAL_EXTENSION_BLOCKS == 5

    def test_slash_rates(self):
        """Test slashing rates."""
        assert NON_REVEAL_SLASH_RATE == 0.10
        assert BACKUP_NON_REVEAL_SLASH_RATE == 0.50


class TestVdfCommitment:
    """Tests for VDF commitment structure."""

    def test_commitment_creation(self):
        """Test VDF commitment creation."""
        commitment = VdfCommitment(
            validator_id="validator_1",
            commitment_hash="abc123",
            stake=1000,
            committed_at=100,
        )

        assert commitment.validator_id == "validator_1"
        assert commitment.stake == 1000


class TestCalculateRevealRatio:
    """Tests for reveal ratio calculation."""

    def test_full_reveal(self):
        """Test 100% reveal ratio."""
        commitments = [
            VdfCommitment("v1", "h1", 1000, 0),
            VdfCommitment("v2", "h2", 1000, 0),
        ]
        reveals = [
            VdfReveal("v1", b"out", b"proof", b"salt"),
            VdfReveal("v2", b"out", b"proof", b"salt"),
        ]

        revealed, total, ratio = calculate_reveal_ratio(commitments, reveals)

        assert revealed == 2000
        assert total == 2000
        assert ratio == 1.0

    def test_partial_reveal(self):
        """Test partial reveal ratio."""
        commitments = [
            VdfCommitment("v1", "h1", 1000, 0),
            VdfCommitment("v2", "h2", 1000, 0),
        ]
        reveals = [
            VdfReveal("v1", b"out", b"proof", b"salt"),
        ]

        revealed, total, ratio = calculate_reveal_ratio(commitments, reveals)

        assert revealed == 1000
        assert total == 2000
        assert ratio == 0.5

    def test_weighted_reveal(self):
        """Test reveal ratio with weighted stakes."""
        commitments = [
            VdfCommitment("v1", "h1", 3000, 0),  # 75% stake
            VdfCommitment("v2", "h2", 1000, 0),  # 25% stake
        ]
        reveals = [
            VdfReveal("v1", b"out", b"proof", b"salt"),  # Only big validator reveals
        ]

        revealed, total, ratio = calculate_reveal_ratio(commitments, reveals)

        assert revealed == 3000
        assert total == 4000
        assert ratio == 0.75

    def test_no_reveals(self):
        """Test zero reveal ratio."""
        commitments = [
            VdfCommitment("v1", "h1", 1000, 0),
        ]
        reveals = []

        revealed, total, ratio = calculate_reveal_ratio(commitments, reveals)

        assert ratio == 0.0


class TestVerifyReveal:
    """Tests for reveal verification."""

    def test_valid_reveal(self):
        """Test valid reveal verification."""
        commitment, reveal = create_mock_commitment("v1", 1000)
        assert verify_reveal(commitment, reveal) is True

    def test_invalid_validator_id(self):
        """Test reveal with wrong validator ID."""
        commitment, reveal = create_mock_commitment("v1", 1000)
        reveal.validator_id = "v2"
        assert verify_reveal(commitment, reveal) is False

    def test_tampered_vdf_output(self):
        """Test reveal with tampered VDF output."""
        commitment, reveal = create_mock_commitment("v1", 1000)
        reveal.vdf_output = b"tampered"
        assert verify_reveal(commitment, reveal) is False


class TestProcessThresholdReveal:
    """Tests for threshold reveal processing."""

    def test_success_above_threshold(self):
        """Test successful collapse when above threshold."""
        commitments = []
        reveals = []

        for i in range(10):
            c, r = create_mock_commitment(f"v{i}", 1000)
            commitments.append(c)
            if i < 7:  # 70% reveal
                reveals.append(r)

        result = process_threshold_reveal(commitments, reveals)

        assert result.status == CollapseStatus.SUCCESS
        assert result.revealed_ratio >= REVEAL_THRESHOLD
        assert len(result.slashed_validators) == 3
        assert result.total_slashed == 300  # 3 * 1000 * 0.10

    def test_below_threshold_no_extension(self):
        """Test below threshold without extension phase."""
        commitments = []
        reveals = []

        for i in range(10):
            c, r = create_mock_commitment(f"v{i}", 1000)
            commitments.append(c)
            if i < 5:  # 50% reveal
                reveals.append(r)

        result = process_threshold_reveal(commitments, reveals, is_extension_phase=False)

        assert result.status == CollapseStatus.BELOW_THRESHOLD
        assert result.revealed_ratio < REVEAL_THRESHOLD
        assert result.total_slashed == 0  # No slashing yet

    def test_backup_activated_after_extension(self):
        """Test backup activation after extension failure."""
        commitments = []
        reveals = []

        for i in range(10):
            c, r = create_mock_commitment(f"v{i}", 1000)
            commitments.append(c)
            if i < 5:  # 50% reveal
                reveals.append(r)

        result = process_threshold_reveal(commitments, reveals, is_extension_phase=True)

        assert result.status == CollapseStatus.BACKUP_ACTIVATED
        assert result.backup_activated is True
        assert result.total_slashed == 2500  # 5 * 1000 * 0.50 (higher rate)

    def test_collapsed_state_seed_generated(self):
        """Test collapsed state seed is generated on success."""
        commitments = []
        reveals = []

        for i in range(3):
            c, r = create_mock_commitment(f"v{i}", 1000)
            commitments.append(c)
            reveals.append(r)

        result = process_threshold_reveal(commitments, reveals)

        assert result.status == CollapseStatus.SUCCESS
        assert result.collapsed_state_seed is not None
        assert len(result.collapsed_state_seed) == 32


class TestCollapseResult:
    """Tests for CollapseResult properties."""

    def test_is_successful_success(self):
        """Test is_successful for SUCCESS status."""
        result = CollapseResult(
            status=CollapseStatus.SUCCESS,
            revealed_stake=7000,
            total_stake=10000,
            revealed_ratio=0.7,
        )
        assert result.is_successful is True

    def test_is_successful_backup(self):
        """Test is_successful for BACKUP_ACTIVATED status."""
        result = CollapseResult(
            status=CollapseStatus.BACKUP_ACTIVATED,
            revealed_stake=5000,
            total_stake=10000,
            revealed_ratio=0.5,
            backup_activated=True,
        )
        assert result.is_successful is True

    def test_is_not_successful(self):
        """Test is_successful for BELOW_THRESHOLD status."""
        result = CollapseResult(
            status=CollapseStatus.BELOW_THRESHOLD,
            revealed_stake=5000,
            total_stake=10000,
            revealed_ratio=0.5,
        )
        assert result.is_successful is False


class TestSimulateCollapseProtocol:
    """Tests for collapse protocol simulation."""

    def test_high_reveal_rate_success(self):
        """Test simulation with high reveal rate succeeds."""
        result = simulate_collapse_protocol(
            n_validators=20,
            reveal_rate=0.9,
            seed=42,
        )

        assert result["result"].is_successful
        assert result["phase"] == 1  # Should succeed in phase 1

    def test_low_reveal_rate_needs_extension(self):
        """Test simulation with low reveal rate needs extension."""
        result = simulate_collapse_protocol(
            n_validators=20,
            reveal_rate=0.5,
            seed=42,
        )

        assert result["extension_needed"] is True


class TestAnalyzeThresholdSensitivity:
    """Tests for threshold sensitivity analysis."""

    def test_analysis_runs(self):
        """Test analysis runs without error."""
        result = analyze_threshold_sensitivity(
            n_validators=10,
            n_simulations=100,
            seed=42,
        )

        assert 0.5 in result
        assert 0.67 in result
        assert 1.0 in result

    def test_higher_rate_higher_success(self):
        """Test higher reveal rate leads to higher success."""
        result = analyze_threshold_sensitivity(
            n_validators=10,
            n_simulations=500,
            seed=42,
        )

        assert result[1.0]["success_rate"] >= result[0.5]["success_rate"]


class TestAdversarialScenario:
    """Tests for adversarial simulation."""

    def test_adversarial_below_threshold(self):
        """Test adversarial validators can prevent threshold."""
        result = simulate_adversarial_scenario(
            n_honest=10,
            n_adversarial=10,  # 50% adversarial
            honest_reveal_rate=0.95,
            adversarial_reveal_rate=0.0,
            seed=42,
        )

        # With 50% honest and 50% adversarial (0% reveal),
        # max reveal is about 47.5%, below 67% threshold
        assert result["revealed_ratio"] < REVEAL_THRESHOLD

    def test_adversarial_minority_ok(self):
        """Test small adversarial minority doesn't prevent success."""
        result = simulate_adversarial_scenario(
            n_honest=17,
            n_adversarial=3,  # 15% adversarial
            honest_reveal_rate=0.95,
            adversarial_reveal_rate=0.0,
            seed=42,
        )

        # With 85% honest at 95% reveal rate, should meet threshold
        assert result["is_successful"] is True


# =============================================================================
# Backup Validator Tests
# =============================================================================

class TestBackupValidatorConstants:
    """Tests for backup validator constants."""

    def test_backup_count(self):
        """Test default backup validator count."""
        assert BACKUP_VALIDATOR_COUNT == 5

    def test_reward_share(self):
        """Test backup reward share."""
        assert BACKUP_REWARD_SHARE == 0.50


class TestBackupValidator:
    """Tests for BackupValidator class."""

    def test_selection_weight(self):
        """Test selection weight calculation."""
        validator = BackupValidator(
            validator_id="v1",
            stake=1000,
            reputation_score=0.8,
        )

        assert validator.selection_weight == 800


class TestSelectBackupValidators:
    """Tests for backup validator selection."""

    def test_select_correct_count(self):
        """Test correct number of validators selected."""
        candidates = create_validator_pool(20, seed=42)
        selected = select_backup_validators(candidates, n_backups=5)

        assert len(selected) == 5

    def test_exclude_validators(self):
        """Test excluded validators are not selected."""
        candidates = create_validator_pool(10, seed=42)
        excluded = [candidates[0].validator_id, candidates[1].validator_id]

        selected = select_backup_validators(
            candidates,
            n_backups=5,
            exclude_validators=excluded,
        )

        selected_ids = {v.validator_id for v in selected}
        assert not set(excluded).intersection(selected_ids)

    def test_deterministic_selection(self):
        """Test selection is deterministic with same seed."""
        candidates = create_validator_pool(20, seed=42)
        seed = b"test_seed_12345678"

        selected1 = select_backup_validators(candidates, selection_seed=seed)
        selected2 = select_backup_validators(candidates, selection_seed=seed)

        ids1 = {v.validator_id for v in selected1}
        ids2 = {v.validator_id for v in selected2}

        assert ids1 == ids2


class TestCalculateBackupRewards:
    """Tests for backup reward calculation."""

    def test_reward_calculation(self):
        """Test reward calculation."""
        total, per_validator = calculate_backup_rewards(
            slashed_amount=10000,
            n_backup_validators=5,
        )

        assert total == 5000  # 50% of slashed
        assert per_validator == 1000

    def test_zero_validators(self):
        """Test zero validators case."""
        total, per_validator = calculate_backup_rewards(
            slashed_amount=10000,
            n_backup_validators=0,
        )

        assert total == 5000
        assert per_validator == 0


class TestActivateBackupValidators:
    """Tests for backup validator activation."""

    def test_activation_creates_record(self):
        """Test activation creates proper record."""
        failed_commitments = [
            VdfCommitment("failed_1", "h1", 1000, 0),
            VdfCommitment("failed_2", "h2", 2000, 0),
        ]
        candidates = create_validator_pool(20, seed=42)

        activation = activate_backup_validators(
            failed_commitments=failed_commitments,
            candidates=candidates,
            activation_block=100,
        )

        assert len(activation.backup_validators) == 5
        assert activation.slashed_from_primary == 1500  # 3000 * 0.50
        assert activation.reward_pool == 750  # 1500 * 0.50


class TestSimulateBackupCollapse:
    """Tests for backup collapse simulation."""

    def test_successful_collapse(self):
        """Test successful backup collapse."""
        activation = BackupActivation(
            activated_at=100,
            backup_validators=["b1", "b2", "b3", "b4", "b5"],
            trigger_reason="test",
            primary_revealed_ratio=0.5,
            slashed_from_primary=5000,
            reward_pool=2500,
        )

        result = simulate_backup_collapse(activation, backup_reveal_rate=1.0, seed=42)

        assert result.status == BackupStatus.COMPLETED
        assert len(result.participating_validators) == 5
        assert result.collapsed_state_seed is not None

    def test_failed_collapse_no_participation(self):
        """Test failed collapse when no backup participates."""
        activation = BackupActivation(
            activated_at=100,
            backup_validators=["b1", "b2"],
            trigger_reason="test",
            primary_revealed_ratio=0.5,
            slashed_from_primary=5000,
            reward_pool=2500,
        )

        result = simulate_backup_collapse(activation, backup_reveal_rate=0.0, seed=42)

        assert result.status == BackupStatus.FAILED
        assert len(result.participating_validators) == 0


class TestBackupCollapseResult:
    """Tests for BackupCollapseResult properties."""

    def test_participation_rate(self):
        """Test participation rate calculation."""
        result = BackupCollapseResult(
            status=BackupStatus.COMPLETED,
            selected_validators=["a", "b", "c", "d", "e"],
            participating_validators=["a", "b", "c"],
        )

        assert result.participation_rate == 0.6


class TestSimulateBackupScenario:
    """Tests for full backup scenario simulation."""

    def test_scenario_runs(self):
        """Test scenario runs without error."""
        result = simulate_backup_scenario(
            n_primary=20,
            n_backup_candidates=30,
            primary_reveal_rate=0.5,
            seed=42,
        )

        assert "primary" in result
        assert "backup" in result
        assert "economics" in result
        assert "outcome" in result


class TestAnalyzeBackupReliability:
    """Tests for backup reliability analysis."""

    def test_analysis_runs(self):
        """Test analysis runs without error."""
        result = analyze_backup_reliability(n_simulations=50, seed=42)

        assert len(result) > 0

    def test_higher_backup_rate_higher_success(self):
        """Test higher backup reveal rate leads to higher success."""
        result = analyze_backup_reliability(n_simulations=200, seed=42)

        # Find results with same primary rate but different backup rates
        high_backup = [k for k in result.keys() if "backup_0.9" in k]
        low_backup = [k for k in result.keys() if "backup_0.5" in k]

        if high_backup and low_backup:
            # High backup reveal should have >= success rate
            assert result[high_backup[0]]["success_rate"] >= result[low_backup[0]]["success_rate"]
