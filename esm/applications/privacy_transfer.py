"""
Privacy Transfer System

Based on ESM Whitepaper v5.3 Section 8.2

Implements privacy-preserving transfers using ESM's superposition states.
The actual recipient remains hidden until collapse, providing:
- Transaction obfuscation
- Multi-recipient distribution
- Probabilistic routing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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
# Data Structures
# =============================================================================

class TransferMode(Enum):
    """Privacy transfer mode."""
    EQUAL = "equal"           # Equal probability for all recipients
    WEIGHTED = "weighted"     # Weighted by specified amounts
    DECOY = "decoy"          # One real recipient + decoys


@dataclass
class PrivateTransfer:
    """
    A privacy-preserving transfer.

    Attributes:
        transfer_id: Unique identifier
        sender: Sender address
        possible_recipients: List of (recipient, amplitude) pairs
        amount: Transfer amount
        psc: Associated PSC
        mode: Transfer mode
        collapsed: Whether transfer has been finalized
        actual_recipient: Final recipient (after collapse)
    """
    transfer_id: str
    sender: str
    possible_recipients: List[Tuple[str, DiscreteAmplitude]]
    amount: Decimal
    psc: PSC
    mode: TransferMode = TransferMode.EQUAL
    collapsed: bool = False
    actual_recipient: Optional[str] = None


@dataclass
class TransferResult(ApplicationResult):
    """
    Result of private transfer.

    Attributes:
        transfer_id: Transfer identifier
        sender: Sender address
        recipient: Actual recipient (after collapse)
        amount: Transfer amount
        n_possible_recipients: Number of possible recipients
        recipient_probability: Probability of selected recipient
    """
    transfer_id: str = ""
    sender: str = ""
    recipient: str = ""
    amount: Decimal = Decimal("0")
    n_possible_recipients: int = 0
    recipient_probability: float = 0.0


# =============================================================================
# Privacy Transfer System
# =============================================================================

class PrivacyTransferSystem(ESMApplication):
    """
    Privacy-preserving transfer system using ESM superposition.

    Creates transfers where the recipient is in superposition
    until collapse, providing transaction privacy.
    """

    def __init__(self):
        super().__init__("PrivacyTransfer")
        self.transfers: Dict[str, PrivateTransfer] = {}
        self.completed_transfers: List[TransferResult] = []

    def create_private_transfer(
        self,
        sender: str,
        recipients: List[str],
        amount: Decimal,
        mode: TransferMode = TransferMode.EQUAL,
        weights: Optional[List[float]] = None,
        real_recipient: Optional[str] = None,
    ) -> PrivateTransfer:
        """
        Create a privacy-preserving transfer.

        The transfer is in superposition until collapse - observers
        cannot determine the actual recipient beforehand.

        Args:
            sender: Sender address
            recipients: List of possible recipients
            amount: Transfer amount
            mode: Transfer mode (equal, weighted, or decoy)
            weights: Custom weights (for weighted mode)
            real_recipient: Real recipient (for decoy mode)

        Returns:
            PrivateTransfer instance
        """
        # Generate transfer ID
        transfer_id = hashlib.sha256(
            f"transfer_{sender}_{self.block_number}_{random.random()}".encode()
        ).hexdigest()[:16]

        # Create PSC
        psc = self.create_application_psc(f"privacy_{transfer_id}")

        # Determine amplitude distribution
        n = len(recipients)

        if mode == TransferMode.EQUAL:
            # Equal probability for all
            amplitudes = distribute_amplitudes(n, "equal")

        elif mode == TransferMode.WEIGHTED and weights:
            # Weighted distribution
            amplitudes = distribute_amplitudes(n, "weighted", weights)

        elif mode == TransferMode.DECOY and real_recipient:
            # One real recipient, others are decoys with very low probability
            real_idx = recipients.index(real_recipient) if real_recipient in recipients else 0

            # Real gets 99%, decoys share 1%
            decoy_weight = 0.01 / (n - 1) if n > 1 else 0
            weights = [decoy_weight] * n
            weights[real_idx] = 0.99
            amplitudes = distribute_amplitudes(n, "weighted", weights)

        else:
            amplitudes = distribute_amplitudes(n, "equal")

        # Create branches for each recipient
        possible_recipients = []
        for i, recipient in enumerate(recipients):
            mag, phase = amplitudes[i]

            state_data = {
                "type": "transfer",
                "sender": sender,
                "recipient": recipient,
                "amount": float(amount),
            }

            branch = create_branch(
                state_data=state_data,
                magnitude=mag,
                phase=phase,
                creator=sender,
            )
            psc.add_branch(branch)

            amp = DiscreteAmplitude(mag, phase)
            possible_recipients.append((recipient, amp))

        transfer = PrivateTransfer(
            transfer_id=transfer_id,
            sender=sender,
            possible_recipients=possible_recipients,
            amount=amount,
            psc=psc,
            mode=mode,
            collapsed=False,
        )

        self.transfers[transfer_id] = transfer
        return transfer

    def get_recipient_probabilities(
        self,
        transfer: PrivateTransfer,
    ) -> Dict[str, Decimal]:
        """
        Get current probability distribution for recipients.

        Args:
            transfer: Private transfer

        Returns:
            Dictionary mapping recipient to probability
        """
        probs = transfer.psc.get_probabilities()

        result = {}
        for recipient, amp in transfer.possible_recipients:
            # Find matching branch
            for branch in transfer.psc.branches:
                if branch.state_data.get("recipient") == recipient:
                    prob = probs.get(branch.state_id, 0)
                    result[recipient] = Decimal(str(prob))
                    break

        return result

    def collapse_transfer(
        self,
        transfer: PrivateTransfer,
        seed: Optional[int] = None,
    ) -> TransferResult:
        """
        Collapse transfer to determine actual recipient.

        Args:
            transfer: Private transfer to collapse
            seed: Optional random seed

        Returns:
            TransferResult with final recipient
        """
        if transfer.collapsed:
            return TransferResult(
                status=ApplicationStatus.FAILED,
                psc_id=transfer.psc.id,
                metadata={"error": "Transfer already collapsed"},
            )

        # Get probabilities before collapse
        probs = self.get_recipient_probabilities(transfer)

        # Collapse PSC
        selected_state, selected_branch = transfer.psc.collapse(seed=seed)

        # Extract recipient
        actual_recipient = selected_branch.state_data["recipient"]
        transfer.actual_recipient = actual_recipient
        transfer.collapsed = True

        # Create result
        result = TransferResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=transfer.psc.id,
            selected_outcome=actual_recipient,
            probability_distribution={k: float(v) for k, v in probs.items()},
            metadata={
                "mode": transfer.mode.value,
                "sender": transfer.sender,
            },
            transfer_id=transfer.transfer_id,
            sender=transfer.sender,
            recipient=actual_recipient,
            amount=transfer.amount,
            n_possible_recipients=len(transfer.possible_recipients),
            recipient_probability=float(probs.get(actual_recipient, 0)),
        )

        self.completed_transfers.append(result)
        return result

    def create_decoy_transfer(
        self,
        sender: str,
        real_recipient: str,
        decoy_recipients: List[str],
        amount: Decimal,
    ) -> PrivateTransfer:
        """
        Create a transfer with decoys for enhanced privacy.

        The real recipient has 99% probability, decoys share 1%.

        Args:
            sender: Sender address
            real_recipient: Actual intended recipient
            decoy_recipients: List of decoy addresses
            amount: Transfer amount

        Returns:
            PrivateTransfer with decoys
        """
        all_recipients = [real_recipient] + decoy_recipients
        return self.create_private_transfer(
            sender=sender,
            recipients=all_recipients,
            amount=amount,
            mode=TransferMode.DECOY,
            real_recipient=real_recipient,
        )

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "pending_transfers": len([t for t in self.transfers.values() if not t.collapsed]),
            "completed_transfers": len(self.completed_transfers),
            "total_transfers": len(self.transfers),
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_private_transfer(
    recipients: List[str] = None,
    mode: TransferMode = TransferMode.EQUAL,
    seed: int = 42,
) -> TransferResult:
    """
    Run a private transfer simulation.

    Args:
        recipients: List of recipients (default: 5)
        mode: Transfer mode
        seed: Random seed

    Returns:
        TransferResult
    """
    if recipients is None:
        recipients = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    system = PrivacyTransferSystem()

    transfer = system.create_private_transfer(
        sender="Sender",
        recipients=recipients,
        amount=Decimal("100"),
        mode=mode,
    )

    print(f"Transfer created: {transfer.transfer_id}")
    print(f"Mode: {mode.value}")
    print(f"Recipients: {len(recipients)}")

    # Show probabilities before collapse
    probs = system.get_recipient_probabilities(transfer)
    print("\nProbabilities before collapse:")
    for r, p in probs.items():
        print(f"  {r}: {float(p)*100:.1f}%")

    # Collapse
    result = system.collapse_transfer(transfer, seed=seed)

    print(f"\nCollapse result: {result.recipient}")
    print(f"Probability was: {result.recipient_probability*100:.1f}%")

    return result


def run_demo():
    """Run privacy transfer demo."""
    print("=" * 60)
    print("Privacy Transfer Demo")
    print("=" * 60)
    print()

    # Demo 1: Equal distribution
    print("Demo 1: Equal Distribution")
    print("-" * 40)
    simulate_private_transfer(mode=TransferMode.EQUAL)
    print()

    # Demo 2: Decoy mode
    print("Demo 2: Decoy Mode (real recipient = Alice)")
    print("-" * 40)
    system = PrivacyTransferSystem()
    transfer = system.create_decoy_transfer(
        sender="Sender",
        real_recipient="Alice",
        decoy_recipients=["Bob", "Charlie", "Diana"],
        amount=Decimal("100"),
    )

    probs = system.get_recipient_probabilities(transfer)
    print("Probabilities:")
    for r, p in sorted(probs.items(), key=lambda x: -float(x[1])):
        print(f"  {r}: {float(p)*100:.1f}%")

    result = system.collapse_transfer(transfer, seed=42)
    print(f"Collapse result: {result.recipient}")
    print()


if __name__ == "__main__":
    run_demo()
