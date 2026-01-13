"""
Decentralized Insurance

Based on ESM Whitepaper v5.3 Section 8.4

Implements parametric insurance using ESM's conditional states.
Insurance payouts are represented as superposition states that
collapse when oracle conditions are met.

Key features:
- Conditional payout states
- Oracle-triggered amplitude adjustments
- Premium-based probability modification
- Automatic payout on condition fulfillment
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from decimal import Decimal
from enum import Enum
import hashlib

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_branch
from esm.core.psc import PSC, create_psc
from esm.applications.base import (
    ESMApplication,
    ApplicationResult,
    ApplicationStatus,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PAYOUT_PROBABILITY = Decimal("0.1")  # 10% initial payout probability
PREMIUM_RATE = Decimal("0.05")  # 5% annual premium
CONDITION_BOOST = Decimal("0.3")  # 30% probability boost per condition met


class PolicyStatus(Enum):
    """Insurance policy status."""
    ACTIVE = "active"
    CLAIMED = "claimed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InsuranceCondition:
    """
    A condition that triggers payout boost.

    Attributes:
        condition_id: Unique identifier
        description: Human-readable description
        met: Whether condition has been met
        boost_factor: Probability boost when met
    """
    condition_id: str
    description: str
    met: bool = False
    boost_factor: Decimal = CONDITION_BOOST


@dataclass
class InsurancePolicy:
    """
    An insurance policy.

    Attributes:
        policy_id: Unique identifier
        policyholder: Policy owner address
        coverage_amount: Maximum payout amount
        premium_paid: Total premium paid
        conditions: List of payout conditions
        psc: Associated PSC
        payout_branch_amplitude: Current payout amplitude
        no_payout_branch_amplitude: Current no-payout amplitude
        status: Policy status
        payout_amount: Final payout (if claimed)
    """
    policy_id: str
    policyholder: str
    coverage_amount: Decimal
    premium_paid: Decimal
    conditions: List[InsuranceCondition]
    psc: PSC
    payout_branch_amplitude: Decimal
    no_payout_branch_amplitude: Decimal
    status: PolicyStatus = PolicyStatus.ACTIVE
    payout_amount: Optional[Decimal] = None


@dataclass
class InsuranceResult(ApplicationResult):
    """
    Result of insurance claim.

    Attributes:
        policy_id: Policy identifier
        policyholder: Policy owner
        coverage_amount: Coverage amount
        payout_amount: Actual payout
        conditions_met: Number of conditions met
        payout_probability: Probability of payout
    """
    policy_id: str = ""
    policyholder: str = ""
    coverage_amount: Decimal = Decimal("0")
    payout_amount: Decimal = Decimal("0")
    conditions_met: int = 0
    payout_probability: float = 0.0


# =============================================================================
# Decentralized Insurance
# =============================================================================

class DecentralizedInsurance(ESMApplication):
    """
    Decentralized parametric insurance using ESM.

    Policies have conditional payouts represented as superposition
    states. Oracle conditions adjust amplitudes, affecting
    payout probability.
    """

    def __init__(self):
        super().__init__("DecentralizedInsurance")
        self.policies: Dict[str, InsurancePolicy] = {}
        self.claims: List[InsuranceResult] = []
        self.insurance_pool: Decimal = Decimal("0")

    def create_policy(
        self,
        policyholder: str,
        coverage_amount: Decimal,
        conditions: List[str],
        initial_payout_prob: Decimal = DEFAULT_PAYOUT_PROBABILITY,
    ) -> InsurancePolicy:
        """
        Create a new insurance policy.

        The policy is in superposition between payout and no-payout
        states. Oracle conditions adjust the amplitudes.

        Args:
            policyholder: Policy owner address
            coverage_amount: Maximum payout amount
            conditions: List of condition descriptions
            initial_payout_prob: Initial payout probability

        Returns:
            New InsurancePolicy
        """
        policy_id = hashlib.sha256(
            f"policy_{policyholder}_{self.block_number}".encode()
        ).hexdigest()[:16]

        # Create PSC
        psc = self.create_application_psc(f"insurance_{policy_id}")

        # Create conditions
        condition_objects = [
            InsuranceCondition(
                condition_id=f"cond_{i}",
                description=desc,
            )
            for i, desc in enumerate(conditions)
        ]

        # Initial amplitudes based on payout probability
        # |payout|^2 = prob, so |payout| = sqrt(prob)
        payout_amp = float(initial_payout_prob) ** 0.5
        no_payout_amp = float(1 - initial_payout_prob) ** 0.5

        # Create payout branch
        payout_data = {
            "outcome": "payout",
            "amount": float(coverage_amount),
            "policyholder": policyholder,
        }
        payout_branch = create_branch(
            state_data=payout_data,
            magnitude=payout_amp,
            phase=DiscretePhase.P0,
            creator=policyholder,
        )
        psc.add_branch(payout_branch)

        # Create no-payout branch
        no_payout_data = {
            "outcome": "no_payout",
            "amount": 0,
            "policyholder": policyholder,
        }
        no_payout_branch = create_branch(
            state_data=no_payout_data,
            magnitude=no_payout_amp,
            phase=DiscretePhase.P0,
            creator=policyholder,
        )
        psc.add_branch(no_payout_branch)

        policy = InsurancePolicy(
            policy_id=policy_id,
            policyholder=policyholder,
            coverage_amount=coverage_amount,
            premium_paid=Decimal("0"),
            conditions=condition_objects,
            psc=psc,
            payout_branch_amplitude=Decimal(str(payout_amp)),
            no_payout_branch_amplitude=Decimal(str(no_payout_amp)),
        )

        self.policies[policy_id] = policy
        return policy

    def pay_premium(
        self,
        policy: InsurancePolicy,
        amount: Decimal,
    ) -> bool:
        """
        Pay premium for policy.

        Premium payment slightly increases payout probability.

        Args:
            policy: Target policy
            amount: Premium amount

        Returns:
            True if payment was successful
        """
        if policy.status != PolicyStatus.ACTIVE:
            return False

        policy.premium_paid += amount
        self.insurance_pool += amount

        # Boost payout amplitude slightly (premium earns coverage)
        boost = float(amount / policy.coverage_amount) * 0.1
        self._adjust_amplitudes(policy, boost)

        return True

    def oracle_condition_met(
        self,
        policy: InsurancePolicy,
        condition_id: str,
    ) -> bool:
        """
        Oracle reports that a condition has been met.

        Meeting conditions increases payout probability.

        Args:
            policy: Target policy
            condition_id: Condition that was met

        Returns:
            True if condition was marked as met
        """
        if policy.status != PolicyStatus.ACTIVE:
            return False

        for condition in policy.conditions:
            if condition.condition_id == condition_id and not condition.met:
                condition.met = True

                # Boost payout amplitude
                boost = float(condition.boost_factor)
                self._adjust_amplitudes(policy, boost)

                return True

        return False

    def _adjust_amplitudes(
        self,
        policy: InsurancePolicy,
        payout_boost: float,
    ) -> None:
        """
        Adjust policy amplitudes.

        Args:
            policy: Target policy
            payout_boost: Amount to boost payout amplitude
        """
        # Find payout branch and boost it
        for branch in policy.psc.branches:
            if branch.state_data.get("outcome") == "payout":
                new_mag = branch.amplitude.magnitude + payout_boost
                branch.amplitude = DiscreteAmplitude(new_mag, branch.amplitude.phase)
                policy.payout_branch_amplitude = Decimal(str(new_mag))
                break

        # Invalidate cache
        policy.psc._interference_dirty = True

    def get_payout_probability(
        self,
        policy: InsurancePolicy,
    ) -> Decimal:
        """
        Get current payout probability.

        Args:
            policy: Target policy

        Returns:
            Payout probability (0-1)
        """
        probs = policy.psc.get_probabilities()

        for branch in policy.psc.branches:
            if branch.state_data.get("outcome") == "payout":
                return Decimal(str(probs.get(branch.state_id, 0)))

        return Decimal("0")

    def trigger_collapse(
        self,
        policy: InsurancePolicy,
        seed: Optional[int] = None,
    ) -> InsuranceResult:
        """
        Trigger policy collapse to determine payout.

        Args:
            policy: Policy to collapse
            seed: Random seed

        Returns:
            InsuranceResult with payout information
        """
        if policy.status != PolicyStatus.ACTIVE:
            return InsuranceResult(
                status=ApplicationStatus.FAILED,
                psc_id=policy.psc.id,
                metadata={"error": "Policy not active"},
            )

        # Get probability before collapse
        payout_prob = self.get_payout_probability(policy)

        # Collapse PSC
        selected_state, selected_branch = policy.psc.collapse(seed=seed)

        # Determine outcome
        outcome = selected_branch.state_data.get("outcome")
        payout = Decimal(str(selected_branch.state_data.get("amount", 0)))

        policy.payout_amount = payout
        policy.status = PolicyStatus.CLAIMED

        # Count conditions met
        conditions_met = sum(1 for c in policy.conditions if c.met)

        result = InsuranceResult(
            status=ApplicationStatus.COMPLETED,
            psc_id=policy.psc.id,
            selected_outcome=outcome,
            probability_distribution={
                "payout": float(payout_prob),
                "no_payout": float(1 - payout_prob),
            },
            metadata={
                "conditions": [
                    {"id": c.condition_id, "met": c.met}
                    for c in policy.conditions
                ],
            },
            policy_id=policy.policy_id,
            policyholder=policy.policyholder,
            coverage_amount=policy.coverage_amount,
            payout_amount=payout,
            conditions_met=conditions_met,
            payout_probability=float(payout_prob),
        )

        self.claims.append(result)
        return result

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "active_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]),
            "claimed_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.CLAIMED]),
            "total_policies": len(self.policies),
            "insurance_pool": self.insurance_pool,
            "block_number": self.block_number,
        }


# =============================================================================
# Simulation Helpers
# =============================================================================

def simulate_insurance_claim(
    coverage: Decimal = Decimal("10000"),
    conditions: List[str] = None,
    conditions_met: List[int] = None,
    seed: int = 42,
) -> InsuranceResult:
    """
    Simulate an insurance claim.

    Args:
        coverage: Coverage amount
        conditions: List of condition descriptions
        conditions_met: Indices of conditions that were met
        seed: Random seed

    Returns:
        InsuranceResult
    """
    if conditions is None:
        conditions = [
            "Flight delayed > 2 hours",
            "Weather emergency declared",
            "Airline bankruptcy",
        ]

    if conditions_met is None:
        conditions_met = [0]  # First condition met by default

    insurance = DecentralizedInsurance()

    # Create policy
    policy = insurance.create_policy(
        policyholder="Alice",
        coverage_amount=coverage,
        conditions=conditions,
    )

    print(f"Policy created: {policy.policy_id}")
    print(f"Coverage: {coverage}")
    print(f"Conditions: {len(conditions)}")

    # Initial probability
    prob = insurance.get_payout_probability(policy)
    print(f"\nInitial payout probability: {float(prob)*100:.1f}%")

    # Pay premium
    premium = coverage * PREMIUM_RATE
    insurance.pay_premium(policy, premium)
    print(f"Premium paid: {premium}")

    prob = insurance.get_payout_probability(policy)
    print(f"Probability after premium: {float(prob)*100:.1f}%")

    # Trigger conditions
    for idx in conditions_met:
        if idx < len(policy.conditions):
            cond = policy.conditions[idx]
            insurance.oracle_condition_met(policy, cond.condition_id)
            print(f"Condition met: {cond.description}")

    prob = insurance.get_payout_probability(policy)
    print(f"Probability after conditions: {float(prob)*100:.1f}%")

    # Trigger collapse
    result = insurance.trigger_collapse(policy, seed=seed)

    print(f"\nClaim result: {'PAYOUT' if result.payout_amount > 0 else 'NO PAYOUT'}")
    if result.payout_amount > 0:
        print(f"Payout amount: {result.payout_amount}")

    return result


def run_demo():
    """Run insurance demo."""
    print("=" * 60)
    print("Decentralized Insurance Demo")
    print("=" * 60)
    print()

    # Demo 1: Condition met, payout likely
    print("Demo 1: Single condition met")
    print("-" * 40)
    simulate_insurance_claim(
        conditions_met=[0],
        seed=42,
    )
    print()

    # Demo 2: Multiple conditions met
    print("Demo 2: Multiple conditions met")
    print("-" * 40)
    simulate_insurance_claim(
        conditions_met=[0, 1],
        seed=42,
    )
    print()


if __name__ == "__main__":
    run_demo()
