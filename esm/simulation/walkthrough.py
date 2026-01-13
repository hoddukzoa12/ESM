"""
Step-by-Step Transaction Walkthrough

Based on ESM Whitepaper v5.3 Section 8.1 (MEV DEX Example)

Provides detailed step-by-step simulation with real numbers,
similar to Ethereum whitepaper's "Suppose a transaction is sent with 10 ether..."

Key features:
- Concrete numbers (100 ESM → TOKEN swap)
- Gas fees, interference deposits, phase calculations
- ASCII-art PSC state visualization
- MEV attack scenario demonstration
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import hashlib

from esm.core.phase import DiscretePhase, COS_TABLE, SIN_TABLE
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch, create_branch
from esm.core.psc import PSC, create_psc
from esm.core.deposit import calculate_deposit_with_buffer, DEFAULT_BUFFER_PERCENT


# =============================================================================
# Constants (from Whitepaper v5.3 Section 7.7)
# =============================================================================

# 1 ESM = 10^6 amp (smallest unit)
AMP_PER_ESM = Decimal("1000000")

# Fee structure (in ESM)
PSC_CREATION_FEE = Decimal("0.00001")  # 10 amp
BRANCH_FEE_BASE = Decimal("0.0001")    # 100 amp base
BRANCH_FEE_PER_KB = Decimal("0.00001") # per KB of data
READ_FEE = Decimal("0.000001")         # 1 amp per read

# Interference deposit rate (as fraction of tx value)
INTERFERENCE_DEPOSIT_RATE = Decimal("0.001")  # 0.1% of value
INTERFERENCE_BUFFER = Decimal("0.20")          # 20% buffer

# MEV timing thresholds (ms)
MEV_CANCEL_THRESHOLD_MS = 100
MEV_PARTIAL_THRESHOLD_MS = 500
MEV_ORTHOGONAL_THRESHOLD_MS = 1000

# Gas fee estimate (ESM)
GAS_FEE_ESTIMATE = Decimal("0.01")


class WalkthroughType(Enum):
    """Type of walkthrough scenario."""
    SIMPLE_TRANSFER = "simple_transfer"
    DEX_SWAP = "dex_swap"
    MEV_ATTACK = "mev_attack"
    PRIVACY_TRANSFER = "privacy_transfer"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WalkthroughStep:
    """
    A single step in the transaction walkthrough.

    Attributes:
        block: Block number
        timestamp_ms: Timestamp in milliseconds
        actor: Who is performing the action (Alice, Bot, System)
        action: Description of the action
        input_amount: Amount being transacted
        gas_fee: Gas fee for this step
        interference_deposit: Deposit paid for interference
        phase: Phase assigned to this transaction
        psc_state: Current PSC state snapshot
        amplitude_calculation: Detailed amplitude calculation string
        notes: Additional notes or observations
    """
    block: int
    timestamp_ms: int
    actor: str
    action: str
    input_amount: Decimal
    gas_fee: Decimal
    interference_deposit: Decimal
    phase: DiscretePhase
    psc_state: Dict[str, Any]
    amplitude_calculation: str
    notes: str = ""


@dataclass
class WalkthroughResult:
    """
    Complete result of a walkthrough simulation.

    Attributes:
        scenario_type: Type of scenario simulated
        steps: List of walkthrough steps
        final_outcome: Description of final outcome
        alice_receives: Amount Alice receives at the end
        bot_profit: Bot's profit (if applicable)
        mev_extracted: MEV extracted (traditional chain comparison)
        total_gas_paid: Total gas fees paid
        total_deposits: Total interference deposits
        deposit_refunds: Deposit refunds after collapse
    """
    scenario_type: WalkthroughType
    steps: List[WalkthroughStep]
    final_outcome: str
    alice_receives: Decimal
    bot_profit: Decimal
    mev_extracted: Decimal
    total_gas_paid: Decimal
    total_deposits: Decimal
    deposit_refunds: Decimal


@dataclass
class PSCSnapshot:
    """Snapshot of PSC state for visualization."""
    psc_id: str
    branches: List[Dict[str, Any]]
    interference_result: Dict[str, Any]
    probabilities: Dict[str, Decimal]


# =============================================================================
# Main Walkthrough Class
# =============================================================================

class TransactionWalkthrough:
    """
    Complete transaction walkthrough simulation.

    Demonstrates ESM mechanics with real numbers, following the
    "Alice sends 100 ESM" example from whitepaper v5.3.
    """

    def __init__(self):
        self.steps: List[WalkthroughStep] = []
        self.psc: Optional[PSC] = None
        self.current_block = 1000
        self.current_time_ms = 0

    # =========================================================================
    # MEV DEX Swap Walkthrough
    # =========================================================================

    def simulate_mev_dex_swap(
        self,
        alice_input: Decimal = Decimal("100"),
        bot_input: Decimal = Decimal("1000"),
        market_price: Decimal = Decimal("9.5"),  # TOKEN per ESM
        attack_delay_ms: int = 50,
        seed: Optional[int] = None,
    ) -> WalkthroughResult:
        """
        Simulate complete MEV attack scenario with real numbers.

        This demonstrates how ESM's phase-based interference
        neutralizes MEV extraction in DEX swaps.

        Args:
            alice_input: Alice's swap input in ESM
            bot_input: Bot's front-run amount in ESM
            market_price: TOKEN/ESM exchange rate
            attack_delay_ms: Bot's timing delay in ms
            seed: Random seed for deterministic collapse

        Returns:
            WalkthroughResult with complete simulation data
        """
        self.steps = []
        self.current_block = 1000
        self.current_time_ms = 0

        # Calculate expected outputs
        alice_expected_output = alice_input * market_price
        bot_expected_output = bot_input * market_price

        # =====================================================================
        # Step 1: Alice submits swap transaction
        # =====================================================================

        # Calculate fees for Alice
        alice_gas = GAS_FEE_ESTIMATE
        alice_base_deposit = alice_input * INTERFERENCE_DEPOSIT_RATE
        alice_deposit = alice_base_deposit * (1 + INTERFERENCE_BUFFER)

        # Create PSC
        psc_id = hashlib.sha256(f"swap_{self.current_block}".encode()).hexdigest()[:16]
        self.psc = create_psc(psc_id)

        # Create Alice's branch
        alice_state_data = {
            "type": "swap",
            "from": "ESM",
            "to": "TOKEN",
            "input": float(alice_input),
            "output": float(alice_expected_output),
            "trader": "Alice",
        }
        alice_branch = create_branch(
            state_data=alice_state_data,
            magnitude=1.0,
            phase=DiscretePhase.P0,  # Normal mode
            creator="Alice",
            tx_type="victim",
            interference_deposit=int(alice_deposit * AMP_PER_ESM),
        )
        self.psc.add_branch(alice_branch)

        # Calculate amplitude
        alice_amp = alice_branch.amplitude
        alice_r, alice_i = alice_amp.to_cartesian()

        alice_calc = (
            f"alpha_Alice = {alice_amp.magnitude:.1f} * (cos({alice_amp.phase.name[1:]}deg) + i*sin({alice_amp.phase.name[1:]}deg))\n"
            f"           = {alice_amp.magnitude:.1f} * ({COS_TABLE[alice_amp.phase]:.4f} + {SIN_TABLE[alice_amp.phase]:.4f}i)\n"
            f"           = {alice_r:.4f} + {alice_i:.4f}i"
        )

        psc_state_1 = self._capture_psc_state()

        step1 = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor="Alice",
            action=f"Submits swap: {alice_input} ESM -> {alice_expected_output} TOKEN",
            input_amount=alice_input,
            gas_fee=alice_gas,
            interference_deposit=alice_deposit,
            phase=DiscretePhase.P0,
            psc_state=psc_state_1,
            amplitude_calculation=alice_calc,
            notes=f"Phase: P0 (Normal) | Expected output: {alice_expected_output} TOKEN @ {market_price} TOKEN/ESM",
        )
        self.steps.append(step1)

        # =====================================================================
        # Step 2: MEV Bot front-running attack
        # =====================================================================

        self.current_time_ms += attack_delay_ms

        # Determine phase based on timing
        if attack_delay_ms < MEV_CANCEL_THRESHOLD_MS:
            bot_phase = DiscretePhase.P180  # Full cancellation
            phase_reason = f"delay {attack_delay_ms}ms < {MEV_CANCEL_THRESHOLD_MS}ms -> P180 (Counter)"
        elif attack_delay_ms < MEV_PARTIAL_THRESHOLD_MS:
            bot_phase = DiscretePhase.P135  # Partial cancellation
            phase_reason = f"delay {attack_delay_ms}ms < {MEV_PARTIAL_THRESHOLD_MS}ms -> P135 (PartialCounter)"
        elif attack_delay_ms < MEV_ORTHOGONAL_THRESHOLD_MS:
            bot_phase = DiscretePhase.P90   # Orthogonal
            phase_reason = f"delay {attack_delay_ms}ms < {MEV_ORTHOGONAL_THRESHOLD_MS}ms -> P90 (Independent)"
        else:
            bot_phase = DiscretePhase.P0    # Normal
            phase_reason = f"delay {attack_delay_ms}ms >= {MEV_ORTHOGONAL_THRESHOLD_MS}ms -> P0 (Normal)"

        # Calculate bot fees
        bot_gas = GAS_FEE_ESTIMATE
        bot_base_deposit = bot_input * INTERFERENCE_DEPOSIT_RATE
        bot_deposit = bot_base_deposit * (1 + INTERFERENCE_BUFFER)

        # Create bot's branch with SAME state_id to cause interference
        bot_state_data = {
            "type": "swap",
            "from": "ESM",
            "to": "TOKEN",
            "input": float(bot_input),
            "output": float(bot_expected_output),
            "trader": "Bot",
        }
        bot_branch = create_branch(
            state_data=bot_state_data,
            magnitude=1.0,
            phase=bot_phase,
            creator="Bot",
            tx_type="attacker",
            interference_deposit=int(bot_deposit * AMP_PER_ESM),
        )
        # Set same state_id to cause interference
        bot_branch.state_id = alice_branch.state_id
        self.psc.add_branch(bot_branch)

        # Calculate amplitude and interference
        bot_amp = bot_branch.amplitude
        bot_r, bot_i = bot_amp.to_cartesian()

        # Combined amplitude calculation
        total_r = alice_r + bot_r
        total_i = alice_i + bot_i
        total_mag = (total_r**2 + total_i**2)**0.5

        bot_calc = (
            f"alpha_Bot = {bot_amp.magnitude:.1f} * (cos({bot_phase.value * 45}deg) + i*sin({bot_phase.value * 45}deg))\n"
            f"         = {bot_amp.magnitude:.1f} * ({COS_TABLE[bot_phase]:.4f} + {SIN_TABLE[bot_phase]:.4f}i)\n"
            f"         = {bot_r:.4f} + {bot_i:.4f}i\n\n"
            f"Interference Calculation:\n"
            f"  alpha_total = alpha_Alice + alpha_Bot\n"
            f"             = ({alice_r:.4f} + {alice_i:.4f}i) + ({bot_r:.4f} + {bot_i:.4f}i)\n"
            f"             = {total_r:.4f} + {total_i:.4f}i\n"
            f"  |alpha_total|^2 = {total_mag**2:.4f}"
        )

        psc_state_2 = self._capture_psc_state()

        step2 = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor="MEV Bot",
            action=f"Front-run attack: {bot_input} ESM -> {bot_expected_output} TOKEN",
            input_amount=bot_input,
            gas_fee=bot_gas,
            interference_deposit=bot_deposit,
            phase=bot_phase,
            psc_state=psc_state_2,
            amplitude_calculation=bot_calc,
            notes=phase_reason,
        )
        self.steps.append(step2)

        # =====================================================================
        # Step 3: VDF and Threshold Reveal
        # =====================================================================

        self.current_block += 100  # VDF computation blocks

        vdf_step = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor="System",
            action="VDF Complete -> Threshold Reveal -> Collapse",
            input_amount=Decimal("0"),
            gas_fee=Decimal("0"),
            interference_deposit=Decimal("0"),
            phase=DiscretePhase.P0,
            psc_state=self._capture_psc_state(),
            amplitude_calculation=(
                f"VDF Seed: SHA256(BlockHeader || StateRoot)\n"
                f"Reveal Threshold: 67% stake met\n"
                f"Combined VDF output determines collapse"
            ),
            notes="Validators reveal VDF results, 67% threshold reached",
        )
        self.steps.append(vdf_step)

        # =====================================================================
        # Step 4: Calculate final probabilities and collapse
        # =====================================================================

        # Get interference results
        interference = self.psc.calculate_interference()
        probs = self.psc.get_probabilities()

        # Collapse PSC
        if seed is not None:
            selected_state, selected_branch = self.psc.collapse(seed=seed)
        else:
            selected_state, selected_branch = self.psc.collapse()

        # Determine outcome
        if bot_phase == DiscretePhase.P180:
            # Full destructive interference
            alice_prob = Decimal("1.0")  # Alice wins deterministically
            bot_prob = Decimal("0.0")
            outcome_desc = "Destructive interference: Bot's attack completely cancelled"
        else:
            # Get actual probabilities
            alice_prob = Decimal(str(probs.get(alice_branch.state_id, 0.5)))
            bot_prob = Decimal("1") - alice_prob
            outcome_desc = f"Partial interference: Alice {alice_prob*100:.1f}%, Bot {bot_prob*100:.1f}%"

        # Calculate final amounts
        if selected_branch.state_data.get("trader") == "Alice":
            alice_receives = alice_expected_output
            bot_profit = -bot_gas - bot_deposit  # Bot loses everything
            winner = "Alice"
        else:
            alice_receives = Decimal("0")
            bot_profit = bot_expected_output - bot_gas - bot_deposit
            winner = "Bot"

        # MEV extracted (compared to traditional chain)
        traditional_mev = alice_input * Decimal("0.03")  # 3% slippage

        collapse_calc = (
            f"Final Probabilities (after interference):\n"
            f"  Alice: |{alice_prob:.4f}|^2 = {float(alice_prob)*100:.1f}%\n"
            f"  Bot:   |{bot_prob:.4f}|^2 = {float(bot_prob)*100:.1f}%\n\n"
            f"Selected Branch: {winner} (deterministic due to interference)"
        )

        collapse_step = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor="System",
            action=f"PSC Collapsed -> Winner: {winner}",
            input_amount=Decimal("0"),
            gas_fee=Decimal("0"),
            interference_deposit=Decimal("0"),
            phase=DiscretePhase.P0,
            psc_state=self._capture_psc_state(),
            amplitude_calculation=collapse_calc,
            notes=outcome_desc,
        )
        self.steps.append(collapse_step)

        # =====================================================================
        # Create final result
        # =====================================================================

        total_gas = alice_gas + bot_gas
        total_deposits = alice_deposit + bot_deposit

        # Refunds (winning branch gets deposit back)
        if winner == "Alice":
            refunds = alice_deposit
        else:
            refunds = bot_deposit

        # Determine final outcome description
        if bot_phase == DiscretePhase.P180 and winner == "Alice":
            final_outcome = (
                f"MEV Attack NEUTRALIZED\n"
                f"Alice receives full expected output ({alice_receives} TOKEN)\n"
                f"Bot loses gas ({bot_gas} ESM) + deposit ({bot_deposit} ESM) = {bot_gas + bot_deposit} ESM"
            )
        else:
            final_outcome = f"Winner: {winner} receives output"

        return WalkthroughResult(
            scenario_type=WalkthroughType.MEV_ATTACK,
            steps=self.steps,
            final_outcome=final_outcome,
            alice_receives=alice_receives if winner == "Alice" else Decimal("0"),
            bot_profit=bot_profit,
            mev_extracted=traditional_mev if winner == "Bot" else Decimal("0"),
            total_gas_paid=total_gas,
            total_deposits=total_deposits,
            deposit_refunds=refunds,
        )

    def _capture_psc_state(self) -> Dict[str, Any]:
        """Capture current PSC state for visualization."""
        if self.psc is None:
            return {}

        branches_data = []
        for b in self.psc.branches:
            r, i = b.amplitude.to_cartesian()
            branches_data.append({
                "trader": b.state_data.get("trader", "Unknown"),
                "output": b.state_data.get("output", 0),
                "magnitude": b.amplitude.magnitude,
                "phase": b.amplitude.phase.name,
                "cartesian": f"{r:.4f} + {i:.4f}i",
                "probability": b.probability(),
            })

        # Calculate interference if multiple branches
        interference = self.psc.calculate_interference()
        probs = self.psc.get_probabilities()

        return {
            "psc_id": self.psc.id[:8],
            "branches": branches_data,
            "total_branches": len(self.psc.branches),
            "probabilities": probs,
            "interference_impact": self.psc.interference_impact(),
        }

    # =========================================================================
    # Simple Transfer Walkthrough
    # =========================================================================

    def simulate_simple_transfer(
        self,
        amount: Decimal = Decimal("100"),
        sender: str = "Alice",
        recipient: str = "Bob",
    ) -> WalkthroughResult:
        """
        Simulate a simple ESM transfer with real numbers.

        Args:
            amount: Transfer amount in ESM
            sender: Sender name
            recipient: Recipient name

        Returns:
            WalkthroughResult with transfer details
        """
        self.steps = []
        self.current_block = 1000
        self.current_time_ms = 0

        # Calculate fees
        gas_fee = GAS_FEE_ESTIMATE
        base_deposit = amount * INTERFERENCE_DEPOSIT_RATE
        deposit = base_deposit * (1 + INTERFERENCE_BUFFER)

        # Create PSC
        psc_id = hashlib.sha256(f"transfer_{self.current_block}".encode()).hexdigest()[:16]
        self.psc = create_psc(psc_id)

        # Create transfer branch
        transfer_branch = create_branch(
            state_data={
                "type": "transfer",
                "from": sender,
                "to": recipient,
                "amount": float(amount),
            },
            magnitude=1.0,
            phase=DiscretePhase.P0,
            creator=sender,
            tx_type="normal",
            interference_deposit=int(deposit * AMP_PER_ESM),
        )
        self.psc.add_branch(transfer_branch)

        amp = transfer_branch.amplitude
        r, i = amp.to_cartesian()

        step = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor=sender,
            action=f"Transfer {amount} ESM to {recipient}",
            input_amount=amount,
            gas_fee=gas_fee,
            interference_deposit=deposit,
            phase=DiscretePhase.P0,
            psc_state=self._capture_psc_state(),
            amplitude_calculation=f"alpha = {r:.4f} + {i:.4f}i (single branch, no interference)",
            notes="Simple transfer with 100% probability",
        )
        self.steps.append(step)

        # Immediate collapse (single branch)
        self.current_block += 1
        collapse_step = WalkthroughStep(
            block=self.current_block,
            timestamp_ms=self.current_time_ms,
            actor="System",
            action=f"Collapse -> {recipient} receives {amount} ESM",
            input_amount=Decimal("0"),
            gas_fee=Decimal("0"),
            interference_deposit=Decimal("0"),
            phase=DiscretePhase.P0,
            psc_state=self._capture_psc_state(),
            amplitude_calculation="Single branch: deterministic collapse",
            notes="No interference, immediate finality",
        )
        self.steps.append(collapse_step)

        return WalkthroughResult(
            scenario_type=WalkthroughType.SIMPLE_TRANSFER,
            steps=self.steps,
            final_outcome=f"{recipient} receives {amount} ESM",
            alice_receives=amount if recipient == "Alice" else Decimal("0"),
            bot_profit=Decimal("0"),
            mev_extracted=Decimal("0"),
            total_gas_paid=gas_fee,
            total_deposits=deposit,
            deposit_refunds=deposit,  # Full refund for successful transfer
        )

    # =========================================================================
    # Formatting Methods
    # =========================================================================

    def format_walkthrough(self, result: WalkthroughResult) -> str:
        """
        Format walkthrough result as ASCII-art output.

        Args:
            result: WalkthroughResult to format

        Returns:
            Formatted string for terminal display
        """
        lines = []
        width = 75

        # Header
        lines.append("=" * width)
        if result.scenario_type == WalkthroughType.MEV_ATTACK:
            lines.append("  ESM v5.4 Walkthrough: Alice sends 100 ESM -> TOKEN Swap")
            lines.append("  MEV Attack Scenario with Real Numbers")
        elif result.scenario_type == WalkthroughType.SIMPLE_TRANSFER:
            lines.append("  ESM v5.4 Walkthrough: Simple Transfer")
        else:
            lines.append(f"  ESM v5.4 Walkthrough: {result.scenario_type.value}")
        lines.append("=" * width)
        lines.append("")

        # Steps
        for i, step in enumerate(result.steps):
            lines.append(f"Block {step.block} (T+{step.timestamp_ms}ms): {step.actor}")
            lines.append("-" * width)

            if step.input_amount > 0:
                lines.append(f"  Action:              {step.action}")
                lines.append(f"  Input:               {step.input_amount:,.6f} ESM")
                lines.append(f"  Gas Fee:             {step.gas_fee:,.6f} ESM")
                lines.append(f"  Interference Deposit:{step.interference_deposit:,.6f} ESM")
                lines.append(f"  Phase:               {step.phase.name}")
            else:
                lines.append(f"  Action: {step.action}")

            if step.notes:
                lines.append(f"  Notes: {step.notes}")

            # PSC State visualization
            if step.psc_state and step.psc_state.get("branches"):
                lines.append("")
                lines.append("  PSC State:")
                lines.append("  " + "+" + "-" * 67 + "+")
                lines.append(f"  | PSC #{step.psc_state.get('psc_id', 'unknown'):<60}|")

                for j, branch in enumerate(step.psc_state["branches"]):
                    trader = branch["trader"]
                    output = branch["output"]
                    phase = branch["phase"]
                    mag = branch["magnitude"]
                    prob = branch["probability"] * 100

                    lines.append(f"  | Branch {j+1}: {trader} -> {output:.0f} TOKEN{' ' * (40 - len(trader) - len(str(int(output))))}|")
                    lines.append(f"  |           amplitude: ({mag:.1f}, {phase})   probability: {prob:.0f}%{' ' * (15 - len(f'{prob:.0f}'))}|")

                # Interference info
                if step.psc_state.get("interference_impact", 1.0) < 0.99:
                    lines.append(f"  |{' ' * 67}|")
                    lines.append(f"  | Interference Effect: {step.psc_state['interference_impact']:.2f}x{' ' * 44}|")

                lines.append("  " + "+" + "-" * 67 + "+")

            # Amplitude calculation
            if step.amplitude_calculation and "alpha" in step.amplitude_calculation.lower():
                lines.append("")
                lines.append("  Amplitude Calculation:")
                for calc_line in step.amplitude_calculation.split("\n"):
                    lines.append(f"    {calc_line}")

            lines.append("")

        # Outcome Summary
        lines.append("=" * width)
        lines.append("  OUTCOME SUMMARY")
        lines.append("=" * width)

        if result.scenario_type == WalkthroughType.MEV_ATTACK:
            if result.alice_receives > 0:
                lines.append(f"  [OK] Alice receives:     {result.alice_receives:,.6f} TOKEN")
                lines.append(f"  [X]  Bot profit:         {result.bot_profit:,.6f} ESM")
                lines.append(f"")
                lines.append(f"  Bot losses:")
                gas_loss = GAS_FEE_ESTIMATE
                deposit_loss = result.total_deposits - result.deposit_refunds
                lines.append(f"     - Gas fee:            {gas_loss:,.6f} ESM")
                lines.append(f"     - Forfeited deposit:  {deposit_loss:,.6f} ESM")
                lines.append(f"     - Total loss:         {gas_loss + deposit_loss:,.6f} ESM")
            else:
                lines.append(f"  [X]  Alice receives:     0 TOKEN")
                lines.append(f"  [OK] Bot profit:         {result.bot_profit:,.6f} ESM")

            lines.append(f"")
            lines.append(f"  [$$] MEV Extracted:      ${float(result.mev_extracted):.2f} (vs ${float(result.alice_receives * Decimal('0.03')):.2f} on traditional chain)")
            lines.append(f"  [--] Attack ROI:         {'Negative (failed)' if result.bot_profit < 0 else 'Positive'}")
        else:
            lines.append(f"  Final outcome: {result.final_outcome}")
            lines.append(f"  Total gas paid: {result.total_gas_paid:,.6f} ESM")
            lines.append(f"  Deposit refund: {result.deposit_refunds:,.6f} ESM")

        lines.append("=" * width)

        return "\n".join(lines)


# =============================================================================
# Simulation Helpers
# =============================================================================

def run_mev_walkthrough(
    alice_amount: Decimal = Decimal("100"),
    attack_delay_ms: int = 50,
    market_price: Decimal = Decimal("9.5"),
    verbose: bool = True,
) -> WalkthroughResult:
    """
    Run MEV attack walkthrough with default parameters.

    Args:
        alice_amount: Alice's input amount in ESM
        attack_delay_ms: Bot's attack timing delay
        market_price: TOKEN/ESM exchange rate
        verbose: Whether to print formatted output

    Returns:
        WalkthroughResult
    """
    walkthrough = TransactionWalkthrough()
    result = walkthrough.simulate_mev_dex_swap(
        alice_input=alice_amount,
        bot_input=alice_amount * 10,  # Bot uses 10x
        market_price=market_price,
        attack_delay_ms=attack_delay_ms,
        seed=42,
    )

    if verbose:
        print(walkthrough.format_walkthrough(result))

    return result


def compare_timing_scenarios(
    alice_amount: Decimal = Decimal("100"),
) -> Dict[str, WalkthroughResult]:
    """
    Compare different attack timing scenarios.

    Args:
        alice_amount: Alice's input amount

    Returns:
        Dictionary of timing -> result
    """
    scenarios = {
        "fast_50ms": 50,
        "medium_300ms": 300,
        "slow_800ms": 800,
        "very_slow_1500ms": 1500,
    }

    results = {}
    walkthrough = TransactionWalkthrough()

    for name, delay in scenarios.items():
        result = walkthrough.simulate_mev_dex_swap(
            alice_input=alice_amount,
            attack_delay_ms=delay,
            seed=42,
        )
        results[name] = result

    return results


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Run walkthrough demonstration."""
    print("=" * 75)
    print("ESM v5.4 Transaction Walkthrough Demo")
    print("=" * 75)
    print()

    # Run MEV attack scenario
    print("Running MEV Attack Scenario...")
    print()

    result = run_mev_walkthrough(
        alice_amount=Decimal("100"),
        attack_delay_ms=50,
        verbose=True,
    )

    print()
    print("Comparing different attack timings:")
    print("-" * 50)

    comparisons = compare_timing_scenarios()
    for name, res in comparisons.items():
        status = "BLOCKED" if res.alice_receives > 0 else "SUCCESS"
        print(f"  {name}: Bot {status}, profit = {res.bot_profit:.4f} ESM")

    return result


if __name__ == "__main__":
    run_demo()
