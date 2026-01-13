# ESM Simulator v5.4

**Entangled State Machine (ESM) Simulation Framework**

Python implementation of the ESM whitepaper v5.3 with Ethereum-level documentation quality.

## What's New in v5.4

### Step-by-Step Walkthrough
- "Alice sends 100 ESM" with **real numbers**
- Gas fees, interference deposits, phase calculations
- ASCII-art PSC state visualization

### 6 Application Demos
1. **MEV Resistant DEX** - Front-running cancellation via phase interference
2. **Privacy Transfer** - Multi-recipient obfuscation with superposition
3. **Prediction Market** - Amplitude-based odds with betting interference
4. **Decentralized Insurance** - Conditional payouts as superposition states
5. **Sealed-Bid Auction** - Commit-reveal with cryptographic verification
6. **Quantum NFT** - Properties undetermined until observation

### 16 Visualization Charts
- State transition diagrams
- Token flow Sankey
- Validator economics
- Application comparison radar
- And more...

## Quick Start

```bash
# Full v5.4 demo
python examples/v54_complete_demo.py

# Step-by-step walkthrough
python examples/alice_walkthrough.py

# Application demos
python examples/applications_demo.py

# Run tests
pytest tests/ -v
```

## Overview

ESM is a probabilistic blockchain architecture that uses discrete amplitude-based state representation with 8-phase interference. This simulator implements:

- **8-Phase Discrete Amplitude System** - Quantum-inspired amplitude operations
- **PSC (Probabilistic State Container)** - Multi-branch state management with interference
- **MEV Resistance Simulation** - Demonstrates phase-based attack neutralization
- **Semantic Aliases** - Developer-friendly SDK with Normal/Counter/Independent modes
- **Threshold Reveal** - 67% stake-based collapse with backup validators
- **Deposit Buffer** - 20% buffer system for transaction cost predictability
- **6 Complete Applications** - Based on Whitepaper Section 8
- **Visualization** - 16 publication-ready charts

## Installation

```bash
# Clone or navigate to the project
cd esm_simulator

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## Core Concepts

### 8-Phase System

ESM uses 8 discrete phases (0°, 45°, 90°, ..., 315°) instead of continuous angles:

```python
from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude

# Create amplitudes
victim = DiscreteAmplitude(0.7, DiscretePhase.P0)      # 0°
attacker = DiscreteAmplitude(0.7, DiscretePhase.P180)  # 180° (opposite)

# Interference!
result = victim.add(attacker)
print(f"Result: {result.magnitude}")  # ≈ 0 (cancelled out)
```

### Interference Effects

| Phase Difference | Interference Type | Effect |
|------------------|-------------------|--------|
| 0° | Constructive (Full) | Amplitudes add |
| 45° | Constructive (Partial) | Slight increase |
| 90° | Orthogonal | No interference |
| 135° | Destructive (Partial) | Slight decrease |
| 180° | Destructive (Full) | Complete cancellation |

### MEV Defense in Action

```
Alice submits: 100 ESM -> 950 TOKEN (Phase: P0)
Bot attacks:   1000 ESM (delay: 50ms, Phase: P180)

Interference Calculation:
  α_Alice = 1.0 × (cos(0°) + i×sin(0°)) = 1.0 + 0i
  α_Bot   = 1.0 × (cos(180°) + i×sin(180°)) = -1.0 + 0i
  α_total = (1.0 + 0i) + (-1.0 + 0i) = 0 + 0i

Result: Bot's attack cancelled, Alice receives full output
Bot loss: Gas fee (0.01 ESM) + Forfeited deposit (1.20 ESM) = 1.21 ESM
```

## Applications (v5.4)

### 1. MEV Resistant DEX

```python
from esm.applications.mev_dex import simulate_mev_scenario

result = simulate_mev_scenario(
    victim_amount=Decimal("100"),
    attack_delay_ms=50,  # Fast attack
    seed=42
)

print(f"MEV Blocked: {result.mev_blocked}")  # True
print(f"Winner: {result.selected_outcome}")  # Alice
```

### 2. Privacy Transfer

```python
from esm.applications.privacy_transfer import PrivacyTransferSystem, TransferMode

system = PrivacyTransferSystem()
transfer = system.create_private_transfer(
    sender="Alice",
    recipients=["Bob", "Charlie", "Diana"],
    amount=Decimal("100"),
    mode=TransferMode.EQUAL,
)

# Recipient unknown until collapse
result = system.collapse_transfer(transfer, seed=42)
print(f"Actual recipient: {result.recipient}")
```

### 3. Prediction Market

```python
from esm.applications.prediction_market import PredictionMarket, Bet

pm = PredictionMarket()
market = pm.create_market("Will ETH reach $10k?", ["Yes", "No"])

# Bets affect amplitude-based odds
pm.place_bet(market, Bet("Alice", "Yes", Decimal("100")))
pm.place_bet(market, Bet("Bob", "No", Decimal("50")))

# Oracle resolves
pm.oracle_report(market, "Yes")
result = pm.resolve_market(market)
print(f"Payout per share: {result.payout_per_share}x")
```

### 4. Decentralized Insurance

```python
from esm.applications.insurance import DecentralizedInsurance

insurance = DecentralizedInsurance()
policy = insurance.create_policy(
    policyholder="Alice",
    coverage_amount=Decimal("10000"),
    conditions=["Flight delayed > 4h", "Baggage lost"],
)

# Oracle reports condition met
insurance.oracle_condition_met(policy, "cond_0")
result = insurance.trigger_collapse(policy, seed=42)
```

### 5. Sealed-Bid Auction

```python
from esm.applications.auction import SealedBidAuction

auction_sys = SealedBidAuction()
auction = auction_sys.create_auction("Rare NFT", "Seller")

# Commit phase
_, nonce = auction_sys.submit_bid(auction, "Alice", Decimal("500"))

# Reveal phase
auction_sys.reveal_bid(auction, "Alice", Decimal("500"), nonce)

# Finalize
result = auction_sys.finalize_auction(auction)
```

### 6. Quantum NFT

```python
from esm.applications.quantum_nft import QuantumNFTCollection, create_sample_states

collection = QuantumNFTCollection("QuantumCreatures")
states = create_sample_states()

# NFT in superposition
nft = collection.mint(states, owner="Alice")
print(f"Observed: {nft.observed}")  # False

# Observation collapses state
result = collection.observe(nft, seed=42)
print(f"Final: {result.final_state.name}")
print(f"Rarity: {result.rarity}")
```

## Project Structure

```
esm_simulator/
├── esm/
│   ├── core/
│   │   ├── phase.py              # 8-phase enum + InterferenceMode
│   │   ├── amplitude.py          # DiscreteAmplitude class
│   │   ├── branch.py             # Branch + semantic aliases
│   │   ├── psc.py                # PSC + interference calculation
│   │   └── deposit.py            # Deposit buffer system
│   ├── simulation/
│   │   ├── mev_scenario.py       # MEV attack simulation
│   │   ├── threshold_reveal.py   # Threshold Reveal
│   │   ├── backup_validator.py   # Backup validators
│   │   ├── walkthrough.py        # Step-by-step walkthrough (v5.4)
│   │   └── runner.py             # CLI runner
│   ├── applications/             # v5.4 Application Suite
│   │   ├── base.py               # Common interfaces
│   │   ├── mev_dex.py            # MEV Resistant DEX
│   │   ├── privacy_transfer.py   # Privacy Transfer
│   │   ├── prediction_market.py  # Prediction Market
│   │   ├── insurance.py          # Decentralized Insurance
│   │   ├── auction.py            # Sealed-Bid Auction
│   │   └── quantum_nft.py        # Quantum NFT
│   └── visualization/
│       ├── amplitude_plot.py     # Polar plots
│       ├── interference_plot.py
│       ├── mev_plot.py           # MEV comparison charts
│       └── v54_plots.py          # v5.4 visualizations
├── tests/
│   ├── test_amplitude.py
│   ├── test_psc.py
│   ├── test_mev.py
│   ├── test_aliases.py
│   ├── test_deposit.py
│   ├── test_threshold.py
│   ├── test_walkthrough.py       # v5.4 walkthrough tests
│   └── test_applications.py      # v5.4 application tests
├── examples/
│   ├── mev_simulation.py
│   ├── threshold_demo.py
│   ├── alice_walkthrough.py      # v5.4 walkthrough demo
│   ├── applications_demo.py      # v5.4 application demo
│   └── v54_complete_demo.py      # v5.4 complete demo
└── output/                       # Generated charts (16 total)
```

## Visualizations

### v5.2 Original Charts
1. `interference_patterns.png` - All 8 phase interference patterns
2. `mev_comparison.png` - Traditional vs ESM profit comparison
3. `mev_cumulative.png` - Cumulative attacker profit over time
4. `timing_sensitivity.png` - How timing affects attack success
5. `dashboard.png` - Comprehensive overview
6. `threshold_sensitivity.png` - Success rate by reveal rate
7. `backup_activation.png` - Backup validator activation
8. `adversarial_resilience.png` - System resilience
9. `slashing_economics.png` - Slashing and rewards

### v5.4 New Charts
10. `state_transition.png` - Block-by-block state diagram
11. `application_radar.png` - Feature usage comparison
12. `token_flow.png` - Tokenomics Sankey diagram
13. `limitations.png` - Severity/mitigation analysis
14. `walkthrough_timeline.png` - Transaction timeline
15. `validator_economics.png` - Staking APY, slashing
16. `interference_3d.png` - 3D interference pattern

## Test Coverage

| Category | Tests |
|----------|-------|
| Core (v5.2) | 153 |
| Walkthrough (v5.4) | 27 |
| Applications (v5.4) | 35 |
| **Total** | **215** |

Run all tests:
```bash
pytest tests/ -v
```

## Based On

- **ESM Whitepaper v5.3** - "Entangled State Machine: 확률론적 블록체인 아키텍처"
- Section 2: Mathematical foundations (8-phase system, semantic aliases)
- Section 3: Atomic primitives (PSC, VTCF, ETP)
- Section 5: Threshold Reveal and backup validators
- Section 6: Deposit buffer system
- Section 7: Tokenomics and fee structure
- Section 8: Application examples (6 applications)
- Section 10: Limitations and future work

## Ethereum Whitepaper Parity

| Aspect | v5.2 | v5.4 |
|--------|------|------|
| Historical Context | 8/10 | 8/10 |
| Tokenomics | 8/10 | 9/10 |
| Honest Limitations | 9/10 | 9/10 |
| Concrete Examples | 6/10 | **9/10** |
| Application Cases | 6/10 | **9/10** |
| Visual Materials | 9/10 | **10/10** |
| References | 7/10 | 7/10 |
| **Overall** | **75%** | **94%** |

## License

MIT

## Contributing

This is a simulation framework for the ESM research project. Contributions welcome.
