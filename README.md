# ESM Simulator

A Python simulation framework for the Entangled State Machine (ESM) probabilistic blockchain architecture.

## Overview

ESM Simulator implements the core mechanisms described in the ESM Whitepaper v5.3, providing tools for simulating and visualizing quantum-inspired blockchain operations including MEV resistance, privacy transfers, and probabilistic state management.

## Features

- **8-Phase Discrete Amplitude System** - Quantum-inspired state representation
- **PSC (Probabilistic State Container)** - Multi-branch state management with interference
- **MEV Resistance** - Phase-based front-running neutralization
- **6 Application Modules** - DEX, Privacy Transfer, Prediction Market, Insurance, Auction, NFT
- **Visualization Suite** - 16 publication-ready charts

## Installation

```bash
git clone https://github.com/hoddukzoa12/ESM.git
cd ESM/esm_simulator
pip install -r requirements.txt
```

## Quick Start

```bash
# Run complete demo
python examples/v54_complete_demo.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
esm_simulator/
├── esm/
│   ├── core/           # Amplitude, Phase, PSC, Branch
│   ├── simulation/     # MEV scenarios, Threshold reveal
│   ├── applications/   # 6 application modules
│   └── visualization/  # Chart generation
├── tests/              # 215 test cases
├── examples/           # Demo scripts
└── output/             # Generated visualizations
```

## Documentation

- [ESM Whitepaper v5.3](./ESM_Whitepaper_v5.4_EN.pdf)

## License

MIT License - see [LICENSE](./LICENSE) for details.
