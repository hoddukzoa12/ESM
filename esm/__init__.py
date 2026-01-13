"""
ESM Simulator - Entangled State Machine Simulation Framework

A Python implementation of the ESM whitepaper v5.1 core mechanisms,
including 8-phase discrete amplitude operations, PSC interference calculation,
and MEV resistance simulation.
"""

__version__ = "0.1.0"
__author__ = "ESM Research Group"

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch
from esm.core.psc import PSC

__all__ = [
    "DiscretePhase",
    "DiscreteAmplitude",
    "Branch",
    "PSC",
]
