"""
Amplitude Visualization - Polar Plots

Visualizes 8-phase discrete amplitudes on polar coordinates.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

from esm.core.phase import DiscretePhase, COS_TABLE, SIN_TABLE
from esm.core.amplitude import DiscreteAmplitude
from esm.core.branch import Branch
from esm.core.psc import PSC


# Color scheme
COLORS = {
    'victim': '#2ecc71',      # Green
    'attacker': '#e74c3c',    # Red
    'normal': '#3498db',      # Blue
    'result': '#9b59b6',      # Purple
    'cancelled': '#95a5a6',   # Gray
}

PHASE_COLORS = [
    '#e74c3c',  # P0: Red
    '#e67e22',  # P45: Orange
    '#f1c40f',  # P90: Yellow
    '#2ecc71',  # P135: Green
    '#1abc9c',  # P180: Teal
    '#3498db',  # P225: Blue
    '#9b59b6',  # P270: Purple
    '#e91e63',  # P315: Pink
]


def plot_amplitude_polar(
    amplitudes: List[Tuple[DiscreteAmplitude, str, str]],
    title: str = "8-Phase Amplitude Visualization",
    show_result: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot amplitudes on a polar coordinate system.

    Args:
        amplitudes: List of (amplitude, label, color) tuples
        title: Plot title
        show_result: Whether to show the sum (interference result)
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Draw 8-phase reference lines
    for i in range(8):
        angle = i * np.pi / 4
        ax.axvline(angle, color='lightgray', linestyle='--', alpha=0.5)

    # Plot each amplitude
    for amp, label, color in amplitudes:
        angle = amp.phase.value * np.pi / 4
        r = amp.magnitude

        # Draw arrow from origin
        ax.annotate('', xy=(angle, r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Add label
        ax.annotate(label, xy=(angle, r * 1.1),
                    ha='center', va='bottom', fontsize=10, color=color)

    # Calculate and show result if requested
    if show_result and len(amplitudes) > 1:
        result = amplitudes[0][0]
        for amp, _, _ in amplitudes[1:]:
            result = result.add(amp)

        result_angle = result.phase.value * np.pi / 4
        result_r = result.magnitude

        ax.annotate('', xy=(result_angle, result_r), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLORS['result'],
                                   lw=3, linestyle='--'))
        ax.annotate(f'Result\n|α|={result_r:.3f}',
                    xy=(result_angle, result_r * 1.15),
                    ha='center', va='bottom', fontsize=10,
                    color=COLORS['result'], fontweight='bold')

    # Customize plot
    ax.set_theta_zero_location('E')  # 0° on the right
    ax.set_theta_direction(1)  # Counter-clockwise

    # Set phase labels
    phase_labels = [f'{i*45}°\n(P{i*45})' for i in range(8)]
    ax.set_xticks(np.arange(8) * np.pi / 4)
    ax.set_xticklabels(phase_labels)

    # Set radial limits
    max_r = max(amp.magnitude for amp, _, _ in amplitudes) * 1.3
    ax.set_ylim(0, max_r)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_interference_demo(
    phase_diff: int = 4,
    magnitude: float = 0.7,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Demonstrate interference between two amplitudes.

    Args:
        phase_diff: Phase difference in 45° steps (0-7)
        magnitude: Magnitude of both amplitudes
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             subplot_kw={'projection': 'polar'})

    amp1 = DiscreteAmplitude(magnitude, DiscretePhase.P0)
    amp2 = DiscreteAmplitude(magnitude, DiscretePhase(phase_diff))
    result = amp1.add(amp2)

    # Before interference
    ax1 = axes[0]
    for i in range(8):
        ax1.axvline(i * np.pi / 4, color='lightgray', linestyle='--', alpha=0.5)

    ax1.annotate('', xy=(0, magnitude), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=COLORS['victim'], lw=2))
    ax1.annotate('', xy=(phase_diff * np.pi / 4, magnitude), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=COLORS['attacker'], lw=2))

    ax1.set_title(f'Before Interference\nΔθ = {phase_diff * 45}°', fontsize=12)
    ax1.set_ylim(0, magnitude * 1.5)

    # After interference
    ax2 = axes[1]
    for i in range(8):
        ax2.axvline(i * np.pi / 4, color='lightgray', linestyle='--', alpha=0.5)

    result_angle = result.phase.value * np.pi / 4
    color = COLORS['result'] if result.magnitude > 0.1 else COLORS['cancelled']

    ax2.annotate('', xy=(result_angle, result.magnitude), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=3))

    prob_before = 2 * magnitude ** 2
    prob_after = result.probability()
    change = (prob_after - prob_before) / prob_before * 100

    ax2.set_title(f'After Interference\n|α|={result.magnitude:.3f}, P={prob_after:.3f}\n({change:+.1f}%)',
                 fontsize=12)
    ax2.set_ylim(0, magnitude * 1.5)

    plt.suptitle('8-Phase Interference Demonstration', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_all_interference_patterns(
    magnitude: float = 0.7,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show all 8 interference patterns in a grid.

    Args:
        magnitude: Magnitude of amplitudes
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize,
                             subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    interference_types = [
        "Constructive (Full)",
        "Constructive (Partial)",
        "Orthogonal",
        "Destructive (Partial)",
        "Destructive (Full)",
        "Destructive (Partial)",
        "Orthogonal",
        "Constructive (Partial)",
    ]

    for i, ax in enumerate(axes):
        amp1 = DiscreteAmplitude(magnitude, DiscretePhase.P0)
        amp2 = DiscreteAmplitude(magnitude, DiscretePhase(i))
        result = amp1.add(amp2)

        # Draw reference
        for j in range(8):
            ax.axvline(j * np.pi / 4, color='lightgray', linestyle='--', alpha=0.3)

        # Draw amplitudes
        ax.annotate('', xy=(0, magnitude), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['victim'], lw=1.5, alpha=0.5))
        ax.annotate('', xy=(i * np.pi / 4, magnitude), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['attacker'], lw=1.5, alpha=0.5))

        # Draw result
        result_angle = result.phase.value * np.pi / 4
        result_color = PHASE_COLORS[result.phase.value] if result.magnitude > 0.05 else COLORS['cancelled']
        ax.annotate('', xy=(result_angle, result.magnitude), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=result_color, lw=2.5))

        ax.set_title(f'{i*45}°: {interference_types[i]}\n|α|={result.magnitude:.2f}',
                    fontsize=9)
        ax.set_ylim(0, magnitude * 2.5)
        ax.set_xticklabels([])

    plt.suptitle('All 8-Phase Interference Patterns (α₁ at 0°, α₂ varying)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_psc_state(
    psc: PSC,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize PSC branches and interference.

    Args:
        psc: PSC to visualize
        title: Optional title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Polar plot of all branches
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')

    for i in range(8):
        ax1.axvline(i * np.pi / 4, color='lightgray', linestyle='--', alpha=0.5)

    max_mag = 0
    for branch in psc.branches:
        angle = branch.amplitude.phase.value * np.pi / 4
        mag = branch.amplitude.magnitude
        max_mag = max(max_mag, mag)

        color = COLORS.get(branch.tx_type, COLORS['normal'])
        ax1.annotate('', xy=(angle, mag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))

    ax1.set_ylim(0, max_mag * 1.3)
    ax1.set_title('Branch Amplitudes', fontsize=12)

    # Right: Bar chart of probabilities
    ax2 = axes[1]
    probs = psc.get_probabilities()

    state_ids = list(probs.keys())
    prob_values = [probs[sid] for sid in state_ids]
    short_ids = [sid[:8] + '...' for sid in state_ids]

    bars = ax2.bar(short_ids, prob_values, color=COLORS['normal'], alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('State ID')
    ax2.set_title('State Probabilities (After Interference)', fontsize=12)
    ax2.set_ylim(0, 1.0)

    # Add value labels
    for bar, prob in zip(bars, prob_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=9)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
