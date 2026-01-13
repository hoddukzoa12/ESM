"""
Interference Visualization

Before/After comparison charts for interference effects.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from esm.core.phase import DiscretePhase
from esm.core.amplitude import DiscreteAmplitude
from esm.core.psc import PSC


# Colors
COLORS = {
    'before': '#3498db',
    'after': '#2ecc71',
    'cancelled': '#e74c3c',
    'constructive': '#27ae60',
    'destructive': '#c0392b',
    'neutral': '#95a5a6',
}


def plot_interference_comparison(
    psc: PSC,
    title: str = "Interference Effect Analysis",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare probabilities before and after interference.

    Args:
        psc: PSC to analyze
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get report
    report = psc.get_interference_report()

    # Extract data
    state_ids = list(report['states'].keys())
    before_probs = []
    after_probs = []
    branch_counts = []

    for sid in state_ids:
        state_data = report['states'][sid]
        # Sum of individual branch probabilities
        before_prob = sum(b['probability_contribution'] for b in state_data['branches'])
        after_prob = state_data['result_amplitude'].probability()
        before_probs.append(before_prob)
        after_probs.append(after_prob)
        branch_counts.append(state_data['branch_count'])

    # Normalize
    total_before = sum(before_probs)
    total_after = sum(after_probs)

    if total_before > 0:
        before_probs = [p / total_before for p in before_probs]
    if total_after > 0:
        after_probs = [p / total_after for p in after_probs]

    # Left plot: Grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(state_ids))
    width = 0.35

    bars1 = ax1.bar(x - width/2, before_probs, width, label='Before Interference',
                   color=COLORS['before'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, after_probs, width, label='After Interference',
                   color=COLORS['after'], alpha=0.8)

    ax1.set_xlabel('State')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{sid[:6]}...' for sid in state_ids], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, max(max(before_probs), max(after_probs)) * 1.2)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # Right plot: Change visualization
    ax2 = axes[1]
    changes = [(after - before) for before, after in zip(before_probs, after_probs)]

    colors = []
    for change in changes:
        if change > 0.01:
            colors.append(COLORS['constructive'])
        elif change < -0.01:
            colors.append(COLORS['destructive'])
        else:
            colors.append(COLORS['neutral'])

    bars = ax2.bar(x, changes, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Probability Change')
    ax2.set_title('Interference Effect (After - Before)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{sid[:6]}...' for sid in state_ids], rotation=45, ha='right')

    # Add legend for change colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['constructive'], label='Constructive (+)'),
        Patch(facecolor=COLORS['destructive'], label='Destructive (-)'),
        Patch(facecolor=COLORS['neutral'], label='Neutral'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_interference_matrix(
    magnitude: float = 0.7,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show interference results as a heatmap matrix.

    X-axis: Phase of amplitude 1
    Y-axis: Phase of amplitude 2
    Color: Resulting magnitude

    Args:
        magnitude: Magnitude of both amplitudes
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate interference matrix
    matrix = np.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            amp1 = DiscreteAmplitude(magnitude, DiscretePhase(i))
            amp2 = DiscreteAmplitude(magnitude, DiscretePhase(j))
            result = amp1.add(amp2)
            matrix[i, j] = result.magnitude

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='equal',
                   vmin=0, vmax=2*magnitude)

    # Set labels
    phase_labels = [f'{i*45}°' for i in range(8)]
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(phase_labels)
    ax.set_yticklabels(phase_labels)
    ax.set_xlabel('Phase of α₂')
    ax.set_ylabel('Phase of α₁')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Resulting Magnitude |α₁ + α₂|')

    # Add text annotations
    for i in range(8):
        for j in range(8):
            text = f'{matrix[i, j]:.2f}'
            color = 'white' if matrix[i, j] < magnitude else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    ax.set_title(f'Interference Matrix (|α₁| = |α₂| = {magnitude})',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_phase_histogram(
    phases: List[DiscretePhase],
    title: str = "Phase Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a circular histogram of phase distribution.

    Args:
        phases: List of phases
        title: Plot title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             subplot_kw={'projection': 'polar'})

    # Count phases
    counts = np.zeros(8)
    for phase in phases:
        counts[phase.value] += 1

    # Normalize
    if sum(counts) > 0:
        proportions = counts / sum(counts)
    else:
        proportions = counts

    # Left: Circular bar chart
    ax1 = axes[0]
    theta = np.arange(8) * np.pi / 4
    width = np.pi / 4 * 0.8

    colors = plt.cm.hsv(np.linspace(0, 1, 8))
    bars = ax1.bar(theta, proportions, width=width, bottom=0,
                   color=colors, alpha=0.8, edgecolor='black')

    ax1.set_theta_zero_location('E')
    ax1.set_xticks(theta)
    ax1.set_xticklabels([f'{i*45}°' for i in range(8)])
    ax1.set_title('Polar Histogram')

    # Right: Regular bar chart
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(8)
    bars2 = ax2.bar(x, counts, color=colors, alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'P{i*45}' for i in range(8)])
    ax2.set_title('Bar Chart')

    # Add count labels
    for bar, count in zip(bars2, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(count)}', ha='center', va='bottom')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
