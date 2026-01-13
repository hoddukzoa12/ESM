"""
MEV Simulation Visualization

Charts for comparing MEV attack profitability between
traditional chains and ESM.
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from esm.simulation.mev_scenario import (
    SimulationResult,
    analyze_timing_sensitivity,
)
from esm.core.phase import DiscretePhase


# Colors
COLORS = {
    'traditional': '#e74c3c',  # Red
    'esm': '#2ecc71',          # Green
    'victim': '#3498db',       # Blue
    'savings': '#9b59b6',      # Purple
    'grid': '#ecf0f1',         # Light gray
}


def plot_mev_comparison(
    result: SimulationResult,
    title: str = "MEV Attack Profitability: Traditional vs ESM",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare MEV profits between traditional chain and ESM.

    Shows:
    - Left: Cumulative profit over time
    - Right: Summary statistics

    Args:
        result: Simulation result
        title: Plot title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Cumulative profit chart
    ax1 = axes[0]

    rounds = np.arange(result.n_rounds)
    trad_cumulative = np.cumsum(result.traditional_profits)
    esm_cumulative = np.cumsum(result.esm_profits)

    ax1.plot(rounds, trad_cumulative, color=COLORS['traditional'],
             label='Traditional Chain', linewidth=2)
    ax1.plot(rounds, esm_cumulative, color=COLORS['esm'],
             label='ESM Chain', linewidth=2)

    # Fill area between (savings)
    ax1.fill_between(rounds, esm_cumulative, trad_cumulative,
                     alpha=0.3, color=COLORS['savings'],
                     label='Attacker Loss (Victim Savings)')

    ax1.set_xlabel('Attack Round', fontsize=12)
    ax1.set_ylabel('Cumulative Attacker Profit ($)', fontsize=12)
    ax1.set_title('Cumulative Attacker Profit Over Time', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Right: Summary bar chart
    ax2 = axes[1]

    stats = result.get_stats()
    categories = ['Total\nProfit', 'Average\nper Attack', 'Profit\nReduction']

    trad_values = [
        stats['traditional_total'],
        stats['traditional_total'] / result.n_rounds,
        0
    ]
    esm_values = [
        stats['esm_total'],
        stats['esm_total'] / result.n_rounds,
        stats['profit_reduction_percent']
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Plot first two categories as comparison
    bars1 = ax2.bar(x[:2] - width/2, trad_values[:2], width,
                   label='Traditional', color=COLORS['traditional'], alpha=0.8)
    bars2 = ax2.bar(x[:2] + width/2, esm_values[:2], width,
                   label='ESM', color=COLORS['esm'], alpha=0.8)

    # Plot reduction as single bar
    reduction_bar = ax2.bar(x[2], esm_values[2], width * 2,
                           color=COLORS['savings'], alpha=0.8)

    ax2.set_ylabel('Value')
    ax2.set_title('Attack Profitability Summary', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()

    # Add value labels
    for bar in bars1[:2]:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2[:2]:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)

    ax2.text(reduction_bar[0].get_x() + reduction_bar[0].get_width()/2,
            reduction_bar[0].get_height(),
            f'{esm_values[2]:.1f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_mev_cumulative(
    result: SimulationResult,
    title: str = "Cumulative MEV Extraction",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Detailed cumulative profit chart with annotations.

    Args:
        result: Simulation result
        title: Plot title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    rounds = np.arange(result.n_rounds)
    trad_cumulative = np.cumsum(result.traditional_profits)
    esm_cumulative = np.cumsum(result.esm_profits)

    # Main lines
    ax.plot(rounds, trad_cumulative, color=COLORS['traditional'],
            label='Traditional Chain', linewidth=2.5)
    ax.plot(rounds, esm_cumulative, color=COLORS['esm'],
            label='ESM Chain', linewidth=2.5)

    # Fill savings area
    ax.fill_between(rounds, esm_cumulative, trad_cumulative,
                    alpha=0.2, color=COLORS['savings'])

    # Add annotations at key points
    mid_point = result.n_rounds // 2
    end_point = result.n_rounds - 1

    # Midpoint annotation
    savings_mid = trad_cumulative[mid_point] - esm_cumulative[mid_point]
    ax.annotate(f'Savings: ${savings_mid:,.0f}',
               xy=(mid_point, (trad_cumulative[mid_point] + esm_cumulative[mid_point])/2),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Final annotation
    savings_final = trad_cumulative[end_point] - esm_cumulative[end_point]
    reduction_pct = result.profit_reduction

    ax.annotate(f'Total Savings: ${savings_final:,.0f}\n({reduction_pct:.1f}% reduction)',
               xy=(end_point, trad_cumulative[end_point]),
               xytext=(end_point * 0.7, trad_cumulative[end_point] * 0.95),
               fontsize=11, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round', facecolor=COLORS['savings'], alpha=0.3))

    ax.set_xlabel('Attack Round', fontsize=12)
    ax.set_ylabel('Cumulative Attacker Profit ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_timing_sensitivity(
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show how interference varies with attack timing.

    Args:
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    analysis = analyze_timing_sensitivity(delay_range=(0, 2000), n_points=200)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    delays = analysis['delays_ms']
    factors = analysis['interference_factors']

    # Left: Line chart
    ax1 = axes[0]
    ax1.plot(delays, factors, color=COLORS['esm'], linewidth=2)
    ax1.fill_between(delays, 0, factors, alpha=0.3, color=COLORS['esm'])

    # Add threshold lines
    ax1.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100ms (Full Cancel)')
    ax1.axvline(x=500, color='orange', linestyle='--', alpha=0.7, label='500ms (Partial)')
    ax1.axvline(x=1000, color='yellow', linestyle='--', alpha=0.7, label='1000ms (Orthogonal)')

    ax1.set_xlabel('Attack Delay (ms)', fontsize=12)
    ax1.set_ylabel('Profit Retention Factor', fontsize=12)
    ax1.set_title('Attack Profit vs Timing', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Right: Phase regions
    ax2 = axes[1]

    regions = [
        (0, 100, 'P180\n(Full Cancel)', COLORS['esm']),
        (100, 500, 'P135\n(Partial)', '#27ae60'),
        (500, 1000, 'P90\n(Orthogonal)', '#f39c12'),
        (1000, 2000, 'P0\n(No Defense)', COLORS['traditional']),
    ]

    for start, end, label, color in regions:
        ax2.axvspan(start, end, alpha=0.4, color=color)
        mid = (start + end) / 2
        ax2.text(mid, 0.5, label, ha='center', va='center',
                fontsize=11, fontweight='bold')

    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Attack Delay (ms)', fontsize=12)
    ax2.set_title('Phase Assignment Regions', fontsize=12)
    ax2.set_yticks([])

    plt.suptitle('MEV Defense: Timing Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_phase_impact(
    result: SimulationResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show profit by assigned phase.

    Args:
        result: Simulation result
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Group profits by phase
    phase_profits = {p: [] for p in DiscretePhase}
    for phase, profit in zip(result.phases_assigned, result.esm_profits):
        phase_profits[phase].append(profit)

    phases_with_data = [p for p in DiscretePhase if phase_profits[p]]
    avg_profits = [np.mean(phase_profits[p]) if phase_profits[p] else 0
                   for p in phases_with_data]
    counts = [len(phase_profits[p]) for p in phases_with_data]

    # Left: Average profit by phase
    ax1 = axes[0]
    x = np.arange(len(phases_with_data))
    colors = ['#e74c3c' if p == DiscretePhase.P0 else '#2ecc71'
              for p in phases_with_data]

    bars = ax1.bar(x, avg_profits, color=colors, alpha=0.8)
    ax1.set_xlabel('Assigned Phase')
    ax1.set_ylabel('Average Profit per Attack ($)')
    ax1.set_title('Attack Profit by Assigned Phase')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.name for p in phases_with_data])

    # Add value labels
    for bar, profit in zip(bars, avg_profits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${profit:.0f}', ha='center', va='bottom', fontsize=9)

    # Right: Phase distribution
    ax2 = axes[1]
    ax2.bar(x, counts, color='#3498db', alpha=0.8)
    ax2.set_xlabel('Assigned Phase')
    ax2.set_ylabel('Number of Attacks')
    ax2.set_title('Attack Distribution by Phase')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.name for p in phases_with_data])

    plt.suptitle('Phase-Based MEV Defense Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_dashboard(
    result: SimulationResult,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive dashboard with all visualizations.

    Args:
        result: Simulation result
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # 1. Cumulative profit (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    rounds = np.arange(result.n_rounds)
    trad_cum = np.cumsum(result.traditional_profits)
    esm_cum = np.cumsum(result.esm_profits)

    ax1.plot(rounds, trad_cum, color=COLORS['traditional'],
             label='Traditional', linewidth=2)
    ax1.plot(rounds, esm_cum, color=COLORS['esm'],
             label='ESM', linewidth=2)
    ax1.fill_between(rounds, esm_cum, trad_cum, alpha=0.2, color=COLORS['savings'])
    ax1.set_title('Cumulative Attacker Profit', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Profit ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Summary stats (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    stats = result.get_stats()

    labels = ['Traditional', 'ESM']
    values = [stats['traditional_total'], stats['esm_total']]
    colors_bar = [COLORS['traditional'], COLORS['esm']]

    bars = ax2.bar(labels, values, color=colors_bar, alpha=0.8)
    ax2.set_title('Total Attacker Profit', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Profit ($)')

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10)

    # Add reduction annotation
    ax2.text(0.5, max(values) * 0.5,
            f'{stats["profit_reduction_percent"]:.1f}%\nReduction',
            ha='center', fontsize=14, fontweight='bold',
            color=COLORS['savings'])

    # 3. Phase distribution (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    phase_dist = stats['phase_distribution']
    phases = list(phase_dist.keys())
    counts = list(phase_dist.values())

    ax3.pie(counts, labels=phases, autopct='%1.1f%%',
           colors=plt.cm.Set3(np.linspace(0, 1, len(phases))))
    ax3.set_title('Phase Distribution', fontsize=12, fontweight='bold')

    # 4. Timing sensitivity (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    analysis = analyze_timing_sensitivity(delay_range=(0, 1500), n_points=100)
    ax4.plot(analysis['delays_ms'], analysis['interference_factors'],
            color=COLORS['esm'], linewidth=2)
    ax4.fill_between(analysis['delays_ms'], 0, analysis['interference_factors'],
                    alpha=0.3, color=COLORS['esm'])
    ax4.axvline(100, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(500, color='orange', linestyle='--', alpha=0.5)
    ax4.axvline(1000, color='yellow', linestyle='--', alpha=0.5)
    ax4.set_title('Profit Retention vs Timing', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Delay (ms)')
    ax4.set_ylabel('Retention Factor')
    ax4.grid(True, alpha=0.3)

    # 5. Key metrics (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    metrics_text = f"""
    ESM MEV Defense Metrics
    ───────────────────────────

    Simulation Rounds:     {stats['n_rounds']:,}

    Traditional Profit:    ${stats['traditional_total']:,.2f}
    ESM Profit:           ${stats['esm_total']:,.2f}

    Profit Reduction:      {stats['profit_reduction_percent']:.1f}%

    Avg Interference:      {stats['average_interference_factor']:.3f}

    Most Common Phase:     {max(phase_dist, key=phase_dist.get)}
    """

    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('ESM MEV Resistance Dashboard', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
