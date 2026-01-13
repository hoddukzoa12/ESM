"""
ESM v5.4 Visualizations

Additional visualizations for ESM Whitepaper v5.3 documentation quality.
Includes state transitions, application comparisons, tokenomics, and more.

Based on the plan to achieve Ethereum whitepaper-level documentation.
"""

import os
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Try to import optional dependencies
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False


# =============================================================================
# Constants
# =============================================================================

# ESM Color Palette
COLORS = {
    "primary": "#667eea",       # Main blue-purple
    "secondary": "#764ba2",     # Purple
    "success": "#48bb78",       # Green
    "warning": "#ed8936",       # Orange
    "danger": "#f56565",        # Red
    "info": "#4299e1",          # Blue
    "neutral": "#a0aec0",       # Gray
    "dark": "#2d3748",          # Dark gray
    "light": "#f7fafc",         # Light gray
}

# Application names
APPLICATIONS = [
    "MEV DEX",
    "Privacy Transfer",
    "Prediction Market",
    "Insurance",
    "Auction",
    "Quantum NFT",
]

# ESM features for radar chart
ESM_FEATURES = [
    "MEV Resistance",
    "Privacy",
    "Probability",
    "Conditional",
    "Commit-Reveal",
    "Superposition",
]

# Tokenomics distribution (v5.3 Section 7)
TOKEN_DISTRIBUTION = {
    "Validator Rewards": 0.40,
    "Ecosystem Fund": 0.25,
    "Team": 0.15,
    "Public Sale": 0.15,
    "Reserve": 0.05,
}

# Fee distribution
FEE_DISTRIBUTION = {
    "Burn": 0.50,
    "Validators": 0.30,
    "Ecosystem": 0.20,
}

# Limitations (v5.3 Section 10)
LIMITATIONS = {
    "Complexity": {"severity": 0.7, "mitigation": 0.4},
    "Scaling": {"severity": 0.8, "mitigation": 0.3},
    "VDF Central.": {"severity": 0.6, "mitigation": 0.5},
    "Finality": {"severity": 0.5, "mitigation": 0.6},
    "Oracle Risk": {"severity": 0.7, "mitigation": 0.4},
    "Collusion": {"severity": 0.6, "mitigation": 0.5},
}


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_output_dir(path: str = "output") -> str:
    """Ensure output directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_figure(fig: plt.Figure, filename: str, output_dir: str = "output") -> str:
    """Save figure to output directory."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    return filepath


# =============================================================================
# 1. State Transition Flowchart
# =============================================================================

def plot_state_transition_flowchart(
    steps: Optional[List[Dict]] = None,
    save_path: str = "output/state_transition.png",
) -> plt.Figure:
    """
    Block-by-block state transition visualization.

    Creates a flowchart showing how PSC state evolves through blocks,
    similar to Ethereum whitepaper state transition diagrams.

    Args:
        steps: List of step dictionaries with block, actor, action, phase info
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    # Default example steps if none provided
    if steps is None:
        steps = [
            {"block": "N", "actor": "Alice", "action": "Submit TX", "phase": "P0", "psc_state": "Created"},
            {"block": "N+1", "actor": "MEV Bot", "action": "Front-run", "phase": "P180", "psc_state": "2 branches"},
            {"block": "N+50", "actor": "VDF", "action": "Complete", "phase": "-", "psc_state": "Locked"},
            {"block": "N+100", "actor": "Validators", "action": "Collapse", "phase": "-", "psc_state": "Resolved"},
        ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, len(steps) - 0.5 + 1)
    ax.set_ylim(-2, 3)
    ax.axis('off')

    # Title
    ax.text(
        len(steps) / 2, 2.7,
        "ESM State Transition: MEV Attack Scenario",
        ha='center', va='center',
        fontsize=16, fontweight='bold', color=COLORS["dark"]
    )

    # Draw boxes and arrows
    box_width = 0.8
    box_height = 1.2

    for i, step in enumerate(steps):
        x = i
        y = 0.5

        # Box color based on phase
        if step.get("phase") == "P0":
            color = COLORS["success"]
        elif step.get("phase") == "P180":
            color = COLORS["danger"]
        else:
            color = COLORS["info"]

        # Draw box
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=COLORS["dark"],
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)

        # Block label
        ax.text(x, y + 0.4, f"Block {step['block']}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Actor
        ax.text(x, y + 0.1, step['actor'], ha='center', va='center',
                fontsize=9, color='white')

        # Action
        ax.text(x, y - 0.2, step['action'], ha='center', va='center',
                fontsize=8, color='white', style='italic')

        # Phase
        if step.get("phase") and step["phase"] != "-":
            ax.text(x, y - 0.45, f"Phase: {step['phase']}", ha='center', va='center',
                    fontsize=8, color='white')

        # PSC state below
        ax.text(x, y - 1.0, f"PSC: {step['psc_state']}", ha='center', va='center',
                fontsize=8, color=COLORS["dark"],
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS["neutral"]))

        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate(
                '', xy=(i + 0.55, y), xytext=(i + box_width/2 + 0.05, y),
                arrowprops=dict(arrowstyle='->', color=COLORS["dark"], lw=2)
            )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["success"], label='Normal (P0)'),
        mpatches.Patch(facecolor=COLORS["danger"], label='Counter (P180)'),
        mpatches.Patch(facecolor=COLORS["info"], label='System'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Footer
    ax.text(
        len(steps) / 2, -1.7,
        "P180 causes destructive interference → MEV attack cancelled",
        ha='center', va='center', fontsize=10, color=COLORS["dark"], style='italic'
    )

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 2. Application Comparison Radar
# =============================================================================

def plot_application_radar(
    save_path: str = "output/application_radar.png",
) -> plt.Figure:
    """
    Radar chart comparing ESM feature usage across 6 applications.

    Shows how each application leverages different ESM capabilities:
    MEV Resistance, Privacy, Probability, Conditional States, Commit-Reveal, Superposition

    Args:
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    # Feature scores for each application (0-1 scale)
    # [MEV, Privacy, Probability, Conditional, Commit-Reveal, Superposition]
    app_scores = {
        "MEV DEX": [1.0, 0.3, 0.5, 0.4, 0.2, 0.6],
        "Privacy Transfer": [0.2, 1.0, 0.7, 0.3, 0.1, 0.9],
        "Prediction Market": [0.3, 0.2, 1.0, 0.5, 0.3, 0.7],
        "Insurance": [0.2, 0.3, 0.8, 1.0, 0.4, 0.7],
        "Auction": [0.4, 0.5, 0.3, 0.4, 1.0, 0.5],
        "Quantum NFT": [0.1, 0.2, 0.6, 0.3, 0.2, 1.0],
    }

    # Number of features
    n_features = len(ESM_FEATURES)
    angles = [n / float(n_features) * 2 * np.pi for n in range(n_features)]
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Colors for each application
    app_colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["danger"],
        COLORS["info"],
    ]

    # Plot each application
    for idx, (app_name, scores) in enumerate(app_scores.items()):
        values = scores + scores[:1]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, label=app_name,
                color=app_colors[idx], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=app_colors[idx])

    # Set feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ESM_FEATURES, size=10, fontweight='bold')

    # Set y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color=COLORS["neutral"])

    # Title
    ax.set_title(
        "ESM Feature Usage by Application",
        size=14, fontweight='bold', color=COLORS["dark"], pad=20
    )

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 3. Token Flow Sankey Diagram
# =============================================================================

def plot_token_flow_sankey(
    save_path: str = "output/token_flow.png",
) -> plt.Figure:
    """
    Token flow visualization for v5.3 Section 7 tokenomics.

    Shows initial supply distribution and fee flow mechanisms.

    Args:
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # === Left: Initial Distribution ===
    ax1.set_title("Initial Token Distribution\n(1 Billion ESM)", fontsize=12, fontweight='bold')

    # Pie chart for distribution
    sizes = list(TOKEN_DISTRIBUTION.values())
    labels = [f"{k}\n({v*100:.0f}%)" for k, v in TOKEN_DISTRIBUTION.items()]
    colors = [COLORS["primary"], COLORS["success"], COLORS["warning"],
              COLORS["info"], COLORS["neutral"]]

    wedges, texts = ax1.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, wedgeprops=dict(width=0.5, edgecolor='white')
    )

    # Make labels bold
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')

    ax1.axis('equal')

    # === Right: Fee Flow ===
    ax2.set_title("Transaction Fee Flow", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Draw flow diagram
    # Source box (TX Fee)
    source_box = FancyBboxPatch(
        (0.5, 4), 2, 2,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["primary"],
        edgecolor=COLORS["dark"],
        linewidth=2
    )
    ax2.add_patch(source_box)
    ax2.text(1.5, 5, "TX Fee\n100%", ha='center', va='center',
             fontsize=10, fontweight='bold', color='white')

    # Destination boxes
    destinations = [
        ("Burn", 0.5, COLORS["danger"], 8),
        ("Validators", 0.3, COLORS["success"], 5),
        ("Ecosystem", 0.2, COLORS["info"], 2),
    ]

    for name, pct, color, y_pos in destinations:
        # Box
        dest_box = FancyBboxPatch(
            (7, y_pos - 0.75), 2.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=COLORS["dark"],
            linewidth=2
        )
        ax2.add_patch(dest_box)
        ax2.text(8.25, y_pos, f"{name}\n{pct*100:.0f}%", ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')

        # Arrow with width proportional to percentage
        ax2.annotate(
            '', xy=(7, y_pos), xytext=(2.5, 5),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=1 + pct * 5, alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            )
        )

    # Description
    ax2.text(5, 0.5, "Burn mechanism creates deflationary pressure",
             ha='center', va='center', fontsize=9, style='italic', color=COLORS["dark"])

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 4. Limitations Analysis
# =============================================================================

def plot_limitations_analysis(
    save_path: str = "output/limitations.png",
) -> plt.Figure:
    """
    Visualization of v5.3 Section 10 limitations.

    Shows severity and mitigation progress for each limitation.

    Args:
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    limitations = list(LIMITATIONS.keys())
    n = len(limitations)
    x = np.arange(n)
    width = 0.35

    severities = [LIMITATIONS[l]["severity"] for l in limitations]
    mitigations = [LIMITATIONS[l]["mitigation"] for l in limitations]

    # Bars
    bars1 = ax.bar(x - width/2, severities, width, label='Severity',
                   color=COLORS["danger"], alpha=0.8)
    bars2 = ax.bar(x + width/2, mitigations, width, label='Mitigation Progress',
                   color=COLORS["success"], alpha=0.8)

    # Labels
    ax.set_ylabel('Score (0-1)', fontweight='bold')
    ax.set_title('ESM v5.3 Limitations Analysis\n(Section 10)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(limitations, rotation=15, ha='right')
    ax.legend()

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)

    # Horizontal reference line
    ax.axhline(y=0.5, color=COLORS["neutral"], linestyle=':', alpha=0.5)
    ax.text(n - 0.5, 0.52, 'Medium threshold', fontsize=8, color=COLORS["neutral"])

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 5. Walkthrough Timeline
# =============================================================================

def plot_walkthrough_timeline(
    walkthrough_data: Optional[Dict] = None,
    save_path: str = "output/walkthrough_timeline.png",
) -> plt.Figure:
    """
    Timeline diagram for Alice→TOKEN swap walkthrough.

    Shows chronological sequence of events in a transaction.

    Args:
        walkthrough_data: Walkthrough result data
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    # Default timeline events
    events = [
        {"time": "T+0ms", "event": "Alice Submit", "type": "tx", "details": "100 ESM swap"},
        {"time": "T+50ms", "event": "Bot Attack", "type": "attack", "details": "1000 ESM front-run"},
        {"time": "T+100ms", "event": "Phase Assign", "type": "system", "details": "Bot→P180"},
        {"time": "T+5s", "event": "VDF Start", "type": "system", "details": "Seed generation"},
        {"time": "T+60s", "event": "VDF Complete", "type": "system", "details": "Locked state"},
        {"time": "T+65s", "event": "Threshold", "type": "system", "details": "67% reveals"},
        {"time": "T+70s", "event": "Collapse", "type": "result", "details": "Alice wins!"},
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, len(events) - 0.5 + 0.5)
    ax.set_ylim(-1.5, 2)
    ax.axis('off')

    # Title
    ax.text(
        len(events) / 2, 1.7,
        "Transaction Timeline: Alice Sends 100 ESM",
        ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLORS["dark"]
    )

    # Timeline base line
    ax.plot([0, len(events) - 1], [0, 0], 'k-', lw=3, alpha=0.3)

    # Event markers
    type_colors = {
        "tx": COLORS["success"],
        "attack": COLORS["danger"],
        "system": COLORS["info"],
        "result": COLORS["primary"],
    }

    for i, event in enumerate(events):
        color = type_colors.get(event["type"], COLORS["neutral"])

        # Marker
        circle = Circle((i, 0), 0.15, color=color, zorder=10)
        ax.add_patch(circle)

        # Time label (below)
        ax.text(i, -0.4, event["time"], ha='center', va='top',
                fontsize=8, color=COLORS["dark"])

        # Event label (above, alternating)
        y_offset = 0.8 if i % 2 == 0 else 0.5
        ax.text(i, y_offset, event["event"], ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)

        # Details
        ax.text(i, y_offset - 0.25, event["details"], ha='center', va='top',
                fontsize=7, style='italic', color=COLORS["dark"])

        # Vertical line to event
        ax.plot([i, i], [0.15, y_offset - 0.35], '-', color=color, lw=1, alpha=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["success"],
               label='Transaction', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["danger"],
               label='Attack', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["info"],
               label='System', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["primary"],
               label='Result', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Footer
    ax.text(
        len(events) / 2, -1.2,
        "MEV attack blocked through phase-based interference mechanism",
        ha='center', va='center', fontsize=9, color=COLORS["dark"], style='italic'
    )

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 6. Validator Economics
# =============================================================================

def plot_validator_economics(
    save_path: str = "output/validator_economics.png",
) -> plt.Figure:
    """
    Validator economics visualization for v5.3 Section 7.6.

    Shows staking APY, slashing scenarios, and reward distribution.

    Args:
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === Panel 1: Staking APY Curve ===
    ax1 = axes[0]
    ax1.set_title("Staking APY vs. Total Staked", fontsize=11, fontweight='bold')

    # APY decreases as more is staked
    total_staked = np.linspace(0.1, 0.8, 100)  # 10% to 80% of supply
    # APY formula: base_rate / sqrt(staked_ratio)
    base_apy = 0.08
    apy = base_apy / np.sqrt(total_staked) * 0.5

    ax1.plot(total_staked * 100, apy * 100, color=COLORS["primary"], lw=2)
    ax1.fill_between(total_staked * 100, 0, apy * 100, alpha=0.2, color=COLORS["primary"])

    # Highlight current range
    ax1.axvspan(30, 50, alpha=0.1, color=COLORS["success"])
    ax1.text(40, 6, "Target\nRange", ha='center', va='center', fontsize=8, color=COLORS["success"])

    ax1.set_xlabel("Total Staked (%)", fontweight='bold')
    ax1.set_ylabel("APY (%)", fontweight='bold')
    ax1.set_xlim(10, 80)
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3)

    # === Panel 2: Slashing Scenarios ===
    ax2 = axes[1]
    ax2.set_title("Slashing Penalties", fontsize=11, fontweight='bold')

    scenarios = ["Downtime\n(<24h)", "Downtime\n(>24h)", "Double\nSign", "Data\nWithhold"]
    penalties = [0.1, 1.0, 5.0, 10.0]
    colors = [COLORS["warning"], COLORS["warning"], COLORS["danger"], COLORS["danger"]]

    bars = ax2.bar(scenarios, penalties, color=colors, alpha=0.8, edgecolor=COLORS["dark"])

    for bar, penalty in zip(bars, penalties):
        height = bar.get_height()
        ax2.annotate(f'{penalty}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel("Stake Slashed (%)", fontweight='bold')
    ax2.set_ylim(0, 12)

    # === Panel 3: Reward Distribution ===
    ax3 = axes[2]
    ax3.set_title("Validator Reward Sources", fontsize=11, fontweight='bold')

    sources = ["Block\nRewards", "TX Fees", "MEV\nProtection"]
    percentages = [50, 30, 20]
    colors = [COLORS["primary"], COLORS["success"], COLORS["info"]]

    wedges, texts, autotexts = ax3.pie(
        percentages, labels=sources, autopct='%1.0f%%',
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )

    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax3.axis('equal')

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# 7. Interference Pattern 3D
# =============================================================================

def plot_interference_3d(
    save_path: str = "output/interference_3d.png",
) -> plt.Figure:
    """
    3D visualization of 8-phase interference patterns.

    Shows interference results for all phase combinations.

    Args:
        save_path: Path to save the figure

    Returns:
        matplotlib Figure
    """
    if not HAS_3D:
        # Fallback to 2D heatmap if 3D not available
        return _plot_interference_2d(save_path)

    # 8 discrete phases
    phases = [0, 45, 90, 135, 180, 225, 270, 315]
    n = len(phases)

    # Create meshgrid
    X, Y = np.meshgrid(range(n), range(n))

    # Calculate interference values
    # |A1 + A2|² where A1 = e^(i*θ1), A2 = e^(i*θ2)
    Z = np.zeros((n, n))
    for i, p1 in enumerate(phases):
        for j, p2 in enumerate(phases):
            theta1 = np.radians(p1)
            theta2 = np.radians(p2)
            # Complex amplitudes
            a1 = np.exp(1j * theta1)
            a2 = np.exp(1j * theta2)
            # Interference magnitude squared
            Z[i, j] = np.abs(a1 + a2) ** 2

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='RdYlGn',
        alpha=0.8,
        edgecolor='none'
    )

    # Labels
    ax.set_xlabel('Phase 1 (degrees)', fontweight='bold')
    ax.set_ylabel('Phase 2 (degrees)', fontweight='bold')
    ax.set_zlabel('Interference |A1+A2|²', fontweight='bold')

    # Custom tick labels
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'P{p}' for p in phases], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'P{p}' for p in phases], fontsize=8)

    # Title
    ax.set_title('ESM 8-Phase Interference Pattern', fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Probability')

    # Add annotations for key points
    # Max (constructive): same phase
    ax.scatter([0], [0], [4], color=COLORS["success"], s=100, marker='o')
    ax.text(0, 0, 4.2, 'Max (Constructive)', fontsize=8)

    # Min (destructive): opposite phase (P0 vs P180 = index 0 vs 4)
    ax.scatter([0], [4], [0], color=COLORS["danger"], s=100, marker='o')
    ax.text(0, 4, 0.3, 'Min (Destructive)', fontsize=8)

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


def _plot_interference_2d(save_path: str) -> plt.Figure:
    """2D fallback for interference visualization."""
    phases = [0, 45, 90, 135, 180, 225, 270, 315]
    n = len(phases)

    # Calculate interference values
    Z = np.zeros((n, n))
    for i, p1 in enumerate(phases):
        for j, p2 in enumerate(phases):
            theta1 = np.radians(p1)
            theta2 = np.radians(p2)
            a1 = np.exp(1j * theta1)
            a2 = np.exp(1j * theta2)
            Z[i, j] = np.abs(a1 + a2) ** 2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(Z, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'P{p}' for p in phases])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'P{p}' for p in phases])
    ax.set_xlabel('Phase 2', fontweight='bold')
    ax.set_ylabel('Phase 1', fontweight='bold')
    ax.set_title('ESM 8-Phase Interference Pattern (Heatmap)', fontsize=14, fontweight='bold')

    # Colorbar
    plt.colorbar(im, ax=ax, label='Interference |A1+A2|²')

    # Add value annotations
    for i in range(n):
        for j in range(n):
            text_color = 'white' if Z[i, j] < 2 else 'black'
            ax.text(j, i, f'{Z[i, j]:.1f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    plt.tight_layout()
    save_figure(fig, os.path.basename(save_path), os.path.dirname(save_path) or "output")
    return fig


# =============================================================================
# Generate All Visualizations
# =============================================================================

def generate_all_v54_visualizations(output_dir: str = "output") -> Dict[str, str]:
    """
    Generate all v5.4 visualizations.

    Args:
        output_dir: Output directory for images

    Returns:
        Dictionary mapping visualization name to file path
    """
    ensure_output_dir(output_dir)

    results = {}

    print("Generating v5.4 visualizations...")

    # 1. State Transition
    print("  [1/7] State Transition Flowchart...")
    fig = plot_state_transition_flowchart(save_path=f"{output_dir}/state_transition.png")
    plt.close(fig)
    results["state_transition"] = f"{output_dir}/state_transition.png"

    # 2. Application Radar
    print("  [2/7] Application Radar...")
    fig = plot_application_radar(save_path=f"{output_dir}/application_radar.png")
    plt.close(fig)
    results["application_radar"] = f"{output_dir}/application_radar.png"

    # 3. Token Flow
    print("  [3/7] Token Flow...")
    fig = plot_token_flow_sankey(save_path=f"{output_dir}/token_flow.png")
    plt.close(fig)
    results["token_flow"] = f"{output_dir}/token_flow.png"

    # 4. Limitations
    print("  [4/7] Limitations Analysis...")
    fig = plot_limitations_analysis(save_path=f"{output_dir}/limitations.png")
    plt.close(fig)
    results["limitations"] = f"{output_dir}/limitations.png"

    # 5. Walkthrough Timeline
    print("  [5/7] Walkthrough Timeline...")
    fig = plot_walkthrough_timeline(save_path=f"{output_dir}/walkthrough_timeline.png")
    plt.close(fig)
    results["walkthrough_timeline"] = f"{output_dir}/walkthrough_timeline.png"

    # 6. Validator Economics
    print("  [6/7] Validator Economics...")
    fig = plot_validator_economics(save_path=f"{output_dir}/validator_economics.png")
    plt.close(fig)
    results["validator_economics"] = f"{output_dir}/validator_economics.png"

    # 7. Interference 3D
    print("  [7/7] Interference 3D...")
    fig = plot_interference_3d(save_path=f"{output_dir}/interference_3d.png")
    plt.close(fig)
    results["interference_3d"] = f"{output_dir}/interference_3d.png"

    print(f"Done! Generated {len(results)} visualizations in {output_dir}/")

    return results


def run_demo():
    """Run visualization demo."""
    print("=" * 60)
    print("ESM v5.4 Visualization Demo")
    print("=" * 60)
    print()

    results = generate_all_v54_visualizations()

    print()
    print("Generated files:")
    for name, path in results.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    run_demo()
