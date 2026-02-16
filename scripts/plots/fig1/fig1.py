# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pillow"]
# ///
"""
Figure 1: Domain mixing workflow diagram.

A timeline showing the evolution of domain mixing with stages:
Initial mix -> Add -> Transform -> Remove -> Partition -> Start training LM

Usage:
    uv run fig1.py [--icon PATH_TO_ICON.png] [--output OUTPUT.png] [--show]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

MANROPE_BASE_PATH = (Path(__file__).parent / "manrope").absolute()
for path in MANROPE_BASE_PATH.iterdir():
    if path.suffix == ".ttf":
        fontManager.addfont(str(path))


plt.rcParams["font.family"] = "Manrope"


# import matplotlib.font_manager as fm

# fm.fontManager.addfont(
#   "/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf"
# )

import matplotlib.pyplot as plt

# plt.rcParams['font.family'] = 'Comic Sans MS'


# =============================================================================
# Colors (approximate from reference image)
# =============================================================================
BLUE_BLOCK = "#aed6f1"
ORANGE_BLOCK = "#f8c471"
GREEN_BLOCK = "#abebc6"
PINK_BLOCK = "#CF56AD"

RED_TEXT = "#d9534f"
BLUE_TEXT = "#337ab7"
BLACK = "#2c3e50"


BACKGROUND = "#FAF2E9"


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class TimelineStage:
    """Represents a stage on the timeline."""

    tick_label: str
    tick_label_color: str
    header: str | None = None
    content_fn: Callable[[plt.Axes, float, float], None] | None = None


# =============================================================================
# Drawing primitives
# =============================================================================
def draw_block(
    ax: plt.Axes,
    x: float,
    y: float,
    color: str,
    width: float = 1.0,
    height: float = 1.2,
) -> float:
    """Draw a single domain block. Returns the right edge x position."""
    rect = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=BLACK,
        linewidth=1.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(rect)
    return x + width


def draw_blocks(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    colors: list[str],
    width: float = 1.0,
    height: float = 1.2,
    gap: float = 0.15,
) -> None:
    """Draw multiple domain blocks centered at (center_x, center_y)."""
    n = len(colors)
    total_width = n * width + (n - 1) * gap
    start_x = center_x - total_width / 2
    for i, color in enumerate(colors):
        draw_block(ax, start_x + i * (width + gap), center_y, color, width, height)


def draw_small_arrow(ax: plt.Axes, x1: float, x2: float, y: float) -> None:
    """Draw a small horizontal arrow between blocks."""
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.5,
        color=BLACK,
    )
    ax.add_patch(arrow)


def draw_plus(ax: plt.Axes, x: float, y: float, size: float = 0.4) -> None:
    """Draw a plus sign."""
    ax.plot([x - size, x + size], [y, y], color=BLACK, lw=2, solid_capstyle="round")
    ax.plot([x, x], [y - size, y + size], color=BLACK, lw=2, solid_capstyle="round")


def draw_minus(ax: plt.Axes, x: float, y: float, size: float = 0.4) -> None:
    """Draw a minus sign."""
    ax.plot([x - size, x + size], [y, y], color=BLACK, lw=2, solid_capstyle="round")


def draw_circled_number(ax: plt.Axes, x: float, y: float, number: int, color: str) -> None:
    """Draw a circled number."""
    circle = Circle((x, y), 0.5, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, str(number), ha="center", va="center", fontsize=12, color=color)


def place_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    icon_path: Path | None,
    zoom: float = 0.15,
) -> None:
    """Place an icon image, or a placeholder if not available."""
    if icon_path and icon_path.exists():
        img = Image.open(icon_path)
        oi = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(oi, (x, y), frameon=False)
        ax.add_artist(ab)
    else:
        # Placeholder dashed box
        box = FancyBboxPatch(
            (x - 1.2, y - 1.2),
            2.4,
            2.4,
            boxstyle="round,pad=0.2",
            fill=False,
            edgecolor=BLACK,
            linestyle="--",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y, "ICON", ha="center", va="center", fontsize=9, color=BLACK)


# =============================================================================
# Stage content functions (what to draw above each tick)
# =============================================================================
def content_initial(ax: plt.Axes, x: float, y: float) -> None:
    """Initial mix: two blocks (pink + blue)."""
    draw_blocks(ax, x, y, [PINK_BLOCK, BLUE_BLOCK])


def content_add(ax: plt.Axes, x: float, y: float) -> None:
    """Add: 4 blocks (pink + blue + 2 orange)."""
    draw_blocks(ax, x, y, [PINK_BLOCK, BLUE_BLOCK, ORANGE_BLOCK, ORANGE_BLOCK])


def content_transform(ax: plt.Axes, x: float, y: float) -> None:
    """Transform: 4 blocks (pink + blue + 1 orange + 1 green) - one orange became green."""
    draw_blocks(ax, x, y, [PINK_BLOCK, BLUE_BLOCK, ORANGE_BLOCK, GREEN_BLOCK])


def content_remove(ax: plt.Axes, x: float, y: float) -> None:
    """Remove: 3 blocks (1 blue + 1 orange + 1 green) - removed one blue."""
    draw_blocks(ax, x, y, [BLUE_BLOCK, ORANGE_BLOCK, GREEN_BLOCK])


def draw_block_with_pattern(
    ax: plt.Axes,
    x: float,
    y: float,
    color: str,
    width: float,
    height: float,
    hatch: str | None = None,
    hatch_color: str = "#d0d0d0",  # Light gray for subtle pattern
) -> None:
    """Draw a block with optional hatch pattern."""
    import matplotlib as mpl

    # Temporarily set hatch color to a lighter shade
    old_hatch_color = mpl.rcParams.get("hatch.color", "black")
    mpl.rcParams["hatch.color"] = hatch_color

    rect = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=BLACK,
        linewidth=1.5,
        hatch=hatch,
    )
    ax.add_patch(rect)

    # Restore original hatch color
    mpl.rcParams["hatch.color"] = old_hatch_color


def content_partition(ax: plt.Axes, x: float, y: float) -> None:
    """Partition: 4 blocks - Blue, Orange (hatched), Orange (dotted), Green."""
    width = 1.0
    height = 1.2
    gap = 0.15
    thin_width = 0.45
    thin_gap = 0.1

    # Total width: blue + gap + thin_orange1 + thin_gap + thin_orange2 + gap + green
    total_width = width + gap + thin_width + thin_gap + thin_width + gap + width
    start_x = x - total_width / 2

    # Blue block
    draw_block(ax, start_x, y, BLUE_BLOCK, width, height)

    # Two thin orange blocks with dense patterns (fine hatching and dots)
    thin_start = start_x + width + gap
    # First thin orange with dense diagonal hatching
    draw_block_with_pattern(ax, thin_start, y, ORANGE_BLOCK, thin_width, height, hatch="//////")
    # Second thin orange with dense dots
    draw_block_with_pattern(ax, thin_start + thin_width + thin_gap, y, ORANGE_BLOCK, thin_width, height, hatch="......")

    # Green block on the right
    green_x = thin_start + thin_width + thin_gap + thin_width + gap
    draw_block(ax, green_x, y, GREEN_BLOCK, width, height)


# =============================================================================
# Main figure construction
# =============================================================================
def build_figure(left_icon_path: Path | None, right_icon_path: Path | None) -> plt.Figure:
    """Build the complete figure."""
    # Hand-drawn sketch style
    plt.xkcd(scale=0.5, length=100, randomness=2)
    plt.rcParams["font.family"] = ["Manrope", "DejaVu Sans"]

    fig, ax = plt.subplots(
        figsize=(20, 2.8),
        dpi=1000,
        facecolor=BACKGROUND,  # figure background
    )
    ax.set_facecolor(BACKGROUND)  # axes background

    ax.set_xlim(0, 42)
    ax.set_ylim(-0.8, 7.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Layout parameters (tighter vertical spacing)
    timeline_y = 2.5
    content_y = 4.0
    header_y = 4.85  # Even closer to content_y to reduce gap between operators and boxes
    tick_label_y = 1.5

    # Define the stages
    stages: list[TimelineStage] = [
        TimelineStage(
            tick_label=r"Initial mix" + "\n" + r"$\rightarrow P_0$",
            tick_label_color=RED_TEXT,
            header=None,  # Header is "Configure mixing method" shown separately
            content_fn=content_initial,
        ),
        TimelineStage(
            tick_label=r"Recompute mix" + "\n" + r"$\rightarrow P_1$",
            tick_label_color=BLUE_TEXT,
            header="Add",
            content_fn=content_add,
        ),
        TimelineStage(
            tick_label=r"Recompute mix" + "\n" + r"$\rightarrow P_2$",
            tick_label_color=BLUE_TEXT,
            header="Revise",
            content_fn=content_transform,
        ),
        TimelineStage(
            tick_label=r"Recompute mix" + "\n" + r"$\rightarrow P_3$",
            tick_label_color=BLUE_TEXT,
            header="Remove",
            content_fn=content_remove,
        ),
        TimelineStage(
            tick_label=r"Recompute mix" + "\n" + r"$\rightarrow P_{\mathrm{final}}$",
            tick_label_color=BLUE_TEXT,
            header="Partition",
            content_fn=content_partition,
        ),
    ]

    # Evenly space the ticks
    n_stages = len(stages)
    timeline_start = 8.0
    timeline_end = 32.0
    tick_spacing = (timeline_end - timeline_start) / (n_stages - 1)
    tick_positions = [timeline_start + i * tick_spacing for i in range(n_stages)]

    # Calculate box dimensions (used for positioning)
    box_left = tick_positions[0] - 2.5
    box_right = tick_positions[-1] + 3.5
    box_bottom = tick_label_y - 1.5  # Lower to not clip P0, P1, P2... text
    box_top = header_y + 0.65

    # Draw main timeline arrow (from left icon area to right icon area)
    arrow = FancyArrowPatch(
        (box_left - 1.5, timeline_y),
        (box_right + 1.5, timeline_y),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(arrow)

    # Draw each stage
    for stage, tx in zip(stages, tick_positions):
        # Tick mark
        ax.plot(
            [tx, tx],
            [timeline_y - 0.4, timeline_y + 0.4],
            color=BLACK,
            lw=2,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )

        # Tick label (below timeline)
        ax.text(
            tx,
            tick_label_y,
            stage.tick_label,
            ha="center",
            va="top",
            fontsize=11,
            color=stage.tick_label_color,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )

        # Header (above content)
        if stage.header:
            ax.text(
                tx,
                header_y,
                stage.header,
                ha="center",
                va="bottom",
                fontsize=13,
                color=BLACK,
                path_effects=[
                    pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                    pe.Normal(),
                ],
            )

        # Content (blocks, arrows, etc.)
        if stage.content_fn:
            stage.content_fn(ax, tx, content_y)

    # ==========================================================================
    # Dotted box around timeline content (Initial mix through last Recompute mix)
    # ==========================================================================
    dotted_box = FancyBboxPatch(
        (box_left, box_bottom),
        box_right - box_left,
        box_top - box_bottom,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        fill=False,
        edgecolor=BLUE_TEXT,
        linestyle="--",
        linewidth=1.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(dotted_box)

    # ==========================================================================
    # Left side: Icon + "Configure mixing method" (mirrors the right side)
    # ==========================================================================
    left_icon_x = box_left - 2.5

    # Icon centered vertically with timeline (same as right side)
    place_icon(ax, left_icon_x, timeline_y, left_icon_path, zoom=0.4)

    # Label below, aligned with tick labels
    ax.text(
        left_icon_x,
        tick_label_y,
        "(1) Configure\nmixing method",
        ha="center",
        va="top",
        fontsize=11,
        color=RED_TEXT,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # ==========================================================================
    # Right side: "Start training LM" + icon (mirrors left side spacing)
    # ==========================================================================
    # Icon centered vertically with timeline, symmetric distance from box
    right_icon_x = box_right + 2.5
    place_icon(ax, right_icon_x, timeline_y, right_icon_path, zoom=0.4)

    # Label below timeline, aligned with tick labels
    ax.text(
        right_icon_x,
        tick_label_y,
        r"Start training" + "\n" + r"LM with $P_{\mathrm{final}}$",
        ha="center",
        va="top",
        fontsize=11,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # ==========================================================================
    # Bottom: Legend and subtitle
    # ==========================================================================
    # Legend box (upper left, above the left icon)
    legend_x = 1.0
    legend_y = header_y - 0.3
    rect = Rectangle(
        (legend_x - 0.4, legend_y - 0.5),
        0.8,
        1.0,
        fill=False,
        edgecolor=BLACK,
        linewidth=1.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(rect)
    ax.text(
        legend_x + 0.8,
        legend_y,
        "= domain",
        ha="left",
        va="center",
        fontsize=11,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # Subtitle at bottom (below the dotted box)
    ax.text(
        (timeline_start + timeline_end) / 2,
        -0.4,
        "(2) Mixing throughout iterative LM development",
        ha="center",
        va="center",
        fontsize=11,
        color=BLUE_TEXT,
        bbox=dict(
            facecolor=BACKGROUND,
            edgecolor=BACKGROUND,
            pad=2,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        ),
        # This is the key: paint over the halo with the block color
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    return fig


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1: Domain mixing workflow")
    parser.add_argument(
        "--left-icon",
        type=Path,
        default=Path("icons8-configuration-100.png"),
        help="Path to left icon PNG (transparent background)",
    )
    parser.add_argument(
        "--right-icon",
        type=Path,
        default=Path("icons8-rocket-100.png"),
        help="Path to right icon PNG (transparent background)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fig1_output_beige.png"),
        help="Output filename",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )
    args = parser.parse_args()

    fig = build_figure(args.left_icon, args.right_icon)
    # Use transparent background for PDF, white for other formats
    if str(args.output).endswith(".pdf"):
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    else:
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
