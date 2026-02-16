# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Figure: Mixture reuse spectrum diagram (hand-drawn XKCD style)

Usage:
    uv run fig_mixreuse.py --output fig_mixreuse.pdf
    uv run fig_mixreuse.py --output fig_mixreuse.png --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import fontManager

import matplotlib.patheffects as pe

# -----------------------------
# Font (optional: matches your style)
# -----------------------------
MANROPE_BASE_PATH = (Path(__file__).parent / "manrope").absolute()
if MANROPE_BASE_PATH.exists():
    for path in MANROPE_BASE_PATH.iterdir():
        if path.suffix == ".ttf":
            fontManager.addfont(str(path))
plt.rcParams["font.family"] = "Manrope"

# -----------------------------
# Colors
# -----------------------------
BLACK = "#2c3e50"
BLUE = "#aed6f1"   # unaffected domains
GREEN= "#abebc6"
ORANGE = "#f8c471"  # affected domains
BACKGROUND="#FAF2E9"


# -----------------------------
# Drawing primitives
# -----------------------------
def rounded_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str = BLACK,
    lw: float = 1.8,
    text: str | None = None,
    fontsize: int = 14,
    bold: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        mutation_aspect=1.0,
        antialiased=False,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(patch)
    if text is not None:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=BLACK,
            fontweight="bold" if bold else "normal",
            # This is the key: paint over the halo with the block color
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=facecolor),
                pe.Normal(),
            ],
        )

def draw_domain_row(
    ax: plt.Axes,
    x0: float,
    y0: float,
    labels: list[str],
    colors: list[str],
    block_w: float = 1.50,
    block_h: float = 1.50,
    gap: float = 0.10,
    lw: float = 1.8,
    thick_outline_idx: set[int] | None = None,
) -> tuple[float, float]:
    """
    Draw a row of labeled blocks starting at (x0, y0) (bottom-left of first block).
    Returns (x_left, x_right) of the whole group.
    """
    thick_outline_idx = thick_outline_idx or set()
    x = x0
    for i, (lab, col) in enumerate(zip(labels, colors)):
        this_lw = 3.2 if i in thick_outline_idx else lw
        rounded_block(ax, x, y0, block_w, block_h, col, lw=this_lw, text=lab, fontsize=14)
        x += block_w + gap
    x_left = x0
    x_right = x - gap
    return x_left, x_right


def connector(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, lw: float = 1.6) -> None:
    ax.plot([x1, x2], [y1, y2], color=BLACK, lw=lw, solid_capstyle="round", path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def arrow(ax, x1, y1, x2, y2, ms=18, lw=2.0, two_way=False):
    style = "<->" if two_way else "-|>"
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=BLACK,
            path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
        )
    )



def draw_olmixbase_call(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    labels: list[str],
    colors: list[str],
    thick_outline_idx: set[int] | None = None,
    title_left: str = "Olmix\nBase",
    block_w: float = 1.50,
    block_h: float = 1.50,
    gap: float = 0.10,
) -> tuple[float, float, float, float]:
    """
    Draw:  OlmixBase( [blocks...] )
    Centered at (center_x, center_y) with text + parentheses.
    Returns (x_left, x_right, y_bottom, y_top) of the blocks (not including text).
    """
    n = len(labels)
    total_w = n * block_w + (n - 1) * gap
    x_blocks_left = center_x - total_w / 2
    y_blocks_bottom = center_y - block_h / 2

    # left label
    ax.text(center_x - total_w / 2 - 3.5, center_y, title_left, ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # left parenthesis
    ax.text(x_blocks_left - 0.55, center_y, "(", ha="center", va="center", fontsize=32, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # blocks
    x_left, x_right = draw_domain_row(
        ax,
        x_blocks_left,
        y_blocks_bottom,
        labels,
        colors,
        block_w=block_w,
        block_h=block_h,
        gap=gap,
        thick_outline_idx=thick_outline_idx,
    )

    # right parenthesis
    ax.text(x_right + 0.55, center_y, ")", ha="center", va="center", fontsize=32, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    return x_left, x_right, y_blocks_bottom, y_blocks_bottom + block_h


def legend_item(ax: plt.Axes, x: float, y: float, label: str, facecolor: str | None, outline: bool = False) -> None:
    w, h = 1.0, 1.0
    if outline:
        rounded_block(ax, x, y - h / 2, w, h, facecolor="white", lw=3.0, text=r"$\mathcal{D}_{\mathrm{fix}}$", fontsize=8)
    else:
        rounded_block(ax, x, y - h / 2, w, h, facecolor=facecolor or "white", lw=1.6)
    ax.text(x + w + 0.45, y, label, ha="left", va="center", fontsize=12, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


# -----------------------------
# Figure
# -----------------------------
def build_figure() -> plt.Figure:
    plt.xkcd(scale=0.5, length=100, randomness=2)
    plt.rcParams["font.family"] = ["Manrope", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(12.5, 6.3), dpi=1000, facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal", adjustable="box")  # <- key line
    ax.axis("off")


    # -----------------------------
    # Layout knobs (tune these)
    # -----------------------------
    # Column centers
    x_left, x_mid, x_right = 6.5, 20, 32.0

    # Y levels
    y_top_titles = 15.0
    y_top_blocks = 13.2
    y_mid_titles = 9.7
    y_mid_blocks = 7.5
    y_bottom = 1.7
    # -----------------------------
    # Top row: Before / After update
    # -----------------------------
    ax.text(x_left- 0.5, y_top_titles, r"Before ($\mathcal{D}$)", ha="center", va="center", fontsize=20, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(x_mid - 0.5, y_top_titles, r"After ($\mathcal{D}'$)", ha="center", va="center", fontsize=20, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    ax.text((x_mid + x_left)/2 , y_top_titles-1.0, "'Partition' update", ha="center", va="center", fontsize=14, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    arrow(ax, x_left+3.0, y_top_titles-1.5, x_mid-3.3, y_top_titles-1.5, ms=18, lw=1)


    # before blocks: D1 D2 D3 (unaffected)
    draw_domain_row(
        ax,
        x_left - 3.0,
        y_top_blocks - 0.45,
        labels=[r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4$"],
        colors=[BLUE, BLUE, BLUE, GREEN],
    )

    # after blocks: D1 D2 D3 unaffected, D4' affected
    draw_domain_row(
        ax,
        x_mid - 3.2,
        y_top_blocks - 0.45,
        labels=[r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4'$", r"$D_5'$"],
        colors=[BLUE, BLUE, BLUE, ORANGE, ORANGE],
    )

    # -----------------------------
    # Legend (top-right)
    # -----------------------------
    legend_x = 30.0
    legend_y0 = 15.0
    legend_item(ax, legend_x, legend_y0, r"unaffected domains", facecolor=BLUE)
    legend_item(ax, legend_x, legend_y0 - 1.2, r"affected domains (old)", facecolor=GREEN)
    legend_item(ax, legend_x, legend_y0 - 2.4, r"affected domains (new)", facecolor=ORANGE)
    legend_item(ax, legend_x, legend_y0 - 3.6, r"reuse ratios within $\mathcal{D}_{\mathrm{fix}}$", facecolor=None, outline=True)

    # -----------------------------
    # Middle row: three strategies
    # -----------------------------
    ax.text(x_left, y_mid_titles, "Full recomputation", ha="center", va="center", fontsize=22, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(x_mid, y_mid_titles, "Partial mixture reuse", ha="center", va="center", fontsize=22, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(x_right, y_mid_titles, "Full mixture reuse", ha="center", va="center", fontsize=22, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # Full recomputation: OlmixBase(D1 D2 D3 D4')
    draw_olmixbase_call(
        ax,
        center_x=x_left+2,
        center_y=y_mid_blocks-0.3,
        labels=[r"$D_1$", r"$D_2$", r"$D_3$", r"$D_4'$", r"$D_5'$"],
        colors=[BLUE, BLUE, BLUE, ORANGE, ORANGE],
    )


    # Partial reuse: OlmixBase(D_fix, D2, D4')


    # bottom small box: (D1, D3) -> D_fix (connector)
    small_center_x = x_mid - 0.3
    small_center_y = y_mid_blocks-1

    s_xL, s_xR = draw_domain_row(
        ax,
        small_center_x - 4,
        small_center_y,
        labels=[r"$\mathcal{D}_{\mathrm{fix}}$"],
        colors=[BLUE],
        block_w=1.50,
        block_h=1.50,
        gap=0.10,
        thick_outline_idx={0}
    )


    ax.text(s_xR+0.3, small_center_y+0.6, "=", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    s_xL, s_xR = draw_domain_row(
        ax,
        small_center_x - 1.3,
        small_center_y,
        labels=[r"$D_1$", r"$D_3$"],
        colors=[BLUE, BLUE],
        block_w=1.50,
        block_h=1.50,
        gap=0.10,
    )

    xL, xR, yB, yT = draw_olmixbase_call(
        ax,
        center_x=x_mid+1.5,
        center_y=y_mid_blocks-2.5,
        labels=[r"$\mathcal{D}_{\mathrm{fix}}$", r"$D_2$", r"$D_4'$", r"$D_5'$"],
        colors=[BLUE, BLUE, ORANGE, ORANGE],
        thick_outline_idx={0},
    )



    # connector from small box to center of D_fix block
    # D_fix block is the first block in the main call:
    #dfix_center_x = xL + 1.50 / 2
    #dfix_center_y = (yB + yT) / 2
    #connector(ax, (s_xL + s_xR) / 2, small_center_y + 0.45, dfix_center_x, dfix_center_y - 0.55)


    #connector(ax, s_xL, small_center_y + 0.6, dfix_center_x, dfix_center_y - 0.9)
    #connector(ax, s_xR, small_center_y + 0.6, dfix_center_x, dfix_center_y - 0.9)


    # Full reuse: OlmixBase( D_fix , D4' ) with thick outline around D_fix


    # bottom small box: (D1, D2, D3) -> D_fix
    small_center_x2 = x_right - 0.3
    small_center_y2 = y_mid_blocks-1


    s_xL, s_xR = draw_domain_row(
        ax,
        small_center_x2 - 4,
        small_center_y2,
        labels=[r"$\mathcal{D}_{\mathrm{fix}}$"],
        colors=[BLUE],
        block_w=1.50,
        block_h=1.50,
        gap=0.10,
        thick_outline_idx={0}
    )

    ax.text(s_xR+0.3, small_center_y2+0.6, "=", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


    s2_xL, s2_xR = draw_domain_row(
        ax,
        small_center_x2 - 1.3,
        small_center_y2,
        labels=[r"$D_1$", r"$D_2$", r"$D_3$"],
        colors=[BLUE, BLUE, BLUE],
        block_w=1.50,
        block_h=1.50,
        gap=0.10,
    )

    xL2, xR2, yB2, yT2 = draw_olmixbase_call(
        ax,
        center_x=x_right+1.5,
        center_y=y_mid_blocks-2.5,
        labels=[r"$\mathcal{D}_{\mathrm{fix}}$", r"$D_4'$", r"$D_5'$"],
        colors=[BLUE, ORANGE, ORANGE],
        thick_outline_idx={0},
    )


    #dfix_center_x2 = xL2 + 1.50 / 2
    #dfix_center_y2 = (yB2 + yT2) / 2

    #connector(ax, s2_xL, small_center_y2 + 0.6, dfix_center_x2, dfix_center_y2 - 0.9)
    #connector(ax, s2_xR, small_center_y2 + 0.6, dfix_center_x2, dfix_center_y2 - 0.9)

    # -----------------------------
    # Bottom: cost/performance axis
    # -----------------------------
    ax.text(1.0, y_bottom + 1.2, "Higher cost", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(1.0, y_bottom - 1.2, "Higher performance", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    ax.text(32.0, y_bottom + 1.2, "Lower cost", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(24.0, y_bottom - 1.2, "Potentially lower performance", ha="left", va="center", fontsize=18, color=BLACK, path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    # baseline arrow (left -> right)
    arrow(ax, 2, y_bottom, 36.0, y_bottom, ms=22, lw=2.2, two_way=True)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("fig_mixreuse_beige.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fig = build_figure()
    if str(args.output).endswith(".pdf"):
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    else:
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
