# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Figure: Swarm -> Regression -> Optimization (hand-drawn XKCD style)

Usage:
    uv run fig_sro.py --output fig_sro.png
    uv run fig_sro.py --output fig_sro.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

MANROPE_BASE_PATH = (Path(__file__).parent / "manrope").absolute()
for path in MANROPE_BASE_PATH.iterdir():
    if path.suffix == ".ttf":
        fontManager.addfont(str(path))


plt.rcParams["font.family"] = "Manrope"

# Y_COLOR = "#d9534f"  # warm red (nice contrast with black)
# FHAT_COLOR = "#7b1fa2"  # purple


Y_COLOR = "#d9534f"
FHAT_COLOR = "#337ab7"
BACKGROUND = "#FAF2E9"


BLACK = "#2c3e50"


from matplotlib.patches import Circle, Wedge


def draw_pie_at(
    ax,
    x: float,
    y: float,
    fracs: list[float],
    colors: list[str],
    r: float = 0.55,
    outline_color: str = "#2c3e50",
    outline_lw: float = 2.0,
    startangle: float = 90,
):
    """Draw a small pie chart centered at (x,y) in data coords, robust under xkcd sketch."""
    total = sum(fracs)
    if total <= 0:
        return
    fracs = [f / total for f in fracs]

    theta = startangle

    for f, c in zip(fracs, colors):
        theta2 = theta + 360 * f
        ax.add_patch(
            Wedge(
                (x, y),
                r,
                theta,
                theta2,
                facecolor=c,
                edgecolor=c,
                linewidth=0.6,
                joinstyle="round",
                path_effects=[
                    pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                    pe.Normal(),
                ],
            )
        )
        theta = theta2

    ax.add_patch(
        Circle(
            (x, y),
            r * 0.06,
            facecolor=colors[0],
            edgecolor="none",
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )
    )

    ax.add_patch(
        Circle(
            (x, y),
            r,
            fill=False,
            edgecolor=outline_color,
            linewidth=outline_lw,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )
    )


def draw_circled_number(ax: plt.Axes, x: float, y: float, number: int) -> None:
    circ = Circle(
        (x, y),
        0.8,
        fill=False,
        edgecolor=BLACK,
        linewidth=2,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(circ)
    ax.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        fontsize=14,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, ms: float = 14) -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=ms,
        linewidth=2,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(arr)


def draw_proxy_box(ax, x, y, w: float = 2.4, h: float = 1.2):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor="white",
        edgecolor=BLACK,
        linewidth=2,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        "proxy",
        ha="center",
        va="center",
        fontsize=13,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    return (x - w / 2, x + w / 2)


def panel_titles(ax: plt.Axes) -> None:
    draw_circled_number(ax, 1.0, 8.5, 1)
    ax.text(
        2.2,
        8.5,
        "Swarm",
        ha="left",
        va="center",
        fontsize=18,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    draw_circled_number(ax, 14.0, 8.5, 2)
    ax.text(
        15.2,
        8.5,
        "Regression",
        ha="left",
        va="center",
        fontsize=18,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    df = 2
    draw_circled_number(ax, 26.5 - df, 8.5, 3)
    ax.text(
        27.7 - df,
        8.5,
        "Optimization",
        ha="left",
        va="center",
        fontsize=18,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def draw_swarm_panel(ax: plt.Axes) -> None:
    PIE_COLORS = ["#aed6f1", "#f8c471", "#abebc6", "#CF56AD"]

    draw_pie_at(ax, 1.15, 5.3, fracs=[0.55, 0.25, 0.15, 0.05], colors=PIE_COLORS, r=0.75)
    ax.text(
        1.15,
        3.8,
        r"$\vdots$",
        ha="center",
        va="center",
        fontsize=18,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    draw_pie_at(ax, 1.15, 2.4, fracs=[0.70, 0.10, 0.15, 0.05], colors=PIE_COLORS, r=0.75)

    proxy_left1, proxy_right1 = draw_proxy_box(ax, 4.2, 5.3)
    proxy_left2, proxy_right2 = draw_proxy_box(ax, 4.2, 2.4)

    L = 1.2

    draw_arrow(ax, proxy_left1 - L, 5.3, proxy_left1, 5.3)
    draw_arrow(ax, proxy_left2 - L, 2.4, proxy_left2, 2.4)

    draw_arrow(ax, proxy_right1, 5.3, proxy_right1 + L, 5.3)
    draw_arrow(ax, proxy_right2, 2.4, proxy_right2 + L, 2.4)
    y_text_x = proxy_right1 + L + 0.2

    ax.text(
        y_text_x,
        5.3,
        r"$y_{11},\ldots,y_{n1}$",
        ha="left",
        va="center",
        fontsize=16,
        color=Y_COLOR,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(
        y_text_x,
        2.4,
        r"$y_{1K},\ldots,y_{nK}$",
        ha="left",
        va="center",
        fontsize=16,
        color=Y_COLOR,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    ax.text(
        0.2,
        6.5,
        r"Mixes $p$",
        ha="left",
        va="center",
        fontsize=16,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(
        y_text_x,
        6.5,
        "Performance",
        ha="left",
        va="center",
        fontsize=16,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def draw_regression_panel(ax: plt.Axes) -> None:
    # Skinnier regression panel
    x0, y0 = 14.8, 2.2
    w, h = 7.5, 5.0

    ax.plot(
        [x0, x0],
        [y0, y0 + h],
        color=BLACK,
        lw=2.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.plot(
        [x0, x0 + w],
        [y0, y0],
        color=BLACK,
        lw=2.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    ax.plot(
        [x0 + 0.6, x0 + w - 0.6],
        [y0 + 0.6, y0 + h - 0.6],
        color=FHAT_COLOR,
        lw=2.5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    rng = np.random.default_rng(4)

    xs = np.linspace(0.9, w - 0.9, 18) + rng.normal(0, 0.2, 18)

    m = (h - 1.2) / (w - 1.2)
    b = 0.6 - m * 0.6

    ys = m * xs + b + rng.normal(0, 0.45, 18)

    xs = np.clip(xs, 0.7, w - 0.7)
    ys = np.clip(ys, 0.7, h - 0.7)

    pts = list(zip(xs, ys))

    for px, py in pts:
        ax.plot(
            x0 + px,
            y0 + py,
            marker=".",
            markersize=6,
            color=Y_COLOR,
            lw=0,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )

    ax.text(
        x0 + w / 2,
        y0 - 1.0,
        r"Predicted $\hat{f}(p)$",
        ha="center",
        va="center",
        fontsize=16,
        color=FHAT_COLOR,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(
        x0 - 1.0,
        y0 + h / 2,
        r"True $f(p)$",
        ha="center",
        va="center",
        rotation=90,
        fontsize=16,
        color=Y_COLOR,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def draw_math_segments(ax, x, y, segments, fontsize=20, va="center"):
    renderer = ax.figure.canvas.get_renderer()
    cur_x = x
    for s, c in segments:
        t = ax.text(
            cur_x,
            y,
            s,
            ha="left",
            va=va,
            fontsize=fontsize,
            color=c,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )
        ax.figure.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        w_data = inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]
        cur_x += w_data


from matplotlib.patches import FancyArrowPatch


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
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            "ICON",
            ha="center",
            va="center",
            fontsize=9,
            color=BLACK,
            path_effects=[
                pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
                pe.Normal(),
            ],
        )


def draw_optimization_panel(ax: plt.Axes) -> None:
    # Three corner points

    dx = 2.0

    top_x, top_y = 30.5 - dx, 4.3
    bl_x, bl_y = 26.8 - dx, 2.4
    br_x, br_y = 34.2 - dx, 2.8

    t = np.linspace(0, 1, 150)

    def quad_bezier(p0, p1, p2):
        x = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t**2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t**2 * p2[1]
        return x, y

    left_ctrl = ((top_x + bl_x) / 2 - 0.8, (top_y + bl_y) / 2 - 1.4)
    xl, yl = quad_bezier((top_x, top_y), left_ctrl, (bl_x, bl_y))
    ax.plot(
        xl,
        yl,
        color=FHAT_COLOR,
        lw=1.8,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=1.8, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    right_ctrl = ((top_x + br_x) / 2 + 0.8, (top_y + br_y) / 2 - 1.4)
    xr, yr = quad_bezier((top_x, top_y), right_ctrl, (br_x, br_y))
    ax.plot(
        xr,
        yr,
        color=FHAT_COLOR,
        lw=1.8,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=1.8, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    bot_ctrl = ((bl_x + br_x) / 2 - 0.3, min(bl_y, br_y) - 1.6)
    xb, yb = quad_bezier((bl_x, bl_y), bot_ctrl, (br_x, br_y))
    ax.plot(
        xb,
        yb,
        color=FHAT_COLOR,
        lw=1.8,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=1.8, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    #### Grid lines
    new_top_x = top_x + 0.7
    new_top_y = top_y - 0.5
    new_bl_x = bl_x + 1.25
    new_bl_y = bl_y - 0.45
    left_ctrl = ((new_top_x + new_bl_x) / 2 + 0.2, (new_top_y + new_bl_y) / 2 - 0.8)
    xl, yl = quad_bezier((new_top_x, new_top_y), left_ctrl, (new_bl_x, new_bl_y))
    ax.plot(
        xl,
        yl,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    new_top_x = top_x + 1.42
    new_top_y = top_y - 0.98
    new_bl_x = bl_x + 2.5
    new_bl_y = bl_y - 0.7
    left_ctrl = ((new_top_x + new_bl_x) / 2 + 0.2, (new_top_y + new_bl_y) / 2 - 0.8)
    xl, yl = quad_bezier((new_top_x, new_top_y), left_ctrl, (new_bl_x, new_bl_y))
    ax.plot(
        xl,
        yl,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    new_top_x = top_x + 2.2
    new_top_y = top_y - 1.4
    new_bl_x = bl_x + 4.8
    new_bl_y = bl_y - 0.5
    left_ctrl = ((new_top_x + new_bl_x) / 2 + 0.2, (new_top_y + new_bl_y) / 2 - 0.2)
    xl, yl = quad_bezier((new_top_x, new_top_y), left_ctrl, (new_bl_x, new_bl_y))
    ax.plot(
        xl,
        yl,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    new_top_x = top_x - 0.65
    new_top_y = top_y - 0.55
    new_br_x = br_x - 1.4
    new_br_y = br_y - 0.55
    right_ctrl = ((new_top_x + new_br_x) / 2 - 0, (new_top_y + new_br_y) / 2 - 0.8)
    xr, yr = quad_bezier((new_top_x, new_top_y), right_ctrl, (new_br_x, new_br_y))
    ax.plot(
        xr,
        yr,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    new_top_x = top_x - 1.45
    new_top_y = top_y - 1.15
    new_br_x = br_x - 2.3
    new_br_y = br_y - 0.8
    right_ctrl = ((new_top_x + new_br_x) / 2 - 0, (new_top_y + new_br_y) / 2 - 0.8)
    xr, yr = quad_bezier((new_top_x, new_top_y), right_ctrl, (new_br_x, new_br_y))
    ax.plot(
        xr,
        yr,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    new_top_x = top_x - 2.3
    new_top_y = top_y - 1.75
    new_br_x = br_x - 4
    new_br_y = br_y - 1.1
    right_ctrl = ((new_top_x + new_br_x) / 2 - 0, (new_top_y + new_br_y) / 2 - 0.4)
    xr, yr = quad_bezier((new_top_x, new_top_y), right_ctrl, (new_br_x, new_br_y))
    ax.plot(
        xr,
        yr,
        color=FHAT_COLOR,
        lw=0.5,
        solid_capstyle="round",
        path_effects=[
            pe.withStroke(linewidth=0.5, foreground=FHAT_COLOR),
            pe.Normal(),
        ],
    )

    # place_icon(ax, top_x, top_y-1, Path("bowl.png"), zoom=0.2)

    # p* dot at the centroid
    pstar_x = (top_x + bl_x + br_x) / 3
    pstar_y = (top_y + bl_y + br_y) / 3
    ax.plot(
        pstar_x,
        pstar_y - 1,
        marker="o",
        markersize=5,
        color=BLACK,
        zorder=5,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )
    ax.text(
        pstar_x + 0.45,
        pstar_y - 0.8,
        r"$p^*$",
        ha="left",
        va="center",
        fontsize=14,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    # Text above
    draw_math_segments(
        ax,
        27.5 - dx,
        6.5,
        [
            ("minimize  ", BLACK),
            (r"$\hat{f}(p)$", FHAT_COLOR),
        ],
        fontsize=16,
    )

    ax.text(
        28.5 - dx,
        5.5,
        r"$p \in S$",
        ha="left",
        va="center",
        fontsize=16,
        color=BLACK,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )

    ax.text(
        pstar_x + 4,
        pstar_y - 0.5,
        r"$\hat{f}$",
        ha="left",
        va="center",
        fontsize=14,
        color=FHAT_COLOR,
        path_effects=[
            pe.withStroke(linewidth=3.2, foreground=BACKGROUND),
            pe.Normal(),
        ],
    )


def build_figure() -> plt.Figure:
    plt.xkcd(scale=0.5, length=100, randomness=2)
    plt.rcParams["font.family"] = ["Manrope", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(11, 4), dpi=1000)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    panel_titles(ax)
    draw_swarm_panel(ax)
    draw_regression_panel(ax)
    draw_optimization_panel(ax)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Swarm/Regression/Optimization figure")
    parser.add_argument("--output", type=Path, default=Path("fig_sro_beige.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fig = build_figure()
    if str(args.output).lower().endswith(".pdf"):
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    else:
        fig.savefig(args.output, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
