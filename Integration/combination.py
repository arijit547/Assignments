"""
combination.py
==============
Implementation of Combination (Hybrid) Numerical Integration for Datasets.
Combines Trapezoidal Rule and Simpson's 1/3 Rule based on spacing patterns
and error minimization principles.

Assignment Requirement:
c. Combination of Trapezoidal and Simpson’s 1/3 Method. In that case, you will be
given the dataset and based on the pattern of the dataset, you must decide
whether you should use Trapezoidal, Simpson’s 1/3 or both. If you decide to use
both, in that case, which segment uses what method, you must decide that too.
Remember, our main intention is to reduce the error.

Algorithm Strategy:
1. Identify contiguous blocks of equal spacing h (using scale-aware tolerance).
2. For each block with m segments:
   - If m is even (m >= 2):
     Apply Multiple-segment Simpson's 1/3 rule (O(h^4) global error) to minimize error.
   - If m is odd (m >= 3, m = 2k + 1):
     Apply Simpson's 1/3 on 2k segments, and Trapezoidal on 1 segment.
     To minimize error, estimate second derivative curvature |f''| at both ends:
     Place Trapezoidal rule on the segment where |f''| is smaller, and Simpson's 1/3 on the rest.
   - If m == 1:
     Apply single-segment Trapezoidal rule (Simpson requires >= 2 equal segments).
3. Produces:
   - Full Pattern Recognition & Decision Verdict
   - Formula & Step-by-Step Algebraic Substitution
   - Node-by-Node Weight & Contribution Tables (1, 4, 2, 4, 1)
   - Decision Breakdown Table
   - Comparison vs Pure Trapezoidal Baseline
   - Parabolic & Linear Segment Visualization Plots
   - CSV exports for reports
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class NodeContribution:
    node_name: str
    index: int
    x: float
    y: float
    weight: int
    contribution: float


@dataclass
class SegmentDecision:
    block_id: int
    start_idx: int
    end_idx: int
    x_start: float
    x_end: float
    num_segments: int
    h: float
    method: str  # "Simpson's 1/3" or "Trapezoidal"
    reason: str
    sub_integral: float
    formula: str
    substitution: str
    nodes: list[NodeContribution]


# ================================================================
# CORE HANDWRITTEN INTEGRATORS & NODE BREAKDOWN
# ================================================================

def build_trapezoid_block(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    global_start_idx: int,
) -> tuple[float, str, str, list[NodeContribution]]:
    """Handwritten single or multiple-segment Trapezoidal block."""
    n = len(x_pts) - 1
    lower = x_pts[0]
    upper = x_pts[-1]
    h = (upper - lower) / n

    nodes = []
    weighted_sum = 0.0
    for i in range(len(x_pts)):
        weight = 1 if (i == 0 or i == n) else 2
        contrib = weight * y_pts[i]
        weighted_sum += contrib
        nodes.append(NodeContribution(
            node_name=f"x{global_start_idx + i}",
            index=global_start_idx + i,
            x=x_pts[i],
            y=y_pts[i],
            weight=weight,
            contribution=contrib,
        ))

    if n == 1:
        val = (upper - lower) * (y_pts[0] + y_pts[1]) / 2.0
        formula = "I = (b - a) / 2 * [y0 + y1]"
        substitution = f"I = ({upper:g} - {lower:g}) / 2 * [{y_pts[0]:.4f} + {y_pts[1]:.4f}] = {val:.4f}"
    else:
        val = (h / 2.0) * weighted_sum
        formula = "I = h / 2 * [y0 + 2*sum(y_interior) + yn]"
        substitution = f"I = {h:g} / 2 * [sum of weighted nodes = {weighted_sum:.4f}] = {val:.4f}"

    return float(val), formula, substitution, nodes


def build_simpson_block(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    global_start_idx: int,
) -> tuple[float, str, str, list[NodeContribution]]:
    """Handwritten Simpson's 1/3 block (even number of segments)."""
    n = len(x_pts) - 1
    if n < 2 or n % 2 != 0:
        raise ValueError(f"Simpson's 1/3 requires an even number of segments. Got {n}.")

    h = (x_pts[-1] - x_pts[0]) / n
    nodes = []
    weighted_sum = 0.0

    for i in range(len(x_pts)):
        if i == 0 or i == n:
            weight = 1
        elif i % 2 == 1:
            weight = 4
        else:
            weight = 2

        contrib = weight * y_pts[i]
        weighted_sum += contrib
        nodes.append(NodeContribution(
            node_name=f"x{global_start_idx + i}",
            index=global_start_idx + i,
            x=x_pts[i],
            y=y_pts[i],
            weight=weight,
            contribution=contrib,
        ))

    val = (h / 3.0) * weighted_sum

    if n == 2:
        formula = "I = h / 3 * [y0 + 4*y1 + y2]"
        substitution = (
            f"I = {h:g} / 3 * [{y_pts[0]:.4f} + 4({y_pts[1]:.4f}) + {y_pts[2]:.4f}] "
            f"= {h:g} / 3 * [{weighted_sum:.4f}] = {val:.4f}"
        )
    else:
        formula = "I = h / 3 * [y0 + 4*sum(y_odd) + 2*sum(y_even) + yn]"
        substitution = f"I = {h:g} / 3 * [sum of weighted nodes = {weighted_sum:.4f}] = {val:.4f}"

    return float(val), formula, substitution, nodes


# ================================================================
# PATTERN RECOGNITION & DECISION ENGINE
# ================================================================

def analyze_and_integrate_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-7,
) -> tuple[float, list[SegmentDecision], str]:
    """
    Analyzes the pattern of the dataset and decides which method to use for each segment.
    Applies curvature analysis to place the leftover trapezoid where error is minimized.

    Returns:
        total_integral, list_of_decisions, overall_verdict_string
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have identical lengths.")
    if len(x_data) < 2:
        raise ValueError("At least 2 data points are required.")

    # Ensure strictly increasing sorted data
    sort_idx = np.argsort(x_data)
    x = np.array(x_data[sort_idx], dtype=float)
    y = np.array(y_data[sort_idx], dtype=float)

    n_total_segments = len(x) - 1
    spacings = np.diff(x)

    # 1. Group segments into contiguous blocks of equal spacing h
    blocks = []
    curr_start = 0

    for i in range(1, n_total_segments):
        h_prev = spacings[i - 1]
        h_curr = spacings[i]
        if not math.isclose(h_prev, h_curr, rel_tol=rel_tol, abs_tol=abs_tol):
            blocks.append((curr_start, i))
            curr_start = i
    blocks.append((curr_start, n_total_segments))

    # 2. Make decisions for each block to minimize error
    decisions: list[SegmentDecision] = []
    total_integral = 0.0
    used_simpson = False
    used_trapezoid = False
    block_counter = 1

    for start_seg, end_seg in blocks:
        m = end_seg - start_seg
        p = start_seg
        q = end_seg
        h = spacings[p]

        # Case A: Even number of segments (m >= 2) -> Full Simpson's 1/3
        if m >= 2 and m % 2 == 0:
            val, formula, subst, nodes = build_simpson_block(x[p:q + 1], y[p:q + 1], p)
            decisions.append(SegmentDecision(
                block_id=block_counter,
                start_idx=p,
                end_idx=q,
                x_start=x[p],
                x_end=x[q],
                num_segments=m,
                h=h,
                method="Simpson's 1/3",
                reason=f"Block has {m} equal segments (even). Simpson's 1/3 applied to maximize accuracy O(h^4).",
                sub_integral=val,
                formula=formula,
                substitution=subst,
                nodes=nodes,
            ))
            total_integral += val
            used_simpson = True
            block_counter += 1

        # Case B: Odd number of segments (m >= 3, m = 2k + 1) -> Curvature-optimized split
        elif m >= 3 and m % 2 != 0:
            curv_start = abs(y[p + 2] - 2.0 * y[p + 1] + y[p])
            curv_end = abs(y[q] - 2.0 * y[q - 1] + y[q - 2])

            if curv_start < curv_end:
                # First segment uses Trapezoidal
                val_t, form_t, sub_t, nodes_t = build_trapezoid_block(x[p:p + 2], y[p:p + 2], p)
                decisions.append(SegmentDecision(
                    block_id=block_counter,
                    start_idx=p,
                    end_idx=p + 1,
                    x_start=x[p],
                    x_end=x[p + 1],
                    num_segments=1,
                    h=h,
                    method="Trapezoidal",
                    reason=f"Odd block ({m} segs): Trapezoidal placed at start because estimated curvature |f''|={curv_start:.3e} is lower than at end |f''|={curv_end:.3e}.",
                    sub_integral=val_t,
                    formula=form_t,
                    substitution=sub_t,
                    nodes=nodes_t,
                ))
                block_counter += 1

                # Remaining m-1 (even) segments use Simpson's 1/3
                val_s, form_s, sub_s, nodes_s = build_simpson_block(x[p + 1:q + 1], y[p + 1:q + 1], p + 1)
                decisions.append(SegmentDecision(
                    block_id=block_counter,
                    start_idx=p + 1,
                    end_idx=q,
                    x_start=x[p + 1],
                    x_end=x[q],
                    num_segments=m - 1,
                    h=h,
                    method="Simpson's 1/3",
                    reason=f"Odd block ({m} segs): Remaining {m-1} equal segments (even) use Simpson's 1/3 (O(h^4)).",
                    sub_integral=val_s,
                    formula=form_s,
                    substitution=sub_s,
                    nodes=nodes_s,
                ))
                block_counter += 1
                total_integral += val_t + val_s
            else:
                # First m-1 (even) segments use Simpson's 1/3
                val_s, form_s, sub_s, nodes_s = build_simpson_block(x[p:q], y[p:q], p)
                decisions.append(SegmentDecision(
                    block_id=block_counter,
                    start_idx=p,
                    end_idx=q - 1,
                    x_start=x[p],
                    x_end=x[q - 1],
                    num_segments=m - 1,
                    h=h,
                    method="Simpson's 1/3",
                    reason=f"Odd block ({m} segs): First {m-1} equal segments (even) use Simpson's 1/3 (O(h^4)).",
                    sub_integral=val_s,
                    formula=form_s,
                    substitution=sub_s,
                    nodes=nodes_s,
                ))
                block_counter += 1

                # Last segment uses Trapezoidal
                val_t, form_t, sub_t, nodes_t = build_trapezoid_block(x[q - 1:q + 1], y[q - 1:q + 1], q - 1)
                decisions.append(SegmentDecision(
                    block_id=block_counter,
                    start_idx=q - 1,
                    end_idx=q,
                    x_start=x[q - 1],
                    x_end=x[q],
                    num_segments=1,
                    h=h,
                    method="Trapezoidal",
                    reason=f"Odd block ({m} segs): Trapezoidal placed at end because estimated curvature |f''|={curv_end:.3e} is lower than at start |f''|={curv_start:.3e}.",
                    sub_integral=val_t,
                    formula=form_t,
                    substitution=sub_t,
                    nodes=nodes_t,
                ))
                block_counter += 1
                total_integral += val_s + val_t

            used_simpson = True
            used_trapezoid = True

        # Case C: Single isolated segment (m == 1) -> Trapezoidal rule
        else:
            val_t, form_t, sub_t, nodes_t = build_trapezoid_block(x[p:q + 1], y[p:q + 1], p)
            decisions.append(SegmentDecision(
                block_id=block_counter,
                start_idx=p,
                end_idx=q,
                x_start=x[p],
                x_end=x[q],
                num_segments=1,
                h=h,
                method="Trapezoidal",
                reason="Single isolated segment with unique spacing; Simpson's 1/3 requires >= 2 equal segments.",
                sub_integral=val_t,
                formula=form_t,
                substitution=sub_t,
                nodes=nodes_t,
            ))
            total_integral += val_t
            used_trapezoid = True
            block_counter += 1

    if used_simpson and not used_trapezoid:
        verdict = "Pure Simpson's 1/3 Rule (Dataset has equal spacing with an even number of segments)."
    elif used_trapezoid and not used_simpson:
        verdict = "Pure Trapezoidal Rule (Dataset has irregular spacing with no 2 consecutive equal segments)."
    else:
        verdict = "Combination Method (Both Simpson's 1/3 and Trapezoidal rules strategically selected to minimize error)."

    return total_integral, decisions, verdict


# ================================================================
# FORMATTING & DISPLAY HELPERS
# ================================================================

def format_table_console(df: pd.DataFrame, title: str) -> str:
    """Format DataFrame as a clean, aligned, ASCII-safe console table."""
    headers = list(df.columns)
    formatted_rows = []
    for _, row in df.iterrows():
        f_row = []
        for i, val in enumerate(row):
            if isinstance(val, (int, np.integer)):
                s = f"{val}"
            elif isinstance(val, (float, np.floating)):
                if "%" in headers[i] or "|et|" in headers[i]:
                    s = f"{val:.4f}%"
                elif "h" in headers[i] or "Width" in headers[i]:
                    s = f"{val:g}"
                elif abs(val) < 1e-4 and val != 0:
                    s = f"{val:.6e}"
                else:
                    s = f"{val:.4f}"
            else:
                s = str(val)
            f_row.append(s)
        formatted_rows.append(f_row)

    widths = [
        max(len(h), max((len(r[i]) for r in formatted_rows), default=0))
        for i, h in enumerate(headers)
    ]

    header_str = " | ".join(h.rjust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    line_width = len(header_str)

    lines = [
        "",
        "=" * line_width,
        title.center(line_width),
        "=" * line_width,
        header_str,
        sep_str,
    ]
    for r in formatted_rows:
        lines.append(" | ".join(r[i].rjust(widths[i]) for i in range(len(headers))))
    lines.append("=" * line_width)
    return "\n".join(lines)


def print_block_detailed_calculations(decisions: list[SegmentDecision]) -> None:
    """
    Displays step-by-step formula substitutions and the node weight tables
    (with weights 1, 4, 2, 4, 1 and weighted contributions) for each block.
    """
    print("\n" + "=" * 90)
    print("STEP-BY-STEP BLOCK CALCULATIONS & NODE WEIGHT CONTRIBUTIONS".center(90))
    print("=" * 90)

    for d in decisions:
        print(f"\n--- Block {d.block_id}: Range [{d.x_start:g}, {d.x_end:g}] ({d.method}, {d.num_segments} segs, h = {d.h:g}) ---")
        print(f"  Formula     : {d.formula}")
        print(f"  Substitution: {d.substitution}")
        print(f"  Sub-Integral: {d.sub_integral:.6f}")

        # Build Node Contribution Table
        node_rows = []
        for nd in d.nodes:
            node_rows.append({
                "Node": nd.node_name,
                "x_i": nd.x,
                "y_i": nd.y,
                "Weight": nd.weight,
                "Contribution (w*y)": nd.contribution,
            })
        df_nodes = pd.DataFrame(node_rows)

        # Print Node table with summary sum row
        headers = ["Node", "x_i", "y_i", "Weight", "Contribution (w*y)"]
        rows_fmt = []
        for r in node_rows:
            rows_fmt.append([
                r["Node"],
                f"{r['x_i']:.4f}",
                f"{r['y_i']:.4f}",
                str(r["Weight"]),
                f"{r['Contribution (w*y)']:.4f}"
            ])
        sum_contrib = sum(r["Contribution (w*y)"] for r in node_rows)
        rows_fmt.append(["SUM", "---", "---", "---", f"{sum_contrib:.4f}"])

        widths = [max(len(headers[i]), max(len(row[i]) for row in rows_fmt)) for i in range(len(headers))]
        hdr_str = " | ".join(headers[i].rjust(widths[i]) for i in range(len(headers)))
        sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))

        print("    " + hdr_str)
        print("    " + sep_str)
        for r in rows_fmt[:-1]:
            print("    " + " | ".join(r[i].rjust(widths[i]) for i in range(len(headers))))
        print("    " + sep_str)
        print("    " + " | ".join(rows_fmt[-1][i].rjust(widths[i]) for i in range(len(headers))))


def generate_decision_dataframe(decisions: list[SegmentDecision]) -> pd.DataFrame:
    """Generate detailed tabular report of decisions made for each interval."""
    rows = []
    accum = 0.0
    for d in decisions:
        accum += d.sub_integral
        rows.append({
            "Block": d.block_id,
            "Interval [a, b]": f"[{d.x_start:g}, {d.x_end:g}]",
            "Points": f"x{d.start_idx}-x{d.end_idx}",
            "Segments": d.num_segments,
            "Spacing h": d.h,
            "Method Chosen": d.method,
            "Sub-Integral": d.sub_integral,
            "Running Total": accum,
            "Rationale (Error Minimization)": d.reason,
        })
    return pd.DataFrame(rows)


def generate_comparison_report(
    x_data: np.ndarray,
    y_data: np.ndarray,
    exact_value: float | None = None,
    output_dir: Path | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """
    Runs both the Combination Method and Full Trapezoidal Rule,
    producing a comparison table showing the error reduction.
    """
    comb_val, decisions, verdict = analyze_and_integrate_dataset(x_data, y_data)
    pure_trap_val = sum((x_data[i+1]-x_data[i])*(y_data[i]+y_data[i+1])/2.0 for i in range(len(x_data)-1))

    rows = []
    # Row 1: Pure Trapezoidal
    row_trap = {
        "Integration Method": "Pure Trapezoidal Rule (Baseline)",
        "Segments Used": f"{len(x_data)-1} Trapezoids",
        "Integral Value": pure_trap_val,
    }
    if exact_value is not None:
        err_t = exact_value - pure_trap_val
        pct_t = (abs(err_t) / abs(exact_value) * 100) if exact_value != 0 else np.nan
        row_trap["True Error (Et)"] = err_t
        row_trap["|et| (%)"] = pct_t
    rows.append(row_trap)

    # Row 2: Combination Method
    simp_segs = sum(d.num_segments for d in decisions if d.method == "Simpson's 1/3")
    trap_segs = sum(d.num_segments for d in decisions if d.method == "Trapezoidal")
    row_comb = {
        "Integration Method": "Combination (Simpson + Trapezoid)",
        "Segments Used": f"{simp_segs} Simpson + {trap_segs} Trap",
        "Integral Value": comb_val,
    }
    if exact_value is not None:
        err_c = exact_value - comb_val
        pct_c = (abs(err_c) / abs(exact_value) * 100) if exact_value != 0 else np.nan
        row_comb["True Error (Et)"] = err_c
        row_comb["|et| (%)"] = pct_c
    rows.append(row_comb)

    df_comp = pd.DataFrame(rows)

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        df_dec = generate_decision_dataframe(decisions)
        df_dec.to_csv(output_dir / "combination_decision_table.csv", index=False)
        df_comp.to_csv(output_dir / "combination_vs_trapezoidal_comparison.csv", index=False)

    return comb_val, pure_trap_val, df_comp


# ================================================================
# PLOTTING COMBINED INTEGRATION DATASET
# ================================================================

def plot_combination_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    decisions: list[SegmentDecision],
    title: str = "Combined Numerical Integration of Dataset",
    output_dir: Path | None = None,
):
    """
    Visualizes the dataset with green parabolic shading for Simpson's 1/3
    and blue linear secant shading for Trapezoidal segments.
    """
    plt.figure(figsize=(11, 6))

    # Plot original data points
    plt.plot(x_data, y_data, "ko", markersize=7, zorder=5, label="Data Points")

    simpson_labeled = False
    trap_labeled = False

    for d in decisions:
        xs = x_data[d.start_idx:d.end_idx + 1]
        ys = y_data[d.start_idx:d.end_idx + 1]

        if d.method == "Simpson's 1/3":
            label = "Simpson's 1/3 Segments (Parabolic)" if not simpson_labeled else None
            simpson_labeled = True

            # Fit parabolas across each pair
            for p in range(d.num_segments // 2):
                xp = xs[2*p:2*p+3]
                yp = ys[2*p:2*p+3]
                poly = np.polyfit(xp, yp, deg=2)
                xf = np.linspace(xp[0], xp[-1], 60)
                yf = np.polyval(poly, xf)
                plt.fill_between(xf, 0, yf, color="#34a853", alpha=0.3)
                plt.plot(xf, yf, color="#1e8e3e", linewidth=2.2, label=label if p == 0 else None)
                label = None

        else:
            label = "Trapezoidal Segments (Linear)" if not trap_labeled else None
            trap_labeled = True
            for i in range(len(xs) - 1):
                xf = [xs[i], xs[i+1]]
                yf = [ys[i], ys[i+1]]
                plt.fill_between(xf, 0, yf, color="#4285f4", alpha=0.3)
                plt.plot(xf, yf, color="#1a73e8", linewidth=2, linestyle="--", label=label if i == 0 else None)
                label = None

    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "combination_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# PRESET DATASETS FOR DEMONSTRATION & TESTING
# ================================================================

def get_preset_datasets():
    """Returns sample datasets representing all combinations and patterns."""
    def rocket_f(t):
        return 2000.0 * np.log(140000.0 / (140000.0 - 2100.0 * t)) - 9.8 * t

    t1 = np.linspace(8, 30, 7)
    y1 = np.array([rocket_f(t) for t in t1])

    t2 = np.array([8, 12, 16, 20, 24, 27, 30, 32], dtype=float)
    y2 = np.array([rocket_f(t) for t in t2])

    t3 = np.array([8, 12, 16, 20, 25], dtype=float)
    y3 = np.array([rocket_f(t) for t in t3])

    x4 = np.array([0.0, 0.12, 0.22, 0.32, 0.36, 0.40, 0.44, 0.54, 0.64, 0.70, 0.80], dtype=float)
    y4 = np.array([0.2000, 1.3097, 1.3052, 1.7434, 1.9000, 2.0200, 2.1100, 2.0500, 1.8300, 1.6300, 1.2200], dtype=float)

    return {
        "1": {
            "title": "Preset 1: Rocket Velocity (Equal Spacing, Even n=6) -> Pure Simpson's 1/3",
            "x": t1, "y": y1, "exact": 11061.34
        },
        "2": {
            "title": "Preset 2: Rocket Velocity Mixed Spacing (4 segs h=4, 2 segs h=3, 1 seg h=2) -> Full Combination",
            "x": t2, "y": y2, "exact": 13038.56
        },
        "3": {
            "title": "Preset 3: Rocket Velocity Odd Segment Block (3 segs h=4, 1 seg h=5) -> Curvature-Optimized Combination",
            "x": t3, "y": y3, "exact": 7329.83
        },
        "4": {
            "title": "Preset 4: Chapra Textbook Table 21.3 (Unequally Spaced Data)",
            "x": x4, "y": y4, "exact": 1.4026
        },
    }


# ================================================================
# CLI INTERFACE WITH UNDO (B / BACK) SUPPORT
# ================================================================

class BackRequested(Exception):
    pass


def prompt_input_with_undo(prompt: str) -> str:
    """Prompt user for input and raise BackRequested if user enters B or BACK."""
    val = input(f"{prompt} (or B/BACK to return): ").strip()
    if val.upper() in {"B", "BACK"}:
        raise BackRequested
    return val


def parse_float_list(text: str) -> list[float]:
    parts = [p for p in re.split(r"[\s,]+", text.strip()) if p]
    if not parts:
        raise ValueError("List cannot be empty.")
    return [float(p) for p in parts]


def run_cli():
    presets = get_preset_datasets()
    out_dir = Path("integration_results/combination")
    saved_dataset: Optional[tuple[np.ndarray, np.ndarray, Optional[float]]] = None

    while True:
        print("\n==========================================================================")
        print("      COMBINATION OF TRAPEZOIDAL AND SIMPSON'S 1/3 METHOD (DATASETS)      ")
        print("==========================================================================")
        print("1. Preset 1: Equal Spacing Even Segments (Pure Simpson's 1/3)")
        print("2. Preset 2: Mixed Spacings (4 segs h=4, 2 segs h=3, 1 seg h=2)")
        print("3. Preset 3: Odd Equal-Spacing Block (Curvature-Optimized Decision)")
        print("4. Preset 4: Chapra Table 21.3 Unequally Spaced Dataset")
        print("5. Enter Custom Dataset (with Undo B/BACK support)")
        print("6. Load Dataset from CSV File")
        if saved_dataset is not None:
            print("7. Re-run Previous Dataset")
        print("0. Exit to Master Menu")

        choice = input("\nSelect an option: ").strip()
        if choice in {"0", "exit", "quit"}:
            break

        if choice in presets:
            data = presets[choice]
            title = data["title"]
            x_data = data["x"]
            y_data = data["y"]
            exact = data.get("exact", None)
            saved_dataset = (x_data, y_data, exact)

        elif choice == "5":
            try:
                title = "User-Entered Custom Dataset"
                x_str = prompt_input_with_undo("Enter x values (comma/space-separated)")
                x_vals = parse_float_list(x_str)

                y_str = prompt_input_with_undo("Enter y values (comma/space-separated)")
                y_vals = parse_float_list(y_str)

                if len(x_vals) != len(y_vals):
                    print(f"\n[ERROR]: x has {len(x_vals)} values but y has {len(y_vals)} values. Must be equal.")
                    continue

                exact_str = prompt_input_with_undo("Enter exact value if known (press Enter to skip)")
                exact = float(exact_str) if exact_str else None

                x_data = np.array(x_vals)
                y_data = np.array(y_vals)
                saved_dataset = (x_data, y_data, exact)
            except BackRequested:
                print("\n-> Returned to menu.")
                continue
            except Exception as e:
                print(f"\n[ERROR]: Invalid input: {e}")
                continue

        elif choice == "6":
            try:
                file_path = prompt_input_with_undo("Enter path to CSV file")
                df_file = pd.read_csv(file_path)
                x_data = df_file.iloc[:, 0].values
                y_data = df_file.iloc[:, 1].values
                title = f"Dataset from {file_path}"
                exact = None
                saved_dataset = (x_data, y_data, exact)
            except BackRequested:
                continue
            except Exception as e:
                print(f"\n[ERROR]: Failed to load CSV: {e}")
                continue

        elif choice == "7" and saved_dataset is not None:
            x_data, y_data, exact = saved_dataset
            title = "Re-running Previous Dataset"

        else:
            print("Invalid selection.")
            continue

        print(f"\n>>> Analyzing: {title}")
        print(f"Total points: {len(x_data)}, Total segments: {len(x_data) - 1}")

        # Run Analysis
        total_val, decisions, verdict = analyze_and_integrate_dataset(x_data, y_data)
        print(f"\n[DECISION ENGINE VERDICT]:\n-> {verdict}")

        # 1. Step-by-Step Block Calculations & Node Weights Table
        print_block_detailed_calculations(decisions)

        # 2. Generate Decision Table
        df_dec = generate_decision_dataframe(decisions)
        print(format_table_console(df_dec, "SEGMENT-BY-SEGMENT PATTERN & DECISION BREAKDOWN"))

        # 3. Generate Comparison Table (Combination vs Pure Trapezoidal)
        comb_val, pure_trap_val, df_comp = generate_comparison_report(x_data, y_data, exact, out_dir)
        print(format_table_console(df_comp, "ACCURACY COMPARISON: COMBINATION METHOD VS PURE TRAPEZOIDAL"))

        # 4. Final Combination Formula
        block_sum_str = " + ".join(f"I{d.block_id}({d.sub_integral:.4f})" for d in decisions)
        print("\n" + "-" * 70)
        print(f"TOTAL HYBRID INTEGRAL I = {block_sum_str}")
        print(f"                        = {total_val:.6f}")
        print("-" * 70)

        # 5. Plot
        plot_combination_dataset(x_data, y_data, decisions, title=title, output_dir=out_dir)
        print(f"\nAll combination tables and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
