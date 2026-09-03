"""
trapezoidal.py
==============
Implementation of Trapezoidal Rule for Numerical Integration:
a. Single segment:
   I = (b - a) * [f(a) + f(b)] / 2
b. Multiple segments:
   I = (h / 2) * [f(x0) + 2 * sum(f(x_i)) + f(xn)]
   where h = (b - a) / n

Includes:
- Direct function integration
- Step-by-step iteration/segment breakdown
- Summary table across segment counts (matching slide Table 1)
- Visualization plots
- Text and CSV exports
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp


# ================================================================
# CORE NUMERICAL METHODS (HANDWRITTEN, NO BUILT-IN INTEGRATOR)
# ================================================================

def trapezoidal_single(f, a: float, b: float) -> float:
    """
    Single-segment Trapezoidal Rule:
        I = (b - a) * [f(a) + f(b)] / 2
    """
    fa = float(f(a))
    fb = float(f(b))
    return (b - a) * (fa + fb) / 2.0


def trapezoidal_multiple(f, a: float, b: float, n: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Multiple-segment Trapezoidal Rule:
        I = (h / 2) * [f(x0) + 2 * sum_{i=1}^{n-1} f(x_i) + f(xn)]
        where h = (b - a) / n

    Returns:
        integral_value, x_points, y_points
    """
    if n < 1:
        raise ValueError("Number of segments n must be at least 1.")

    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts = np.array([float(f(x)) for x in x_pts])

    if n == 1:
        integral = (b - a) * (y_pts[0] + y_pts[-1]) / 2.0
    else:
        interior_sum = np.sum(y_pts[1:-1])
        integral = (h / 2.0) * (y_pts[0] + 2.0 * interior_sum + y_pts[-1])

    return float(integral), x_pts, y_pts


def trapezoidal_dataset(x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Trapezoidal rule applied to discrete data points (x_i, y_i):
        I = sum_{i=0}^{n-1} (x_{i+1} - x_i) * [y_i + y_{i+1}] / 2
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")
    if len(x_data) < 2:
        raise ValueError("At least 2 points are required for integration.")

    total = 0.0
    for i in range(len(x_data) - 1):
        h_i = x_data[i + 1] - x_data[i]
        total += h_i * (y_data[i] + y_data[i + 1]) / 2.0
    return float(total)


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
                elif "Segment" in headers[i] or "Step" in headers[i]:
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


# ================================================================
# DETAILED SEGMENT BREAKDOWN TABLE
# ================================================================

def generate_segment_table(f, a: float, b: float, n: int, output_dir: Path | None = None) -> pd.DataFrame:
    """Generate step-by-step breakdown table for each trapezoid segment."""
    h = (b - a) / n
    rows = []
    accumulated = 0.0

    for i in range(n):
        x_left = a + i * h
        x_right = x_left + h
        y_left = float(f(x_left))
        y_right = float(f(x_right))
        segment_area = (h / 2.0) * (y_left + y_right)
        accumulated += segment_area

        rows.append({
            "Segment i": i + 1,
            "x_left": x_left,
            "x_right": x_right,
            "f(x_left)": y_left,
            "f(x_right)": y_right,
            "Segment Area": segment_area,
            "Running Total": accumulated,
        })

    df = pd.DataFrame(rows)
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_dir / f"trapezoidal_segments_n{n}.csv", index=False)
    return df


# ================================================================
# SUMMARY TABLE ACROSS MULTIPLE SEGMENT COUNTS
# ================================================================

def generate_summary_table(
    f,
    a: float,
    b: float,
    segment_counts: list[int],
    exact_value: float | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate summary table across multiple segment counts n."""
    rows = []
    prev_val = None

    for n in segment_counts:
        val, _, _ = trapezoidal_multiple(f, a, b, n)
        h = (b - a) / n

        row = {
            "Segments (n)": n,
            "Segment Width (h)": h,
            "Approximate Value": val,
        }

        if exact_value is not None:
            true_error = exact_value - val
            rel_error = (abs(true_error) / abs(exact_value) * 100) if exact_value != 0 else np.nan
            row["True Error (Et)"] = true_error
            row["|et| (%)"] = rel_error

        if prev_val is not None:
            approx_error = abs((val - prev_val) / val * 100) if val != 0 else np.nan
            row["|ea| (%)"] = approx_error
        else:
            row["|ea| (%)"] = np.nan

        prev_val = val
        rows.append(row)

    df = pd.DataFrame(rows)
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_dir / "trapezoidal_summary.csv", index=False)
    return df


# ================================================================
# PLOTTING FUNCTIONS
# ================================================================

def plot_trapezoidal(f, a: float, b: float, n: int, output_dir: Path | None = None):
    """Plot the function along with trapezoidal approximation segments."""
    val, x_pts, y_pts = trapezoidal_multiple(f, a, b, n)

    x_fine = np.linspace(a, b, 400)
    y_fine = [float(f(x)) for x in x_fine]

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, "b-", linewidth=2.5, label="f(x) Integrand")

    # Plot trapezoids
    for i in range(n):
        xs = [x_pts[i], x_pts[i], x_pts[i + 1], x_pts[i + 1]]
        ys = [0, y_pts[i], y_pts[i + 1], 0]
        plt.fill(xs, ys, color="#4285f4", alpha=0.25, edgecolor="#1a73e8", linewidth=1.5)

    plt.plot(x_pts, y_pts, "ro--", markersize=6, label=f"Trapezoidal (n={n})")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.title(f"Trapezoidal Rule Approximation (n = {n} segments, Area = {val:.4f})", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / f"trapezoidal_plot_n{n}.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# DEFAULT PROBLEM FROM LECTURE SLIDES
# ================================================================

def get_slide_default_problem():
    """
    Example 1 & 2 from slides:
        f(t) = 2000 * ln(140000 / (140000 - 2100*t)) - 9.8*t
        from a = 8 to b = 30
        exact = 11061.34 m
    """
    def f(t):
        return 2000.0 * np.log(140000.0 / (140000.0 - 2100.0 * t)) - 9.8 * t

    return {
        "name": "Rocket Vertical Distance (Lecture Slide Example)",
        "func": f,
        "a": 8.0,
        "b": 30.0,
        "exact": 11061.34,
        "default_n": [1, 2, 3, 4, 5, 6, 7, 8],
    }


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    prob = get_slide_default_problem()
    out_dir = Path("integration_results/trapezoidal")

    print("\n==================================================")
    print("      TRAPEZOIDAL RULE NUMERICAL INTEGRATION      ")
    print("==================================================")
    print("1. Slide Example (Rocket: t=8 to 30)")
    print("2. Custom Function Input")
    print("3. Discrete Dataset Integration")
    print("4. Exit")

    choice = input("\nSelect an option (1, 2, 3, 4): ").strip()
    if choice == "4":
        return

    if choice == "1":
        f = prob["func"]
        a, b = prob["a"], prob["b"]
        exact = prob["exact"]

        print(f"\nIntegrating Rocket Function from a={a} to b={b} (Exact = {exact})")

        # 1. Single Segment
        val_single = trapezoidal_single(f, a, b)
        err_single = exact - val_single
        pct_single = abs(err_single) / exact * 100
        print(f"\n[Single Segment]: Value = {val_single:.4f}, True Error Et = {err_single:.4f}, |et| = {pct_single:.4f}%")

        # 2. Summary Table across multiple segments
        summary_df = generate_summary_table(f, a, b, prob["default_n"], exact, out_dir)
        print(format_table_console(summary_df, "MULTIPLE SEGMENT TRAPEZOIDAL RULE SUMMARY"))

        # 3. Detailed Step Table for n=4
        detail_df = generate_segment_table(f, a, b, 4, out_dir)
        print(format_table_console(detail_df, "STEP-BY-STEP SEGMENT BREAKDOWN (n = 4)"))

        plot_trapezoidal(f, a, b, 4, out_dir)
        print(f"\nResults saved to: {out_dir.resolve()}")

    elif choice == "2":
        expr_str = input("Enter f(x) (e.g., 300*x / (1 + exp(x))): ").strip()
        x_sym = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={"x": x_sym})
        fn = sp.lambdify(x_sym, expr, modules=["numpy"])
        f = lambda val: float(fn(val))

        a = float(input("Enter lower limit a: ").strip())
        b = float(input("Enter upper limit b: ").strip())
        n = int(input("Enter number of segments n (e.g. 4): ").strip())

        val, _, _ = trapezoidal_multiple(f, a, b, n)
        print(f"\nTrapezoidal integral (n={n}) = {val:.6f}")
        detail_df = generate_segment_table(f, a, b, n, out_dir)
        print(format_table_console(detail_df, f"STEP-BY-STEP BREAKDOWN (n = {n})"))
        plot_trapezoidal(f, a, b, n, out_dir)

    elif choice == "3":
        x_input = input("Enter x values separated by commas: ").strip()
        y_input = input("Enter y values separated by commas: ").strip()
        x_data = np.array([float(x.strip()) for x in x_input.split(",")])
        y_data = np.array([float(y.strip()) for y in y_input.split(",")])

        result = trapezoidal_dataset(x_data, y_data)
        print(f"\nTrapezoidal Integral over {len(x_data) - 1} segments = {result:.6f}")


if __name__ == "__main__":
    run_cli()
