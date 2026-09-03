"""
simpson.py
==========
Implementation of Simpson's 1/3 Rule for Numerical Integration:
b. Simpson's 1/3 rule:
   i. 2 segments (single application):
      I = (h / 3) * [f(x0) + 4 * f(x1) + f(x2)]
      where h = (b - a) / 2, x1 = (a + b) / 2
   ii. Multiple segments (n must be even, n >= 2):
      I = (h / 3) * [f(x0) + 4 * sum_{odd} f(x_i) + 2 * sum_{even} f(x_i) + f(xn)]
      where h = (b - a) / n

Includes:
- Direct function integration
- Step-by-step pair-segment breakdown
- Summary table across even segment counts (matching slide Table 1, Page 54)
- Parabolic segment visualization plots
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

def simpson_single_2seg(f, a: float, b: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    2-Segment Simpson's 1/3 Rule (Single Application):
        h = (b - a) / 2
        x0 = a, x1 = a + h = (a + b) / 2, x2 = b
        I = (h / 3) * [f(x0) + 4 * f(x1) + f(x2)]

    Returns:
        integral_value, x_points, y_points
    """
    h = (b - a) / 2.0
    x_pts = np.array([a, a + h, b], dtype=float)
    y_pts = np.array([float(f(x)) for x in x_pts], dtype=float)

    integral = (h / 3.0) * (y_pts[0] + 4.0 * y_pts[1] + y_pts[2])
    return float(integral), x_pts, y_pts


def simpson_multiple(f, a: float, b: float, n: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Multiple-segment Simpson's 1/3 Rule:
        n must be an even integer >= 2.
        h = (b - a) / n
        I = (h / 3) * [f(x0) + 4 * sum_{i=1,3,...}^{n-1} f(x_i) + 2 * sum_{i=2,4,...}^{n-2} f(x_i) + f(xn)]

    Returns:
        integral_value, x_points, y_points
    """
    if n < 2:
        raise ValueError("Simpson's 1/3 Rule requires at least 2 segments.")
    if n % 2 != 0:
        raise ValueError(f"Simpson's 1/3 Rule requires an even number of segments. Received n = {n}.")

    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts = np.array([float(f(x)) for x in x_pts], dtype=float)

    odd_sum = np.sum(y_pts[1:n:2])
    even_sum = np.sum(y_pts[2:n-1:2]) if n > 2 else 0.0

    integral = (h / 3.0) * (y_pts[0] + 4.0 * odd_sum + 2.0 * even_sum + y_pts[-1])
    return float(integral), x_pts, y_pts


def simpson_dataset(x_data: np.ndarray, y_data: np.ndarray) -> float:
    """
    Simpson's 1/3 rule on equally-spaced discrete dataset points:
        len(x_data) must be odd (even number of segments n = len(x_data) - 1).
    """
    n = len(x_data) - 1
    if n < 2 or n % 2 != 0:
        raise ValueError(f"Simpson's 1/3 rule requires an even number of segments (odd number of points). Got {n} segments.")

    h = x_data[1] - x_data[0]
    odd_sum = np.sum(y_data[1:n:2])
    even_sum = np.sum(y_data[2:n-1:2]) if n > 2 else 0.0

    integral = (h / 3.0) * (y_data[0] + 4.0 * odd_sum + 2.0 * even_sum + y_data[-1])
    return float(integral)


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
                elif "Segment" in headers[i] or "Pair" in headers[i] or "Step" in headers[i]:
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
# DETAILED PAIR-OF-SEGMENTS BREAKDOWN TABLE
# ================================================================

def generate_segment_table(f, a: float, b: float, n: int, output_dir: Path | None = None) -> pd.DataFrame:
    """Generate breakdown table for each 2-segment parabolic interval."""
    if n % 2 != 0:
        raise ValueError("n must be even.")

    h = (b - a) / n
    num_pairs = n // 2
    rows = []
    accumulated = 0.0

    for p in range(num_pairs):
        i = 2 * p
        x0 = a + i * h
        x1 = x0 + h
        x2 = x0 + 2 * h
        y0 = float(f(x0))
        y1 = float(f(x1))
        y2 = float(f(x2))
        pair_area = (h / 3.0) * (y0 + 4.0 * y1 + y2)
        accumulated += pair_area

        rows.append({
            "Pair p": p + 1,
            "Segments": f"[{i}, {i+2}]",
            "x0": x0,
            "x1 (mid)": x1,
            "x2": x2,
            "f(x0)": y0,
            "f(x1)": y1,
            "f(x2)": y2,
            "Pair Area": pair_area,
            "Running Total": accumulated,
        })

    df = pd.DataFrame(rows)
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_dir / f"simpson_pairs_n{n}.csv", index=False)
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
    """Generate summary table across even segment counts n."""
    rows = []
    prev_val = None

    for n in segment_counts:
        if n % 2 != 0:
            continue
        val, _, _ = simpson_multiple(f, a, b, n)
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
        df.to_csv(output_dir / "simpson_summary.csv", index=False)
    return df


# ================================================================
# PLOTTING FUNCTIONS (PARABOLIC INTERPOLANTS)
# ================================================================

def plot_simpson(f, a: float, b: float, n: int, output_dir: Path | None = None):
    """Plot function along with Simpson's 1/3 piecewise quadratic parabolas."""
    val, x_pts, y_pts = simpson_multiple(f, a, b, n)

    x_fine = np.linspace(a, b, 400)
    y_fine = [float(f(x)) for x in x_fine]

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, "k-", linewidth=2.5, label="f(x) Exact Integrand")

    # Fit quadratic polynomial over each pair of segments
    colors = ["#4285f4", "#34a853", "#fbbc05", "#ea4335", "#9c27b0"]
    for p in range(n // 2):
        i = 2 * p
        xp = x_pts[i:i + 3]
        yp = y_pts[i:i + 3]

        poly = np.polyfit(xp, yp, deg=2)
        x_quad = np.linspace(xp[0], xp[2], 100)
        y_quad = np.polyval(poly, x_quad)

        col = colors[p % len(colors)]
        plt.fill_between(x_quad, 0, y_quad, color=col, alpha=0.25)
        plt.plot(x_quad, y_quad, color=col, linestyle="--", linewidth=2, label=f"Parabola Pair {p+1}" if p < 4 else None)

    plt.plot(x_pts, y_pts, "ro", markersize=6, label=f"Grid points (n={n})")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

    plt.title(f"Simpson's 1/3 Rule Approximation (n = {n} segments, Area = {val:.4f})", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / f"simpson_plot_n{n}.png", dpi=300, bbox_inches="tight")
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
        "default_n": [2, 4, 6, 8, 10],
    }


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    prob = get_slide_default_problem()
    out_dir = Path("integration_results/simpson")

    print("\n==================================================")
    print("     SIMPSON'S 1/3 RULE NUMERICAL INTEGRATION     ")
    print("==================================================")
    print("1. Slide Example (Rocket: t=8 to 30)")
    print("2. Custom Function Input")
    print("3. Discrete Dataset Integration (even segments)")
    print("4. Exit")

    choice = input("\nSelect an option (1, 2, 3, 4): ").strip()
    if choice == "4":
        return

    if choice == "1":
        f = prob["func"]
        a, b = prob["a"], prob["b"]
        exact = prob["exact"]

        print(f"\nIntegrating Rocket Function from a={a} to b={b} (Exact = {exact})")

        # 1. 2-Segment Single Application
        val_2seg, _, _ = simpson_single_2seg(f, a, b)
        err_2seg = exact - val_2seg
        pct_2seg = abs(err_2seg) / exact * 100
        print(f"\n[2-Segment Single Application]: Value = {val_2seg:.4f}, True Error Et = {err_2seg:.4f}, |et| = {pct_2seg:.4f}%")

        # 2. Summary Table across multiple even segments
        summary_df = generate_summary_table(f, a, b, prob["default_n"], exact, out_dir)
        print(format_table_console(summary_df, "MULTIPLE SEGMENT SIMPSON'S 1/3 RULE SUMMARY"))

        # 3. Detailed Step Table for n=4
        detail_df = generate_segment_table(f, a, b, 4, out_dir)
        print(format_table_console(detail_df, "STEP-BY-STEP PAIR BREAKDOWN (n = 4)"))

        plot_simpson(f, a, b, 4, out_dir)
        print(f"\nResults saved to: {out_dir.resolve()}")

    elif choice == "2":
        expr_str = input("Enter f(x) (e.g., 300*x / (1 + exp(x))): ").strip()
        x_sym = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={"x": x_sym})
        fn = sp.lambdify(x_sym, expr, modules=["numpy"])
        f = lambda val: float(fn(val))

        a = float(input("Enter lower limit a: ").strip())
        b = float(input("Enter upper limit b: ").strip())
        n = int(input("Enter even number of segments n (e.g. 4): ").strip())
        if n % 2 != 0:
            print("Error: n must be even!")
            return

        val, _, _ = simpson_multiple(f, a, b, n)
        print(f"\nSimpson's 1/3 integral (n={n}) = {val:.6f}")
        detail_df = generate_segment_table(f, a, b, n, out_dir)
        print(format_table_console(detail_df, f"STEP-BY-STEP BREAKDOWN (n = {n})"))
        plot_simpson(f, a, b, n, out_dir)

    elif choice == "3":
        x_input = input("Enter x values separated by commas: ").strip()
        y_input = input("Enter y values separated by commas: ").strip()
        x_data = np.array([float(x.strip()) for x in x_input.split(",")])
        y_data = np.array([float(y.strip()) for y in y_input.split(",")])

        result = simpson_dataset(x_data, y_data)
        print(f"\nSimpson's 1/3 Integral over {len(x_data) - 1} segments = {result:.6f}")


if __name__ == "__main__":
    run_cli()
