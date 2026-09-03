"""
direct_interpolation.py
=======================
Direct Method of Interpolation:
a. Direct method of interpolation:
   i. Linear (Order 1, 2 data points):
      v(t) = a0 + a1*t
   ii. Quadratic (Order 2, 3 data points):
      v(t) = a0 + a1*t + a2*t^2
   iii. Cubic (Order 3, 4 data points):
      v(t) = a0 + a1*t + a2*t^2 + a3*t^3

Zero built-in equation solvers used:
- Handwritten Gaussian Elimination with Partial Pivoting for Vandermonde linear systems.
- Step-by-step matrix displays and polynomial evaluation.
- Relative approximate error (|ea|%).
- Analytical derivative (acceleration) & analytical integral (distance).
- Full ASCII table formatting and visualization plots.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# HANDWRITTEN GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING
# ================================================================

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian Elimination with partial pivoting.
    Strictly handwritten without np.linalg.solve or scipy.
    """
    n = len(b)
    # Form augmented matrix [A | b]
    M = np.hstack([A.astype(float), b.astype(float).reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for p in range(n):
        # Find pivot row with largest absolute value in current column
        max_row = p + int(np.argmax(np.abs(M[p:, p])))
        if max_row != p:
            M[[p, max_row]] = M[[max_row, p]]

        pivot = M[p, p]
        if abs(pivot) < 1e-14:
            raise ValueError(f"Matrix is singular or near-singular at pivot row {p}.")

        for i in range(p + 1, n):
            factor = M[i, p] / pivot
            M[i, p:] -= factor * M[p, p:]

    # Back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:])) / M[i, i]

    return x


# ================================================================
# POINT SELECTION HELPER (TRUE BRACKETING & CONTIGUOUS EXPANSION)
# ================================================================

def select_bracketed_closest_points(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Selects num_points points from (x_data, y_data) that strictly bracket
    x_target whenever possible, expanding contiguously by picking the nearest neighbor.
    
    Returns:
        (x_selected, y_selected, is_extrapolated)
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    n = len(x_arr)
    if n < num_points:
        raise ValueError(f"Need at least {num_points} distinct data points. Got {n}.")

    # Sort in ascending order of x
    sort_idx = np.argsort(x_arr)
    x_s = x_arr[sort_idx]
    y_s = y_arr[sort_idx]

    # Validate distinct x-values
    diffs = np.diff(x_s)
    if np.any(diffs <= 1e-14):
        dup_indices = np.where(diffs <= 1e-14)[0]
        dup_vals = [f"{x_s[idx]:g}" for idx in dup_indices]
        raise ValueError(f"Duplicate x-values detected at: {', '.join(dup_vals)}. All x-coordinates must be distinct.")

    # Check for extrapolation (target outside dataset domain)
    if x_target <= x_s[0]:
        return x_s[:num_points].copy(), y_s[:num_points].copy(), True
    if x_target >= x_s[-1]:
        return x_s[-num_points:].copy(), y_s[-num_points:].copy(), True

    # Interior Bracketing:
    # 1. Locate fundamental bracket interval [x_i, x_{i+1}] enclosing x_target
    i = int(np.searchsorted(x_s, x_target)) - 1
    L, R = i, i + 1  # Window containing at least 2 points bracketing x_target

    # 2. Expand window to num_points by iteratively choosing closer adjacent neighbor
    while (R - L + 1) < num_points:
        can_left = (L > 0)
        can_right = (R < n - 1)
        if can_left and can_right:
            d_left = abs(x_s[L - 1] - x_target)
            d_right = abs(x_s[R + 1] - x_target)
            if d_left <= d_right:
                L -= 1
            else:
                R += 1
        elif can_left:
            L -= 1
        elif can_right:
            R += 1
        else:
            break

    return x_s[L:R + 1].copy(), y_s[L:R + 1].copy(), False


def select_closest_points(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper returning (x_sel, y_sel)."""
    xs, ys, _ = select_bracketed_closest_points(x_data, y_data, x_target, num_points)
    return xs, ys


# ================================================================
# DIRECT METHOD INTERPOLATOR
# ================================================================

@dataclass
class DirectInterpolationResult:
    order: int
    name: str
    x_points: np.ndarray
    y_points: np.ndarray
    x_target: float
    coefficients: np.ndarray  # [a0, a1, a2, ...]
    interpolated_value: float
    approx_error_percent: float | None = None
    is_extrapolated: bool = False
    matrix_A: np.ndarray | None = None
    vector_b: np.ndarray | None = None

    @property
    def polynomial_string(self) -> str:
        terms = []
        for i, c in enumerate(self.coefficients):
            if i == 0:
                terms.append(f"{c:.6f}")
            elif i == 1:
                terms.append(f"+ {c:.6f}*x" if c >= 0 else f"- {abs(c):.6f}*x")
            else:
                terms.append(f"+ {c:.6f}*x^{i}" if c >= 0 else f"- {abs(c):.6f}*x^{i}")
        return "P(x) = " + " ".join(terms)

    def evaluate(self, x_val: float | np.ndarray) -> float | np.ndarray:
        result = 0.0
        for i, c in enumerate(self.coefficients):
            result += c * (x_val ** i)
        return result

    def derivative(self, x_val: float) -> float:
        """Exact analytical first derivative P'(x) (e.g. acceleration from velocity profile)."""
        res = 0.0
        for i in range(1, len(self.coefficients)):
            res += i * self.coefficients[i] * (x_val ** (i - 1))
        return res

    def integrate(self, a: float, b: float) -> float:
        """Exact analytical definite integral of P(x) from a to b (e.g. distance from velocity profile)."""
        res = 0.0
        for i, c in enumerate(self.coefficients):
            res += (c / (i + 1)) * (b ** (i + 1) - a ** (i + 1))
        return res


def direct_interpolate_order(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    x_target: float,
    order: int,
    prev_value: float | None = None,
    is_extrapolated: bool = False,
) -> DirectInterpolationResult:
    """
    Fits polynomial of degree `order` through `order + 1` points.
    order = 1 (Linear, 2 points)
    order = 2 (Quadratic, 3 points)
    order = 3 (Cubic, 4 points)
    """
    num_pts = order + 1
    if len(x_pts) != num_pts or len(y_pts) != num_pts:
        raise ValueError(f"Order {order} requires exactly {num_pts} data points. Got {len(x_pts)}.")

    # Construct Vandermonde matrix V[i, j] = (x_pts[i])^j
    V = np.zeros((num_pts, num_pts), dtype=float)
    for j in range(num_pts):
        V[:, j] = x_pts ** j

    # Solve V * a = y using handwritten Gaussian elimination
    coeffs = gaussian_elimination(V, y_pts)

    # Evaluate at x_target
    val = float(sum(coeffs[i] * (x_target ** i) for i in range(num_pts)))

    # Compute relative approximate error
    ea = None
    if prev_value is not None:
        ea = abs(val - prev_value) / abs(val) * 100.0 if val != 0 else 0.0

    names = {1: "Linear (1st Order)", 2: "Quadratic (2nd Order)", 3: "Cubic (3rd Order)"}
    name = names.get(order, f"{order}-th Order")

    return DirectInterpolationResult(
        order=order,
        name=name,
        x_points=x_pts,
        y_points=y_pts,
        x_target=x_target,
        coefficients=coeffs,
        interpolated_value=val,
        approx_error_percent=ea,
        is_extrapolated=is_extrapolated,
        matrix_A=V,
        vector_b=y_pts,
    )


# ================================================================
# COMPLETE DIRECT METHOD SUITE (DYNAMIC DATASET SIZE ADAPTATION)
# ================================================================

def run_direct_interpolation_suite(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    output_dir: Path | None = None,
) -> list[DirectInterpolationResult]:
    """
    Runs available interpolation orders based on dataset size:
    - 2 points: Linear only
    - 3 points: Linear & Quadratic
    - >= 4 points: Linear, Quadratic, and Cubic
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    n = len(x_arr)
    if n < 2:
        raise ValueError(f"At least 2 data points required for interpolation. Provided: {n}")

    # Maximum feasible order given dataset size (up to cubic)
    max_order = min(3, n - 1)
    available_orders = list(range(1, max_order + 1))

    results = []
    prev_val = None

    for order in available_orders:
        x_sel, y_sel, is_ext = select_bracketed_closest_points(x_arr, y_arr, x_target, num_points=order + 1)
        res = direct_interpolate_order(x_sel, y_sel, x_target, order, prev_val, is_extrapolated=is_ext)
        results.append(res)
        prev_val = res.interpolated_value

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        summary_df = generate_summary_dataframe(results)
        summary_df.to_csv(output_dir / "direct_interpolation_summary.csv", index=False)

    return results


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
                if "%" in headers[i] or "|ea|" in headers[i]:
                    s = f"{val:.5f}%" if not math.isnan(val) else "---"
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


def print_step_by_step_direct(res: DirectInterpolationResult):
    """Prints full handwritten system of equations and solution steps."""
    print(f"\n--- {res.name} Direct Interpolation (x* = {res.x_target:g}) ---")
    print("Selected Data Points:")
    for i, (xp, yp) in enumerate(zip(res.x_points, res.y_points)):
        print(f"  Point {i}: ({xp:g}, {yp:g})")

    print("\nSet of Linear Equations [V] {a} = {y}:")
    for i in range(len(res.x_points)):
        terms = []
        for j in range(len(res.x_points)):
            if j == 0:
                terms.append("a0")
            elif j == 1:
                terms.append(f"a1*({res.x_points[i]:g})")
            else:
                terms.append(f"a{j}*({res.x_points[i]:g})^{j}")
        row_str = " + ".join(terms)
        print(f"  Equation {i+1}: {row_str} = {res.y_points[i]:g}")

    print("\nSolved Polynomial Coefficients (via Gaussian Elimination):")
    for j, c in enumerate(res.coefficients):
        print(f"  a{j} = {c:.6f}")

    print(f"\nConstructed Polynomial:\n  {res.polynomial_string}")
    print(f"Interpolated Value at x = {res.x_target:g}:\n  f({res.x_target:g}) = {res.interpolated_value:.4f}")
    if res.approx_error_percent is not None:
        print(f"Absolute Relative Approximate Error |ea|: {res.approx_error_percent:.5f}%")


def generate_summary_dataframe(results: list[DirectInterpolationResult]) -> pd.DataFrame:
    """Generate summary table across Linear, Quadratic, and Cubic orders."""
    rows = []
    for r in results:
        rows.append({
            "Order": r.order,
            "Method Type": r.name,
            "Points Used": ", ".join(f"{x:g}" for x in r.x_points),
            f"f({r.x_target:g})": r.interpolated_value,
            "|ea| (%)": r.approx_error_percent if r.approx_error_percent is not None else np.nan,
            "Polynomial Expression": r.polynomial_string,
        })
    return pd.DataFrame(rows)


# ================================================================
# PLOTTING FUNCTIONS
# ================================================================

def plot_direct_interpolation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    results: list[DirectInterpolationResult],
    x_target: float,
    output_dir: Path | None = None,
):
    """Plots data points and Linear, Quadratic, and Cubic polynomial curves."""
    plt.figure(figsize=(10, 6))

    # Plot full dataset points
    plt.plot(x_all, y_all, "ko", markersize=7, label="Original Data Points", zorder=5)

    # Plot curves for each order
    x_min = min(r.x_points[0] for r in results)
    x_max = max(r.x_points[-1] for r in results)
    margin = (x_max - x_min) * 0.1
    x_fine = np.linspace(max(0, x_min - margin), x_max + margin, 300)

    colors = ["#1a73e8", "#34a853", "#ea4335"]
    styles = ["--", "-.", "-"]

    for res, col, st in zip(results, colors, styles):
        y_fine = res.evaluate(x_fine)
        plt.plot(x_fine, y_fine, color=col, linestyle=st, linewidth=2, label=f"{res.name}: {res.interpolated_value:.2f}")

    # Mark query target
    for res, col in zip(results, colors):
        plt.plot(x_target, res.interpolated_value, marker="*", markersize=11, color=col)

    plt.axvline(x_target, color="gray", linestyle=":", alpha=0.7, label=f"Query Point x* = {x_target:g}")
    plt.title(f"Direct Method of Interpolation (Target x = {x_target:g})", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "direct_interpolation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# DEFAULT LECTURE SLIDE PROBLEM (ROCKET VELOCITY)
# ================================================================

def get_slide_default_problem():
    """
    Lecture Slide Example:
        t = [0, 10, 15, 20, 22.5, 30]
        v = [0, 227.04, 362.78, 517.35, 602.97, 901.67]
        Target: t = 16.0 s
    """
    t = np.array([0.0, 10.0, 15.0, 20.0, 22.5, 30.0], dtype=float)
    v = np.array([0.0, 227.04, 362.78, 517.35, 602.97, 901.67], dtype=float)
    return {
        "title": "Rocket Velocity vs Time (Lecture Slide Example)",
        "x_name": "t (s)",
        "y_name": "v(t) (m/s)",
        "x": t,
        "y": v,
        "target": 16.0,
    }


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    prob = get_slide_default_problem()
    out_dir = Path("interpolation_results/direct")

    print("\n==================================================================")
    print("           DIRECT METHOD OF NUMERICAL INTERPOLATION               ")
    print("==================================================================")
    print("1. Slide Benchmark Problem (Rocket Velocity at t = 16 s)")
    print("2. Enter Custom Dataset and Query Point")
    print("3. Exit")

    choice = input("\nSelect an option (1, 2, 3): ").strip()
    if choice == "3":
        return

    if choice == "1":
        x_data = prob["x"]
        y_data = prob["y"]
        x_target = prob["target"]
        title = prob["title"]
    elif choice == "2":
        x_str = input("Enter x values separated by commas/spaces: ").strip()
        y_str = input("Enter y values separated by commas/spaces: ").strip()
        x_data = np.array([float(x) for x in x_str.replace(",", " ").split() if x])
        y_data = np.array([float(y) for y in y_str.replace(",", " ").split() if y])
        x_target = float(input("Enter target x value to interpolate: ").strip())
        title = "Custom User Dataset"
    else:
        print("Invalid choice.")
        return

    print(f"\n>>> Running Direct Interpolation on: {title}")
    results = run_direct_interpolation_suite(x_data, y_data, x_target, out_dir)

    # Step-by-step breakdown for each order
    for r in results:
        print_step_by_step_direct(r)

    # Summary Table across orders
    df_summary = generate_summary_dataframe(results)
    print(format_table_console(df_summary, f"DIRECT INTERPOLATION SUMMARY TABLE (Target = {x_target:g})"))

    # Slide Extra: Acceleration and Distance for Cubic order
    if len(results) >= 3:
        cubic_res = results[2]
        deriv_val = cubic_res.derivative(x_target)
        int_val = cubic_res.integrate(11.0, 16.0)
        print("\n--- Physical Derivatives & Integrals from Cubic Profile (Slide Extra) ---")
        print(f"  Derivative at x = {x_target:g} (e.g. Acceleration a(16)): {deriv_val:.3f}")
        print(f"  Definite Integral from 11 to 16 (e.g. Distance s(16)-s(11)): {int_val:.2f}")

    # Plot
    plot_direct_interpolation(x_data, y_data, results, x_target, out_dir)
    print(f"\nAll Direct method tables and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
