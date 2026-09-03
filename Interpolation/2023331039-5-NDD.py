"""
newton_divided_difference.py
============================
Newton's Divided Difference Method of Interpolation:
c. Newton's Divided Difference method:
   i. Linear (Order 1, 2 points):
      f1(x) = b0 + b1*(x - x0)
      where b0 = f[x0], b1 = f[x1, x0] = (f[x1] - f[x0]) / (x1 - x0)
   ii. Quadratic (Order 2, 3 points):
      f2(x) = b0 + b1*(x - x0) + b2*(x - x0)*(x - x1)
      where b2 = f[x2, x1, x0]
   iii. Cubic (Order 3, 4 points):
      f3(x) = b0 + b1*(x - x0) + b2*(x - x0)*(x - x1) + b3*(x - x0)*(x - x1)*(x - x2)
      where b3 = f[x3, x2, x1, x0]

Zero built-in equation solvers used:
- Handwritten 2D triangular divided difference table construction.
- Formatted triangular table display in ASCII.
- Step-by-step coefficient tracking (b0, b1, b2, b3).
- Relative approximate error (|ea|%).
- Analytical derivative & integral.
- Full ASCII table formatting and visualization plots.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

_mod_dm = importlib.import_module("2023331039-5-DM")
select_closest_points = _mod_dm.select_closest_points
select_bracketed_closest_points = _mod_dm.select_bracketed_closest_points
format_table_console = _mod_dm.format_table_console


# ================================================================
# HANDWRITTEN DIVIDED DIFFERENCE TABLE BUILDER
# ================================================================

def build_divided_difference_table(x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    """
    Constructs the n x n divided difference table:
    Column 0: f[x_i]
    Column 1: f[x_i, x_{i+1}] (1st divided difference)
    Column 2: f[x_i, x_{i+1}, x_{i+2}] (2nd divided difference)
    ...
    Column n-1: (n-1)-th divided difference
    """
    n = len(x_pts)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y_pts.astype(float)

    for col in range(1, n):
        for row in range(n - col):
            num = table[row + 1, col - 1] - table[row, col - 1]
            den = x_pts[row + col] - x_pts[row]
            if abs(den) < 1e-15:
                raise ValueError(f"Duplicate x values detected: x[{row}] == x[{row+col}].")
            table[row, col] = num / den

    return table


# ================================================================
# NEWTON INTERPOLATION RESULT CLASS
# ================================================================

@dataclass
class NewtonTermEvaluation:
    order: int
    coeff_b: float
    product_str: str
    evaluated_product: float
    term_value: float


def newton_to_standard_poly(x_pts: np.ndarray, b_coeffs: np.ndarray) -> np.ndarray:
    """
    Expands Newton's divided difference polynomial:
    P(x) = b0 + b1*(x - x0) + b2*(x - x0)*(x - x1) + ...
    into standard canonical polynomial coefficients [a0, a1, ..., an]
    where P(x) = sum(a_i * x^i).
    """
    n = len(b_coeffs)
    current_poly = np.array([b_coeffs[0]], dtype=float)
    term_poly = np.array([1.0], dtype=float)

    for i in range(1, n):
        term_poly = np.convolve(term_poly, [-x_pts[i - 1], 1.0])
        padded_current = np.pad(current_poly, (0, len(term_poly) - len(current_poly)))
        current_poly = padded_current + b_coeffs[i] * term_poly

    return current_poly


@dataclass
class NewtonInterpolationResult:
    order: int
    name: str
    x_points: np.ndarray
    y_points: np.ndarray
    x_target: float
    table: np.ndarray          # Full triangular divided difference table
    coefficients_b: np.ndarray  # [b0, b1, b2, ...]
    term_evaluations: list[NewtonTermEvaluation]
    interpolated_value: float
    approx_error_percent: float | None = None
    is_extrapolated: bool = False
    canonical_coefficients: np.ndarray | None = None

    def __post_init__(self):
        if self.canonical_coefficients is None:
            self.canonical_coefficients = newton_to_standard_poly(self.x_points, self.coefficients_b)

    @property
    def newton_formula_string(self) -> str:
        terms = []
        for i, b in enumerate(self.coefficients_b):
            if i == 0:
                terms.append(f"{b:.6f}")
            else:
                factors = "".join(f"(x - {self.x_points[k]:g})" for k in range(i))
                sign = "+ " if b >= 0 else "- "
                terms.append(f"{sign}{abs(b):.6f}*{factors}")
        return "P(x) = " + " ".join(terms)

    @property
    def polynomial_string(self) -> str:
        """Canonical expanded polynomial string P(x) = a0 + a1*x + ..."""
        terms = []
        for i, c in enumerate(self.canonical_coefficients):
            if i == 0:
                terms.append(f"{c:.6f}")
            elif i == 1:
                terms.append(f"+ {c:.6f}*x" if c >= 0 else f"- {abs(c):.6f}*x")
            else:
                terms.append(f"+ {c:.6f}*x^{i}" if c >= 0 else f"- {abs(c):.6f}*x^{i}")
        return "P(x) = " + " ".join(terms)

    def evaluate(self, x_val: float | np.ndarray) -> float | np.ndarray:
        """Horner-like evaluation of Newton's divided difference polynomial."""
        n = len(self.coefficients_b)
        if isinstance(x_val, np.ndarray):
            result = np.full_like(x_val, self.coefficients_b[-1], dtype=float)
            for i in range(n - 2, -1, -1):
                result = result * (x_val - self.x_points[i]) + self.coefficients_b[i]
            return result
        else:
            result = float(self.coefficients_b[-1])
            for i in range(n - 2, -1, -1):
                result = result * (x_val - self.x_points[i]) + self.coefficients_b[i]
            return float(result)

    def derivative(self, x_val: float) -> float:
        """Exact analytical first derivative P'(x) from expanded canonical polynomial."""
        res = 0.0
        for i in range(1, len(self.canonical_coefficients)):
            res += i * self.canonical_coefficients[i] * (x_val ** (i - 1))
        return res

    def integrate(self, a: float, b: float) -> float:
        """Exact analytical definite integral of P(x) from a to b."""
        res = 0.0
        for i, c in enumerate(self.canonical_coefficients):
            res += (c / (i + 1)) * (b ** (i + 1) - a ** (i + 1))
        return res


def newton_interpolate_order(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    x_target: float,
    order: int,
    prev_value: float | None = None,
    is_extrapolated: bool = False,
) -> NewtonInterpolationResult:
    """
    Fits Newton polynomial of degree `order` using `order + 1` points.
    Builds divided difference table and evaluates each term b_k * prod(x - x_j).
    """
    num_pts = order + 1
    if len(x_pts) != num_pts or len(y_pts) != num_pts:
        raise ValueError(f"Order {order} requires exactly {num_pts} data points. Got {len(x_pts)}.")

    table = build_divided_difference_table(x_pts, y_pts)
    coeffs_b = table[0, :num_pts].copy()

    # Step-by-step term evaluation at x_target
    term_evals = []
    total_val = 0.0
    accum_prod = 1.0

    for i in range(num_pts):
        b = coeffs_b[i]
        if i == 0:
            prod_str = "1"
            term_prod = 1.0
        else:
            accum_prod *= (x_target - x_pts[i - 1])
            term_prod = accum_prod
            prod_str = " * ".join(f"({x_target:g} - {x_pts[k]:g})" for k in range(i))

        term_val = b * term_prod
        total_val += term_val

        term_evals.append(NewtonTermEvaluation(
            order=i,
            coeff_b=b,
            product_str=prod_str,
            evaluated_product=term_prod,
            term_value=term_val,
        ))

    ea = None
    if prev_value is not None:
        ea = abs(total_val - prev_value) / abs(total_val) * 100.0 if total_val != 0 else 0.0

    names = {1: "Linear (1st Order)", 2: "Quadratic (2nd Order)", 3: "Cubic (3rd Order)"}
    name = names.get(order, f"{order}-th Order")

    return NewtonInterpolationResult(
        order=order,
        name=name,
        x_points=x_pts,
        y_points=y_pts,
        x_target=x_target,
        table=table,
        coefficients_b=coeffs_b,
        term_evaluations=term_evals,
        interpolated_value=total_val,
        approx_error_percent=ea,
        is_extrapolated=is_extrapolated,
    )


# ================================================================
# COMPLETE NEWTON SUITE (DYNAMIC DATASET SIZE ADAPTATION)
# ================================================================

def run_newton_interpolation_suite(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    output_dir: Path | None = None,
) -> list[NewtonInterpolationResult]:
    """Runs available Newton interpolation orders based on dataset size."""
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    n = len(x_arr)
    if n < 2:
        raise ValueError(f"At least 2 data points required for interpolation. Provided: {n}")

    # select_bracketed_closest_points imported from 2023331039-5-DM at module level

    max_order = min(3, n - 1)
    available_orders = list(range(1, max_order + 1))

    results = []
    prev_val = None

    for order in available_orders:
        x_sel, y_sel, is_ext = select_bracketed_closest_points(x_arr, y_arr, x_target, num_points=order + 1)
        res = newton_interpolate_order(x_sel, y_sel, x_target, order, prev_val, is_extrapolated=is_ext)
        results.append(res)
        prev_val = res.interpolated_value

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        summary_df = generate_summary_dataframe(results)
        summary_df.to_csv(output_dir / "newton_interpolation_summary.csv", index=False)

    return results


# ================================================================
# FORMATTED TRIANGULAR TABLE & STEP DISPLAY
# ================================================================

def format_divided_difference_table(x_pts: np.ndarray, table: np.ndarray) -> str:
    """Formats the upper-triangular divided difference table in clean ASCII."""
    n = len(x_pts)
    col_names = ["i", "x_i", "f[x_i]"]
    for k in range(1, n):
        suffix = "st" if k == 1 else ("nd" if k == 2 else ("rd" if k == 3 else "th"))
        col_names.append(f"{k}{suffix} Diff")

    rows = []
    for r in range(n):
        row = [f"{r}", f"{x_pts[r]:.4f}"]
        for c in range(n):
            if c < n - r:
                val = table[r, c]
                if abs(val) < 1e-4 and val != 0:
                    row.append(f"{val:.5e}")
                else:
                    row.append(f"{val:.4f}")
            else:
                row.append("")
        rows.append(row)

    widths = [max(len(col_names[c]), max(len(rows[r][c]) for r in range(n))) for c in range(len(col_names))]
    hdr_str = " | ".join(col_names[c].rjust(widths[c]) for c in range(len(col_names)))
    sep_str = "-+-".join("-" * widths[c] for c in range(len(col_names)))

    out = [hdr_str, sep_str]
    for r in rows:
        out.append(" | ".join(r[c].rjust(widths[c]) for c in range(len(col_names))))
    return "\n".join(out)


def print_step_by_step_newton(res: NewtonInterpolationResult):
    """Prints full triangular divided difference table and coefficient steps."""
    print(f"\n--- {res.name} Newton Divided Difference Interpolation (x* = {res.x_target:g}) ---")
    print("Selected Data Points:")
    for i, (xp, yp) in enumerate(zip(res.x_points, res.y_points)):
        print(f"  Point {i}: ({xp:g}, {yp:g})")

    print("\nDivided Difference Table:")
    print(format_divided_difference_table(res.x_points, res.table))

    print("\nNewton Polynomial Coefficients (Diagonal b_k):")
    for k, b in enumerate(res.coefficients_b):
        suffix = "st" if k == 1 else ("nd" if k == 2 else ("rd" if k == 3 else "th"))
        col_desc = "f[x0]" if k == 0 else f"{k}{suffix} diff f[x0...x{k}]"
        print(f"  b{k} = {b:.6f}  ({col_desc})")

    print(f"\nConstructed Newton Form:\n  {res.newton_formula_string}")

    print(f"\nStep-by-Step Evaluation at x = {res.x_target:g}:")
    for t in res.term_evaluations:
        print(f"  Order {t.order}: b{t.order} * [{t.product_str}] = {t.coeff_b:.6f} * {t.evaluated_product:.4f} = {t.term_value:.4f}")

    print(f"Interpolated Value at x = {res.x_target:g}:\n  f({res.x_target:g}) = {res.interpolated_value:.4f}")
    if res.approx_error_percent is not None:
        print(f"Absolute Relative Approximate Error |ea|: {res.approx_error_percent:.5f}%")


def generate_summary_dataframe(results: list[NewtonInterpolationResult]) -> pd.DataFrame:
    """Generate summary table across Linear, Quadratic, and Cubic orders."""
    rows = []
    for r in results:
        coeffs_str = ", ".join(f"b{k}={b:.4f}" for k, b in enumerate(r.coefficients_b))
        rows.append({
            "Order": r.order,
            "Method Type": r.name,
            "Points Used": ", ".join(f"{x:g}" for x in r.x_points),
            f"f({r.x_target:g})": r.interpolated_value,
            "|ea| (%)": r.approx_error_percent if r.approx_error_percent is not None else np.nan,
            "Divided Diff Coeffs b_k": coeffs_str,
        })
    return pd.DataFrame(rows)


# ================================================================
# PLOTTING FUNCTIONS
# ================================================================

def plot_newton_interpolation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    results: list[NewtonInterpolationResult],
    x_target: float,
    output_dir: Path | None = None,
):
    """Plots data points and Linear, Quadratic, and Cubic Newton curves."""
    plt.figure(figsize=(10, 6))

    plt.plot(x_all, y_all, "ko", markersize=7, label="Original Data Points", zorder=5)

    x_min = min(r.x_points[0] for r in results)
    x_max = max(r.x_points[-1] for r in results)
    margin = (x_max - x_min) * 0.1
    x_fine = np.linspace(max(0, x_min - margin), x_max + margin, 300)

    colors = ["#1a73e8", "#34a853", "#ea4335"]
    styles = ["--", "-.", "-"]

    for res, col, st in zip(results, colors, styles):
        y_fine = res.evaluate(x_fine)
        plt.plot(x_fine, y_fine, color=col, linestyle=st, linewidth=2, label=f"{res.name}: {res.interpolated_value:.2f}")

    for res, col in zip(results, colors):
        plt.plot(x_target, res.interpolated_value, marker="*", markersize=11, color=col)

    plt.axvline(x_target, color="gray", linestyle=":", alpha=0.7, label=f"Query Point x* = {x_target:g}")
    plt.title(f"Newton's Divided Difference Interpolation (Target x = {x_target:g})", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "newton_interpolation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    t = np.array([0.0, 10.0, 15.0, 20.0, 22.5, 30.0], dtype=float)
    v = np.array([0.0, 227.04, 362.78, 517.35, 602.97, 901.67], dtype=float)
    out_dir = Path("interpolation_results/newton")

    print("\n==================================================================")
    print("   NEWTON'S DIVIDED DIFFERENCE NUMERICAL INTERPOLATION            ")
    print("==================================================================")
    print("1. Slide Benchmark Problem (Rocket Velocity at t = 16 s)")
    print("2. Enter Custom Dataset and Query Point")
    print("3. Exit")

    choice = input("\nSelect an option (1, 2, 3): ").strip()
    if choice == "3":
        return

    if choice == "1":
        x_data = t
        y_data = v
        x_target = 16.0
        title = "Rocket Velocity vs Time (Lecture Slide Example)"
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

    print(f"\n>>> Running Newton Divided Difference Interpolation on: {title}")
    results = run_newton_interpolation_suite(x_data, y_data, x_target, out_dir)

    for r in results:
        print_step_by_step_newton(r)

    df_summary = generate_summary_dataframe(results)
    print(format_table_console(df_summary, f"NEWTON INTERPOLATION SUMMARY TABLE (Target = {x_target:g})"))

    if len(results) >= 3:
        cubic_res = results[2]
        deriv_val = cubic_res.derivative(x_target)
        int_val = cubic_res.integrate(11.0, 16.0)
        print("\n--- Physical Derivatives & Integrals from Cubic Profile (Slide Extra) ---")
        print(f"  Derivative at x = {x_target:g} (Acceleration a(16)): {deriv_val:.3f}")
        print(f"  Definite Integral from 11 to 16 (Distance s(16)-s(11)): {int_val:.2f}")

    plot_newton_interpolation(x_data, y_data, results, x_target, out_dir)
    print(f"\nAll Newton method tables and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
