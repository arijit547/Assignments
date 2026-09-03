"""
lagrange_interpolation.py
=========================
Lagrange Method of Interpolation:
b. Lagrange method of interpolation:
   i. Linear (Order 1, 2 points):
      f1(x) = L0(x)*y0 + L1(x)*y1
   ii. Quadratic (Order 2, 3 points):
      f2(x) = L0(x)*y0 + L1(x)*y1 + L2(x)*y2
   iii. Cubic (Order 3, 4 points):
      f3(x) = sum_{i=0}^3 L_i(x)*y_i

Zero built-in equation solvers used:
- Handwritten nested loops for Lagrange basis polynomials L_i(x).
- Step-by-step basis weight tables and contribution breakdown.
- Relative approximate error (|ea|%).
- Analytical derivative & integral.
- Full ASCII table formatting and visualization plots.
"""

from __future__ import annotations

import importlib
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
# HANDWRITTEN LAGRANGE BASIS & EVALUATION
# ================================================================

@dataclass
class LagrangeBasisTerm:
    index: int
    xi: float
    yi: float
    formula_str: str
    evaluated_weight: float  # L_i(x*)
    contribution: float      # L_i(x*) * y_i

def lagrange_to_standard_poly(x_pts: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    """
    Expands the Lagrange basis polynomial sum into standard canonical polynomial
    coefficients [a0, a1, ..., an] where P(x) = sum(a_i * x^i).
    """
    n = len(x_pts)
    total_poly = np.zeros(n, dtype=float)
    for i in range(n):
        poly_num = np.array([1.0], dtype=float)
        den = 1.0
        for j in range(n):
            if i != j:
                poly_num = np.convolve(poly_num, [-x_pts[j], 1.0])
                den *= (x_pts[i] - x_pts[j])
        total_poly += y_pts[i] * (poly_num / den)
    return total_poly


@dataclass
class LagrangeInterpolationResult:
    order: int
    name: str
    x_points: np.ndarray
    y_points: np.ndarray
    x_target: float
    basis_terms: list[LagrangeBasisTerm]
    interpolated_value: float
    approx_error_percent: float | None = None
    is_extrapolated: bool = False
    canonical_coefficients: np.ndarray | None = None

    def __post_init__(self):
        if self.canonical_coefficients is None:
            self.canonical_coefficients = lagrange_to_standard_poly(self.x_points, self.y_points)

    def evaluate(self, x_val: float | np.ndarray) -> float | np.ndarray:
        """Evaluate Lagrange polynomial at any arbitrary x."""
        n = len(self.x_points)
        if isinstance(x_val, np.ndarray):
            y_out = np.zeros_like(x_val, dtype=float)
            for idx, x in enumerate(x_val):
                val = 0.0
                for i in range(n):
                    L_i = 1.0
                    for j in range(n):
                        if i != j:
                            L_i *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                    val += L_i * self.y_points[i]
                y_out[idx] = val
            return y_out
        else:
            val = 0.0
            for i in range(n):
                L_i = 1.0
                for j in range(n):
                    if i != j:
                        L_i *= (x_val - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                val += L_i * self.y_points[i]
            return float(val)

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

    @property
    def polynomial_string(self) -> str:
        terms = []
        for i, c in enumerate(self.canonical_coefficients):
            if i == 0:
                terms.append(f"{c:.6f}")
            elif i == 1:
                terms.append(f"+ {c:.6f}*x" if c >= 0 else f"- {abs(c):.6f}*x")
            else:
                terms.append(f"+ {c:.6f}*x^{i}" if c >= 0 else f"- {abs(c):.6f}*x^{i}")
        return "P(x) = " + " ".join(terms)


def lagrange_interpolate_order(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    x_target: float,
    order: int,
    prev_value: float | None = None,
    is_extrapolated: bool = False,
) -> LagrangeInterpolationResult:
    """
    Fits Lagrange polynomial of degree `order` using `order + 1` points.
    Calculates explicit basis weights L_i(x*) and terms L_i(x*) * y_i.
    """
    num_pts = order + 1
    if len(x_pts) != num_pts or len(y_pts) != num_pts:
        raise ValueError(f"Order {order} requires exactly {num_pts} data points. Got {len(x_pts)}.")

    basis_terms = []
    total_val = 0.0

    for i in range(num_pts):
        numerator_factors = []
        denominator_factors = []
        L_i = 1.0

        for j in range(num_pts):
            if i != j:
                diff_target = x_target - x_pts[j]
                diff_node = x_pts[i] - x_pts[j]
                numerator_factors.append(f"({x_target:g} - {x_pts[j]:g})")
                denominator_factors.append(f"({x_pts[i]:g} - {x_pts[j]:g})")
                L_i *= diff_target / diff_node

        formula_str = " * ".join(numerator_factors) + " / [" + " * ".join(denominator_factors) + "]"
        contrib = L_i * y_pts[i]
        total_val += contrib

        basis_terms.append(LagrangeBasisTerm(
            index=i,
            xi=x_pts[i],
            yi=y_pts[i],
            formula_str=formula_str,
            evaluated_weight=L_i,
            contribution=contrib,
        ))

    ea = None
    if prev_value is not None:
        ea = abs(total_val - prev_value) / abs(total_val) * 100.0 if total_val != 0 else 0.0

    names = {1: "Linear (1st Order)", 2: "Quadratic (2nd Order)", 3: "Cubic (3rd Order)"}
    name = names.get(order, f"{order}-th Order")

    return LagrangeInterpolationResult(
        order=order,
        name=name,
        x_points=x_pts,
        y_points=y_pts,
        x_target=x_target,
        basis_terms=basis_terms,
        interpolated_value=total_val,
        approx_error_percent=ea,
        is_extrapolated=is_extrapolated,
    )


# ================================================================
# COMPLETE LAGRANGE METHOD SUITE (DYNAMIC DATASET SIZE ADAPTATION)
# ================================================================

def run_lagrange_interpolation_suite(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    output_dir: Path | None = None,
) -> list[LagrangeInterpolationResult]:
    """Runs available Lagrange interpolation orders based on dataset size."""
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
        res = lagrange_interpolate_order(x_sel, y_sel, x_target, order, prev_val, is_extrapolated=is_ext)
        results.append(res)
        prev_val = res.interpolated_value

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        summary_df = generate_summary_dataframe(results)
        summary_df.to_csv(output_dir / "lagrange_interpolation_summary.csv", index=False)

    return results


# ================================================================
# STEP-BY-STEP DISPLAY & SUMMARY TABLES
# ================================================================

def print_step_by_step_lagrange(res: LagrangeInterpolationResult):
    """Prints full handwritten Lagrange basis weights and terms."""
    print(f"\n--- {res.name} Lagrange Interpolation (x* = {res.x_target:g}) ---")
    print("Selected Data Points:")
    for i, (xp, yp) in enumerate(zip(res.x_points, res.y_points)):
        print(f"  Point {i}: ({xp:g}, {yp:g})")

    print(f"\nLagrange Basis Polynomials L_i({res.x_target:g}) and Weighted Contributions:")
    headers = ["Basis Term", "x_i", "y_i", f"L_i({res.x_target:g}) Weight", "Weighted Value L_i*y_i"]
    rows = []
    for term in res.basis_terms:
        rows.append([
            f"L_{term.index}(x)",
            f"{term.xi:.4f}",
            f"{term.yi:.4f}",
            f"{term.evaluated_weight:.6f}",
            f"{term.contribution:.4f}",
        ])
    sum_weights = sum(t.evaluated_weight for t in res.basis_terms)
    sum_contrib = sum(t.contribution for t in res.basis_terms)
    rows.append(["SUM (P(x))", "---", "---", f"{sum_weights:.4f} (=1.0000)", f"{sum_contrib:.4f}"])

    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]
    hdr_str = " | ".join(headers[i].rjust(widths[i]) for i in range(len(headers)))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print("  " + hdr_str)
    print("  " + sep_str)
    for r in rows[:-1]:
        print("  " + " | ".join(r[i].rjust(widths[i]) for i in range(len(headers))))
    print("  " + sep_str)
    print("  " + " | ".join(rows[-1][i].rjust(widths[i]) for i in range(len(headers))))

    print(f"\nInterpolated Value at x = {res.x_target:g}:\n  f({res.x_target:g}) = {res.interpolated_value:.4f}")
    if res.approx_error_percent is not None:
        print(f"Absolute Relative Approximate Error |ea|: {res.approx_error_percent:.5f}%")


def generate_summary_dataframe(results: list[LagrangeInterpolationResult]) -> pd.DataFrame:
    """Generate summary table across Linear, Quadratic, and Cubic orders."""
    rows = []
    for r in results:
        weights_str = ", ".join(f"L{t.index}={t.evaluated_weight:.4f}" for t in r.basis_terms)
        rows.append({
            "Order": r.order,
            "Method Type": r.name,
            "Points Used": ", ".join(f"{x:g}" for x in r.x_points),
            f"f({r.x_target:g})": r.interpolated_value,
            "|ea| (%)": r.approx_error_percent if r.approx_error_percent is not None else np.nan,
            "Lagrange Basis Weights L_i": weights_str,
        })
    return pd.DataFrame(rows)


# ================================================================
# PLOTTING FUNCTIONS
# ================================================================

def plot_lagrange_interpolation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    results: list[LagrangeInterpolationResult],
    x_target: float,
    output_dir: Path | None = None,
):
    """Plots data points and Linear, Quadratic, and Cubic Lagrange curves."""
    plt.figure(figsize=(10, 6))

    # Plot full dataset points
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
    plt.title(f"Lagrange Method of Interpolation (Target x = {x_target:g})", fontsize=14, fontweight="bold")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "lagrange_interpolation_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    # Slide default:
    t = np.array([0.0, 10.0, 15.0, 20.0, 22.5, 30.0], dtype=float)
    v = np.array([0.0, 227.04, 362.78, 517.35, 602.97, 901.67], dtype=float)
    out_dir = Path("interpolation_results/lagrange")

    print("\n==================================================================")
    print("          LAGRANGE METHOD OF NUMERICAL INTERPOLATION              ")
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

    print(f"\n>>> Running Lagrange Interpolation on: {title}")
    results = run_lagrange_interpolation_suite(x_data, y_data, x_target, out_dir)

    for r in results:
        print_step_by_step_lagrange(r)

    df_summary = generate_summary_dataframe(results)
    print(format_table_console(df_summary, f"LAGRANGE INTERPOLATION SUMMARY TABLE (Target = {x_target:g})"))

    if len(results) >= 3:
        cubic_res = results[2]
        deriv_val = cubic_res.derivative(x_target)
        int_val = cubic_res.integrate(11.0, 16.0)
        print("\n--- Physical Derivatives & Integrals from Cubic Profile (Slide Extra) ---")
        print(f"  Derivative at x = {x_target:g} (Acceleration a(16)): {deriv_val:.3f}")
        print(f"  Definite Integral from 11 to 16 (Distance s(16)-s(11)): {int_val:.2f}")

    plot_lagrange_interpolation(x_data, y_data, results, x_target, out_dir)
    print(f"\nAll Lagrange method tables and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
