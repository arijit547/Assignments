"""
interpolation_comparison.py
===========================
Comprehensive Performance Comparison Suite for Numerical Interpolation:
1. Compares Direct Method, Lagrange Method, and Newton's Divided Difference Method
   across Linear (1st Order), Quadratic (2nd Order), and Cubic (3rd Order) polynomials.
2. Evaluates convergence rate via Absolute Relative Approximate Error (|ea|%).
3. Evaluates and demonstrates the Uniqueness of Interpolating Polynomials:
   P_Direct(x) == P_Lagrange(x) == P_Newton(x) (within floating-point eps).
4. Algorithmic Complexity and Computational Trade-off Analysis:
   - Direct: O(n^3) solution + O(n) evaluation.
   - Lagrange: O(n^2) evaluation, no matrix needed, but adding a point requires full recomputation.
   - Newton: O(n^2) table setup + O(n) Horner evaluation, incremental point addition without recomputation!
5. Multi-curve comparison plots and CSV exports.
"""

from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from direct_interpolation import (
    run_direct_interpolation_suite,
    format_table_console,
    get_slide_default_problem,
)
from lagrange_interpolation import run_lagrange_interpolation_suite
from newton_divided_difference import run_newton_interpolation_suite


# ================================================================
# COMPARISON MATRIX RUNNER
# ================================================================

def run_full_comparison(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_target: float,
    exact_func=None,
    output_dir: Path | None = None,
) -> dict:
    """
    Executes Direct, Lagrange, and Newton methods for Orders 1, 2, 3
    and records values, errors, and execution timings.
    """
    # 1. Run Direct Method
    t0 = time.perf_counter()
    direct_res = run_direct_interpolation_suite(x_data, y_data, x_target)
    t_direct = (time.perf_counter() - t0) * 1000.0  # ms

    # 2. Run Lagrange Method
    t0 = time.perf_counter()
    lagrange_res = run_lagrange_interpolation_suite(x_data, y_data, x_target)
    t_lagrange = (time.perf_counter() - t0) * 1000.0  # ms

    # 3. Run Newton's Divided Difference Method
    t0 = time.perf_counter()
    newton_res = run_newton_interpolation_suite(x_data, y_data, x_target)
    t_newton = (time.perf_counter() - t0) * 1000.0  # ms

    # Compile Table 1: Interpolated Values & Approximate Errors
    comparison_rows = []
    methods_data = [
        ("Direct Method", direct_res, t_direct, "O(n^3)", "Solves Vandermonde linear system"),
        ("Lagrange Method", lagrange_res, t_lagrange, "O(n^2)", "Evaluates products of basis weights"),
        ("Newton's Divided Diff.", newton_res, t_newton, "O(n^2)", "Constructs triangular difference table"),
    ]

    for m_name, res_list, t_exec, complexity, notes in methods_data:
        v1 = res_list[0].interpolated_value
        v2 = res_list[1].interpolated_value
        v3 = res_list[2].interpolated_value
        ea2 = res_list[1].approx_error_percent
        ea3 = res_list[2].approx_error_percent

        comparison_rows.append({
            "Method": m_name,
            "Linear (n=1)": v1,
            "Quadratic (n=2)": v2,
            "|ea| (1->2) (%)": ea2,
            "Cubic (n=3)": v3,
            "|ea| (2->3) (%)": ea3,
            "Complexity": complexity,
        })

    df_comparison = pd.DataFrame(comparison_rows)

    # Verification of Uniqueness Theorem
    diff_dir_lag = abs(direct_res[2].interpolated_value - lagrange_res[2].interpolated_value)
    diff_dir_new = abs(direct_res[2].interpolated_value - newton_res[2].interpolated_value)
    uniqueness_verified = max(diff_dir_lag, diff_dir_new) < 1e-10

    # True error if exact function provided
    df_true_error = None
    if exact_func is not None:
        y_true = float(exact_func(x_target))
        te_rows = []
        for m_name, res_list, _, _, _ in methods_data:
            te_row = {"Method": m_name}
            for r in res_list:
                et = abs(y_true - r.interpolated_value) / abs(y_true) * 100.0 if y_true != 0 else 0.0
                te_row[f"Order {r.order} |et| (%)"] = et
            te_rows.append(te_row)
        df_true_error = pd.DataFrame(te_rows)

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        df_comparison.to_csv(output_dir / "master_interpolation_comparison.csv", index=False)
        if df_true_error is not None:
            df_true_error.to_csv(output_dir / "true_error_comparison.csv", index=False)

    return {
        "df_comparison": df_comparison,
        "df_true_error": df_true_error,
        "direct_res": direct_res,
        "lagrange_res": lagrange_res,
        "newton_res": newton_res,
        "uniqueness_verified": uniqueness_verified,
        "max_discrepancy": max(diff_dir_lag, diff_dir_new),
    }


# ================================================================
# MULTI-CURVE COMPARISON PLOT
# ================================================================

def plot_comprehensive_comparison(
    x_all: np.ndarray,
    y_all: np.ndarray,
    direct_res: list,
    x_target: float,
    output_dir: Path | None = None,
):
    """
    Plots the dataset and the 1st, 2nd, and 3rd order interpolating polynomials,
    annotating the convergence towards the target point.
    """
    plt.figure(figsize=(11, 7))

    # Raw points
    plt.scatter(x_all, y_all, color="black", s=70, zorder=6, label="Given Discrete Dataset Points")

    x_min = min(r.x_points[0] for r in direct_res)
    x_max = max(r.x_points[-1] for r in direct_res)
    margin = (x_max - x_min) * 0.12
    x_dense = np.linspace(max(0, x_min - margin), x_max + margin, 400)

    curve_configs = [
        (direct_res[0], "#1a73e8", "--", "Linear (Order 1, 2 pts)"),
        (direct_res[1], "#0d904f", "-.", "Quadratic (Order 2, 3 pts)"),
        (direct_res[2], "#d93025", "-", "Cubic (Order 3, 4 pts)"),
    ]

    for r, col, ls, lbl in curve_configs:
        y_dense = r.evaluate(x_dense)
        plt.plot(x_dense, y_dense, color=col, linestyle=ls, linewidth=2.2,
                 label=f"{lbl} -> f({x_target:g}) = {r.interpolated_value:.2f}")
        plt.plot(x_target, r.interpolated_value, marker="o", markersize=9, color=col, markeredgecolor="black")

    # Target indicator line
    plt.axvline(x_target, color="#5f6368", linestyle=":", alpha=0.8, linewidth=1.5,
                label=f"Query Point x* = {x_target:g}")

    plt.title(f"Comparison of Numerical Interpolation Orders (Target x = {x_target:g})",
              fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Independent Variable x (e.g. Time t)", fontsize=12)
    plt.ylabel("Dependent Variable y (e.g. Velocity v)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "master_interpolation_comparison_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


# ================================================================
# CLI INTERFACE
# ================================================================

def run_cli():
    prob = get_slide_default_problem()
    out_dir = Path("interpolation_results/comparison")

    print("\n==================================================================")
    print("      NUMERICAL INTERPOLATION - MASTER COMPARISON SUITE          ")
    print("==================================================================")
    print("1. Run Benchmark Problem Comparison (Rocket Velocity at t = 16 s)")
    print("2. Enter Custom Dataset and Target Point")
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

    print(f"\n>>> Running Full Comparative Analysis on: {title}")
    comp = run_full_comparison(x_data, y_data, x_target, output_dir=out_dir)

    print(format_table_console(comp["df_comparison"], f"INTERPOLATION PERFORMANCE COMPARISON (Target x = {x_target:g})"))

    print("\n--- Theoretical Property Verification: Uniqueness of Interpolating Polynomial ---")
    if comp["uniqueness_verified"]:
        print("  [PASSED] Uniqueness Theorem Verified:")
        print(f"  P_Direct(x*) == P_Lagrange(x*) == P_Newton(x*) within floating-point tolerance ({comp['max_discrepancy']:.2e}).")
        print("  Theoretical Insight: For any set of n+1 distinct points, there exists a UNIQUE")
        print("  polynomial of degree <= n that passes through all points.")
        print("  The Direct, Lagrange, and Newton methods are simply three different algebraic")
        print("  representations of the exact same interpolating polynomial!")
    else:
        print(f"  [DISCREPANCY DETECTED]: Max difference = {comp['max_discrepancy']:.6e}")

    print("\n--- Algorithmic Performance & Complexity Comparison ---")
    print("  1. Direct Method:")
    print("     - Computational Cost: O(n^3) to set up and solve the Vandermonde linear system.")
    print("     - Condition Number: Vandermonde matrices become severely ill-conditioned for higher orders (n > 5).")
    print("     - Best For: Finding explicit polynomial coefficients a0, a1, ..., an.")
    print("  2. Lagrange Method:")
    print("     - Computational Cost: O(n^2) to evaluate basis polynomials directly.")
    print("     - Sensitivity: Extremely simple to code; does not suffer from matrix ill-conditioning.")
    print("     - Limitation: Adding a new data point requires recomputing all basis polynomials from scratch.")
    print("  3. Newton's Divided Difference Method (Recommended):")
    print("     - Computational Cost: O(n^2) to construct the triangular table, O(n) to evaluate via Horner scheme.")
    print("     - Modularity: Highly advantageous for incremental data. Adding a new point (x_{n+1}, y_{n+1})")
    print("       only requires computing one additional diagonal entry without discarding previous work!")

    plot_comprehensive_comparison(x_data, y_data, comp["direct_res"], x_target, out_dir)
    print(f"\nAll comparison tables and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
