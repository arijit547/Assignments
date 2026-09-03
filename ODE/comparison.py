"""
comparison.py
=============
Comparison of all 5 ODE numerical methods:
1. Euler's Method (1st Order Runge-Kutta)
2. Heun's Method (2nd Order Runge-Kutta, a2 = 1/2)
3. Midpoint Method (2nd Order Runge-Kutta, a2 = 1)
4. Ralston's Method (2nd Order Runge-Kutta, a2 = 2/3)
5. Classical Runge-Kutta 4th Order Method (RK4)

Produces:
- Numerical solution comparison table at xf
- Absolute relative true error (|et|%) comparison table (Table 2 in slides)
- Comparison plot of trajectories vs exact solution
- CSV export for all comparison tables
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from euler import euler_method, ODEProblem, compute_reference_solution
from Heun import heun_method
from Midpoint_method import midpoint_method
from Ralston_method import ralston_method
from RK4_method import rk4_method


# ================================================================
# TABLE FORMATTING HELPER
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
                elif "Step" in headers[i] or headers[i].startswith("i"):
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
# COMPARISON RUNNER
# ================================================================

def run_comparison(problem: ODEProblem, reference_function, output_dir: str = "comparison_results"):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)

    dep = problem.dependent_name
    indep = problem.independent_name
    xf = problem.xf

    # Reference exact value at final time
    y_exact = float(reference_function(xf))

    # Lambdify equation RHS safely for scalar/array evaluation
    fn = sp.lambdify(
        (problem.independent_symbol, problem.dependent_symbol),
        problem.expression,
        modules=["numpy"],
    )

    def rhs(x_val, y_val):
        y_scalar = y_val[0] if isinstance(y_val, (list, np.ndarray)) else y_val
        return float(fn(x_val, float(y_scalar)))

    solution_rows = []
    error_rows = []
    combined_rows = []
    results = {}

    sorted_steps = sorted(problem.step_sizes, reverse=True)

    for h in sorted_steps:
        sol_euler = euler_method(rhs, problem.x0, problem.y0, problem.xf, h)
        sol_heun = heun_method(rhs, problem.x0, problem.y0, problem.xf, h)
        sol_mid = midpoint_method(rhs, problem.x0, problem.y0, problem.xf, h)
        sol_ralston = ralston_method(rhs, problem.x0, problem.y0, problem.xf, h)
        sol_rk4 = rk4_method(rhs, problem.x0, problem.y0, problem.xf, h)

        y_euler = sol_euler[1][-1]
        y_heun = sol_heun[1][-1]
        y_mid = sol_mid[1][-1]
        y_ralston = sol_ralston[1][-1]
        y_rk4 = sol_rk4[1][-1]

        # Calculate |et|% (Absolute relative true error %)
        err_euler = (abs(y_exact - y_euler) / abs(y_exact) * 100) if y_exact != 0 else np.nan
        err_heun = (abs(y_exact - y_heun) / abs(y_exact) * 100) if y_exact != 0 else np.nan
        err_mid = (abs(y_exact - y_mid) / abs(y_exact) * 100) if y_exact != 0 else np.nan
        err_ralston = (abs(y_exact - y_ralston) / abs(y_exact) * 100) if y_exact != 0 else np.nan
        err_rk4 = (abs(y_exact - y_rk4) / abs(y_exact) * 100) if y_exact != 0 else np.nan

        solution_rows.append({
            "Step size, h": h,
            f"Euler {dep}({xf:g})": y_euler,
            f"Heun {dep}({xf:g})": y_heun,
            f"Midpoint {dep}({xf:g})": y_mid,
            f"Ralston {dep}({xf:g})": y_ralston,
            f"RK4 {dep}({xf:g})": y_rk4,
            f"Exact {dep}({xf:g})": y_exact,
        })

        error_rows.append({
            "Step size, h": h,
            "Euler |et|%": err_euler,
            "Heun |et|%": err_heun,
            "Midpoint |et|%": err_mid,
            "Ralston |et|%": err_ralston,
            "RK4 |et|%": err_rk4,
        })

        combined_rows.append({
            "Step size, h": h,
            "Euler": y_euler,
            "Euler |et|%": err_euler,
            "Heun": y_heun,
            "Heun |et|%": err_heun,
            "Midpoint": y_mid,
            "Midpoint |et|%": err_mid,
            "Ralston": y_ralston,
            "Ralston |et|%": err_ralston,
            "RK4": y_rk4,
            "RK4 |et|%": err_rk4,
            "Exact": y_exact,
        })

        results[h] = {
            "euler": sol_euler,
            "heun": sol_heun,
            "midpoint": sol_mid,
            "ralston": sol_ralston,
            "rk4": sol_rk4,
            "exact": y_exact,
        }

    df_solutions = pd.DataFrame(solution_rows)
    df_errors = pd.DataFrame(error_rows)
    df_combined = pd.DataFrame(combined_rows)

    # 1. Print formatted tables to console
    print(format_table_console(
        df_solutions,
        f"TABLE 1: COMPARISON OF NUMERICAL METHOD SOLUTIONS AT {dep.upper()}({xf:g})"
    ))

    print(format_table_console(
        df_errors,
        "TABLE 2: COMPARISON OF ABSOLUTE RELATIVE TRUE ERROR |et| (%)"
    ))

    # 2. Save tables to CSV
    df_solutions.to_csv(output_path / "comparison_solutions.csv", index=False)
    df_errors.to_csv(output_path / "comparison_errors.csv", index=False)
    df_combined.to_csv(output_path / "comparison_complete_results.csv", index=False)

    # 3. Plot comparison graph
    if len(problem.step_sizes) > 1:
        plot_h = sorted_steps[1]  # Second largest step size for clear visualization
    else:
        plot_h = sorted_steps[0]

    sol_h = results[plot_h]

    plt.figure(figsize=(10, 6))

    x_fine = np.linspace(problem.x0, problem.xf, 500)
    y_fine = [reference_function(x) for x in x_fine]
    plt.plot(x_fine, y_fine, "k-", linewidth=2.5, label="Exact / Reference")

    plt.plot(sol_h["euler"][0], sol_h["euler"][1], "ro--", label=f"Euler (h={plot_h:g})")
    plt.plot(sol_h["heun"][0], sol_h["heun"][1], "bs-.", label=f"Heun (h={plot_h:g})")
    plt.plot(sol_h["midpoint"][0], sol_h["midpoint"][1], "g^-.", label=f"Midpoint (h={plot_h:g})")
    plt.plot(sol_h["ralston"][0], sol_h["ralston"][1], "md-.", label=f"Ralston (h={plot_h:g})")
    plt.plot(sol_h["rk4"][0], sol_h["rk4"][1], "c*-", linewidth=2, markersize=8, label=f"RK4 (h={plot_h:g})")

    plt.title(f"Comparison of Numerical Methods vs Exact Solution (h = {plot_h:g})", fontsize=14, fontweight="bold")
    plt.xlabel(f"{problem.independent_name}", fontsize=12)
    plt.ylabel(f"{problem.dependent_name}", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = output_path / f"comparison_plot_h{plot_h:g}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nComparison results and plot saved in: {output_path.resolve()}")
    return df_solutions, df_errors
