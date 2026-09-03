"""
comparison.py
=============
Comprehensive Numerical Comparison & Convergence Analysis for ODEs:
1. Euler's Method (1st Order Runge-Kutta)
2. Heun's Method (2nd Order Runge-Kutta, a2 = 1/2)
3. Midpoint Method (2nd Order Runge-Kutta, a2 = 1)
4. Ralston's Method (2nd Order Runge-Kutta, a2 = 2/3)
5. Classical Runge-Kutta 4th Order Method (RK4)

Produces:
- Table 1: Numerical solution comparison at xf (with exact/high-accuracy reference)
- Table 2: Absolute relative true error (|et|%) comparison
- Table 3: Maximum pointwise absolute error (E_max) & Root-Mean-Square Error (RMSE)
- Table 4: Computational cost & RHS function evaluations
- Table 5: Empirical vs Theoretical Order of Convergence
- Publication-quality plots:
    1. Trajectory comparison vs Reference
    2. Pointwise error distribution vs Independent variable (x)
    3. Log-Log convergence plot (Error vs h) with order slopes
- Comprehensive CSV exports for all metrics
- Standalone execution support with interactive CLI
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from ode_input_gui import ODEProblem, get_problem, get_slide_default_problem
from euler import euler_method, compute_reference_solution
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
                if np.isnan(val):
                    s = "Undefined (Exact=0)"
                elif "%" in headers[i] or "|et|" in headers[i]:
                    s = f"{val:.4f}%"
                elif "Step" in headers[i] or "Order" in headers[i]:
                    s = f"{val:.4f}" if isinstance(val, float) else f"{val}"
                elif abs(val) < 1e-4 and val != 0:
                    s = f"{val:.6e}"
                else:
                    s = f"{val:.6f}" if abs(val) < 1.0 else f"{val:.4f}"
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
# COMPREHENSIVE COMPARISON ENGINE
# ================================================================

def run_comparison(
    problem: ODEProblem,
    reference_function=None,
    output_dir: str = "comparison_results",
):
    """
    Runs all 5 numerical ODE methods across all configured step sizes,
    calculates comprehensive error metrics, computational costs,
    empirical convergence rates, and exports formatted tables and plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    dep = problem.dependent_name
    indep = problem.independent_name
    xf = problem.xf
    x0 = problem.x0
    y0 = problem.y0

    # Ensure reference solution
    if reference_function is None:
        reference_function = compute_reference_solution(problem)

    ref_label = (
        "Exact Analytical Solution"
        if problem.exact_solution is not None
        else "High-Accuracy Reference (SciPy DOP853)"
    )

    y_exact_final = float(reference_function(xf))

    # Lambdify ODE RHS
    fn = sp.lambdify(
        (problem.independent_symbol, problem.dependent_symbol),
        problem.expression,
        modules=["numpy"],
    )

    def rhs(x_val, y_val):
        y_scalar = y_val[0] if isinstance(y_val, (list, np.ndarray)) else y_val
        return float(fn(x_val, float(y_scalar)))

    methods = [
        ("Euler", euler_method, 1, 1, "ro--"),
        ("Heun", heun_method, 2, 2, "bs-."),
        ("Midpoint", midpoint_method, 2, 2, "g^-."),
        ("Ralston", ralston_method, 2, 2, "md-."),
        ("RK4", rk4_method, 4, 4, "c*-"),
    ]

    sorted_steps = sorted(problem.step_sizes, reverse=True)

    solution_rows = []
    error_rows = []
    advanced_metric_rows = []
    cost_rows = []
    combined_rows = []

    # Store trajectory results: results[h][method_name] = (x_vals, y_vals, max_err, rmse, runtime)
    results = {h: {} for h in sorted_steps}

    for h in sorted_steps:
        n_steps = round((xf - x0) / h)
        
        sol_row = {"Step size, h": h}
        err_row = {"Step size, h": h}
        cost_row = {"Step size, h": h, "Steps (N)": n_steps}
        adv_row = {"Step size, h": h}
        comb_row = {"Step size, h": h}

        for name, fn_method, order, evals_per_step, _ in methods:
            t_start = time.perf_counter()
            x_sol, y_sol = fn_method(rhs, x0, y0, xf, h)
            t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms

            y_final = float(y_sol[-1])
            true_err = y_exact_final - y_final
            rel_err_pct = (
                (abs(true_err) / abs(y_exact_final) * 100.0)
                if y_exact_final != 0
                else np.nan
            )

            # Pointwise grid evaluations for RMSE and Max Error
            y_ref_grid = np.array([float(reference_function(x)) for x in x_sol])
            pt_abs_errors = np.abs(y_ref_grid - y_sol)
            max_abs_err = float(np.max(pt_abs_errors))
            rmse = float(np.sqrt(np.mean(pt_abs_errors ** 2)))

            total_rhs_evals = n_steps * evals_per_step

            results[h][name] = {
                "x": x_sol,
                "y": y_sol,
                "y_final": y_final,
                "true_err": true_err,
                "rel_err_pct": rel_err_pct,
                "max_abs_err": max_abs_err,
                "rmse": rmse,
                "evals_per_step": evals_per_step,
                "total_evals": total_rhs_evals,
                "runtime_ms": t_elapsed,
                "order": order,
            }

            sol_row[f"{name} {dep}({xf:g})"] = y_final
            err_row[f"{name} |et|%"] = rel_err_pct
            adv_row[f"{name} E_max"] = max_abs_err
            adv_row[f"{name} RMSE"] = rmse
            cost_row[f"{name} RHS Calls"] = total_rhs_evals
            cost_row[f"{name} Time (ms)"] = t_elapsed

            comb_row[f"{name} Sol"] = y_final
            comb_row[f"{name} |et|%"] = rel_err_pct
            comb_row[f"{name} E_max"] = max_abs_err
            comb_row[f"{name} RMSE"] = rmse

        sol_row[f"Reference {dep}({xf:g})"] = y_exact_final
        comb_row["Reference Sol"] = y_exact_final

        solution_rows.append(sol_row)
        error_rows.append(err_row)
        advanced_metric_rows.append(adv_row)
        cost_rows.append(cost_row)
        combined_rows.append(comb_row)

    df_solutions = pd.DataFrame(solution_rows)
    df_errors = pd.DataFrame(error_rows)
    df_metrics = pd.DataFrame(advanced_metric_rows)
    df_costs = pd.DataFrame(cost_rows)
    df_combined = pd.DataFrame(combined_rows)

    # Calculate empirical order of convergence if multiple step sizes provided
    convergence_rows = []
    if len(sorted_steps) >= 2:
        for i in range(len(sorted_steps) - 1):
            h1 = sorted_steps[i]
            h2 = sorted_steps[i + 1]
            log_h_ratio = math.log(h1 / h2)
            c_row = {"Step Transition": f"h={h1:g} -> {h2:g}", "h ratio": h1 / h2}
            for name, _, order, _, _ in methods:
                e1 = results[h1][name]["max_abs_err"]
                e2 = results[h2][name]["max_abs_err"]
                if e1 > 0 and e2 > 0:
                    p_obs = math.log(e1 / e2) / log_h_ratio
                    c_row[f"{name} p_obs"] = p_obs
                    c_row[f"{name} p_theory"] = order
                else:
                    c_row[f"{name} p_obs"] = np.nan
                    c_row[f"{name} p_theory"] = order
            convergence_rows.append(c_row)
    df_convergence = pd.DataFrame(convergence_rows)

    # ------------------------------------------------------------
    # PRINT CONSOLE TABLES
    # ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("      COMPREHENSIVE NUMERICAL METHODS COMPARISON & CONVERGENCE SUITE      ")
    print(f"      Problem: d{dep}/d{indep} = {problem.expression}")
    print(f"      Domain: {indep}0 = {x0:g} -> {xf:g} | {dep}0 = {y0:g}")
    print(f"      Reference Mode: {ref_label}")
    print("=" * 78)

    print(format_table_console(
        df_solutions,
        f"TABLE 1: COMPARISON OF NUMERICAL METHOD SOLUTIONS AT {dep.upper()}({xf:g})"
    ))

    print(format_table_console(
        df_errors,
        "TABLE 2: COMPARISON OF ABSOLUTE RELATIVE TRUE ERROR |et| (%)"
    ))

    print(format_table_console(
        df_metrics,
        "TABLE 3: MAXIMUM ABSOLUTE ERROR (E_max) & ROOT-MEAN-SQUARE ERROR (RMSE)"
    ))

    print(format_table_console(
        df_costs,
        "TABLE 4: COMPUTATIONAL COST & RHS FUNCTION EVALUATIONS"
    ))

    if not df_convergence.empty:
        print(format_table_console(
            df_convergence,
            "TABLE 5: EMPIRICAL CONVERGENCE ORDER (p_obs vs p_theory)"
        ))

    # ------------------------------------------------------------
    # SAVE CSV EXPORTS
    # ------------------------------------------------------------
    df_solutions.to_csv(output_path / "comparison_solutions.csv", index=False)
    df_errors.to_csv(output_path / "comparison_errors.csv", index=False)
    df_metrics.to_csv(output_path / "comparison_metrics_summary.csv", index=False)
    df_costs.to_csv(output_path / "comparison_computational_cost.csv", index=False)
    if not df_convergence.empty:
        df_convergence.to_csv(output_path / "comparison_convergence_order.csv", index=False)
    df_combined.to_csv(output_path / "comparison_complete_results.csv", index=False)

    # ------------------------------------------------------------
    # PLOT 1: TRAJECTORY COMPARISON (PRIMARY / FINEST STEP SIZE)
    # ------------------------------------------------------------
    # Default to finest step size for highest accuracy representation
    primary_h = sorted_steps[-1]
    
    plt.figure(figsize=(11, 6))
    x_fine = np.linspace(x0, xf, 500)
    y_fine = [reference_function(x) for x in x_fine]
    plt.plot(x_fine, y_fine, "k-", linewidth=2.5, label=f"{ref_label}")

    for name, _, _, _, fmt in methods:
        res_m = results[primary_h][name]
        plt.plot(res_m["x"], res_m["y"], fmt, markersize=5, label=f"{name} (h={primary_h:g})")

    plt.title(f"Numerical Solutions vs Reference Trajectory (h = {primary_h:g})", fontsize=14, fontweight="bold")
    plt.xlabel(f"{indep}", fontsize=12)
    plt.ylabel(f"{dep}", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # PLOT 2: POINTWISE ERROR DISTRIBUTION VS INDEPENDENT VARIABLE
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    for name, _, _, _, fmt in methods:
        res_m = results[primary_h][name]
        x_m = res_m["x"]
        y_ref_m = np.array([float(reference_function(x)) for x in x_m])
        err_m = np.abs(y_ref_m - res_m["y"])
        plt.semilogy(x_m, np.maximum(err_m, 1e-16), fmt, markersize=5, label=f"{name} Error (h={primary_h:g})")

    plt.title(f"Pointwise Absolute Error Distribution |y_ref(x) - y(x)| vs {indep} (h = {primary_h:g})", fontsize=14, fontweight="bold")
    plt.xlabel(f"{indep}", fontsize=12)
    plt.ylabel("Absolute Error (log scale)", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_error_vs_x.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # PLOT 3: LOG-LOG CONVERGENCE PLOT (ERROR VS STEP SIZE h)
    # ------------------------------------------------------------
    if len(sorted_steps) >= 2:
        plt.figure(figsize=(10, 6))
        step_arr = np.array(sorted_steps)
        for name, _, order, _, fmt in methods:
            errs = [results[h][name]["max_abs_err"] for h in sorted_steps]
            # Replace zeros with tiny positive epsilon for log plot
            errs_clean = np.maximum(errs, 1e-16)
            plt.loglog(step_arr, errs_clean, fmt, linewidth=1.8, markersize=7, label=f"{name} (Theory: O(h^{order}))")

        plt.title("Empirical Convergence Analysis: Max Absolute Error vs Step Size h", fontsize=14, fontweight="bold")
        plt.xlabel("Step Size h (log scale)", fontsize=12)
        plt.ylabel("Maximum Absolute Error (log scale)", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "comparison_convergence_loglog.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nAll comparison tables and 3 publication-quality plots successfully saved in:\n  {output_path.resolve()}")
    return df_solutions, df_errors, df_metrics, df_costs, df_convergence


# ================================================================
# STANDALONE CLI WORKFLOW
# ================================================================

def main():
    """Standalone CLI runner for the comparison module."""
    print("\n==================================================================")
    print("          ODE NUMERICAL METHODS COMPARISON & BENCHMARK            ")
    print("==================================================================")
    print("1. Slide Benchmark Problem (Radiation Cooling of Sphere)")
    print("2. Launch Interactive GUI for Custom Problem Configuration")
    print("3. Exit")

    choice = input("\nSelect an option (1, 2, 3): ").strip()
    if choice == "1":
        problem = get_slide_default_problem()
        print(f"\nLoaded Slide Benchmark Problem: d{problem.dependent_name}/d{problem.independent_name} = {problem.expression}")
        print(f"Interval: {problem.independent_name} = {problem.x0} -> {problem.xf} | Step sizes: {problem.step_sizes}")
        run_comparison(problem)
    elif choice == "2":
        problem = get_problem("ODE Comparison Suite")
        if problem is not None:
            run_comparison(problem)
        else:
            print("\nProblem configuration cancelled. Exiting.")
    elif choice == "3":
        print("\nExiting ODE Comparison Suite.")
    else:
        print("\nInvalid choice. Exiting.")


if __name__ == "__main__":
    main()
