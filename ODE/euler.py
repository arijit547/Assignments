
"""
euler.py
========
Euler's Method implementation.

Run:
    python euler.py

The program first launches the reusable GUI from ode_input_gui.py.
After successful validation, Euler's method is performed manually.

No built-in ODE solver is used for Euler's actual calculation.
scipy.solve_ivp is used only by the input layer for validation and,
when no analytical exact solution is supplied, as a high-accuracy
reference for comparison.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp

from ode_input_gui import get_problem, ODEProblem


# ================================================================
# EULER'S METHOD
# ================================================================

def euler_method(
    f,
    x0: float,
    y0: float,
    xf: float,
    h: float,
):
    """
    Handwritten Euler method.

        y_(n+1) = y_n + h*f(x_n, y_n)

    Returns:
        x_values, y_values
    """
    if not math.isfinite(h) or h <= 0:
        raise ValueError(
            f"Step size h must be a positive finite number, got {h}."
        )

    number_of_steps_float = (xf - x0) / h

    number_of_steps = round(
        number_of_steps_float
    )

    if not math.isclose(
        number_of_steps_float,
        number_of_steps,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(
            f"h={h:g} does not divide the interval exactly."
        )

    x_values = np.empty(
        number_of_steps + 1,
        dtype=float,
    )

    y_values = np.empty(
        number_of_steps + 1,
        dtype=float,
    )

    x_values[0] = x0
    y_values[0] = y0

    # Safety limit against runaway Euler solutions.
    max_abs_y = 1e100

    for i in range(number_of_steps):

        try:
            slope = float(
                f(
                    x_values[i],
                    y_values[i],
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Euler failed at step {i}, "
                f"x={x_values[i]:g}, "
                f"y={y_values[i]:g}."
            ) from exc

        if not math.isfinite(slope):
            raise FloatingPointError(
                f"Derivative became invalid at "
                f"x={x_values[i]:g}, "
                f"y={y_values[i]:g}."
            )

        x_values[i + 1] = (
            x_values[i] + h
        )

        y_values[i + 1] = (
            y_values[i] + h * slope
        )

        if not math.isfinite(
            y_values[i + 1]
        ):
            raise FloatingPointError(
                f"Euler produced NaN/infinity at "
                f"x={x_values[i+1]:g}."
            )

        if abs(y_values[i + 1]) > max_abs_y:
            raise OverflowError(
                f"Numerical solution exceeded magnitude limit ({max_abs_y:g}); computation stopped at step {i+1} near x={x_values[i+1]:g}."
            )

    return x_values, y_values


# ================================================================
# REFERENCE SOLUTION
# ================================================================

def build_reference_function(problem: ODEProblem):

    fn = sp.lambdify(
        (
            problem.independent_symbol,
            problem.dependent_symbol,
        ),
        problem.expression,
        modules=["numpy"],
    )

    def rhs(x_value, y_value):
        y_val = y_value[0] if isinstance(y_value, (list, __import__('numpy').ndarray)) else y_value
        return float(
            fn(
                x_value,
                float(y_val),
            )
        )

    return rhs


def compute_reference_solution(
    problem: ODEProblem,
    return_label: bool = False,
):
    """
    If the user supplied a valid exact solution, use it.

    Otherwise solve_ivp (DOP853) is used to obtain a high-accuracy
    numerical reference solution for comparison.

    Carries explicit provenance ('exact' vs 'numerical') and descriptive label.
    """

    if problem.exact_solution is not None:
        from ode_input_gui import validate_exact_solution
        is_valid, msg = validate_exact_solution(
            problem.exact_solution,
            problem.expression,
            problem.independent_symbol,
            problem.dependent_symbol,
            problem.x0,
            problem.y0,
            problem.xf,
        )
        if is_valid:
            exact_fn = sp.lambdify(
                problem.independent_symbol,
                problem.exact_solution,
                modules=["numpy"],
            )

            def reference(x_values):
                return np.asarray(
                    exact_fn(x_values),
                    dtype=float,
                )

            reference.provenance = "exact"
            reference.label = "Exact Analytical Solution"
            if return_label:
                return reference, reference.label
            return reference
        else:
            print(f"\n[WARNING] Exact solution failed verification: {msg}")
            print("          Falling back to high-accuracy SciPy DOP853 numerical reference solution.\n")

    rhs = build_reference_function(
        problem
    )

    result = solve_ivp(
        rhs,
        (
            problem.x0,
            problem.xf,
        ),
        [problem.y0],
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        dense_output=True,
        max_step=max(
            (problem.xf - problem.x0) / 1000,
            1e-12,
        ),
    )

    if not result.success:
        raise RuntimeError(
            "Reference solution failed:\n"
            + result.message
        )

    def reference(x_values):
        return result.sol(x_values)[0]

    reference.provenance = "numerical"
    reference.label = "High-Accuracy Reference (SciPy DOP853)"
    if return_label:
        return reference, reference.label
    return reference


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

def create_output_directory():

    output = Path("euler_results")
    output.mkdir(
        exist_ok=True
    )

    return output


# ================================================================
# TABLE FORMATTING & CONSOLE DISPLAY HELPER
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
# TABLE GENERATION
# ================================================================

def generate_results_table(
    problem: ODEProblem,
    reference_function,
    output: Path,
):
    """
    Generate summary table across all step sizes for Euler's method.
    """
    rhs = build_reference_function(problem)
    rows = []
    dep = problem.dependent_name
    xf = problem.xf

    for h in problem.step_sizes:
        x_values, y_values = euler_method(
            rhs,
            problem.x0,
            problem.y0,
            problem.xf,
            h,
        )

        numerical_final = y_values[-1]
        reference_final = float(reference_function(problem.xf))
        error = reference_final - numerical_final

        if reference_final != 0:
            error_percent = (abs(error) / abs(reference_final)) * 100
        else:
            error_percent = np.nan

        ref_col = (
            f"Exact {dep}({xf:g})"
            if getattr(reference_function, "provenance", None) == "exact"
            else f"Reference {dep}({xf:g})"
        )
        rows.append(
            {
                "Step size, h": h,
                f"Euler {dep}({xf:g})": numerical_final,
                ref_col: reference_final,
                "True Error (Et)": error,
                "|et| (%)": error_percent,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(
        output / "euler_results.csv",
        index=False,
    )
    table.to_csv(
        output / "euler_summary_table.csv",
        index=False,
    )
    return table


def generate_iteration_table(
    problem: ODEProblem,
    reference_function,
    h: float,
    output: Path,
):
    """
    Generate detailed step-by-step iteration table for a specific step size h.
    """
    rhs = build_reference_function(problem)
    number_of_steps = round((problem.xf - problem.x0) / h)
    dep = problem.dependent_name
    indep = problem.independent_name

    x = problem.x0
    y = problem.y0
    rows = []

    for i in range(number_of_steps):
        slope = float(rhs(x, y))
        x_next = x + h
        y_next = y + h * slope
        y_exact = float(reference_function(x_next))
        error = y_exact - y_next
        error_percent = (abs(error) / abs(y_exact) * 100) if y_exact != 0 else np.nan

        ref_step_col = (
            f"Exact {dep}_(i+1)"
            if getattr(reference_function, "provenance", None) == "exact"
            else f"Reference {dep}_(i+1)"
        )
        rows.append(
            {
                "Step i": i,
                f"{indep}_i": x,
                f"{dep}_i": y,
                f"f({indep}_i, {dep}_i)": slope,
                f"{dep}_(i+1)": y_next,
                ref_step_col: y_exact,
                "True Error (Et)": error,
                "|et| (%)": error_percent,
            }
        )
        x = x_next
        y = y_next

    table = pd.DataFrame(rows)
    table.to_csv(
        output / f"euler_steps_h{h:g}.csv",
        index=False,
    )
    return table


# ================================================================
# GRAPH 1: GIVEN FUNCTION
# ================================================================

def plot_given_function(
    problem: ODEProblem,
    output: Path,
):

    rhs = build_reference_function(
        problem
    )

    x_values = np.linspace(
        problem.x0,
        problem.xf,
        1000,
    )

    # Plot f(x, y0), which works for both
    # autonomous and non-autonomous equations.
    values = np.array(
        [
            rhs(
                x_value,
                np.array([problem.y0]),
            )
            for x_value in x_values
        ]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        x_values,
        values,
        linewidth=2,
    )

    plt.xlabel(
        problem.independent_name
    )

    plt.ylabel(
        f"f({problem.independent_name}, "
        f"{problem.dependent_name}₀)"
    )

    plt.title(
        "Given ODE Function"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output / "01_given_function.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ================================================================
# GRAPH 2: REFERENCE VS EULER FOR A SELECTED STEP SIZE
# ================================================================

def plot_reference_vs_euler(
    problem: ODEProblem,
    reference_function,
    output: Path,
    comparison_h: float,
):

    rhs = build_reference_function(
        problem
    )

    x_euler, y_euler = (
        euler_method(
            rhs,
            problem.x0,
            problem.y0,
            problem.xf,
            comparison_h,
        )
    )

    x_reference = np.linspace(
        problem.x0,
        problem.xf,
        1000,
    )

    y_reference = reference_function(
        x_reference
    )

    plt.figure(
        figsize=(9, 5)
    )

    ref_label = getattr(reference_function, "label", "Exact / Reference")

    plt.plot(
        x_reference,
        y_reference,
        linewidth=2.5,
        label=ref_label,
    )

    plt.plot(
        x_euler,
        y_euler,
        marker="s",
        linewidth=1.5,
        label=f"Euler, h={comparison_h:g}",
    )

    plt.xlabel(
        problem.independent_name
    )

    plt.ylabel(
        problem.dependent_name
    )

    plt.title(
        f"{ref_label} vs Euler's Method (h={comparison_h:g})"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output / "02_reference_vs_euler.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ================================================================
# GRAPH 3: DIFFERENT STEP SIZES
# ================================================================

def plot_different_step_sizes(
    problem: ODEProblem,
    reference_function,
    output: Path,
):

    rhs = build_reference_function(
        problem
    )

    x_reference = np.linspace(
        problem.x0,
        problem.xf,
        1000,
    )

    y_reference = reference_function(
        x_reference
    )

    plt.figure(
        figsize=(9, 5)
    )

    ref_label = getattr(reference_function, "label", "Exact / Reference")

    plt.plot(
        x_reference,
        y_reference,
        linewidth=2.5,
        label=ref_label,
    )

    for h in problem.step_sizes:

        x_values, y_values = (
            euler_method(
                rhs,
                problem.x0,
                problem.y0,
                problem.xf,
                h,
            )
        )

        plt.plot(
            x_values,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"h={h:g}",
        )

    plt.xlabel(
        problem.independent_name
    )

    plt.ylabel(
        problem.dependent_name
    )

    plt.title(
        "Euler's Method for Different Step Sizes"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output / "03_different_step_sizes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ================================================================
# GRAPH 4: EFFECT OF STEP SIZE
# ================================================================

def plot_step_size_effect(
    table: pd.DataFrame,
    problem: ODEProblem,
    output: Path,
):
    step_col = table.columns[0]
    num_col = table.columns[1]
    ref_col = table.columns[2]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        table[step_col],
        table[num_col],
        marker="o",
        linewidth=2,
        label="Euler",
    )

    plt.axhline(
        table[ref_col].iloc[0],
        linestyle="--",
        linewidth=1.5,
        label="Exact / Reference",
    )

    plt.xlabel(
        "Step size, h"
    )

    plt.ylabel(
        f"{problem.dependent_name}"
        f"({problem.xf:g})"
    )

    plt.title(
        "Effect of Step Size on Euler's Method"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output / "04_step_size_effect.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ================================================================
# DISPLAY TABLE
# ================================================================

def print_results(table: pd.DataFrame):
    title = "EULER'S METHOD SUMMARY TABLE"
    print(format_table_console(table, title))


def print_iteration_results(table: pd.DataFrame, h: float):
    title = f"EULER'S METHOD STEP-BY-STEP ITERATION TABLE (h = {h:g})"
    print(format_table_console(table, title))


# ================================================================
# MAIN
# ================================================================

def main():
    while True:
        print("\n=================================")
        print("ODE Numerical Methods")
        print("=================================")
        print("1. Default equation from slides (theta)")
        print("2. Enter your own equation (GUI)")
        print("3. Compare all methods (Default equation)")
        print("4. Exit")
        print()
    
        choice = input("Select a mode (1, 2, 3, or 4): ").strip()
    
        if choice == "4":
            print("Exiting...")
            break
            
        elif choice == "3":
            t_sym = sp.Symbol("t", real=True)
            theta_sym = sp.Symbol("theta", real=True)
            equation_str = "-2.2067e-12*(theta**4 - 81e8)"
            expr = sp.sympify(equation_str, locals={"t": t_sym, "theta": theta_sym})
            
            problem = ODEProblem(
                expression=expr,
                latex=sp.latex(expr),
                independent_name="t",
                dependent_name="theta",
                independent_symbol=t_sym,
                dependent_symbol=theta_sym,
                x0=0.0,
                y0=1200.0,
                xf=480.0,
                step_sizes=(480, 240, 120, 60, 30),
                exact_solution=None,
            )
            print("\nRunning comparison for default equation.")
            reference_function = compute_reference_solution(problem)
            import comparison
            comparison.run_comparison(problem, reference_function)
            continue
            
        elif choice == "1":
            t_sym = sp.Symbol("t", real=True)
            theta_sym = sp.Symbol("theta", real=True)
            equation_str = "-2.2067e-12*(theta**4 - 81e8)"
            expr = sp.sympify(equation_str, locals={"t": t_sym, "theta": theta_sym})
            
            problem = ODEProblem(
                expression=expr,
                latex=sp.latex(expr),
                independent_name="t",
                dependent_name="theta",
                independent_symbol=t_sym,
                dependent_symbol=theta_sym,
                x0=0.0,
                y0=1200.0,
                xf=480.0,
                step_sizes=(480, 240, 120, 60, 30),
                exact_solution=None,
            )
            print("\nUsing default equation.")
            
        else:
            # ------------------------------------------------------------
            # STEP 1: Input GUI + validation
            # ------------------------------------------------------------
        
            problem = get_problem()
        
            if problem is None:
                print(
                    "Program cancelled: "
                    "no validated ODE problem was supplied."
                )
                continue
        
            print(
                "\nInput validation successful."
            )
    
        print(
            f"Equation: "
            f"d{problem.dependent_name}/"
            f"d{problem.independent_name} = "
            f"{problem.latex}"
        )
    
        # ------------------------------------------------------------
        # STEP 2: Reference solution
        # ------------------------------------------------------------
    
        reference_function = (
            compute_reference_solution(
                problem
            )
        )
    
        # ------------------------------------------------------------
        # STEP 3: Euler calculations
        # ------------------------------------------------------------
    
        output = create_output_directory()
    
        table = generate_results_table(
            problem,
            reference_function,
            output,
        )
    
        # ------------------------------------------------------------
        # STEP 4: Print summary and iteration tables
        # ------------------------------------------------------------
    
        print_results(
            table
        )

        sorted_steps = sorted(
            problem.step_sizes
        )
    
        comparison_h = (
            sorted_steps[len(sorted_steps) // 2]
        )

        iter_table = generate_iteration_table(
            problem,
            reference_function,
            comparison_h,
            output,
        )
        print_iteration_results(
            iter_table,
            comparison_h,
        )
    
        # ------------------------------------------------------------
        # STEP 5: Graphs
        # ------------------------------------------------------------
    
        plot_given_function(
            problem,
            output,
        )
    
        # Standardize on finest step size for the primary method-vs-reference plot
        sorted_steps = sorted(
            problem.step_sizes
        )
    
        finest_h = (
            sorted_steps[0]
        )
    
        plot_reference_vs_euler(
            problem,
            reference_function,
            output,
            finest_h,
        )
    
        plot_different_step_sizes(
            problem,
            reference_function,
            output,
        )
    
        plot_step_size_effect(
            table,
            problem,
            output,
        )
    
        print(
            "\nResults saved in:",
            output.resolve(),
        )
        
        comp_ans = input("\nWould you like to run a comparison of all 5 numerical methods for this problem? (y/n): ").strip().lower()
        if comp_ans == 'y':
            import comparison
            comparison.run_comparison(problem, reference_function)


if __name__ == "__main__":
    main()
