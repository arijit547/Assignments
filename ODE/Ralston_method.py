
"""
Ralston_method.py
========
Ralston's Method implementation.

Run:
    python Ralston_method.py

The program first launches the reusable GUI from ode_input_gui.py.
After successful validation, Ralston's method is performed manually.

No built-in ODE solver is used for Ralston's actual calculation.
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
# RALSTON'S METHOD
# ================================================================

def ralston_method(
    f,
    x0: float,
    y0: float,
    xf: float,
    h: float,
):
    """
    Handwritten Ralston's method.

        y_(n+1) = y_n + h*f(x_n, y_n)

    Returns:
        x_values, y_values
    """

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

    # Safety limit against runaway Ralston solutions.
    max_abs_y = 1e100

    for i in range(number_of_steps):

        try:
            k1 = float(
                f(
                    x_values[i],
                    y_values[i],
                )
            )
            k2 = float(
                f(
                    x_values[i] + 0.75 * h,
                    y_values[i] + 0.75 * k1 * h,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ralston failed at step {i}, "
                f"x={x_values[i]:g}, "
                f"y={y_values[i]:g}."
            ) from exc

        if not math.isfinite(k1) or not math.isfinite(k2):
            raise FloatingPointError(
                f"Derivative became invalid at "
                f"x={x_values[i]:g}, "
                f"y={y_values[i]:g}."
            )

        x_values[i + 1] = (
            x_values[i] + h
        )

        y_values[i + 1] = (
            y_values[i] + (1.0/3.0 * k1 + 2.0/3.0 * k2) * h
        )

        if not math.isfinite(
            y_values[i + 1]
        ):
            raise FloatingPointError(
                f"Ralston produced NaN/infinity at "
                f"x={x_values[i+1]:g}."
            )

        if abs(y_values[i + 1]) > max_abs_y:
            raise FloatingPointError(
                f"Ralston solution became unstable near "
                f"x={x_values[i+1]:g}."
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
):
    """
    If the user supplied an exact solution, use it.

    Otherwise solve_ivp is used only to obtain a high-accuracy
    reference solution for comparison.
    """

    if problem.exact_solution is not None:

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

        return reference

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

    return reference


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

def create_output_directory():

    output = Path("ralston_results")
    output.mkdir(
        exist_ok=True
    )

    return output


# ================================================================
# TABLE GENERATION
# ================================================================

def generate_results_table(
    problem: ODEProblem,
    reference_function,
    output: Path,
):

    rhs = build_reference_function(
        problem
    )

    rows = []

    for h in problem.step_sizes:

        x_values, y_values = (
            ralston_method(
                rhs,
                problem.x0,
                problem.y0,
                problem.xf,
                h,
            )
        )

        numerical_final = y_values[-1]

        reference_final = float(
            reference_function(
                problem.xf
            )
        )

        error = (
            reference_final -
            numerical_final
        )

        if reference_final != 0:
            error_percent = (
                abs(error) /
                abs(reference_final)
            ) * 100
        else:
            error_percent = np.nan

        rows.append(
            {
                "Step size h": h,
                "Ralston y(xf)": numerical_final,
                "Reference y(xf)": reference_final,
                "Error": error,
                "Absolute error %": error_percent,
            }
        )

    table = pd.DataFrame(rows)

    table.to_csv(
        output / "ralston_results.csv",
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

def plot_reference_vs_ralston(
    problem: ODEProblem,
    reference_function,
    output: Path,
    comparison_h: float,
):

    rhs = build_reference_function(
        problem
    )

    x_ralston, y_ralston = (
        ralston_method(
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

    plt.plot(
        x_reference,
        y_reference,
        linewidth=2.5,
        label="Exact / Reference",
    )

    plt.plot(
        x_ralston,
        y_ralston,
        marker="s",
        linewidth=1.5,
        label=f"Ralston, h={comparison_h:g}",
    )

    plt.xlabel(
        problem.independent_name
    )

    plt.ylabel(
        problem.dependent_name
    )

    plt.title(
        "Exact / Reference Solution vs Ralston's Method"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output / "02_reference_vs_ralston.png",
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

    plt.plot(
        x_reference,
        y_reference,
        linewidth=2.5,
        label="Exact / Reference",
    )

    for h in problem.step_sizes:

        x_values, y_values = (
            ralston_method(
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
        "Ralston's Method for Different Step Sizes"
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

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        table["Step size h"],
        table["Ralston y(xf)"],
        marker="o",
        linewidth=2,
        label="Ralston",
    )

    plt.axhline(
        table["Reference y(xf)"].iloc[0],
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
        "Effect of Step Size on Ralston's Method"
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

    print("\n")
    print("=" * 90)
    print("RALSTON'S METHOD RESULTS")
    print("=" * 90)

    print(
        table.to_string(
            index=False,
            formatters={
                "Step size h": "{:.6g}".format,
                "Ralston y(xf)": "{:.8f}".format,
                "Reference y(xf)": "{:.8f}".format,
                "Error": "{:.8f}".format,
                "Absolute error %": "{:.6f}".format,
            },
        )
    )


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
        # STEP 3: Ralston calculations
        # ------------------------------------------------------------
    
        output = create_output_directory()
    
        table = generate_results_table(
            problem,
            reference_function,
            output,
        )
    
        # ------------------------------------------------------------
        # STEP 4: Print table
        # ------------------------------------------------------------
    
        print_results(
            table
        )
    
        # ------------------------------------------------------------
        # STEP 5: Graphs
        # ------------------------------------------------------------
    
        plot_given_function(
            problem,
            output,
        )
    
        # Use the middle step size for the first
        # exact/reference comparison when possible.
        sorted_steps = sorted(
            problem.step_sizes
        )
    
        comparison_h = (
            sorted_steps[len(sorted_steps) // 2]
        )
    
        plot_reference_vs_ralston(
            problem,
            reference_function,
            output,
            comparison_h,
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
