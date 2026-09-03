"""
2023331039-3-MAIN.py
====================
Master runner for the Numerical ODE Solver & Analysis Suite:
1. Euler's Method (1st Order Runge-Kutta)
2. Heun's Method (2nd Order Runge-Kutta, a2 = 1/2)
3. Midpoint Method (2nd Order Runge-Kutta, a2 = 1)
4. Ralston's Method (2nd Order Runge-Kutta, a2 = 2/3)
5. Classical Runge-Kutta 4th Order Method (RK4)
6. Comprehensive Comparison & Convergence Analysis (All 5 Methods)
7. Slide Benchmark Problem (Radiation Cooling of Sphere)
8. Launch Interactive Problem Input GUI

Run:
    python 2023331039-3-MAIN.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import ode_input_gui
euler = importlib.import_module("2023331039-3-EM")
Heun = importlib.import_module("2023331039-3-HM")
Midpoint_method = importlib.import_module("2023331039-3-MM")
Ralston_method = importlib.import_module("2023331039-3-RM")
RK4_method = importlib.import_module("2023331039-3-RK4")
import comparison


def run_slide_benchmark():
    """Runs and compares all 5 methods on the slide radiation cooling problem."""
    problem = ode_input_gui.get_slide_default_problem()
    print("\n==================================================================")
    print("      BENCHMARK PROBLEM: RADIATION COOLING OF A SPHERICAL BODY    ")
    print("==================================================================")
    print(f"Differential Equation: d{problem.dependent_name}/d{problem.independent_name} = {problem.expression}")
    print(f"Initial Condition:     {problem.dependent_name}({problem.x0:g}) = {problem.y0:g} K")
    print(f"Integration Interval:  {problem.independent_name} = {problem.x0:g} -> {problem.xf:g} s")
    print(f"Step Sizes Evaluated:  {problem.step_sizes} s")
    print("==================================================================")
    
    comparison.run_comparison(problem, output_dir="comparison_results/benchmark")


def main():
    while True:
        print("\n=================================================================")
        print("             NUMERICAL ODE SOLVER & ANALYSIS SYSTEM              ")
        print("=================================================================")
        print("1. Euler's Method (1st Order Runge-Kutta)")
        print("2. Heun's Method (2nd Order Runge-Kutta, a2 = 1/2)")
        print("3. Midpoint Method (2nd Order Runge-Kutta, a2 = 1)")
        print("4. Ralston's Method (2nd Order Runge-Kutta, a2 = 2/3)")
        print("5. Classical Runge-Kutta 4th Order Method (RK4)")
        print("6. Comprehensive Comparison & Convergence Analysis (All 5 Methods)")
        print("7. Slide Benchmark Problem (Radiation Cooling of Sphere)")
        print("8. Launch Interactive Problem Input GUI")
        print("9. Exit")

        choice = input("\nSelect an option (1-9): ").strip()

        if choice == "1":
            euler.main()
        elif choice == "2":
            Heun.main()
        elif choice == "3":
            Midpoint_method.main()
        elif choice == "4":
            Ralston_method.main()
        elif choice == "5":
            RK4_method.main()
        elif choice == "6":
            comparison.main()
        elif choice == "7":
            run_slide_benchmark()
        elif choice == "8":
            problem = ode_input_gui.get_problem("ODE Problem Configurator")
            if problem is not None:
                print("\n[SUCCESS] Configured and validated ODE Problem:")
                print(f"  d{problem.dependent_name}/d{problem.independent_name} = {problem.expression}")
                print(f"  Initial: {problem.dependent_name}({problem.x0:g}) = {problem.y0:g}")
                print(f"  Domain:  {problem.x0:g} -> {problem.xf:g}")
                print(f"  Step sizes: {problem.step_sizes}")
                run_comp = input("\nWould you like to run full 5-method comparison on this problem now? (y/n): ").strip().lower()
                if run_comp == "y":
                    comparison.run_comparison(problem)
            else:
                print("\nGUI window closed without submission.")
        elif choice == "9":
            print("\nExiting ODE Numerical Suite. Goodbye!")
            break
        else:
            print("\nInvalid selection. Please enter a number between 1 and 9.")


if __name__ == "__main__":
    main()
