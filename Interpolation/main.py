"""
main.py
=======
Master Terminal Launcher for Numerical Interpolation Assignment:
Implemented Methods:
a. Direct Method of Interpolation (Linear, Quadratic, Cubic)
b. Lagrange Method of Interpolation (Linear, Quadratic, Cubic)
c. Newton's Divided Difference Method (Linear, Quadratic, Cubic)

Features:
- Handwritten algorithms with zero built-in black-box solvers.
- Step-by-step systems of equations, basis weight tables, and divided difference tables.
- Master comparison table verifying polynomial uniqueness.
- Graphical User Interface with live LaTeX formula rendering and interactive Matplotlib curves.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local module imports work seamlessly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import direct_interpolation
import lagrange_interpolation
import newton_divided_difference
import interpolation_comparison


def display_menu():
    print("\n" + "=" * 70)
    print("      NUMERICAL METHODS LAB: POLYNOMIAL INTERPOLATION SUITE       ")
    print("=" * 70)
    print("  1. Direct Method of Interpolation (Linear, Quadratic, Cubic)")
    print("  2. Lagrange Method of Interpolation (Linear, Quadratic, Cubic)")
    print("  3. Newton's Divided Difference Method (Linear, Quadratic, Cubic)")
    print("  4. Master Performance & Convergence Comparison Across All Methods")
    print("  5. Launch Interactive Desktop GUI (Live LaTeX & Curve Visualizer)")
    print("  6. Run Full Benchmark Verification (Slides Example: Rocket at t=16s)")
    print("  7. Exit")
    print("=" * 70)


def run_benchmark():
    prob = direct_interpolation.get_slide_default_problem()
    x_data = prob["x"]
    y_data = prob["y"]
    x_target = prob["target"]
    out_dir = Path("interpolation_results/master_benchmark")

    print("\n>>> EXECUTING COMPREHENSIVE BENCHMARK VERIFICATION (Rocket at t = 16 s)")
    comp = interpolation_comparison.run_full_comparison(x_data, y_data, x_target, output_dir=out_dir)

    print(direct_interpolation.format_table_console(
        comp["df_comparison"], f"BENCHMARK INTERPOLATION COMPARISON TABLE (x* = {x_target:g})"
    ))

    print("\n--- Theoretical Property: Uniqueness of Interpolating Polynomial ---")
    if comp["uniqueness_verified"]:
        print(f"  [PASSED] Uniqueness Theorem Verified! Max discrepancy = {comp['max_discrepancy']:.2e}")
        print("  All 3 methods yield identical interpolated values to within machine precision.")

    print(f"\nBenchmark results and plots saved to: {out_dir.resolve()}")


def main():
    while True:
        display_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            direct_interpolation.run_cli()
        elif choice == "2":
            lagrange_interpolation.run_cli()
        elif choice == "3":
            newton_divided_difference.run_cli()
        elif choice == "4":
            interpolation_comparison.run_cli()
        elif choice == "5":
            try:
                import interpolation_gui
                print("\nLaunching Interactive GUI with Live LaTeX View...")
                app = interpolation_gui.InterpolationGUI()
                app.mainloop()
            except Exception as e:
                print(f"Failed to launch GUI: {e}")
        elif choice == "6":
            run_benchmark()
        elif choice == "7":
            print("\nExiting Numerical Interpolation Suite. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number from 1 to 7.")


if __name__ == "__main__":
    main()
