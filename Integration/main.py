"""
main.py
=======
Master runner for the Numerical Integration Suite:
a. Trapezoidal Rule (Single & Multiple Segments)
b. Simpson's 1/3 Rule (2 Segments & Multiple Segments)
c. Combination of Trapezoidal and Simpson's 1/3 Method for Datasets

Run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import trapezoidal
import simpson
import combination


def compare_all_on_slide_benchmark():
    """Compares Trapezoidal vs Simpson's 1/3 on the Rocket problem from slides."""
    prob = trapezoidal.get_slide_default_problem()
    f = prob["func"]
    a, b = prob["a"], prob["b"]
    exact = prob["exact"]

    segment_counts = [2, 4, 6, 8, 10, 16]

    rows = []
    for n in segment_counts:
        trap_val, _, _ = trapezoidal.trapezoidal_multiple(f, a, b, n)
        simp_val, _, _ = simpson.simpson_multiple(f, a, b, n)

        err_t = exact - trap_val
        pct_t = abs(err_t) / exact * 100

        err_s = exact - simp_val
        pct_s = abs(err_s) / exact * 100

        rows.append({
            "Segments (n)": n,
            "Trapezoidal Value": trap_val,
            "Trap |et| (%)": pct_t,
            "Simpson 1/3 Value": simp_val,
            "Simpson |et| (%)": pct_s,
            "Accuracy Gain": f"{pct_t / pct_s:.1f}x better" if pct_s > 0 else "Optimal",
        })

    df = pd.DataFrame(rows)
    print(trapezoidal.format_table_console(
        df,
        "BENCHMARK COMPARISON: TRAPEZOIDAL VS SIMPSON'S 1/3 (ROCKET PROBLEM)"
    ))
    out_dir = Path("integration_results")
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / "benchmark_trapezoidal_vs_simpson.csv", index=False)
    print(f"\nBenchmark table saved to: {(out_dir / 'benchmark_trapezoidal_vs_simpson.csv').resolve()}")


def main():
    while True:
        print("\n=================================================================")
        print("          NUMERICAL INTEGRATION ASSIGNMENT SYSTEM                ")
        print("=================================================================")
        print("1. Trapezoidal Rule (Single & Multiple Segments)")
        print("2. Simpson's 1/3 Rule (2 Segments & Multiple Segments)")
        print("3. Combination Method: Pattern Recognition & Decision Engine")
        print("4. Benchmark Comparison (Trapezoidal vs Simpson's 1/3)")
        print("5. Launch Interactive Graphical User Interface (GUI)")
        print("6. Exit")

        choice = input("\nSelect an option (1-6): ").strip()

        if choice == "1":
            trapezoidal.run_cli()
        elif choice == "2":
            simpson.run_cli()
        elif choice == "3":
            combination.run_cli()
        elif choice == "4":
            compare_all_on_slide_benchmark()
        elif choice == "5":
            try:
                import integration_gui
                integration_gui.launch_gui()
            except Exception as e:
                print(f"Could not open GUI window: {e}")
        elif choice == "6":
            print("Exiting Integration Suite. Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
