
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pathlib

from euler import euler_method, ODEProblem, compute_reference_solution
from Heun import heun_method
from Midpoint_method import midpoint_method
from Ralston_method import ralston_method
from RK4_method import rk4_method

def run_comparison(problem, reference_function, output_dir="comparison_results"):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n==========================================================================================")
    print("COMPARISON OF ALL METHODS")
    print("==========================================================================================")
    
    # Header
    print(f"{'Step size h':>15} | {'Euler':>10} | {'Heun':>10} | {'Midpoint':>10} | {'Ralston':>10} | {'RK4':>10} | {'Exact':>10}")
    print("-" * 90)
    
    # 1. Print Table
    results = {}
    
    # Lambdify the expression
    f_lambdified = sp.lambdify(
        (problem.independent_symbol, problem.dependent_symbol),
        problem.expression,
        modules=["math"],
    )
    
    for h in sorted(problem.step_sizes, reverse=True):
        sol_euler = euler_method(f_lambdified, problem.x0, problem.y0, problem.xf, h)
        sol_heun = heun_method(f_lambdified, problem.x0, problem.y0, problem.xf, h)
        sol_mid = midpoint_method(f_lambdified, problem.x0, problem.y0, problem.xf, h)
        sol_ralston = ralston_method(f_lambdified, problem.x0, problem.y0, problem.xf, h)
        sol_rk4 = rk4_method(f_lambdified, problem.x0, problem.y0, problem.xf, h)
        
        y_euler = sol_euler[1][-1]
        y_heun = sol_heun[1][-1]
        y_mid = sol_mid[1][-1]
        y_ralston = sol_ralston[1][-1]
        y_rk4 = sol_rk4[1][-1]
        y_exact = reference_function(problem.xf)
        
        results[h] = {
            'euler': sol_euler,
            'heun': sol_heun,
            'midpoint': sol_mid,
            'ralston': sol_ralston,
            'rk4': sol_rk4,
            'exact': y_exact
        }
        
        print(f"{h:>15} | {y_euler:>10.2f} | {y_heun:>10.2f} | {y_mid:>10.2f} | {y_ralston:>10.2f} | {y_rk4:>10.2f} | {y_exact:>10.2f}")
        
    # 2. Plot graph (compare for h=240, or the second to largest step size to make it visible)
    if len(problem.step_sizes) > 1:
        plot_h = sorted(problem.step_sizes, reverse=True)[1]
    else:
        plot_h = problem.step_sizes[0]
        
    sol_h = results[plot_h]
    
    plt.figure(figsize=(10, 6))
    
    # Plot Exact
    x_fine = np.linspace(problem.x0, problem.xf, 200)
    y_fine = [reference_function(x) for x in x_fine]
    plt.plot(x_fine, y_fine, 'k-', linewidth=2, label="Exact")
    
    # Plot numerical
    plt.plot(sol_h['euler'][0], sol_h['euler'][1], 'ro-', label="Euler")
    plt.plot(sol_h['heun'][0], sol_h['heun'][1], 'bs-', label="Heun")
    plt.plot(sol_h['midpoint'][0], sol_h['midpoint'][1], 'g^-', label="Midpoint")
    plt.plot(sol_h['ralston'][0], sol_h['ralston'][1], 'md-', label="Ralston")
    plt.plot(sol_h['rk4'][0], sol_h['rk4'][1], 'cx-', label="RK4", markersize=8)
    
    plt.title(f"Comparison of Numerical Methods (h = {plot_h})")
    plt.xlabel(f"{problem.independent_name}")
    plt.ylabel(f"{problem.dependent_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / f"comparison_plot_h{plot_h}.png")
    plt.show(block=False)
    plt.pause(30.0)
    plt.close()
    
    print(f"\nComparison results saved in: {output_path.resolve()}")
