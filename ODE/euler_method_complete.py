"""
Euler's Method for the ODE problem in the supplied slides.

Only the ODE equation needs to be changed in `equation`.
The remaining requirements are taken from the supplied figures:

    theta(0) = 1200 K
    final time = 480 s
    step sizes = 480, 240, 120, 60, 30 s
    exact theta(480) = 647.57 K

Required packages:
    pip install numpy pandas matplotlib scipy sympy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from pathlib import Path


# ============================================================
# INPUT: ONLY THE ODE EQUATION
# ============================================================
#
# dy/dt = f(t, y)
#
# Enter only the RHS of the differential equation here.
#
equation = "-2.2067e-12*(theta**4 - 81e8)"


# ============================================================
# FIXED PROBLEM DATA FROM THE SUPPLIED SLIDES
# ============================================================

y_name = "theta"
y0 = 1200.0
t0 = 0.0
tf = 480.0

step_sizes = [480, 240, 120, 60, 30]

# Exact value shown in the slide
exact_final = 647.57


# ============================================================
# CONVERT THE ENTERED EQUATION INTO A PYTHON FUNCTION
# ============================================================

t_symbol = sp.symbols("t")
theta_symbol = sp.symbols("theta")

rhs = sp.sympify(
    equation,
    locals={
        "t": t_symbol,
        "theta": theta_symbol
    }
)

f_sym = sp.lambdify(
    (t_symbol, theta_symbol),
    rhs,
    modules="numpy"
)


def f(t, theta):
    return float(f_sym(t, theta))


# ============================================================
# EULER'S METHOD
# ============================================================

def euler_method(f, t0, y0, tf, h):
    n = int(round((tf - t0) / h))

    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    t[0] = t0
    y[0] = y0

    for i in range(n):
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


# ============================================================
# HIGH-ACCURACY REFERENCE SOLUTION
#
# This is used as the "Exact Solution" curve.  The supplied
# slide gives theta(480) = 647.57 K.
#
# A very small maximum step and strict tolerances make this
# reference solution much more accurate than Euler's solution.
# ============================================================

reference = solve_ivp(
    lambda t, y: [f(t, y[0])],
    (t0, tf),
    [y0],
    method="DOP853",
    rtol=1e-12,
    atol=1e-12,
    dense_output=True,
    max_step=0.25
)


def exact_solution(t):
    return reference.sol(t)[0]


# ============================================================
# GENERATE EULER SOLUTIONS
# ============================================================

solutions = {}

for h in step_sizes:
    t, theta = euler_method(
        f,
        t0,
        y0,
        tf,
        h
    )

    solutions[h] = {
        "t": t,
        "theta": theta
    }


# ============================================================
# TABLE 1
# Temperature at 480 seconds as a function of step size
#
# Exactly the structure shown in the supplied slide:
#
# h       theta(480)       Et       |et|%
# ============================================================

rows = []

for h in step_sizes:

    theta_480 = solutions[h]["theta"][-1]

    true_error = exact_final - theta_480

    relative_error = (
        abs(true_error) / exact_final
    ) * 100

    rows.append([
        h,
        theta_480,
        true_error,
        relative_error
    ])


table = pd.DataFrame(
    rows,
    columns=[
        "Step, h",
        "theta(480)",
        "Et",
        "|et|%"
    ]
)


# ============================================================
# DISPLAY TABLE
# ============================================================

print("\n")
print("=" * 70)
print("TABLE 1. Temperature at 480 seconds as a function of step size, h")
print("=" * 70)

print(
    table.to_string(
        index=False,
        formatters={
            "theta(480)": "{:.2f}".format,
            "Et": "{:.2f}".format,
            "|et|%": "{:.4f}".format
        }
    )
)

print("\nExact theta(480) =", exact_final, "K")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output = Path("euler_results")
output.mkdir(exist_ok=True)

table.to_csv(
    output / "euler_step_size_table.csv",
    index=False
)


# ============================================================
# COMMON GRAPH SETTINGS
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# GRAPH 1
#
# Comparing exact and Euler's method for h = 240
#
# Matches the first graph in the supplied figures.
# ============================================================

t_exact = np.linspace(t0, tf, 1000)
theta_exact = exact_solution(t_exact)

t240 = solutions[240]["t"]
theta240 = solutions[240]["theta"]

plt.figure(figsize=(10, 6))

plt.plot(
    t_exact,
    theta_exact,
    linewidth=2.5,
    label="Exact Solution"
)

plt.plot(
    t240,
    theta240,
    marker="s",
    linewidth=1.8,
    label="Euler, h = 240 s"
)

plt.xlabel("Time, t (sec)", fontsize=12)
plt.ylabel("Temperature, θ(K)", fontsize=12)

plt.title(
    "Comparison of Exact and Euler's Method",
    fontsize=16,
    fontweight="bold"
)

plt.xlim(0, 500)
plt.ylim(0, 1400)

plt.xticks(np.arange(0, 501, 100))
plt.yticks(np.arange(0, 1401, 200))

plt.grid(True, alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    output / "figure_3_exact_vs_euler_h240.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 2
#
# Comparison with exact results for h = 120, 240, 480
#
# Matches the second graph in the supplied figures.
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    t_exact,
    theta_exact,
    linewidth=2.5,
    label="Exact solution"
)

markers = {
    120: "D",
    240: "s",
    480: "o"
}

for h in [120, 240, 480]:

    t_h = solutions[h]["t"]
    theta_h = solutions[h]["theta"]

    plt.plot(
        t_h,
        theta_h,
        marker=markers[h],
        linewidth=1.6,
        markersize=5,
        label=f"h = {h}"
    )

plt.xlabel("Time, t (sec)", fontsize=12)
plt.ylabel("Temperature, θ(K)", fontsize=12)

plt.title(
    "Comparison with Exact Results",
    fontsize=16,
    fontweight="bold"
)

plt.xlim(0, 500)
plt.ylim(-1500, 1500)

plt.xticks(np.arange(0, 501, 100))
plt.yticks(np.arange(-1500, 1501, 500))

plt.axhline(
    0,
    linewidth=0.8
)

plt.grid(True, alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    output / "figure_4_exact_vs_euler_different_h.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 3
#
# Effect of step size on Euler's Method
#
# x-axis  : step size h
# y-axis  : theta(480)
# ============================================================

h_values = np.array(step_sizes)
theta_values = np.array([
    solutions[h]["theta"][-1]
    for h in step_sizes
])

plt.figure(figsize=(10, 6))

plt.plot(
    h_values,
    theta_values,
    marker="o",
    linewidth=2,
    markersize=7
)

plt.xlabel(
    "Step size, h (s)",
    fontsize=12
)

plt.ylabel(
    "Temperature, θ(480) (K)",
    fontsize=12
)

plt.title(
    "Effect of Step Size on Euler's Method",
    fontsize=16,
    fontweight="bold"
)

plt.xlim(0, 500)
plt.ylim(-1200, 800)

plt.xticks(np.arange(0, 501, 100))
plt.yticks(np.arange(-1200, 801, 200))

plt.axhline(
    0,
    linewidth=0.8
)

plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.savefig(
    output / "figure_5_effect_of_step_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE A TEXT SUMMARY
# ============================================================

with open(output / "results.txt", "w", encoding="utf-8") as file:

    file.write(
        "Euler's Method Results\n"
        "======================\n\n"
    )

    file.write(
        f"ODE: dtheta/dt = {equation}\n"
    )

    file.write(
        f"Initial condition: theta(0) = {y0} K\n"
    )

    file.write(
        f"Final time: {tf} s\n"
    )

    file.write(
        f"Exact theta(480): {exact_final} K\n\n"
    )

    file.write(
        table.to_string(index=False)
    )

print("\nAll results saved in:")
print(output.resolve())
