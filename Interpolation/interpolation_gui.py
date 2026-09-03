"""
interpolation_gui.py
====================
Interactive Graphical User Interface for Numerical Interpolation:
- Direct Method, Lagrange Method, and Newton's Divided Difference Method.
- Orders: Linear (1st), Quadratic (2nd), Cubic (3rd), or All Orders together.
- Live LaTeX mathematical preview of the fitted polynomial and evaluation.
- Embedded interactive Matplotlib figure with real-time curve plotting.
- Preset problem selector (Rocket Velocity slide problem, Custom dataset).
- Detailed step-by-step tables (Vandermonde system, Lagrange weights, Newton triangular table).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import sympy as sp

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from direct_interpolation import (
    run_direct_interpolation_suite,
    get_slide_default_problem,
    select_closest_points,
    direct_interpolate_order,
)
from lagrange_interpolation import lagrange_interpolate_order, run_lagrange_interpolation_suite
from newton_divided_difference import newton_interpolate_order, run_newton_interpolation_suite, format_divided_difference_table
from interpolation_comparison import run_full_comparison


class InterpolationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Interpolation Suite - Direct, Lagrange & Newton")
        self.geometry("1240x820")
        self.minsize(1050, 720)
        self.configure(bg="#f8f9fa")

        # Set ttk styles
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("TFrame", background="#f8f9fa")
        self.style.configure("Card.TFrame", background="#ffffff", relief="flat")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#1a73e8", foreground="#ffffff")
        self.style.map("Primary.TButton", background=[("active", "#1765cc")])

        # State variables
        self.preset_var = tk.StringVar(value="Rocket Velocity (Lecture Slide: t=16s)")
        self.x_input_var = tk.StringVar(value="0, 10, 15, 20, 22.5, 30")
        self.y_input_var = tk.StringVar(value="0, 227.04, 362.78, 517.35, 602.97, 901.67")
        self.target_var = tk.StringVar(value="16.0")
        self.method_var = tk.StringVar(value="All Methods (Comparison)")
        self.order_var = tk.StringVar(value="Cubic (3rd Order)")
        self.status_var = tk.StringVar(value="Ready. Click 'Compute Interpolation' to calculate and plot.")

        self._build_ui()
        self.compute_interpolation()

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self, bg="#1a73e8", height=50)
        header.pack(fill="x", side="top")
        tk.Label(
            header, text="NUMERICAL INTERPOLATION SUITE",
            font=("Segoe UI", 14, "bold"), bg="#1a73e8", fg="#ffffff", padx=16, pady=10
        ).pack(side="left")
        tk.Label(
            header, text="Direct • Lagrange • Newton's Divided Difference",
            font=("Segoe UI", 10), bg="#1a73e8", fg="#e8f0fe", padx=16
        ).pack(side="left")

        # Main Paned Window (Left: Controls & LaTeX, Right: Plot & Tables)
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#dadce0", sashwidth=4)
        main_pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Left Column Frame
        left_frame = tk.Frame(main_pane, bg="#f8f9fa", width=460)
        main_pane.add(left_frame, minsize=420)

        # Right Column Frame
        right_frame = tk.Frame(main_pane, bg="#ffffff")
        main_pane.add(right_frame, minsize=520)

        # Build Subsections
        self._build_left_controls(left_frame)
        self._build_right_notebook(right_frame)

    # ============================================================
    # LEFT PANEL: DATA INPUT & LATEX PREVIEW
    # ============================================================

    def _build_left_controls(self, parent):
        container = tk.Frame(parent, bg="#f8f9fa")
        container.pack(fill="both", expand=True, padx=6, pady=6)

        # 1. Preset Selector Card
        preset_card = tk.LabelFrame(container, text=" Dataset Presets ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=10, pady=8)
        preset_card.pack(fill="x", pady=(0, 8))

        preset_cb = ttk.Combobox(
            preset_card, textvariable=self.preset_var,
            values=[
                "Rocket Velocity (Lecture Slide: t=16s)",
                "Thermodynamics (Temperature vs Depth)",
                "Custom Dataset",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        preset_cb.pack(fill="x", pady=4)
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_change)

        # 2. Data Entry Card
        data_card = tk.LabelFrame(container, text=" Discrete Dataset & Query Point ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=10, pady=8)
        data_card.pack(fill="x", pady=(0, 8))

        tk.Label(data_card, text="x-values (comma or space separated):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Entry(data_card, textvariable=self.x_input_var, font=("Consolas", 10)).pack(fill="x", pady=(2, 6))

        tk.Label(data_card, text="y-values (comma or space separated):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Entry(data_card, textvariable=self.y_input_var, font=("Consolas", 10)).pack(fill="x", pady=(2, 6))

        target_row = tk.Frame(data_card, bg="#ffffff")
        target_row.pack(fill="x", pady=(2, 4))
        tk.Label(target_row, text="Target x* to Interpolate:", bg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Entry(target_row, textvariable=self.target_var, font=("Consolas", 11, "bold"), width=12).pack(side="left", padx=8)

        # 3. Method & Order Card
        method_card = tk.LabelFrame(container, text=" Method & Polynomial Order ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=10, pady=8)
        method_card.pack(fill="x", pady=(0, 8))

        tk.Label(method_card, text="Interpolation Method:", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        method_cb = ttk.Combobox(
            method_card, textvariable=self.method_var,
            values=[
                "All Methods (Comparison)",
                "Direct Method",
                "Lagrange Method",
                "Newton's Divided Difference",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        method_cb.pack(fill="x", pady=(2, 6))

        tk.Label(method_card, text="Polynomial Order:", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        order_cb = ttk.Combobox(
            method_card, textvariable=self.order_var,
            values=[
                "Linear (1st Order - 2 points)",
                "Quadratic (2nd Order - 3 points)",
                "Cubic (3rd Order - 4 points)",
                "All Orders (Linear, Quad, Cubic)",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        order_cb.pack(fill="x", pady=(2, 6))

        ttk.Button(
            container, text="Compute Interpolation", command=self.compute_interpolation, style="Primary.TButton"
        ).pack(fill="x", ipady=4, pady=(0, 8))

        # 4. LIVE LATEX MATHEMATICAL DISPLAY CARD
        latex_card = tk.LabelFrame(container, text=" Live Mathematical LaTeX View ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=6, pady=6)
        latex_card.pack(fill="both", expand=True)

        self.latex_fig = Figure(figsize=(4.5, 2.2), dpi=100)
        self.latex_fig.patch.set_facecolor("#ffffff")
        self.latex_ax = self.latex_fig.add_subplot(111)
        self.latex_ax.axis("off")

        self.latex_canvas = FigureCanvasTkAgg(self.latex_fig, master=latex_card)
        self.latex_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ============================================================
    # RIGHT PANEL: MATPLOTLIB CURVES & TABBED WORKINGS
    # ============================================================

    def _build_right_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Interactive Curve Plot
        plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(plot_tab, text=" Interpolation Curves ")

        self.plot_fig = Figure(figsize=(6.5, 4.5), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_tab)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(plot_tab, bg="#ffffff")
        toolbar_frame.pack(fill="x", side="bottom")
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, toolbar_frame)
        self.toolbar.update()

        # Tab 2: Master Comparison Table
        comp_tab = ttk.Frame(self.notebook)
        self.notebook.add(comp_tab, text=" Comparison Table ")
        self.comp_text = tk.Text(comp_tab, font=("Consolas", 10), wrap="none", bg="#ffffff", bd=0)
        comp_scroll_y = ttk.Scrollbar(comp_tab, orient="vertical", command=self.comp_text.yview)
        comp_scroll_x = ttk.Scrollbar(comp_tab, orient="horizontal", command=self.comp_text.xview)
        self.comp_text.configure(xscrollcommand=comp_scroll_x.set, yscrollcommand=comp_scroll_y.set)
        comp_scroll_y.pack(side="right", fill="y")
        comp_scroll_x.pack(side="bottom", fill="x")
        self.comp_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Tab 3: Detailed Step-by-Step Workings
        steps_tab = ttk.Frame(self.notebook)
        self.notebook.add(steps_tab, text=" Step-by-Step Workings ")
        self.steps_text = tk.Text(steps_tab, font=("Consolas", 10), wrap="none", bg="#ffffff", bd=0)
        steps_scroll_y = ttk.Scrollbar(steps_tab, orient="vertical", command=self.steps_text.yview)
        steps_scroll_x = ttk.Scrollbar(steps_tab, orient="horizontal", command=self.steps_text.xview)
        self.steps_text.configure(xscrollcommand=steps_scroll_x.set, yscrollcommand=steps_scroll_y.set)
        steps_scroll_y.pack(side="right", fill="y")
        steps_scroll_x.pack(side="bottom", fill="x")
        self.steps_text.pack(fill="both", expand=True, padx=6, pady=6)

    # ============================================================
    # EVENT HANDLERS & COMPUTATION ENGINE
    # ============================================================

    def _on_preset_change(self, event=None):
        choice = self.preset_var.get()
        if "Rocket" in choice:
            self.x_input_var.set("0, 10, 15, 20, 22.5, 30")
            self.y_input_var.set("0, 227.04, 362.78, 517.35, 602.97, 901.67")
            self.target_var.set("16.0")
        elif "Thermodynamics" in choice:
            self.x_input_var.set("0, 1.2, 2.4, 3.6, 4.8, 6.0")
            self.y_input_var.set("18.5, 16.2, 14.8, 13.5, 12.9, 12.5")
            self.target_var.set("2.0")
        self.compute_interpolation()

    def parse_inputs(self) -> tuple[np.ndarray, np.ndarray, float]:
        x_str = self.x_input_var.get().replace(",", " ").strip()
        y_str = self.y_input_var.get().replace(",", " ").strip()
        target_str = self.target_var.get().strip()

        xs = [float(x) for x in x_str.split() if x]
        ys = [float(y) for y in y_str.split() if y]
        target = float(target_str)

        if len(xs) != len(ys):
            raise ValueError(f"Number of x-values ({len(xs)}) must match number of y-values ({len(ys)}).")
        if len(xs) < 2:
            raise ValueError("At least 2 data points are required for interpolation.")

        # Sort dataset by x
        arr_x = np.array(xs, dtype=float)
        arr_y = np.array(ys, dtype=float)
        idx = np.argsort(arr_x)
        return arr_x[idx], arr_y[idx], target

    def compute_interpolation(self):
        try:
            x_data, y_data, x_target = self.parse_inputs()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        method_choice = self.method_var.get()
        order_choice = self.order_var.get()

        # Run comparison suite
        comp = run_full_comparison(x_data, y_data, x_target)
        direct_res = comp["direct_res"]
        lagrange_res = comp["lagrange_res"]
        newton_res = comp["newton_res"]

        # Update Comparison Tab
        self.comp_text.delete("1.0", tk.END)
        self.comp_text.insert(tk.END, "========================================================================================\n")
        self.comp_text.insert(tk.END, f"            NUMERICAL INTERPOLATION PERFORMANCE COMPARISON (x* = {x_target:g})            \n")
        self.comp_text.insert(tk.END, "========================================================================================\n\n")
        self.comp_text.insert(tk.END, comp["df_comparison"].to_string(index=False) + "\n\n")

        self.comp_text.insert(tk.END, "--- Theoretical Uniqueness Verification ---\n")
        if comp["uniqueness_verified"]:
            self.comp_text.insert(tk.END, f"[PASSED] Uniqueness Theorem: P_Direct == P_Lagrange == P_Newton (Diff: {comp['max_discrepancy']:.2e})\n")
            self.comp_text.insert(tk.END, "Insight: Direct, Lagrange, and Newton are 3 distinct computational formulations of the UNIQUE interpolating polynomial.\n\n")

        self.comp_text.insert(tk.END, "--- Method Trade-Offs ---\n")
        self.comp_text.insert(tk.END, "• Direct Method: O(n^3) - Solves linear system; ill-conditioned for n > 5.\n")
        self.comp_text.insert(tk.END, "• Lagrange Method: O(n^2) - Clean basis formula; adding points requires full recalculation.\n")
        self.comp_text.insert(tk.END, "• Newton's Divided Diff: O(n^2) setup, O(n) Horner eval - Incremental updates without recomputing!\n")

        # Update Detailed Steps Tab
        self.steps_text.delete("1.0", tk.END)
        self.steps_text.insert(tk.END, f"=== DETAILED CALCULATION WORKINGS (Target x* = {x_target:g}) ===\n\n")

        for r in direct_res:
            self.steps_text.insert(tk.END, f"--- {r.name} Direct Method ---\n")
            self.steps_text.insert(tk.END, f"Points Used: {', '.join(f'({x:g}, {y:g})' for x, y in zip(r.x_points, r.y_points))}\n")
            self.steps_text.insert(tk.END, f"Coefficients: {', '.join(f'a{i}={c:.6f}' for i, c in enumerate(r.coefficients))}\n")
            self.steps_text.insert(tk.END, f"Polynomial: {r.polynomial_string}\n")
            self.steps_text.insert(tk.END, f"Value at {x_target:g}: {r.interpolated_value:.4f}\n")
            if r.approx_error_percent is not None:
                self.steps_text.insert(tk.END, f"Approx Error |ea|: {r.approx_error_percent:.5f}%\n")
            self.steps_text.insert(tk.END, "\n")

        # Newton table for Cubic
        cubic_newt = newton_res[2]
        self.steps_text.insert(tk.END, "--- Newton's Divided Difference Triangular Table (Order 3) ---\n")
        self.steps_text.insert(tk.END, format_divided_difference_table(cubic_newt.x_points, cubic_newt.table) + "\n\n")

        # Update Plots
        self._update_plot(x_data, y_data, direct_res, x_target, order_choice)

        # Update Live LaTeX View
        self._update_latex_view(direct_res, x_target, order_choice)

    # ============================================================
    # MATPLOTLIB PLOT UPDATER
    # ============================================================

    def _update_plot(self, x_all, y_all, results, x_target, order_choice):
        self.plot_ax.clear()

        # Data points
        self.plot_ax.scatter(x_all, y_all, color="black", s=60, zorder=6, label="Data Points")

        x_min = min(r.x_points[0] for r in results)
        x_max = max(r.x_points[-1] for r in results)
        margin = max(1.0, (x_max - x_min) * 0.12)
        x_dense = np.linspace(max(0, x_min - margin), x_max + margin, 350)

        color_map = {1: "#1a73e8", 2: "#0d904f", 3: "#d93025"}
        style_map = {1: "--", 2: "-.", 3: "-"}

        for r in results:
            should_plot = ("All" in order_choice) or (r.order == 1 and "Linear" in order_choice) or (r.order == 2 and "Quadratic" in order_choice) or (r.order == 3 and "Cubic" in order_choice)
            if should_plot:
                y_dense = r.evaluate(x_dense)
                c = color_map[r.order]
                ls = style_map[r.order]
                self.plot_ax.plot(x_dense, y_dense, color=c, linestyle=ls, linewidth=2.2, label=f"{r.name}: {r.interpolated_value:.2f}")
                self.plot_ax.plot(x_target, r.interpolated_value, marker="o", markersize=8, color=c, markeredgecolor="black")

        self.plot_ax.axvline(x_target, color="#5f6368", linestyle=":", alpha=0.8, label=f"x* = {x_target:g}")
        self.plot_ax.set_title(f"Polynomial Interpolation Curves (Target x* = {x_target:g})", fontsize=12, fontweight="bold")
        self.plot_ax.set_xlabel("x", fontsize=10)
        self.plot_ax.set_ylabel("f(x)", fontsize=10)
        self.plot_ax.grid(True, linestyle="--", alpha=0.4)
        self.plot_ax.legend(loc="best", fontsize=9)
        self.plot_fig.tight_layout()
        self.plot_canvas.draw_idle()

    # ============================================================
    # LIVE LATEX MATHEMATICAL RENDERING
    # ============================================================

    def _update_latex_view(self, results, x_target, order_choice):
        self.latex_ax.clear()
        self.latex_ax.axis("off")

        # Pick active result for display
        if "Linear" in order_choice:
            res = results[0]
        elif "Quadratic" in order_choice:
            res = results[1]
        else:
            res = results[2]  # Cubic or All

        coeffs = res.coefficients
        n = len(coeffs)

        # Generate LaTeX representation of P_n(x)
        terms = []
        for i, c in enumerate(coeffs):
            if i == 0:
                terms.append(f"{c:.4f}")
            elif i == 1:
                sign = "+" if c >= 0 else "-"
                terms.append(f"{sign} {abs(c):.4f}x")
            else:
                sign = "+" if c >= 0 else "-"
                terms.append(f"{sign} {abs(c):.4f}x^{{{i}}}")

        poly_tex = " ".join(terms)
        val_tex = f"f({x_target:g}) = {res.interpolated_value:.4f}"
        error_tex = f"|\\epsilon_a| = {res.approx_error_percent:.4f}\\%" if res.approx_error_percent is not None else "|\\epsilon_a| = \\text{N/A}"

        full_latex = (
            f"$\\mathbf{{{res.name}}}$\n\n"
            f"$P_{{{res.order}}}(x) = {poly_tex}$\n\n"
            f"${val_tex} \\quad ({error_tex})$"
        )

        fsize = 13 if len(poly_tex) > 45 else 15

        self.latex_ax.text(
            0.5, 0.5, full_latex,
            ha="center", va="center", fontsize=fsize, color="#1a73e8"
        )
        self.latex_fig.tight_layout()
        self.latex_canvas.draw_idle()


def main():
    app = InterpolationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
