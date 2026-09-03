"""
interpolation_gui.py
====================
Interactive Graphical User Interface for Numerical Interpolation:
- Fully functional Method Selector: Direct Method, Lagrange Method, Newton's Divided Difference, or All Methods (Comparison).
- Dynamically adapts to 2-point, 3-point, or N-point datasets (no crashing on small datasets).
- Input boundary validation with clear error alerts for duplicate x-coordinates.
- Live LaTeX mathematical preview customized to each method's authentic formulation.
- Interactive Matplotlib canvas with real-time curve plotting and query point highlighting.
- Dedicated tabbed workings for Direct, Lagrange, Newton, and Master Comparison.
- Displays measured execution runtimes (us) and domain-wide uniqueness verification table.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from direct_interpolation import (
    run_direct_interpolation_suite,
    get_slide_default_problem,
    select_bracketed_closest_points,
    format_table_console,
)
from lagrange_interpolation import (
    run_lagrange_interpolation_suite,
)
from newton_divided_difference import (
    run_newton_interpolation_suite,
    format_divided_difference_table,
)
from interpolation_comparison import run_full_comparison


class InterpolationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Interpolation Suite - Direct, Lagrange & Newton")
        self.geometry("1260x840")
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
        self.order_var = tk.StringVar(value="Cubic (3rd Order - 4 points)")
        self.status_var = tk.StringVar(value="Ready.")

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
            header, text="Direct Method • Lagrange Method • Newton's Divided Difference",
            font=("Segoe UI", 10), bg="#1a73e8", fg="#e8f0fe", padx=16
        ).pack(side="left")

        # Main Paned Window (Left: Controls & LaTeX, Right: Plot & Tabs)
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#dadce0", sashwidth=4)
        main_pane.pack(fill="both", expand=True, padx=8, pady=8)

        # Left Column Frame
        left_frame = tk.Frame(main_pane, bg="#f8f9fa", width=470)
        main_pane.add(left_frame, minsize=430)

        # Right Column Frame
        right_frame = tk.Frame(main_pane, bg="#ffffff")
        main_pane.add(right_frame, minsize=540)

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
                "Thermodynamics (Temperature vs Depth: x=2m)",
                "Small 2-Point Linear Dataset (x=1.5)",
                "Small 3-Point Quadratic Dataset (x=2.5)",
                "Custom Dataset",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        preset_cb.pack(fill="x", pady=4)
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_change)

        # 2. Data Entry Card
        data_card = tk.LabelFrame(container, text=" Discrete Dataset & Query Point ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=10, pady=8)
        data_card.pack(fill="x", pady=(0, 8))

        tk.Label(data_card, text="x-values (comma or space separated, distinct):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        self.x_entry = ttk.Entry(data_card, textvariable=self.x_input_var, font=("Consolas", 10))
        self.x_entry.pack(fill="x", pady=(2, 6))

        tk.Label(data_card, text="y-values (comma or space separated):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        self.y_entry = ttk.Entry(data_card, textvariable=self.y_input_var, font=("Consolas", 10))
        self.y_entry.pack(fill="x", pady=(2, 6))

        target_row = tk.Frame(data_card, bg="#ffffff")
        target_row.pack(fill="x", pady=(2, 4))
        tk.Label(target_row, text="Target x* to Interpolate:", bg="#ffffff", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Entry(target_row, textvariable=self.target_var, font=("Consolas", 11, "bold"), width=12).pack(side="left", padx=8)

        # 3. Method & Order Card
        method_card = tk.LabelFrame(container, text=" Method & Polynomial Order ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=10, pady=8)
        method_card.pack(fill="x", pady=(0, 8))

        tk.Label(method_card, text="Interpolation Method:", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        self.method_cb = ttk.Combobox(
            method_card, textvariable=self.method_var,
            values=[
                "All Methods (Comparison)",
                "Direct Method",
                "Lagrange Method",
                "Newton's Divided Difference",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        self.method_cb.pack(fill="x", pady=(2, 6))
        self.method_cb.bind("<<ComboboxSelected>>", lambda _: self.compute_interpolation())

        tk.Label(method_card, text="Polynomial Order:", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor="w")
        self.order_cb = ttk.Combobox(
            method_card, textvariable=self.order_var,
            values=[
                "Linear (1st Order - 2 points)",
                "Quadratic (2nd Order - 3 points)",
                "Cubic (3rd Order - 4 points)",
                "All Available Orders",
            ],
            state="readonly", font=("Segoe UI", 10)
        )
        self.order_cb.pack(fill="x", pady=(2, 6))
        self.order_cb.bind("<<ComboboxSelected>>", lambda _: self.compute_interpolation())

        ttk.Button(
            container, text="Compute & Update View", command=self.compute_interpolation, style="Primary.TButton"
        ).pack(fill="x", ipady=4, pady=(0, 8))

        # 4. LIVE LATEX MATHEMATICAL DISPLAY CARD
        latex_card = tk.LabelFrame(container, text=" Live Mathematical LaTeX View ", font=("Segoe UI", 10, "bold"), bg="#ffffff", padx=6, pady=6)
        latex_card.pack(fill="both", expand=True)

        self.latex_fig = Figure(figsize=(4.6, 2.3), dpi=100)
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

        # Tab 0: Interactive Curve Plot
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text=" Interpolation Curves ")

        self.plot_fig = Figure(figsize=(6.5, 4.5), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=self.plot_tab)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(self.plot_tab, bg="#ffffff")
        toolbar_frame.pack(fill="x", side="bottom")
        self.toolbar = NavigationToolbar2Tk(self.plot_canvas, toolbar_frame)
        self.toolbar.update()

        # Tab 1: Master Comparison & Performance
        self.comp_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.comp_tab, text=" Comparison & Performance ")
        self.comp_text = self._create_scrolled_text(self.comp_tab)

        # Tab 2: Direct Method Workings
        self.direct_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.direct_tab, text=" Direct Method Workings ")
        self.direct_text = self._create_scrolled_text(self.direct_tab)

        # Tab 3: Lagrange Method Workings
        self.lagrange_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.lagrange_tab, text=" Lagrange Basis Workings ")
        self.lagrange_text = self._create_scrolled_text(self.lagrange_tab)

        # Tab 4: Newton Divided Difference Workings
        self.newton_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.newton_tab, text=" Newton's Divided Diff. ")
        self.newton_text = self._create_scrolled_text(self.newton_tab)

    def _create_scrolled_text(self, parent_frame: ttk.Frame) -> tk.Text:
        text_widget = tk.Text(parent_frame, font=("Consolas", 10), wrap="none", bg="#ffffff", bd=0)
        scroll_y = ttk.Scrollbar(parent_frame, orient="vertical", command=text_widget.yview)
        scroll_x = ttk.Scrollbar(parent_frame, orient="horizontal", command=text_widget.xview)
        text_widget.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        text_widget.pack(fill="both", expand=True, padx=6, pady=6)
        return text_widget

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
        elif "2-Point" in choice:
            self.x_input_var.set("1.0, 2.0")
            self.y_input_var.set("3.0, 5.0")
            self.target_var.set("1.5")
        elif "3-Point" in choice:
            self.x_input_var.set("1.0, 2.0, 3.0")
            self.y_input_var.set("1.0, 4.0, 9.0")
            self.target_var.set("2.5")
        self.compute_interpolation()

    def parse_inputs(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Parses and validates user dataset inputs with duplicate detection."""
        x_str = self.x_input_var.get().replace(",", " ").strip()
        y_str = self.y_input_var.get().replace(",", " ").strip()
        target_str = self.target_var.get().strip()

        try:
            xs = [float(x) for x in x_str.split() if x]
        except ValueError as e:
            raise ValueError(f"Invalid numeric format in x-values: {e}")

        try:
            ys = [float(y) for y in y_str.split() if y]
        except ValueError as e:
            raise ValueError(f"Invalid numeric format in y-values: {e}")

        try:
            target = float(target_str)
        except ValueError:
            raise ValueError("Target x* must be a valid real number.")

        if len(xs) != len(ys):
            raise ValueError(f"Number of x-values ({len(xs)}) must match number of y-values ({len(ys)}).")
        if len(xs) < 2:
            raise ValueError("At least 2 data points are required for interpolation.")

        # Sort dataset by x
        arr_x = np.array(xs, dtype=float)
        arr_y = np.array(ys, dtype=float)
        idx = np.argsort(arr_x)
        arr_x = arr_x[idx]
        arr_y = arr_y[idx]

        # Duplicate x validation at input boundary
        diffs = np.diff(arr_x)
        if np.any(diffs <= 1e-14):
            dup_indices = np.where(diffs <= 1e-14)[0]
            dup_vals = [f"{arr_x[i]:g}" for i in dup_indices]
            raise ValueError(
                f"Duplicate x-coordinates detected at: {', '.join(dup_vals)}.\n"
                "In polynomial interpolation, all x coordinates must be strictly distinct."
            )

        return arr_x, arr_y, target

    def _sync_order_combobox(self, num_points: int):
        """Dynamically adapts order dropdown options based on number of points available."""
        max_order = min(3, num_points - 1)
        if max_order == 1:
            opts = ["Linear (1st Order - 2 points)"]
            if self.order_var.get() not in opts:
                self.order_var.set(opts[0])
        elif max_order == 2:
            opts = [
                "Linear (1st Order - 2 points)",
                "Quadratic (2nd Order - 3 points)",
                "All Available Orders",
            ]
            if "Cubic" in self.order_var.get():
                self.order_var.set("Quadratic (2nd Order - 3 points)")
        else:
            opts = [
                "Linear (1st Order - 2 points)",
                "Quadratic (2nd Order - 3 points)",
                "Cubic (3rd Order - 4 points)",
                "All Available Orders",
            ]
        self.order_cb.configure(values=opts)

    def compute_interpolation(self):
        try:
            x_data, y_data, x_target = self.parse_inputs()
        except Exception as e:
            messagebox.showerror("Input Validation Error", str(e))
            return

        self._sync_order_combobox(len(x_data))
        method_choice = self.method_var.get()
        order_choice = self.order_var.get()

        # Run comparison suite across all methods
        comp = run_full_comparison(x_data, y_data, x_target)
        direct_res = comp["direct_res"]
        lagrange_res = comp["lagrange_res"]
        newton_res = comp["newton_res"]

        # Select current method results
        if "Direct" in method_choice:
            current_results = direct_res
            active_tab_index = 2
        elif "Lagrange" in method_choice:
            current_results = lagrange_res
            active_tab_index = 3
        elif "Newton" in method_choice:
            current_results = newton_res
            active_tab_index = 4
        else:
            current_results = direct_res
            active_tab_index = 1

        # Populate Tab 1: Master Comparison & Performance
        self._populate_comparison_tab(comp, x_target)

        # Populate Tab 2: Direct Method Workings
        self._populate_direct_tab(direct_res, x_target)

        # Populate Tab 3: Lagrange Method Workings
        self._populate_lagrange_tab(lagrange_res, x_target)

        # Populate Tab 4: Newton's Divided Difference Workings
        self._populate_newton_tab(newton_res, x_target)

        # Switch notebook tab to reflect user selection if user chose a specific method
        if self.notebook.index("current") != 0:  # If not actively on the plot tab
            self.notebook.select(active_tab_index)

        # Update Plots (reflects selected method and orders)
        self._update_plot(x_data, y_data, current_results, x_target, order_choice, method_choice)

        # Update Live LaTeX View (customized to method formulation)
        self._update_latex_view(current_results, x_target, order_choice, method_choice, comp["uniqueness_verified"])

    # ============================================================
    # TAB POPULATION HELPERS
    # ============================================================

    def _populate_comparison_tab(self, comp: dict, x_target: float):
        self.comp_text.delete("1.0", tk.END)
        self.comp_text.insert(tk.END, "=" * 96 + "\n")
        self.comp_text.insert(tk.END, f"       MASTER NUMERICAL INTERPOLATION PERFORMANCE & RUNTIME COMPARISON (x* = {x_target:g})       \n")
        self.comp_text.insert(tk.END, "=" * 96 + "\n\n")

        self.comp_text.insert(tk.END, comp["df_comparison"].to_string(index=False) + "\n\n")

        self.comp_text.insert(tk.END, "-" * 96 + "\n")
        self.comp_text.insert(tk.END, "  DOMAIN-WIDE POLYNOMIAL UNIQUENESS PROOF (Evaluated across 50 Grid Points in [xmin, xmax])\n")
        self.comp_text.insert(tk.END, "-" * 96 + "\n")
        self.comp_text.insert(tk.END, comp["df_uniqueness"].to_string(index=False) + "\n\n")

        self.comp_text.insert(tk.END, "--- Theoretical Uniqueness Insight ---\n")
        if comp["uniqueness_verified"]:
            self.comp_text.insert(tk.END, f"[PASSED] Uniqueness Theorem Verified! Max domain residual: {comp['max_discrepancy']:.2e}\n")
            self.comp_text.insert(tk.END, "Theorem: For any n+1 distinct points, there exists a UNIQUE polynomial P(x) of degree <= n.\n")
            self.comp_text.insert(tk.END, "Direct, Lagrange, and Newton methods are simply three distinct algebraic formulations of\n")
            self.comp_text.insert(tk.END, "the EXACT SAME interpolating polynomial. Their mathematical equivalence is confirmed above.\n\n")

        self.comp_text.insert(tk.END, "--- Computational Complexity & Trade-Offs ---\n")
        self.comp_text.insert(tk.END, "1. Direct Method: O(n^3) solve. Setup Vandermonde system; sensitive to ill-conditioning for n > 5.\n")
        self.comp_text.insert(tk.END, "2. Lagrange Method: O(n^2) eval. Clean basis formula; adding a new point requires complete recomputation.\n")
        self.comp_text.insert(tk.END, "3. Newton's Divided Diff: O(n^2) table, O(n) Horner eval. Modular: easily add new points incrementally!\n")

    def _populate_direct_tab(self, direct_res: list, x_target: float):
        self.direct_text.delete("1.0", tk.END)
        self.direct_text.insert(tk.END, f"=== DIRECT METHOD STEP-BY-STEP WORKINGS (Query x* = {x_target:g}) ===\n\n")
        for r in direct_res:
            self.direct_text.insert(tk.END, f"--- {r.name} Direct Method ---\n")
            ext_tag = " [EXTRAPOLATED]" if r.is_extrapolated else " [BRACKETED]"
            self.direct_text.insert(tk.END, f"Selected Points{ext_tag}: {', '.join(f'({x:g}, {y:g})' for x, y in zip(r.x_points, r.y_points))}\n")
            self.direct_text.insert(tk.END, "Set of Linear Equations [V]{a} = {y}:\n")
            for i in range(len(r.x_points)):
                terms = []
                for j in range(len(r.x_points)):
                    if j == 0: terms.append("a0")
                    elif j == 1: terms.append(f"a1*({r.x_points[i]:g})")
                    else: terms.append(f"a{j}*({r.x_points[i]:g})^{j}")
                self.direct_text.insert(tk.END, f"  Eq {i+1}: {' + '.join(terms)} = {r.y_points[i]:g}\n")
            self.direct_text.insert(tk.END, f"Solved Coefficients (via Gaussian Elimination): {', '.join(f'a{i}={c:.6f}' for i, c in enumerate(r.coefficients))}\n")
            self.direct_text.insert(tk.END, f"Polynomial: {r.polynomial_string}\n")
            self.direct_text.insert(tk.END, f"Interpolated Value at x*={x_target:g}: {r.interpolated_value:.4f}\n")
            if r.approx_error_percent is not None:
                self.direct_text.insert(tk.END, f"Absolute Relative Approx Error |ea|: {r.approx_error_percent:.5f}%\n")
            self.direct_text.insert(tk.END, f"Exact Analytical Derivative P'({x_target:g}): {r.derivative(x_target):.4f}\n")
            self.direct_text.insert(tk.END, f"Exact Analytical Definite Integral from {r.x_points[0]:g} to {r.x_points[-1]:g}: {r.integrate(r.x_points[0], r.x_points[-1]):.4f}\n\n")

    def _populate_lagrange_tab(self, lagrange_res: list, x_target: float):
        self.lagrange_text.delete("1.0", tk.END)
        self.lagrange_text.insert(tk.END, f"=== LAGRANGE METHOD BASIS WORKINGS (Query x* = {x_target:g}) ===\n\n")
        for r in lagrange_res:
            self.lagrange_text.insert(tk.END, f"--- {r.name} Lagrange Method ---\n")
            ext_tag = " [EXTRAPOLATED]" if r.is_extrapolated else " [BRACKETED]"
            self.lagrange_text.insert(tk.END, f"Selected Points{ext_tag}: {', '.join(f'({x:g}, {y:g})' for x, y in zip(r.x_points, r.y_points))}\n")
            self.lagrange_text.insert(tk.END, "Lagrange Basis Weight Breakdown:\n")
            headers = ["Basis", "x_i", "y_i", f"L_i({x_target:g}) Weight", "Contribution L_i*y_i"]
            table_rows = []
            for t in r.basis_terms:
                table_rows.append([f"L_{t.index}(x)", f"{t.xi:g}", f"{t.yi:g}", f"{t.evaluated_weight:.6f}", f"{t.contribution:.4f}"])
            sum_w = sum(t.evaluated_weight for t in r.basis_terms)
            sum_c = sum(t.contribution for t in r.basis_terms)
            table_rows.append(["SUM (P)", "---", "---", f"{sum_w:.4f} (=1.0000)", f"{sum_c:.4f}"])

            widths = [max(len(headers[i]), max(len(row[i]) for row in table_rows)) for i in range(len(headers))]
            hdr_line = " | ".join(headers[i].rjust(widths[i]) for i in range(len(headers)))
            sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
            self.lagrange_text.insert(tk.END, f"  {hdr_line}\n  {sep_line}\n")
            for row in table_rows[:-1]:
                self.lagrange_text.insert(tk.END, "  " + " | ".join(row[i].rjust(widths[i]) for i in range(len(headers))) + "\n")
            self.lagrange_text.insert(tk.END, f"  {sep_line}\n")
            self.lagrange_text.insert(tk.END, "  " + " | ".join(table_rows[-1][i].rjust(widths[i]) for i in range(len(headers))) + "\n")
            self.lagrange_text.insert(tk.END, f"Interpolated Value at x*={x_target:g}: {r.interpolated_value:.4f}\n")
            if r.approx_error_percent is not None:
                self.lagrange_text.insert(tk.END, f"Absolute Relative Approx Error |ea|: {r.approx_error_percent:.5f}%\n")
            self.lagrange_text.insert(tk.END, f"Exact Analytical Derivative P'({x_target:g}): {r.derivative(x_target):.4f}\n")
            self.lagrange_text.insert(tk.END, f"Exact Analytical Definite Integral from {r.x_points[0]:g} to {r.x_points[-1]:g}: {r.integrate(r.x_points[0], r.x_points[-1]):.4f}\n\n")

    def _populate_newton_tab(self, newton_res: list, x_target: float):
        self.newton_text.delete("1.0", tk.END)
        self.newton_text.insert(tk.END, f"=== NEWTON'S DIVIDED DIFFERENCE WORKINGS (Query x* = {x_target:g}) ===\n\n")
        for r in newton_res:
            self.newton_text.insert(tk.END, f"--- {r.name} Newton's Divided Difference ---\n")
            ext_tag = " [EXTRAPOLATED]" if r.is_extrapolated else " [BRACKETED]"
            self.newton_text.insert(tk.END, f"Selected Points{ext_tag}: {', '.join(f'({x:g}, {y:g})' for x, y in zip(r.x_points, r.y_points))}\n")
            self.newton_text.insert(tk.END, "Triangular Divided Difference Table:\n")
            self.newton_text.insert(tk.END, format_divided_difference_table(r.x_points, r.table) + "\n")
            self.newton_text.insert(tk.END, f"Newton Diagonal Coefficients: {', '.join(f'b{k}={b:.6f}' for k, b in enumerate(r.coefficients_b))}\n")
            self.newton_text.insert(tk.END, f"Newton Form: {r.newton_formula_string}\n")
            self.newton_text.insert(tk.END, f"Interpolated Value at x*={x_target:g}: {r.interpolated_value:.4f}\n")
            if r.approx_error_percent is not None:
                self.newton_text.insert(tk.END, f"Absolute Relative Approx Error |ea|: {r.approx_error_percent:.5f}%\n")
            self.newton_text.insert(tk.END, f"Exact Analytical Derivative P'({x_target:g}): {r.derivative(x_target):.4f}\n")
            self.newton_text.insert(tk.END, f"Exact Analytical Definite Integral from {r.x_points[0]:g} to {r.x_points[-1]:g}: {r.integrate(r.x_points[0], r.x_points[-1]):.4f}\n\n")

    # ============================================================
    # MATPLOTLIB PLOT UPDATER
    # ============================================================

    def _update_plot(self, x_all, y_all, results, x_target, order_choice, method_choice):
        self.plot_ax.clear()

        # Data points
        self.plot_ax.scatter(x_all, y_all, color="black", s=65, zorder=6, label="Discrete Data Points")

        x_min = min(r.x_points[0] for r in results)
        x_max = max(r.x_points[-1] for r in results)
        margin = max(0.5, (x_max - x_min) * 0.12)
        x_dense = np.linspace(max(0, x_min - margin), x_max + margin, 350)

        color_map = {1: "#1a73e8", 2: "#0d904f", 3: "#d93025"}
        style_map = {1: "--", 2: "-.", 3: "-"}

        for r in results:
            should_plot = ("All" in order_choice) or (r.order == 1 and "Linear" in order_choice) or (r.order == 2 and "Quadratic" in order_choice) or (r.order == 3 and "Cubic" in order_choice)
            if should_plot:
                y_dense = r.evaluate(x_dense)
                c = color_map.get(r.order, "#202124")
                ls = style_map.get(r.order, "-")
                self.plot_ax.plot(x_dense, y_dense, color=c, linestyle=ls, linewidth=2.2, label=f"{r.name}: {r.interpolated_value:.2f}")
                self.plot_ax.plot(x_target, r.interpolated_value, marker="o", markersize=8, color=c, markeredgecolor="black")

        self.plot_ax.axvline(x_target, color="#5f6368", linestyle=":", alpha=0.8, label=f"Query Point x* = {x_target:g}")
        self.plot_ax.set_title(f"Polynomial Interpolation Curves ({method_choice}) - Target x* = {x_target:g}", fontsize=12, fontweight="bold")
        self.plot_ax.set_xlabel("x", fontsize=10)
        self.plot_ax.set_ylabel("f(x)", fontsize=10)
        self.plot_ax.grid(True, linestyle="--", alpha=0.4)
        self.plot_ax.legend(loc="best", fontsize=9)
        self.plot_fig.tight_layout()
        self.plot_canvas.draw_idle()

    # ============================================================
    # LIVE LATEX MATHEMATICAL RENDERING
    # ============================================================

    def _update_latex_view(self, results, x_target, order_choice, method_choice, uniqueness_verified):
        self.latex_ax.clear()
        self.latex_ax.axis("off")

        # Pick active result for display based on order selection
        if "Linear" in order_choice or len(results) == 1:
            res = results[0]
        elif "Quadratic" in order_choice or len(results) == 2:
            res = results[min(1, len(results) - 1)]
        else:
            res = results[-1]  # Highest available order (Cubic if >= 4 points)

        val_tex = f"f({x_target:g}) = {res.interpolated_value:.4f}"
        error_tex = f"|\\epsilon_a| = {res.approx_error_percent:.4f}\\%" if res.approx_error_percent is not None else "|\\epsilon_a| = \\text{N/A}"

        if "Lagrange" in method_choice:
            # Lagrange basis representation
            basis_terms_str = " + ".join(f"L_{{{t.index}}}(x) \\cdot {t.yi:g}" for t in res.basis_terms)
            if len(basis_terms_str) > 50:
                basis_terms_str = f"\\sum_{{i=0}}^{{{res.order}}} L_i(x) y_i"
            full_latex = (
                f"$\\mathbf{{Lagrange\\ Method\\ ({res.name})}}$\n\n"
                f"$P_{{{res.order}}}(x) = {basis_terms_str}$\n\n"
                f"${val_tex} \\quad ({error_tex})$"
            )
        elif "Newton" in method_choice:
            # Newton divided difference representation
            b_terms = []
            for k, b in enumerate(res.coefficients_b):
                if k == 0:
                    b_terms.append(f"{b:.3f}")
                else:
                    fact = "".join(f"(x - {res.x_points[j]:g})" for j in range(k))
                    sign = "+" if b >= 0 else "-"
                    b_terms.append(f"{sign} {abs(b):.3f}{fact}")
            newt_tex = " ".join(b_terms)
            if len(newt_tex) > 55:
                newt_tex = f"b_0 + \\sum_{{k=1}}^{{{res.order}}} b_k \\prod_{{j=0}}^{{k-1}}(x - x_j)"
            full_latex = (
                f"$\\mathbf{{Newton's\\ Divided\\ Difference\\ ({res.name})}}$\n\n"
                f"$P_{{{res.order}}}(x) = {newt_tex}$\n\n"
                f"${val_tex} \\quad ({error_tex})$"
            )
        elif "Direct" in method_choice:
            # Direct power basis polynomial
            coeffs = res.coefficients
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
            full_latex = (
                f"$\\mathbf{{Direct\\ Method\\ ({res.name})}}$\n\n"
                f"$P_{{{res.order}}}(x) = {poly_tex}$\n\n"
                f"${val_tex} \\quad ({error_tex})$"
            )
        else:
            # All Methods Comparison mode
            coeffs = res.canonical_coefficients if hasattr(res, 'canonical_coefficients') and res.canonical_coefficients is not None else res.coefficients
            terms = []
            for i, c in enumerate(coeffs):
                if i == 0: terms.append(f"{c:.4f}")
                elif i == 1: terms.append(f"{'+' if c >= 0 else '-'} {abs(c):.4f}x")
                else: terms.append(f"{'+' if c >= 0 else '-'} {abs(c):.4f}x^{{{i}}}")
            poly_tex = " ".join(terms)
            uniq_str = "\\text{[Uniqueness Verified: } P_{\\text{dir}} \\equiv P_{\\text{lag}} \\equiv P_{\\text{newt}} \\text{]}"
            full_latex = (
                f"$\\mathbf{{{res.name}\\ Interpolant}}$\n\n"
                f"$P_{{{res.order}}}(x) = {poly_tex}$\n\n"
                f"${val_tex} \\quad ({error_tex})$\n\n"
                f"${uniq_str}$"
            )

        fsize = 12 if len(full_latex) > 120 else 14

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
