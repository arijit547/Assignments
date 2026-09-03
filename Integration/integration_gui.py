"""
integration_gui.py
==================
Interactive Graphical User Interface for Numerical Integration:
- Trapezoidal Rule (Single & Multiple segments)
- Simpson's 1/3 Rule (2 segments & Multiple segments)
- Combination Method for Datasets (with automatic pattern recognition & error minimization)
- Real-time Matplotlib plotting
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from trapezoidal import trapezoidal_single, trapezoidal_multiple, trapezoidal_dataset, get_slide_default_problem
from simpson import simpson_single_2seg, simpson_multiple, simpson_dataset
from combination import analyze_and_integrate_dataset, get_preset_datasets, generate_decision_dataframe


class IntegrationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Integration Suite - Trapezoidal, Simpson & Combination")
        self.geometry("1100x800")
        self.minsize(950, 680)
        self.configure(bg="#f8f9fa")

        # Style
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background="#f8f9fa")
        style.configure("TLabel", background="#f8f9fa", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#1a73e8")
        style.configure("Subheader.TLabel", font=("Segoe UI", 11), foreground="#5f6368")
        style.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), background="#1a73e8", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#1557b0")])

        self._build_ui()

    def _build_ui(self):
        # Header
        header = ttk.Frame(self, padding=(20, 15, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="Numerical Integration System", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="Trapezoidal Rule | Simpson's 1/3 Rule | Optimal Combination for Datasets", style="Subheader.TLabel").pack(anchor="w")

        # Tabs for Function mode vs Dataset mode
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=(5, 15))

        self.tab_func = ttk.Frame(notebook, padding=15)
        self.tab_data = ttk.Frame(notebook, padding=15)

        notebook.add(self.tab_func, text="  Function Integration (Analytical / Equation)  ")
        notebook.add(self.tab_data, text="  Dataset Integration (Combination Decision Engine)  ")

        self._build_function_tab(self.tab_func)
        self._build_dataset_tab(self.tab_data)

    # ================================================================
    # TAB 1: FUNCTION INTEGRATION
    # ================================================================
    def _build_function_tab(self, parent):
        left = ttk.Frame(parent, width=380)
        left.pack(side="left", fill="y", padx=(0, 15))

        right = ttk.Frame(parent)
        right.pack(side="right", fill="both", expand=True)

        # Controls
        ttk.Label(left, text="Preset Functions:", style="Bold.TLabel").pack(anchor="w", pady=(0, 2))
        self.func_preset_var = tk.StringVar(value="Rocket Velocity: 2000*ln(140000/(140000-2100*t)) - 9.8*t [8 to 30]")
        presets = [
            "Rocket Velocity: 2000*ln(140000/(140000-2100*t)) - 9.8*t [8 to 30]",
            "Example: 300*x / (1 + exp(x)) [0 to 10]",
            "Custom Function (Enter Below)",
        ]
        combo = ttk.Combobox(left, textvariable=self.func_preset_var, values=presets, state="readonly", width=42)
        combo.pack(fill="x", pady=(0, 10))
        combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        ttk.Label(left, text="f(x) Integrand Expression:", style="Bold.TLabel").pack(anchor="w")
        self.expr_var = tk.StringVar(value="2000*log(140000/(140000 - 2100*x)) - 9.8*x")
        ttk.Entry(left, textvariable=self.expr_var, font=("Consolas", 11)).pack(fill="x", pady=(0, 10))

        # Limits
        limits_frame = ttk.Frame(left)
        limits_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(limits_frame, text="a (lower):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.a_var = tk.StringVar(value="8.0")
        ttk.Entry(limits_frame, textvariable=self.a_var, width=10).grid(row=0, column=1, padx=(0, 15))

        ttk.Label(limits_frame, text="b (upper):").grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.b_var = tk.StringVar(value="30.0")
        ttk.Entry(limits_frame, textvariable=self.b_var, width=10).grid(row=0, column=3)

        # Method
        ttk.Label(left, text="Numerical Method:", style="Bold.TLabel").pack(anchor="w", pady=(0, 2))
        self.method_var = tk.StringVar(value="Simpson's 1/3 Rule")
        methods = ["Trapezoidal - Single Segment", "Trapezoidal - Multiple Segments", "Simpson's 1/3 - 2 Segments", "Simpson's 1/3 - Multiple Segments", "Compare Both Across Segments"]
        ttk.Combobox(left, textvariable=self.method_var, values=methods, state="readonly").pack(fill="x", pady=(0, 10))

        ttk.Label(left, text="Number of Segments (n):", style="Bold.TLabel").pack(anchor="w", pady=(0, 2))
        self.n_var = tk.StringVar(value="4")
        ttk.Entry(left, textvariable=self.n_var).pack(fill="x", pady=(0, 15))

        btn = ttk.Button(left, text="Calculate Integral & Plot", style="Primary.TButton", command=self._calculate_function)
        btn.pack(fill="x", ipady=5)

        # Results Text Box
        self.func_result_box = tk.Text(left, height=14, font=("Consolas", 9), wrap="none", bg="#ffffff")
        self.func_result_box.pack(fill="both", expand=True, pady=(15, 0))

        # Matplotlib Plot on Right
        self.func_fig = Figure(figsize=(6, 5), dpi=100)
        self.func_ax = self.func_fig.add_subplot(111)
        self.func_canvas = FigureCanvasTkAgg(self.func_fig, master=right)
        self.func_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_preset_change(self, event=None):
        val = self.func_preset_var.get()
        if "Rocket" in val:
            self.expr_var.set("2000*log(140000/(140000 - 2100*x)) - 9.8*x")
            self.a_var.set("8.0")
            self.b_var.set("30.0")
        elif "300*x" in val:
            self.expr_var.set("300*x / (1 + exp(x))")
            self.a_var.set("0.0")
            self.b_var.set("10.0")

    def _calculate_function(self):
        try:
            x_sym = sp.Symbol("x", real=True)
            expr = sp.sympify(self.expr_var.get(), locals={"x": x_sym, "t": x_sym})
            fn = sp.lambdify(x_sym, expr, modules=["numpy"])
            f = lambda val: float(fn(val))

            a = float(self.a_var.get())
            b = float(self.b_var.get())
            method = self.method_var.get()
            n = int(self.n_var.get())

            self.func_ax.clear()
            self.func_result_box.delete("1.0", tk.END)

            exact = 11061.34 if "Rocket" in self.func_preset_var.get() else None

            if method == "Trapezoidal - Single Segment":
                val = trapezoidal_single(f, a, b)
                self.func_result_box.insert(tk.END, f"METHOD: Single-Segment Trapezoidal\n")
                self.func_result_box.insert(tk.END, f"Integral I = {val:.6f}\n")
                if exact:
                    err = exact - val
                    pct = abs(err) / exact * 100
                    self.func_result_box.insert(tk.END, f"True Error Et = {err:.4f}\n|et| = {pct:.4f}%\n")
                self._plot_func_trapezoid(f, a, b, 1, val)

            elif method == "Trapezoidal - Multiple Segments":
                val, x_pts, y_pts = trapezoidal_multiple(f, a, b, n)
                self.func_result_box.insert(tk.END, f"METHOD: Multiple-Segment Trapezoidal (n={n})\n")
                self.func_result_box.insert(tk.END, f"Integral I = {val:.6f}\n")
                if exact:
                    err = exact - val
                    pct = abs(err) / exact * 100
                    self.func_result_box.insert(tk.END, f"True Error Et = {err:.4f}\n|et| = {pct:.4f}%\n")
                self._plot_func_trapezoid(f, a, b, n, val)

            elif method == "Simpson's 1/3 - 2 Segments":
                val, x_pts, y_pts = simpson_single_2seg(f, a, b)
                self.func_result_box.insert(tk.END, f"METHOD: 2-Segment Simpson's 1/3\n")
                self.func_result_box.insert(tk.END, f"Integral I = {val:.6f}\n")
                if exact:
                    err = exact - val
                    pct = abs(err) / exact * 100
                    self.func_result_box.insert(tk.END, f"True Error Et = {err:.4f}\n|et| = {pct:.4f}%\n")
                self._plot_func_simpson(f, a, b, 2, val)

            elif method == "Simpson's 1/3 - Multiple Segments":
                if n % 2 != 0:
                    messagebox.showerror("Invalid Input", "Simpson's 1/3 rule requires an even number of segments (n).")
                    return
                val, x_pts, y_pts = simpson_multiple(f, a, b, n)
                self.func_result_box.insert(tk.END, f"METHOD: Multiple-Segment Simpson's 1/3 (n={n})\n")
                self.func_result_box.insert(tk.END, f"Integral I = {val:.6f}\n")
                if exact:
                    err = exact - val
                    pct = abs(err) / exact * 100
                    self.func_result_box.insert(tk.END, f"True Error Et = {err:.4f}\n|et| = {pct:.4f}%\n")
                self._plot_func_simpson(f, a, b, n, val)

            else:  # Compare Both
                self.func_result_box.insert(tk.END, f"=== COMPARISON (a={a:g}, b={b:g}) ===\n")
                self.func_result_box.insert(tk.END, f"{'n':>4} | {'Trapezoidal':>12} | {'Simpson 1/3':>12}\n" + "-"*35 + "\n")
                for s in [2, 4, 6, 8, 10]:
                    t_val, _, _ = trapezoidal_multiple(f, a, b, s)
                    s_val, _, _ = simpson_multiple(f, a, b, s)
                    self.func_result_box.insert(tk.END, f"{s:>4} | {t_val:>12.4f} | {s_val:>12.4f}\n")
                self._plot_func_simpson(f, a, b, n if n % 2 == 0 else 4, s_val)

        except Exception as e:
            messagebox.showerror("Calculation Error", str(e))

    def _plot_func_trapezoid(self, f, a, b, n, val):
        x_fine = np.linspace(a, b, 300)
        y_fine = [float(f(x)) for x in x_fine]
        x_pts = np.linspace(a, b, n + 1)
        y_pts = [float(f(x)) for x in x_pts]

        self.func_ax.plot(x_fine, y_fine, "k-", linewidth=2, label="f(x)")
        for i in range(n):
            xs = [x_pts[i], x_pts[i], x_pts[i + 1], x_pts[i + 1]]
            ys = [0, y_pts[i], y_pts[i + 1], 0]
            self.func_ax.fill(xs, ys, color="#4285f4", alpha=0.25)
        self.func_ax.plot(x_pts, y_pts, "ro--", label=f"Trapezoidal (n={n})")
        self.func_ax.set_title(f"Trapezoidal Rule (n = {n}, Integral = {val:.4f})")
        self.func_ax.legend()
        self.func_ax.grid(True, alpha=0.3)
        self.func_canvas.draw()

    def _plot_func_simpson(self, f, a, b, n, val):
        x_fine = np.linspace(a, b, 300)
        y_fine = [float(f(x)) for x in x_fine]
        x_pts = np.linspace(a, b, n + 1)
        y_pts = [float(f(x)) for x in x_pts]

        self.func_ax.plot(x_fine, y_fine, "k-", linewidth=2.5, label="f(x)")
        colors = ["#34a853", "#4285f4", "#fbbc05", "#ea4335"]
        for p in range(n // 2):
            i = 2 * p
            xp = x_pts[i:i + 3]
            yp = y_pts[i:i + 3]
            poly = np.polyfit(xp, yp, deg=2)
            xf = np.linspace(xp[0], xp[2], 60)
            yf = np.polyval(poly, xf)
            col = colors[p % len(colors)]
            self.func_ax.fill_between(xf, 0, yf, color=col, alpha=0.25)
            self.func_ax.plot(xf, yf, color=col, linestyle="--", linewidth=1.8)

        self.func_ax.plot(x_pts, y_pts, "ro", label=f"Nodes (n={n})")
        self.func_ax.set_title(f"Simpson's 1/3 Rule (n = {n}, Integral = {val:.4f})")
        self.func_ax.legend()
        self.func_ax.grid(True, alpha=0.3)
        self.func_canvas.draw()

    # ================================================================
    # TAB 2: DATASET INTEGRATION (COMBINATION METHOD)
    # ================================================================
    def _build_dataset_tab(self, parent):
        left = ttk.Frame(parent, width=420)
        left.pack(side="left", fill="y", padx=(0, 15))

        right = ttk.Frame(parent)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Preset Datasets:", style="Bold.TLabel").pack(anchor="w", pady=(0, 2))
        self.data_preset_var = tk.StringVar(value="Preset 2: Rocket Mixed Spacings (4 segs h=4, 2 segs h=3, 1 seg h=2)")
        presets = [
            "Preset 1: Rocket Equal Spacing (Even n=6) -> Pure Simpson's 1/3",
            "Preset 2: Rocket Mixed Spacings (4 segs h=4, 2 segs h=3, 1 seg h=2)",
            "Preset 3: Rocket Odd Block (3 segs h=4, 1 seg h=5) -> Curvature Decision",
            "Preset 4: Chapra Table 21.3 Unequally Spaced Data",
            "Custom Input (Enter Values Below)",
        ]
        combo = ttk.Combobox(left, textvariable=self.data_preset_var, values=presets, state="readonly", width=42)
        combo.pack(fill="x", pady=(0, 10))
        combo.bind("<<ComboboxSelected>>", self._on_dataset_preset_change)

        ttk.Label(left, text="x values (comma/space separated):", style="Bold.TLabel").pack(anchor="w")
        self.x_data_var = tk.StringVar(value="8, 12, 16, 20, 24, 27, 30, 32")
        ttk.Entry(left, textvariable=self.x_data_var, font=("Consolas", 10)).pack(fill="x", pady=(0, 10))

        ttk.Label(left, text="y values (comma/space separated):", style="Bold.TLabel").pack(anchor="w")
        self.y_data_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.y_data_var, font=("Consolas", 10)).pack(fill="x", pady=(0, 12))

        btn = ttk.Button(left, text="Run Pattern Recognition & Integrate", style="Primary.TButton", command=self._calculate_dataset)
        btn.pack(fill="x", ipady=5)

        # Decision text display
        self.data_result_box = tk.Text(left, height=18, font=("Consolas", 9), wrap="none", bg="#ffffff")
        self.data_result_box.pack(fill="both", expand=True, pady=(15, 0))

        # Matplotlib Plot for dataset on Right
        self.data_fig = Figure(figsize=(6, 5), dpi=100)
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=right)
        self.data_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._on_dataset_preset_change()

    def _on_dataset_preset_change(self, event=None):
        presets = get_preset_datasets()
        choice = self.data_preset_var.get()
        if "Preset 1" in choice:
            d = presets["1"]
        elif "Preset 2" in choice:
            d = presets["2"]
        elif "Preset 3" in choice:
            d = presets["3"]
        elif "Preset 4" in choice:
            d = presets["4"]
        else:
            return
        self.x_data_var.set(", ".join(f"{x:g}" for x in d["x"]))
        self.y_data_var.set(", ".join(f"{y:.4f}" for y in d["y"]))

    def _calculate_dataset(self):
        try:
            x_vals = [float(x) for x in self.x_data_var.get().replace(",", " ").split() if x]
            y_vals = [float(y) for y in self.y_data_var.get().replace(",", " ").split() if y]
            x_data = np.array(x_vals)
            y_data = np.array(y_vals)

            total_val, decisions, verdict = analyze_and_integrate_dataset(x_data, y_data)
            trap_val = sum((x_data[i+1]-x_data[i])*(y_data[i]+y_data[i+1])/2.0 for i in range(len(x_data)-1))

            self.data_result_box.delete("1.0", tk.END)
            self.data_result_box.insert(tk.END, f"=== PATTERN ANALYSIS & DECISION VERDICT ===\n")
            self.data_result_box.insert(tk.END, f"{verdict}\n\n")
            self.data_result_box.insert(tk.END, f"Combination Integral = {total_val:.6f}\n")
            self.data_result_box.insert(tk.END, f"Pure Trapezoidal     = {trap_val:.6f}\n")
            diff = abs(total_val - trap_val)
            self.data_result_box.insert(tk.END, f"Method Difference    = {diff:.6f}\n\n")

            self.data_result_box.insert(tk.END, f"{'Interval':<12} | {'Method':<14} | {'Sub-Area':<10}\n")
            self.data_result_box.insert(tk.END, "-" * 42 + "\n")
            for d in decisions:
                self.data_result_box.insert(tk.END, f"[{d.x_start:g}, {d.x_end:g}]".ljust(12) + f" | {d.method:<14} | {d.sub_integral:<10.4f}\n")

            # Plot
            self.data_ax.clear()
            self.data_ax.plot(x_data, y_data, "ko", markersize=6, zorder=5, label="Data points")

            simp_labeled = False
            trap_labeled = False
            for d in decisions:
                xs = x_data[d.start_idx:d.end_idx + 1]
                ys = y_data[d.start_idx:d.end_idx + 1]
                if d.method == "Simpson's 1/3":
                    lbl = "Simpson's 1/3" if not simp_labeled else None
                    simp_labeled = True
                    for p in range(d.num_segments // 2):
                        xp = xs[2*p:2*p+3]
                        yp = ys[2*p:2*p+3]
                        poly = np.polyfit(xp, yp, deg=2)
                        xf = np.linspace(xp[0], xp[-1], 50)
                        yf = np.polyval(poly, xf)
                        self.data_ax.fill_between(xf, 0, yf, color="#34a853", alpha=0.3)
                        self.data_ax.plot(xf, yf, color="#1e8e3e", linewidth=2, label=lbl if p==0 else None)
                        lbl = None
                else:
                    lbl = "Trapezoidal" if not trap_labeled else None
                    trap_labeled = True
                    for i in range(len(xs) - 1):
                        self.data_ax.fill_between([xs[i], xs[i+1]], 0, [ys[i], ys[i+1]], color="#4285f4", alpha=0.3)
                        self.data_ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color="#1a73e8", linestyle="--", linewidth=2, label=lbl if i==0 else None)
                        lbl = None

            self.data_ax.set_title(f"Combined Integration (Area = {total_val:.4f})")
            self.data_ax.legend(loc="best")
            self.data_ax.grid(True, alpha=0.3)
            self.data_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))


def launch_gui():
    app = IntegrationGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
