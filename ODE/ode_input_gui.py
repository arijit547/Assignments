
"""
ode_input_gui.py
================
Reusable GUI + validation layer for all ODE numerical methods.

Supported problem form:
    dy/dx = f(x, y)

The selected variable names make the displayed equation dynamic, e.g.
    dy/dx
    dtheta/dt
    dT/dt

This file contains:
    1. Calculator-style equation input
    2. LaTeX preview
    3. Equation validation
    4. IVP / step-size validation
    5. Optional exact solution input
    6. scipy.solve_ivp validation
    7. ODEProblem data object

It does NOT implement Euler, Heun, Midpoint, Ralston, or RK4.
"""

from __future__ import annotations

import math
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ================================================================
# DATA MODEL
# ================================================================

@dataclass
class ODEProblem:
    expression: sp.Expr
    latex: str

    independent_name: str
    dependent_name: str
    independent_symbol: sp.Symbol
    dependent_symbol: sp.Symbol

    x0: float
    y0: float
    xf: float
    step_sizes: tuple[float, ...]

    exact_solution: sp.Expr | None = None


# ================================================================
# MATH FUNCTIONS
# ================================================================

FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "csc": sp.csc,
    "sec": sp.sec,
    "cot": sp.cot,

    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "acsc": sp.acsc,
    "asec": sp.asec,
    "acot": sp.acot,

    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,

    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,

    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda arg: sp.log(arg, 10),

    "sqrt": sp.sqrt,
    "cbrt": sp.real_root,
    "abs": sp.Abs,

    "floor": sp.floor,
    "ceil": sp.ceiling,
}

CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
}

BASE_LOCALS = {
    **FUNCTIONS,
    **CONSTANTS,
}


# ================================================================
# TEXT NORMALIZATION
# ================================================================

def normalize_math_text(text: str) -> str:
    """Convert calculator-style text into SymPy-friendly syntax."""
    text = text.strip()

    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "π": "pi",
        "√": "sqrt",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Allow both x^2 and x**2.
    text = text.replace("^", "**")

    # ln(...) -> log(...)
    text = re.sub(r"\bln\s*\(", "log(", text)

    return text


# ================================================================
# SYMBOL CREATION
# ================================================================

def create_symbols(independent_name: str, dependent_name: str):
    indep = independent_name.strip()
    dep = dependent_name.strip()
    if indep == dep:
        raise ValueError(
            f"Independent and dependent variables must be different (both are '{indep}')."
        )
    independent_symbol = sp.Symbol(indep, real=True)
    dependent_symbol = sp.Symbol(dep, real=True)
    return independent_symbol, dependent_symbol


# ================================================================
# EQUATION PARSING
# ================================================================

def parse_ode(
    text: str,
    independent_name: str,
    dependent_name: str,
) -> tuple[sp.Expr, str]:

    text = normalize_math_text(text)

    if not text:
        raise ValueError("The differential equation cannot be empty.")

    independent_symbol, dependent_symbol = create_symbols(
        independent_name,
        dependent_name,
    )

    local_names = dict(BASE_LOCALS)

    # Only the selected variables are permitted.
    local_names[independent_name] = independent_symbol
    local_names[dependent_name] = dependent_symbol

    if independent_name != "x":
        local_names.pop("x", None)

    if dependent_name != "y":
        local_names.pop("y", None)

    try:
        expression = sp.sympify(
            text,
            locals=local_names,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid mathematical expression.\n\n"
            "Check brackets, operators, powers, and function names."
        ) from exc

    allowed_symbols = {
        independent_symbol,
        dependent_symbol,
    }

    unknown_symbols = expression.free_symbols - allowed_symbols

    if unknown_symbols:
        names = ", ".join(
            sorted(str(symbol) for symbol in unknown_symbols)
        )
        raise ValueError(
            f"Unknown variable(s): {names}\n\n"
            f"Only {independent_name} and {dependent_name} are allowed."
        )

    try:
        latex = sp.latex(expression)
    except Exception as exc:
        raise ValueError(
            "The expression could not be converted to LaTeX."
        ) from exc

    return expression, latex


def parse_exact_solution(
    text: str,
    independent_name: str,
) -> sp.Expr:

    text = normalize_math_text(text)

    if not text:
        raise ValueError("Exact solution is empty.")

    independent_symbol = sp.Symbol(
        independent_name,
        real=True,
    )

    local_names = dict(BASE_LOCALS)
    local_names[independent_name] = independent_symbol

    try:
        expression = sp.sympify(
            text,
            locals=local_names,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid exact solution."
        ) from exc

    unknown = expression.free_symbols - {independent_symbol}

    if unknown:
        names = ", ".join(
            sorted(str(symbol) for symbol in unknown)
        )

        raise ValueError(
            f"Exact solution contains unknown variable(s): {names}."
        )

    return expression


def validate_exact_solution(
    exact_expr: sp.Expr,
    ode_expr: sp.Expr,
    independent_symbol: sp.Symbol,
    dependent_symbol: sp.Symbol,
    x0: float,
    y0: float,
    xf: float,
) -> tuple[bool, str]:
    """
    Validates that the supplied exact solution:
      1. Satisfies the initial condition: y_exact(x0) == y0.
      2. Satisfies the differential equation: dy_exact/dx == f(x, y_exact(x)).

    Returns:
        (is_valid: bool, message: str)
    """
    # 1. Check Initial Condition
    try:
        y_at_x0 = float(exact_expr.subs(independent_symbol, x0).evalf())
    except Exception as exc:
        return False, f"Could not evaluate exact solution at initial point {independent_symbol.name}0 = {x0:g}: {exc}"

    init_diff = abs(y_at_x0 - y0)
    init_tol = max(1e-4, 1e-4 * abs(y0))
    if init_diff > init_tol:
        return False, (
            f"Initial condition mismatch: exact solution evaluates to {y_at_x0:.6g} at "
            f"{independent_symbol.name}(0) = {x0:g}, but {dependent_symbol.name}0 is {y0:g} "
            f"(absolute difference = {init_diff:.4e} > tolerance {init_tol:.4e})."
        )

    # 2. Check Differential Equation Satisfaction
    try:
        dy_exact = sp.diff(exact_expr, independent_symbol)
        rhs_subs = ode_expr.subs(dependent_symbol, exact_expr)
        residual_expr = dy_exact - rhs_subs

        # Fast symbolic simplification check
        try:
            simp = sp.simplify(residual_expr)
            if simp == 0:
                return True, "Exact solution verified symbolically and matches initial condition."
        except Exception:
            pass

        # Numerical sampling check across interval [x0, xf]
        test_points = np.linspace(x0, xf, 15)
        res_fn = sp.lambdify(independent_symbol, residual_expr, modules=["numpy"])
        f_fn = sp.lambdify((independent_symbol, dependent_symbol), ode_expr, modules=["numpy"])

        max_res = 0.0
        max_scale = 1.0
        for tp in test_points:
            try:
                r_val = float(res_fn(tp))
                y_val = float(exact_expr.subs(independent_symbol, tp).evalf())
                f_val = float(f_fn(tp, y_val))
                scale = max(1.0, abs(f_val))
                if not math.isfinite(r_val):
                    return False, f"Exact solution derivative produces non-finite residual at {independent_symbol.name} = {tp:g}."
                if abs(r_val) > max_res:
                    max_res = abs(r_val)
                    max_scale = scale
            except Exception as exc:
                return False, f"Could not evaluate exact solution ODE residual at {independent_symbol.name} = {tp:g}: {exc}"

        rel_res = max_res / max_scale
        if rel_res > 1e-3 and max_res > 1e-3:
            return False, (
                f"ODE satisfaction failure: exact solution does not satisfy dy/dx = f(x, y). "
                f"Maximum residual |y'_exact - f(x, y_exact)| = {max_res:.4e} (relative residual = {rel_res:.4e})."
            )

    except Exception as exc:
        return False, f"Error validating ODE differential satisfaction: {exc}"

    return True, "Exact solution verified numerically and matches initial condition."


# ================================================================
# NUMERICAL VALIDATION USING SCIPY
# ================================================================

def validate_ode_with_scipy(
    expression: sp.Expr,
    independent_symbol: sp.Symbol,
    dependent_symbol: sp.Symbol,
    x0: float,
    y0: float,
) -> None:
    """
    Use scipy.solve_ivp only for input/domain validation.

    The actual numerical-method assignment is NOT implemented
    with solve_ivp.
    """

    fn = sp.lambdify(
        (independent_symbol, dependent_symbol),
        expression,
        modules=["numpy"],
    )

    def rhs(x_value, y_value):
        try:
            result = fn(
                x_value,
                float(y_value[0]),
            )

            result = np.asarray(result)

            if np.iscomplexobj(result):
                raise ValueError(
                    "The derivative became complex."
                )

            value = float(result)

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "The ODE could not be evaluated at the initial condition."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                "The ODE returned NaN or infinity."
            )

        return [value]

    try:
        result = solve_ivp(
            rhs,
            (x0, x0 + 1e-6),
            [y0],
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            max_step=1e-7,
        )
    except Exception as exc:
        raise ValueError(
            "SciPy could not validate this ODE at the initial condition."
        ) from exc

    if not result.success:
        raise ValueError(
            "ODE validation failed.\n\n"
            f"SciPy: {result.message}"
        )


# ================================================================
# INPUT PARSING / VALIDATION
# ================================================================

def parse_real_number(
    text: str,
    field_name: str,
) -> float:

    try:
        value = float(text.strip())
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be a real number."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite."
        )

    return value


def parse_step_sizes(text: str) -> tuple[float, ...]:
    parts = [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]

    if not parts:
        raise ValueError(
            "Enter at least one step size."
        )

    values = []

    for part in parts:
        value = parse_real_number(
            part,
            "Step size",
        )

        if value <= 0:
            raise ValueError(
                "Every step size must be greater than zero."
            )

        values.append(value)

    # Remove duplicates, then sort from largest to smallest.
    return tuple(
        sorted(
            set(values),
            reverse=True,
        )
    )


def validate_step_sizes(
    x0: float,
    xf: float,
    step_sizes: tuple[float, ...],
) -> None:

    interval = xf - x0

    if interval <= 0:
        raise ValueError(
            "Final point must be greater than initial point."
        )

    for h in step_sizes:

        if h > interval:
            raise ValueError(
                f"h = {h:g} is larger than the integration interval."
            )

        number_of_steps = interval / h

        if not math.isclose(
            number_of_steps,
            round(number_of_steps),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"h = {h:g} does not divide the interval "
                f"[{x0:g}, {xf:g}] exactly."
            )


# ================================================================
# GUI
# ================================================================

class ODEInputGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("ODE Numerical Methods - Input")
        self.geometry("1180x850")
        self.minsize(1000, 700)
        self.configure(bg="#ffffff")

        # Set up Modern Style
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", font=("Segoe UI", 12))
        style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 14), foreground="#5f6368")
        
        style.configure(
            "Primary.TButton", 
            background="#4285f4", 
            foreground="#ffffff", 
            font=("Segoe UI", 12, "bold"), 
            borderwidth=0,
            padding=10
        )
        style.map("Primary.TButton", background=[("active", "#3367d6")])

        self.problem: ODEProblem | None = None
        self.independent_var = tk.StringVar(value="x")
        self.dependent_var = tk.StringVar(value="y")
        self.equation_var = tk.StringVar()
        self.x0_var = tk.StringVar(value="0")
        self.y0_var = tk.StringVar(value="1")
        self.xf_var = tk.StringVar(value="1")
        self.step_var = tk.StringVar(value="0.1, 0.05, 0.01")
        self.exact_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Enter the differential equation below.")

        self.btn_var_indep = None
        self.btn_var_dep = None
        self.equation_entry = None

        self._build_ui()
        self._update_labels()
        self._update_latex_preview()

    # ============================================================
    # UI CREATION
    # ============================================================

    def _build_ui(self):
        root = ttk.Frame(self, padding=30)
        root.pack(fill="both", expand=True)

        header_frame = ttk.Frame(root)
        header_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(header_frame, text="ODE Numerical Methods", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header_frame, text="Reusable input and validation panel", style="Subheader.TLabel").pack(anchor="w")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 20))

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        self._build_equation_panel(left)
        self._build_keypad(left)
        self._build_problem_panel(right)

    # ============================================================
    # EQUATION PANEL (ENTRY + MATPLOTLIB)
    # ============================================================

    def _build_equation_panel(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=(0, 20))

        variables = ttk.Frame(frame)
        variables.pack(fill="x", pady=(0, 15))

        ttk.Label(variables, text="Independent:", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.independent_combo = ttk.Combobox(
            variables, textvariable=self.independent_var, values=("x", "t", "s", "u"), width=5, font=("Segoe UI", 12)
        )
        self.independent_combo.pack(side="left", padx=(8, 30))
        self.independent_combo.bind("<<ComboboxSelected>>", lambda _: self._variable_changed())

        ttk.Label(variables, text="Dependent:", font=("Segoe UI", 12, "bold")).pack(side="left")
        self.dependent_combo = ttk.Combobox(
            variables, textvariable=self.dependent_var, values=("y", "theta", "θ", "T", "u"), width=5, font=("Segoe UI", 12)
        )
        self.dependent_combo.pack(side="left", padx=(8, 0))
        self.dependent_combo.bind("<<ComboboxSelected>>", lambda _: self._variable_changed())

        # Native Entry for input
        entry_frame = tk.Frame(frame, bg="#ffffff")
        entry_frame.pack(fill="x", pady=(0, 10))
        
        self.equation_entry = ttk.Entry(
            entry_frame, textvariable=self.equation_var, font=("Consolas", 18)
        )
        self.equation_entry.pack(fill="x", ipady=8)
        self.equation_entry.focus_set()
        
        self.equation_var.trace_add("write", lambda *args: self._update_latex_preview())
        self.exact_var.trace_add("write", lambda *args: self._update_latex_preview())
        self.equation_entry.bind("<Return>", lambda e: self.validate_equation())

        # Live Canvas Input Area
        preview_frame = tk.Frame(frame, bg="#f8f9fa", bd=1, relief="solid", highlightbackground="#dadce0", highlightthickness=1)
        preview_frame.pack(fill="both", expand=True)
        
        self.preview_figure = Figure(figsize=(6, 3), dpi=100)
        self.preview_figure.patch.set_facecolor('#f8f9fa')
        self.preview_axes = self.preview_figure.add_subplot(111)
        self.preview_axes.axis("off")

        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_frame)
        tk_canvas = self.preview_canvas.get_tk_widget()
        tk_canvas.pack(fill="both", expand=True)

    # ============================================================
    # KEYPAD (DESMOS STYLE)
    # ============================================================

    def _build_keypad(self, parent):
        frame = tk.Frame(parent, bg="#f8f9fa", bd=1, relief="solid", highlightbackground="#dadce0", highlightthickness=1)
        frame.pack(fill="x", side="bottom")
        
        keypad_inner = tk.Frame(frame, bg="#f8f9fa", padx=10, pady=10)
        keypad_inner.pack(fill="both", expand=True)

        # Fully functional 10-column ODE keypad layout
        keys = [
            ["VAR_X", "VAR_Y", "a²", "a^b", "7", "8", "9", "÷", "funcs"],
            ["(", ")", "ln", "exp", "4", "5", "6", "×", "←", "→"],
            ["|a|", "√", "e", "π", "1", "2", "3", "−", "Backspace"],
            ["Clear", "sin", "cos", "0", ".", ",", "+", "Enter"]
        ]

        def get_btn_bg(k):
            if k.isdigit() or k == ".": return "#ffffff"
            if k == "Clear": return "#fce8e6"
            if k in ["funcs", "Backspace", "←", "→"]: return "#dadce0"
            if k == "Enter": return "#4285f4"
            return "#f1f3f4"
            
        def get_btn_fg(k):
            if k == "Clear": return "#c5221f"
            if k == "Enter": return "#ffffff"
            return "#202124"

        for row_index, row in enumerate(keys):
            col_offset = 0
            for item in row:
                bg = get_btn_bg(item)
                fg = get_btn_fg(item)
                display_text = item
                
                colspan = 1
                if item == "funcs":
                    colspan = 2
                elif item == "Backspace":
                    colspan = 2
                elif item == "Clear":
                    colspan = 2
                elif item == "Enter":
                    colspan = 2
                    display_text = "↵ Enter"
                
                if item == "VAR_X": display_text = self.independent_var.get() or "x"
                if item == "VAR_Y": display_text = self.dependent_var.get() or "y"
                
                btn = tk.Button(
                    keypad_inner, text=display_text, font=("Segoe UI", 11, "bold"),
                    bg=bg, fg=fg, activebackground="#e8eaed", relief="flat", borderwidth=1, highlightbackground="#dadce0",
                    command=lambda v=item: self._keypad_press(v)
                )
                
                btn.grid(row=row_index, column=col_offset, sticky="nsew", padx=3, pady=3, columnspan=colspan)
                
                if item == "VAR_X": self.btn_var_indep = btn
                if item == "VAR_Y": self.btn_var_dep = btn
                
                col_offset += colspan

        for column in range(10):
            keypad_inner.columnconfigure(column, weight=1)
        for row in range(4):
            keypad_inner.rowconfigure(row, weight=1)

    def _show_funcs_popup(self):
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.configure(bg="#ffffff", bd=1, relief="solid")
        
        x = self.winfo_rootx() + 300
        y = self.winfo_rooty() + 200
        popup.geometry(f"+{x}+{y}")
        
        container = tk.Frame(popup, bg="#ffffff", padx=10, pady=10)
        container.pack()
        
        sections = {
            "TRIGONOMETRIC": ["sin", "cos", "tan", "csc", "sec", "cot"],
            "INVERSE TRIGONOMETRIC": ["asin", "acos", "atan", "acsc", "asec", "acot"],
            "HYPERBOLIC": ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"],
            "LOGARITHMIC": ["ln", "log", "log10"],
            "EXPONENTIAL": ["e^x", "exp"],
            "MISCELLANEOUS": ["sqrt", "cbrt", "|x|", "π", "e", "x", "y"]
        }
        
        for title, buttons in sections.items():
            tk.Label(container, text=title, font=("Segoe UI", 10), bg="#ffffff", fg="#5f6368").pack(anchor="w", pady=(10, 5))
            grid = tk.Frame(container, bg="#ffffff")
            grid.pack(fill="x")
            for i, btn_text in enumerate(buttons):
                b = tk.Button(
                    grid, text=btn_text, font=("Segoe UI", 11), bg="#f1f3f4", relief="flat", borderwidth=1,
                    command=lambda v=btn_text: [self._keypad_press(v), popup.destroy()]
                )
                b.grid(row=i//3, column=i%3, sticky="nsew", padx=3, pady=3)
            for c in range(3): grid.columnconfigure(c, weight=1)
            
        popup.bind("<FocusOut>", lambda e: popup.destroy())
        popup.focus_set()

    def _keypad_press(self, key):
        if not self.equation_entry:
            return
            
        if key == "Clear":
            self.equation_var.set("")
            self.equation_entry.focus_set()
            return
        elif key == "Backspace":
            try:
                cursor_idx = self.equation_entry.index(tk.INSERT)
                if cursor_idx > 0:
                    self.equation_entry.delete(cursor_idx - 1)
            except tk.TclError:
                pass
        elif key == "funcs":
            self._show_funcs_popup()
            return
        elif key in ["Enter", "↵"]:
            self.validate_equation()
            return
        elif key in ["←"]:
            try:
                idx = self.equation_entry.index(tk.INSERT)
                self.equation_entry.icursor(max(0, idx - 1))
            except tk.TclError:
                pass
        elif key in ["→"]:
            try:
                idx = self.equation_entry.index(tk.INSERT)
                self.equation_entry.icursor(min(len(self.equation_entry.get()), idx + 1))
            except tk.TclError:
                pass
        else:
            text = ""
            if key == "VAR_X":
                text = self.independent_var.get().strip() or "x"
            elif key == "VAR_Y":
                text = self.dependent_var.get().strip() or "y"
            elif key == "a²":
                text = "^2"
            elif key == "a^b":
                text = "^"
            elif key == "÷":
                text = "/"
            elif key == "×":
                text = "*"
            elif key == "−":
                text = "-"
            elif key == "π":
                text = "pi"
            elif key == "e":
                text = "e"
            elif key == "√" or key == "sqrt":
                text = "sqrt("
            elif key == "cbrt":
                text = "cbrt("
            elif key == "|a|" or key == "|x|":
                text = "abs("
            elif key == "e^x" or key == "exp":
                text = "exp("
            elif key == "ln":
                text = "ln("
            else:
                if key in FUNCTIONS or key in ["sin", "cos", "tan", "asin", "acos", "atan", "csc", "sec", "cot", "acsc", "asec", "acot", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "log", "log10"]:
                    text = key + "("
                else:
                    text = key
            
            # Insert at cursor
            self.equation_entry.insert(tk.INSERT, text)
            
        self.equation_entry.focus_set()

    # ============================================================
    # PROBLEM DATA PANEL
    # ============================================================

    def _build_problem_panel(self, parent):
        frame = tk.Frame(parent, bg="#ffffff", bd=1, relief="solid", highlightbackground="#dadce0", highlightthickness=1)
        frame.pack(fill="x")
        
        tk.Label(frame, text="Problem Data & Presets", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#5f6368").pack(anchor="w", padx=10, pady=(10, 0))

        # Quick Presets Selector
        preset_frame = tk.Frame(frame, bg="#ffffff")
        preset_frame.pack(fill="x", padx=15, pady=(8, 0))
        tk.Label(preset_frame, text="Load Preset Problem:", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#1a73e8").pack(anchor="w")

        self.preset_combobox = ttk.Combobox(
            preset_frame,
            values=[
                "Default Slide: Radiation Cooling (theta, t)",
                "Example 1: Exponential Decay (dy/dx = -2*y)",
                "Example 2: Simple Polynomial (dy/dx = 4*x - 2*y)",
                "Example 3: Harmonic/Trig (dy/dx = cos(x) - y)",
                "Custom / Blank",
            ],
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.preset_combobox.pack(fill="x", pady=(3, 5))
        self.preset_combobox.bind("<<ComboboxSelected>>", self._on_preset_selected)

        grid_frame = tk.Frame(frame, bg="#ffffff")
        grid_frame.pack(fill="x", padx=15, pady=10)

        def create_labeled_entry(row, label_text, var):
            lbl = tk.Label(grid_frame, text=label_text, font=("Segoe UI", 12), bg="#ffffff")
            lbl.grid(row=row, column=0, sticky="w", pady=8)
            entry = tk.Entry(
                grid_frame, textvariable=var, font=("Consolas", 14), 
                bg="#f8f9fa", highlightthickness=1, highlightcolor="#4285f4", 
                highlightbackground="#dadce0", relief="flat", bd=4
            )
            entry.grid(row=row, column=1, sticky="ew", padx=(15, 0))
            return lbl

        self.initial_x_label = create_labeled_entry(0, "Initial x₀", self.x0_var)
        self.initial_y_label = create_labeled_entry(1, "Initial y₀", self.y0_var)
        self.final_x_label = create_labeled_entry(2, "Final x", self.xf_var)
        create_labeled_entry(3, "Step sizes", self.step_var)
        create_labeled_entry(4, "Exact solution (optional)", self.exact_var)

        grid_frame.columnconfigure(1, weight=1)

        ttk.Button(
            frame, text="Validate Problem", command=self.validate_problem, style="Primary.TButton"
        ).pack(fill="x", padx=15, pady=(0, 15))

        status_frame = tk.Frame(parent, bg="#e8f0fe", bd=0, padx=10, pady=10)
        status_frame.pack(fill="x", pady=(20, 0))
        
        tk.Label(
            status_frame, textvariable=self.status_var, wraplength=400, bg="#e8f0fe", fg="#1967d2", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

    def _on_preset_selected(self, event=None):
        choice = self.preset_combobox.get()
        if "Radiation Cooling" in choice:
            self.independent_var.set("t")
            self.dependent_var.set("theta")
            self.equation_var.set("-2.2067e-12*(theta**4 - 81e8)")
            self.x0_var.set("0")
            self.y0_var.set("1200")
            self.xf_var.set("480")
            self.step_var.set("480, 240, 120, 60, 30")
            self.exact_var.set("")
        elif "Exponential Decay" in choice:
            self.independent_var.set("x")
            self.dependent_var.set("y")
            self.equation_var.set("-2*y")
            self.x0_var.set("0")
            self.y0_var.set("1")
            self.xf_var.set("2")
            self.step_var.set("0.5, 0.25, 0.1")
            self.exact_var.set("exp(-2*x)")
        elif "Harmonic" in choice:
            self.independent_var.set("x")
            self.dependent_var.set("y")
            self.equation_var.set("cos(x) - y")
            self.x0_var.set("0")
            self.y0_var.set("0")
            self.xf_var.set("3")
            self.step_var.set("0.5, 0.25, 0.1")
            self.exact_var.set("0.5*sin(x) + 0.5*cos(x) - 0.5*exp(-x)")
        elif "Polynomial" in choice:
            self.independent_var.set("x")
            self.dependent_var.set("y")
            self.equation_var.set("4*x - 2*y")
            self.x0_var.set("0")
            self.y0_var.set("2")
            self.xf_var.set("1")
            self.step_var.set("0.2, 0.1, 0.05")
            self.exact_var.set("2*x - 1 + 3*exp(-2*x)")
        elif "Custom" in choice:
            self.equation_var.set("")
            self.exact_var.set("")

        self._variable_changed()

    # ============================================================
    # LABELS / PREVIEW UPDATES
    # ============================================================

    def _variable_changed(self):
        self._update_labels()
        indep = self.independent_var.get().strip() or "x"
        dep = self.dependent_var.get().strip() or "y"
        if self.btn_var_indep: self.btn_var_indep.configure(text=indep)
        if self.btn_var_dep: self.btn_var_dep.configure(text=dep)
        self._update_latex_preview()

    def _update_labels(self):
        indep = self.independent_var.get().strip() or "x"
        dep = self.dependent_var.get().strip() or "y"
        self.initial_x_label.configure(text=f"Initial {indep}₀")
        self.initial_y_label.configure(text=f"Initial {dep}₀")
        self.final_x_label.configure(text=f"Final {indep}")

    def _update_latex_preview(self):
        if not hasattr(self, 'preview_axes'): return
        self.preview_axes.clear()
        self.preview_axes.axis("off")

        indep = self.independent_var.get().strip() or "x"
        dep = self.dependent_var.get().strip() or "y"
        text = self.equation_var.get().strip()
        exact_text = self.exact_var.get().strip()

        # Greek letter mapping for proper mathematical typography
        greek_map = {
            "theta": r"\theta", "θ": r"\theta",
            "alpha": r"\alpha", "α": r"\alpha",
            "beta": r"\beta", "β": r"\beta",
            "gamma": r"\gamma", "γ": r"\gamma",
            "phi": r"\phi", "φ": r"\phi",
            "psi": r"\psi", "ψ": r"\psi",
            "omega": r"\omega", "ω": r"\omega",
            "delta": r"\delta", "δ": r"\delta",
            "lambda": r"\lambda", "λ": r"\lambda",
            "mu": r"\mu", "μ": r"\mu",
            "nu": r"\nu", "ν": r"\nu",
            "rho": r"\rho", "ρ": r"\rho",
            "sigma": r"\sigma", "σ": r"\sigma",
            "tau": r"\tau", "τ": r"\tau",
        }
        indep_tex = greek_map.get(indep.lower(), indep)
        dep_tex = greek_map.get(dep.lower(), dep)

        is_valid_ode = False
        if not text:
            latex_rhs = r"\text{Enter differential equation...}"
        else:
            try:
                expression, _ = parse_ode(text, indep, dep)
                latex_rhs = sp.latex(expression)
                is_valid_ode = True
            except Exception:
                latex_rhs = None

        exact_latex = None
        if exact_text:
            try:
                exact_expr = parse_exact_solution(exact_text, indep)
                exact_latex = sp.latex(exact_expr)
            except Exception:
                exact_latex = None

        # Dynamically scale font size so long equations never clip
        total_len = len(latex_rhs) if latex_rhs else len(text)
        if exact_latex:
            total_len = max(total_len, len(exact_latex))

        if total_len > 55:
            fsize = 14
        elif total_len > 40:
            fsize = 16
        elif total_len > 25:
            fsize = 18
        else:
            fsize = 21

        try:
            if is_valid_ode or not text:
                ode_str = f"$\\frac{{d{dep_tex}}}{{d{indep_tex}}} = {latex_rhs}$"
                if exact_latex:
                    full_latex = f"{ode_str}\n\n${dep_tex}_{{\\mathrm{{exact}}}}({indep_tex}) = {exact_latex}$"
                    self.preview_axes.text(
                        0.5, 0.5, full_latex,
                        ha="center", va="center", fontsize=fsize, color="#1a73e8"
                    )
                else:
                    self.preview_axes.text(
                        0.5, 0.5, ode_str,
                        ha="center", va="center", fontsize=fsize, color="#1a73e8" if is_valid_ode else "#80868b"
                    )
            else:
                # Safe plain text rendering while mid-typing incomplete expression (never crashes mathtext)
                self.preview_axes.text(
                    0.5, 0.5, f"d{dep}/d{indep} = {text}\n(typing...)",
                    ha="center", va="center", fontsize=max(14, fsize - 2),
                    family="monospace", color="#5f6368"
                )
        except Exception:
            self.preview_axes.text(
                0.5, 0.5, f"d{dep}/d{indep} = {text}",
                ha="center", va="center", fontsize=16, family="monospace", color="#3c4043"
            )

        self.preview_canvas.draw_idle()

    # ============================================================
    # EQUATION VALIDATION
    # ============================================================

    def validate_equation(self):

        try:

            indep = (
                self.independent_var.get().strip()
            )

            dep = (
                self.dependent_var.get().strip()
            )

            if not re.fullmatch(
                r"[A-Za-z\u03B8][A-Za-z0-9_\u03B8]*",
                indep,
            ):
                raise ValueError(
                    "Invalid independent variable name."
                )

            if not re.fullmatch(
                r"[A-Za-z\u03B8][A-Za-z0-9_\u03B8]*",
                dep,
            ):
                raise ValueError(
                    "Invalid dependent variable name."
                )

            if indep == dep:
                raise ValueError(
                    "Independent and dependent variables "
                    "must be different."
                )

            expression, _ = parse_ode(
                self.equation_var.get(),
                indep,
                dep,
            )

            indep_symbol, dep_symbol = (
                create_symbols(
                    indep,
                    dep,
                )
            )

            fn = sp.lambdify(
                (indep_symbol, dep_symbol),
                expression,
                modules=["numpy"],
            )

            # Test domain safety across multiple candidate points to avoid false singularity errors
            candidate_points = []
            try:
                candidate_points.append((float(self.x0_var.get().strip()), float(self.y0_var.get().strip())))
            except Exception:
                pass
            candidate_points.extend([(1.0, 1.0), (0.5, 0.5), (2.0, 2.0), (0.0, 1.0)])

            test_ok = False
            last_err = None
            for tx, ty in candidate_points:
                try:
                    res = float(fn(tx, ty))
                    if math.isfinite(res):
                        test_ok = True
                        break
                except Exception as e:
                    last_err = e

            if not test_ok:
                raise ValueError(
                    f"The equation could not be evaluated at test points: {last_err}"
                )

            self.status_var.set(
                "Equation is valid. Now validate the IVP data."
            )

            messagebox.showinfo(
                "Valid Equation",
                "The differential equation is valid.\n\n"
                "Now enter the initial condition, final point, "
                "and step sizes.",
            )

        except Exception as exc:

            self.status_var.set(
                f"Invalid equation: {exc}"
            )

            messagebox.showerror(
                "Invalid Equation",
                str(exc),
            )

    # ============================================================
    # COMPLETE PROBLEM VALIDATION
    # ============================================================

    def validate_problem(self):

        try:

            indep = (
                self.independent_var.get().strip()
            )

            dep = (
                self.dependent_var.get().strip()
            )

            expression, latex = parse_ode(
                self.equation_var.get(),
                indep,
                dep,
            )

            indep_symbol, dep_symbol = (
                create_symbols(
                    indep,
                    dep,
                )
            )

            x0 = parse_real_number(
                self.x0_var.get(),
                f"Initial {indep}0",
            )

            y0 = parse_real_number(
                self.y0_var.get(),
                f"Initial {dep}0",
            )

            xf = parse_real_number(
                self.xf_var.get(),
                f"Final {indep}",
            )

            if xf <= x0:
                raise ValueError(
                    f"Final {indep} must be greater than "
                    f"initial {indep}0."
                )

            step_sizes = parse_step_sizes(
                self.step_var.get()
            )

            validate_step_sizes(
                x0,
                xf,
                step_sizes,
            )

            # SciPy is used only for equation/domain validation.
            validate_ode_with_scipy(
                expression,
                indep_symbol,
                dep_symbol,
                x0,
                y0,
            )

            exact_text = (
                self.exact_var.get().strip()
            )

            exact_solution = None

            if exact_text:
                exact_solution = parse_exact_solution(
                    exact_text,
                    indep,
                )
                is_valid, msg = validate_exact_solution(
                    exact_solution,
                    expression,
                    indep_symbol,
                    dep_symbol,
                    x0,
                    y0,
                    xf,
                )
                if not is_valid:
                    raise ValueError(f"Exact Solution Verification Failed:\n{msg}")

            self.problem = ODEProblem(
                expression=expression,
                latex=latex,
                independent_name=indep,
                dependent_name=dep,
                independent_symbol=indep_symbol,
                dependent_symbol=dep_symbol,
                x0=x0,
                y0=y0,
                xf=xf,
                step_sizes=step_sizes,
                exact_solution=exact_solution,
            )

            self.status_var.set(
                "All validation checks passed. "
                "Returning the problem to the numerical-method program."
            )

            messagebox.showinfo(
                "Problem Valid",
                "All input validation checks passed.\n\n"
                "The problem is ready for numerical calculation.",
            )

            self.destroy()

        except Exception as exc:

            self.status_var.set(
                f"Invalid problem data: {exc}"
            )

            messagebox.showerror(
                "Invalid Input",
                str(exc),
            )


# ================================================================
# PUBLIC FUNCTIONS & BENCHMARK HELPERS
# ================================================================

def get_slide_default_problem() -> ODEProblem:
    """
    Returns the benchmark ODE problem from lecture slides:
    Radiation cooling of a spherical body:
        dtheta/dt = -2.2067e-12 * (theta^4 - 81e8)
        theta(0) = 1200 K
        tf = 480 s
        step_sizes = (480.0, 240.0, 120.0, 60.0, 30.0)
    """
    indep = "t"
    dep = "theta"
    indep_symbol, dep_symbol = create_symbols(indep, dep)
    expr_str = "-2.2067e-12 * (theta**4 - 81e8)"
    expr, latex = parse_ode(expr_str, indep, dep)
    return ODEProblem(
        expression=expr,
        latex=latex,
        independent_name=indep,
        dependent_name=dep,
        independent_symbol=indep_symbol,
        dependent_symbol=dep_symbol,
        x0=0.0,
        y0=1200.0,
        xf=480.0,
        step_sizes=(480.0, 240.0, 120.0, 60.0, 30.0),
        exact_solution=None,
    )


def get_problem(method_name: str = "ODE Solver") -> ODEProblem | None:
    """
    Launch the GUI and return a validated ODEProblem.

    Every numerical-method file can simply do:

        from ode_input_gui import get_problem
        problem = get_problem("Heun's Method")

    If the user closes the GUI before completing validation,
    None is returned.
    """

    app = ODEInputGUI()
    if method_name:
        app.title(f"{method_name} - Input & Validation")

    def handle_close():
        app.problem = None
        app.destroy()

    app.protocol(
        "WM_DELETE_WINDOW",
        handle_close,
    )

    app.mainloop()

    return app.problem


# ================================================================
# DIRECT EXECUTION
# ================================================================

if __name__ == "__main__":
    problem = get_problem()

    if problem is not None:
        print("\nValidated ODE:")
        print(
            f"d{problem.dependent_name}/"
            f"d{problem.independent_name} = "
            f"{problem.latex}"
        )

        print(
            f"Initial: {problem.independent_name}0 = "
            f"{problem.x0}, "
            f"{problem.dependent_name}0 = "
            f"{problem.y0}"
        )

        print(
            f"Final {problem.independent_name} = "
            f"{problem.xf}"
        )

        print(
            "Step sizes:",
            problem.step_sizes,
        )
