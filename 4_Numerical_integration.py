import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.special import kn
    import pandas as pd
    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return kn, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Numerical integration

    Numerical integration is one of the most useful tools in computational
    materials science and physics.

    We often encounter quantities of the form

    \[
    I = \int_a^b f(x)\,dx
    \]

    for which

    - no convenient analytical solution exists,
    - the function is expensive to evaluate,
    - or the function is known only at discrete points.

    This notebook builds the classic quadrature rules from first principles:

    1. **Rectangle** (midpoint) and **trapezoidal** rules
    2. **Simpson's rule**, obtained by combining the two above
    3. **Convergence** behaviour of each rule
    4. **Adaptive** integration, controlling the error by step doubling
    5. **Romberg** integration, a systematic extrapolation to higher order
    6. **Difficult integrands**: discontinuities and improper integrals
    7. **High-order quadratures**: Newton–Cotes and Clenshaw–Curtis
    8. **Gaussian quadrature**, and its generalization to other weight functions
    9. A physics capstone: the density of a relativistic quantum gas

    Throughout, we track a single reference example so that the methods can be
    compared directly against each other and against the known exact answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    md_ref_title = mo.md(
        """
        ## 1. A reference example
        ---
        """
    )
    return (md_ref_title,)


@app.cell(hide_code=True)
def _(mo):
    from functools import partial
    from wigglystuff import TangleLatex

    def ref_func(x, a, b, c):
        return a * x**4 + b * x + c

    formula_widget = mo.ui.anywidget(
        TangleLatex(
            latex=(r"I = \int_0^2 \left(\tangle{a}x^4 + \tangle{b}x + \tangle{c}\right) dx"),
            parameters={
                "a": {
                    "value": 1.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "a",
                    "label": "Quartic coefficient",
                },
                "b": {
                    "value": -2.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "b",
                    "label": "Linear coefficient",
                },
                "c": {
                    "value": 1.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "c",
                    "label": "Constant term",
                },
            },
            editor="inline",
            reveal_all_on_drag=True,
        )
    )

    def f(x):
        return ref_func(
            x,
            float(formula_widget.values["a"]),
            float(formula_widget.values["b"]),
            float(formula_widget.values["c"]),
        )





    return f, formula_widget


@app.cell(hide_code=True)
def _(f, formula_widget, np):
    interval = [0.0, 2.0]
    a_ref = interval[0]
    b_ref = interval[1]

    a_val = float(formula_widget.values["a"])
    b_val = float(formula_widget.values["b"])
    c_val = float(formula_widget.values["c"])

    I_exact = (
        (a_val / 5.0) * (b_ref**5 - a_ref**5)
        + (b_val / 2.0) * (b_ref**2 - a_ref**2)
        + c_val * (b_ref - a_ref)
    )

    _x = np.linspace(a_ref, b_ref, 400)
    _y = f(_x)
    return I_exact, a_ref, a_val, b_ref, b_val, c_val


@app.cell(hide_code=True)
def _(a_val, b_val, c_val, mo):
    md_ref_text = mo.md(
        rf"""
        Consider $f(x) = {a_val:g}x^4 + {b_val:g}x + {c_val:g}$ on $[0,2]$.
        Its antiderivative is elementary, so we know the exact answer and can
        measure the error of every method below against it:
        """
    )
    return (md_ref_text,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    a_val,
    b_ref,
    b_val,
    c_val,
    f,
    formula_widget,
    md_ref_text,
    md_ref_title,
    mo,
    np,
    plt,
):
    _xplot = np.linspace(a_ref, b_ref, 400)
    _yplot = f(_xplot)

    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, _yplot, color="crimson", lw=2, label=r"$f(x) = ax^4 + bx + c$")
    _ax.fill_between(_xplot, _yplot, alpha=0.2, color="crimson")
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    mo.vstack(
        [
            md_ref_title,
            md_ref_text,
            mo.hstack([formula_widget,mo.md(rf"""
                $$
                \huge
                = \left[\frac{{{a_val:g}}}{{5}}x^5 + \frac{{{b_val:g}}}{{2}}x^2 + {c_val:g}x\right]_0^2 = {I_exact:g}
                $$
                """)],align="center",justify="center"),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_rect_title = mo.md(
        """
        ## 2. Rectangle (midpoint) rule
        ---
        """
    )
    md_rect_text = mo.md(
        r"""
        Approximate the integral over a slice by the area of a rectangle whose
        height is the value of $f$ at the slice's midpoint:
        $$
        \int_a^b f(x)\,dx \approx (b-a)\, f\!\left(\frac{a+b}{2}\right).
        $$
        Splitting $[a,b]$ into $N$ slices of width $h=(b-a)/N$ and applying
        this to each one gives the **composite rectangle rule**
        $$
        \int_a^b f(x)\,dx \approx h \sum_{k=1}^{N} f(x_k), \qquad
        x_k = a + \left(k - \tfrac12\right) h.
        $$
        A Taylor expansion around the midpoint shows that a single slice has
        error $\frac{h^3}{24}f''(\xi)$, so the composite rule converges as
        $\mathcal{O}(h^2)$. The rule is exact for any linear function, since
        $f''=0$ then.
        """
    )
    return md_rect_text, md_rect_title


@app.cell
def _(np):
    def rectangle_rule(f, a, b, n):
        h = (b - a) / n
        xk = a + h / 2.0 + h * np.arange(n)
        return h * np.sum(f(xk))

    return (rectangle_rule,)


@app.cell(hide_code=True)
def _(mo):
    rect_n = mo.ui.slider(1, 40, value=5, step=1, label="Number of slices $N$")
    return (rect_n,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_rect_text,
    md_rect_title,
    mo,
    np,
    plt,
    rect_n,
    rectangle_rule,
):
    _n = rect_n.value
    _h = (b_ref - a_ref) / _n
    _edges = a_ref + _h * np.arange(_n + 1)
    _midpoints = a_ref + _h / 2.0 + _h * np.arange(_n)
    _heights = f(_midpoints)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=2, label="$f(x)$")
    _ax.bar(
        _edges[:-1],
        _heights,
        width=_h,
        align="edge",
        color="steelblue",
        alpha=0.4,
        edgecolor="steelblue",
        label="Rectangles",
    )
    _ax.scatter(_midpoints, _heights, color="steelblue", zorder=3, s=25)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    _estimate = rectangle_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_rect_title,
            md_rect_text,
            mo.hstack([rect_n,mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _estimate):.3e}",
                    )]),
            _fig,
        ],
        justify="center"
    )
    mo.vstack(
        [
            md_rect_title,
            mo.hstack(
                [
                    mo.vstack([md_rect_text,mo.hstack([mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Absolute error",value=f"{abs(I_exact - _estimate):.3e}")])]),
                    mo.vstack([rect_n,_fig],align="center"),

                ],
                widths=[0.5, 0.5],
                align="start"
            )
        ], 

    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_trap_title = mo.md(
        """
        ## 3. Trapezoidal rule
        ---
        """
    )
    md_trap_text = mo.md(
        r"""
        Instead of a rectangle, approximate $f$ on each slice by the straight
        line through its two endpoints. The area under that line is a trapezoid:
        $$
        \int_a^b f(x)\,dx \approx (b-a)\,\frac{f(a)+f(b)}{2}.
        $$
        The composite rule over $N$ slices of width $h=(b-a)/N$ reads
        $$
        \int_a^b f(x)\,dx \approx h \left[\frac{f(x_0)+f(x_N)}{2} +
        \sum_{k=1}^{N-1} f(x_k)\right], \qquad x_k = a + kh.
        $$
        Its leading error term is $-\frac{h^3}{12}f''(\xi)$ per slice — same
        order as the rectangle rule, $\mathcal{O}(h^2)$ overall, but with the
        opposite sign and twice the magnitude. It is likewise exact for
        linear functions.
        """
    )
    return md_trap_text, md_trap_title


@app.cell(hide_code=True)
def _(mo):
    trap_n = mo.ui.slider(1, 40, value=5, step=1, label="Number of slices $N$")
    return (trap_n,)


@app.cell
def _(np):
    def trapezoidal_rule(f, a, b, n):
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    return (trapezoidal_rule,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_trap_text,
    md_trap_title,
    mo,
    np,
    plt,
    trap_n,
    trapezoidal_rule,
):
    _n = trap_n.value
    _x = np.linspace(a_ref, b_ref, _n + 1)
    _y = f(_x)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=3, label="$f(x)$")
    _ax.fill_between(
        _x, _y, color="darkorange", alpha=0.4, label="Trapezoids", step=None
    )
    _ax.plot(_x, _y, color="darkorange", marker="o", ms=4)
    # add vertical lines at the trapezoid edges that stops at the function curve
    for xi in _x:
        _ax.vlines(xi, 0, f(xi), color="darkorange", lw=1, ls="-", alpha=0.9)
        # add linear segments connecting the function values at the trapezoid edges
        if xi != _x[-1]:
            _ax.plot([xi, _x[np.where(_x == xi)[0][0] + 1]], [f(xi), f(_x[np.where(_x == xi)[0][0] + 1])], color="darkorange", lw=1, ls="-", alpha=0.9)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()



    _estimate = trapezoidal_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_trap_title,
            mo.hstack(
                [
                    mo.vstack([md_trap_text,mo.hstack([mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Absolute error",value=f"{abs(I_exact - _estimate):.3e}")])]),
                    mo.vstack([trap_n,_fig],align="center"),

                ],
                widths=[0.5, 0.5],
                align="start"
            )
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_simpson_title = mo.md(
        """
        ## 4. Simpson's rule
        ---
        """
    )
    md_simpson_text = mo.md(
        r"""
        The rectangle and trapezoidal errors have the same order but opposite
        sign and different weight:
        $$
        I - I_{\rm rect} = \frac{h^3}{24}f''(\xi) + \mathcal{O}(h^4), \qquad
        I - I_{\rm trap} = -\frac{h^3}{12}f''(\xi) + \mathcal{O}(h^4).
        $$
        Combining them in the ratio that cancels the $h^2$ term,
        $$
        I_S = \frac{2 I_{\rm rect} + I_{\rm trap}}{3},
        $$
        removes the leading error and leaves an $\mathcal{O}(h^4)$ method:
        **Simpson's rule**. Equivalently, it fits a parabola through the two
        endpoints and the midpoint of each slice. Over $N$ (even) slices with
        $h=(b-a)/N$:
        $$
        \int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0) + f(x_N) +
        4\sum_{k~{\rm odd}} f(x_k) + 2\sum_{k~{\rm even}, \,k\neq 0,N} f(x_k)\right].
        $$
        Because the local parabola matches $f$, $f'$ and $f''$ at the
        midpoint, Simpson's rule integrates any cubic **exactly** — one order
        higher than either of its ingredients.
        """
    )
    return md_simpson_text, md_simpson_title


@app.cell
def _(mo):
    simpson_n = mo.ui.slider(2, 40, value=4, step=2, label="Number of slices $N$ (even)")
    return (simpson_n,)


@app.cell
def _(np):
    def simpson_rule(f, a, b, n):
        if n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of slices.")
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        return (h / 3.0) * (
            y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        )

    return (simpson_rule,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_simpson_text,
    md_simpson_title,
    mo,
    np,
    plt,
    simpson_n,
    simpson_rule,
):
    _n = simpson_n.value
    _h = (b_ref - a_ref) / _n
    _x = a_ref + _h * np.arange(_n + 1)
    _y = f(_x)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=2, label="$f(x)$")

    for _k in range(0, _n, 2):
        _x3 = _x[_k : _k + 3]
        _y3 = _y[_k : _k + 3]
        _coeffs = np.polyfit(_x3, _y3, 2)
        _xfine = np.linspace(_x3[0], _x3[-1], 30)
        _ax.plot(
            _xfine,
            np.polyval(_coeffs, _xfine),
            color="seagreen",
            lw=2,
            label="Local parabola" if _k == 0 else None,
        )
    _ax.plot(_x, _y, "o", color="seagreen", ms=4)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    _estimate = simpson_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_simpson_title,
            md_simpson_text,
            simpson_n,
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _estimate):.3e}",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_exact_title = mo.md(
        """
        ### Exactness demo
        ---
        """
    )
    md_exact_text = mo.md(
        r"""
        Pick the degree of a random polynomial and compare how each rule
        performs on $[-1,2]$ using only $N=6$ slices. Simpson's rule should
        show (near) zero error up to degree 3, while the rectangle and
        trapezoidal rules only do so up to degree 1.
        """
    )
    return md_exact_text, md_exact_title


@app.cell(hide_code=True)
def _(mo):
    exactness_degree = mo.ui.dropdown(
        options={
            "Constant (degree 0)": 0,
            "Linear (degree 1)": 1,
            "Quadratic (degree 2)": 2,
            "Cubic (degree 3)": 3,
            "Quartic (degree 4)": 4,
        },
        value="Cubic (degree 3)",
        label="Polynomial degree:",
    )
    return (exactness_degree,)


@app.cell(hide_code=True)
def _(
    exactness_degree,
    md_exact_text,
    md_exact_title,
    mo,
    np,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _rng = np.random.default_rng(0)
    _coeffs = _rng.uniform(-3.0, 3.0, size=exactness_degree.value + 1)

    def _poly(x):
        return np.polyval(_coeffs, x)

    _a, _b, _n = -1.0, 2.0, 8
    _antideriv = np.polyint(_coeffs) # Antiderivative of the polynomial
    _I_poly_exact = np.polyval(_antideriv, _b) - np.polyval(_antideriv, _a)

    _I_rect = rectangle_rule(_poly, _a, _b, _n)
    _I_trap = trapezoidal_rule(_poly, _a, _b, _n)
    _I_simp = simpson_rule(_poly, _a, _b, _n)

    # add a figure of the polynomial
    _xplot = np.linspace(_a, _b, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, _poly(_xplot), color="crimson", lw=2, label="$f(x)$")
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()
    mo.vstack(
        [
            md_exact_title,
            md_exact_text,
            exactness_degree,
            mo.hstack(
                [
                    mo.stat(
                        label="Rectangle error",
                        value=f"{abs(_I_poly_exact - _I_rect):.2e}",
                    ),
                    mo.stat(
                        label="Trapezoidal error",
                        value=f"{abs(_I_poly_exact - _I_trap):.2e}",
                    ),
                    mo.stat(
                        label="Simpson error",
                        value=f"{abs(_I_poly_exact - _I_simp):.2e}",
                    ),
                ]
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_conv_title = mo.md(
        """
        ## 5. Convergence study
        ---
        """
    )
    md_conv_text = mo.md(
        r"""
        The error scaling predicted above, $\mathcal{O}(h^2)$ for rectangle
        and trapezoidal, $\mathcal{O}(h^4)$ for Simpson, can be verified
        empirically: plotting $\log(\text{error})$ against $\log(h)$ should
        give straight lines whose slopes match those orders.
        """
    )
    return md_conv_text, md_conv_title


@app.cell(hide_code=True)
def _(mo):
    conv_nmax = mo.ui.slider(6, 60, value=30, step=2, label="Maximum $N$ (even)")
    return (conv_nmax,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    conv_nmax,
    f,
    md_conv_text,
    md_conv_title,
    mo,
    np,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _N_values = np.arange(2, conv_nmax.value + 1, 2)
    _h_values = (b_ref - a_ref) / _N_values
    _err_rect = np.array(
        [abs(I_exact - rectangle_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )
    _err_trap = np.array(
        [abs(I_exact - trapezoidal_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )
    _err_simp = np.array(
        [abs(I_exact - simpson_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )

    _order_rect = np.polyfit(np.log(_h_values), np.log(_err_rect), 1)[0]
    _order_trap = np.polyfit(np.log(_h_values), np.log(_err_trap), 1)[0]
    _order_simp = np.polyfit(np.log(_h_values), np.log(_err_simp), 1)[0]

    _fig, _ax = plt.subplots()
    _ax.loglog(_h_values, _err_rect, "o-", label=f"Rectangle (order {_order_rect:.2f})")
    _ax.loglog(_h_values, _err_trap, "s-", label=f"Trapezoidal (order {_order_trap:.2f})")
    _ax.loglog(_h_values, _err_simp, "^-", label=f"Simpson (order {_order_simp:.2f})")
    _ax.set(xlabel="$h$", ylabel="Absolute error")
    _ax.legend()

    mo.vstack(
        [
            md_conv_title,
            md_conv_text,
            conv_nmax,
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_adapt_title = mo.md(
        """
        ## 6. Adaptive integration by step doubling
        ---
        """
    )
    md_adapt_text = mo.md(
        r"""
        In practice we rarely know $f''$, so we cannot predict the error
        directly — but we can *estimate* it by comparing two successive
        refinements. Doubling $N$ halves $h$; since the error scales as
        $\varepsilon = c\,h^p$,
        $$
        \varepsilon_2 = I - I_2 = c\left(\frac{h_1}{2}\right)^p, \qquad
        \varepsilon_1 = I - I_1 = c\,h_1^p = 2^p\,\varepsilon_2,
        $$
        so that
        $$
        \varepsilon_2 \approx \frac{I_2 - I_1}{2^p - 1}.
        $$
        For rectangle/trapezoidal ($p=2$) the divisor is $3$; for Simpson
        ($p=4$) it is $15$. We keep doubling $N$ until this error estimate
        drops below a target tolerance.
        """
    )
    return md_adapt_text, md_adapt_title


@app.cell
def _(np):
    def adaptive_integrate(rule, error_divisor, f, a, b, tol=1e-8, n_start=1, max_iter=24):
        n = n_start
        I_prev = rule(f, a, b, n)
        history = [(n, I_prev, np.nan)]
        for _ in range(max_iter):
            n *= 2
            I_new = rule(f, a, b, n)
            err_est = (I_new - I_prev) / error_divisor
            history.append((n, I_new, err_est))
            if abs(err_est) < tol:
                return I_new, history
            I_prev = I_new
        return I_new, history

    return (adaptive_integrate,)


@app.cell(hide_code=True)
def _(mo):
    adaptive_rule_choice = mo.ui.dropdown(
        options={
            "Rectangle (order 2)": "rectangle",
            "Trapezoidal (order 2)": "trapezoidal",
            "Simpson (order 4)": "simpson",
        },
        value="Rectangle (order 2)",
        label="Rule:",
    )
    adaptive_tol_exp = mo.ui.slider(
        -12, -2, value=-8, step=1, label="Target tolerance, $\\log_{10}(\\rm tol)$"
    )
    return adaptive_rule_choice, adaptive_tol_exp


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    adaptive_integrate,
    adaptive_rule_choice,
    adaptive_tol_exp,
    b_ref,
    f,
    md_adapt_text,
    md_adapt_title,
    mo,
    np,
    pd,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _rule_map = {
        "rectangle": (rectangle_rule, 3, 1),
        "trapezoidal": (trapezoidal_rule, 3, 1),
        "simpson": (simpson_rule, 15, 2),
    }
    _rule_fn, _err_div, _n_start = _rule_map[adaptive_rule_choice.value]
    _tol = 10.0 ** adaptive_tol_exp.value

    _I_final, _history = adaptive_integrate(
        _rule_fn, _err_div, f, a_ref, b_ref, tol=_tol, n_start=_n_start
    )
    _df = pd.DataFrame(_history, columns=["N", "Estimate", "Error estimate"])

    _fig, _ax = plt.subplots()
    _iterations = np.arange(1, len(_history))
    _errs = np.abs([h[2] for h in _history[1:]])
    _ax.semilogy(_iterations, _errs, "o-", color="steelblue")
    _ax.axhline(_tol, color="black", ls="--", label="Target tolerance")
    _ax.set(xlabel="Doubling iteration", ylabel="Error estimate")
    _ax.legend()

    mo.vstack(
        [
            md_adapt_title,
            md_adapt_text,
            mo.hstack([adaptive_rule_choice, adaptive_tol_exp]),
            #mo.ui.table(_df),
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Final $N$", value=str(_history[-1][0])),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _I_final):.3e}",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_romberg_title = mo.md(
        """
        ## 7. Romberg integration
        ---
        """
    )
    md_romberg_text = mo.md(
        r"""
        The trapezoidal rule converges slowly, and Simpson's rule is a
        separate formula built by fitting parabolas. Romberg integration
        instead gets a much better answer out of the *same* trapezoidal
        evaluations, by combining them algebraically to cancel the leading
        error term.

        Doubling $N$ is cheap: the new sample points sit exactly halfway
        between the old ones, so nothing already computed is wasted,
        $$
        I_{2N} = \tfrac12 I_N + h_{2N}\!\!\sum_{k~{\rm odd}}^{1\ldots 2N-1} f(a+k\,h_{2N}).
        $$
        Writing the exact integral as the trapezoidal estimate plus an
        error series in even powers of $h$,
        $$
        I = I_N + c_2 h^2 + c_4 h^4 + \cdots,
        $$
        and eliminating $c_2$ between the estimates at $h$ and $h/2$ gives
        an $\mathcal{O}(h^4)$ estimate for free,
        $$
        R_2 = I_{2N} + \frac{1}{3}\left(I_{2N}-I_N\right),
        $$
        which turns out to be *exactly* Simpson's rule on the same points.
        Nothing stops us from repeating the trick on the extrapolated
        values themselves: combining two neighbouring order-$2m$ estimates
        cancels their shared error term and buys two more orders of
        accuracy,
        $$
        R_{i,m+1} = R_{i,m} + \frac{R_{i,m}-R_{i-1,m}}{4^{m}-1}.
        $$
        Arranged in a triangle — each row one more doubling of $N$, each
        column one more level of extrapolation — column 1 is the plain
        trapezoidal rule, column 2 is Simpson's rule, and the bottom-right
        entry $R_{n,n}$ is by far the best estimate available, accurate to
        roughly $\mathcal{O}(h^{2n})$.
        """
    )
    return md_romberg_text, md_romberg_title


@app.cell
def _(np):
    def trapezoidal_doublings(f, a, b, n_levels):
        h = b - a
        N = 1
        I = 0.5 * h * (f(a) + f(b))
        estimates = [I]
        for _ in range(1, n_levels):
            h_new = h / 2.0
            odd_sum = sum(f(a + (2 * k - 1) * h_new) for k in range(1, N + 1))
            I = 0.5 * I + h_new * odd_sum
            estimates.append(I)
            N *= 2
            h = h_new
        return np.array(estimates)

    def romberg_table(I_list):
        n = len(I_list)
        R = [[0.0] * (i + 1) for i in range(n)]
        for i in range(n):
            R[i][0] = I_list[i]
        for m in range(1, n):
            for i in range(m, n):
                R[i][m] = R[i][m - 1] + (R[i][m - 1] - R[i - 1][m - 1]) / (4.0**m - 1.0)
        return R

    return romberg_table, trapezoidal_doublings


@app.cell
def _(np):
    def runge_fn(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    runge_a, runge_b = -1.0, 1.0
    runge_exact = 0.4 * np.arctan(5.0)
    return runge_a, runge_b, runge_exact, runge_fn


@app.cell(hide_code=True)
def _(mo):
    romberg_target_choice = mo.ui.dropdown(
        options={
            "Reference quartic (smooth, Section 1's widget)": "ref",
            "Runge function 1/(1+25x²) on [-1,1] (harder)": "runge",
        },
        value="Reference quartic (smooth, Section 1's widget)",
        label="Integrand:",
    )
    romberg_levels = mo.ui.slider(2, 10, value=6, step=1, label="Number of doublings")
    return romberg_levels, romberg_target_choice


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_romberg_text,
    md_romberg_title,
    mo,
    np,
    plt,
    romberg_levels,
    romberg_table,
    romberg_target_choice,
    runge_a,
    runge_b,
    runge_exact,
    runge_fn,
    trapezoidal_doublings,
):
    if romberg_target_choice.value == "ref":
        _f, _a, _b, _true = f, a_ref, b_ref, I_exact
    else:
        _f, _a, _b, _true = runge_fn, runge_a, runge_b, runge_exact

    _n = romberg_levels.value
    _I = trapezoidal_doublings(_f, _a, _b, _n)
    _R = romberg_table(list(_I))

    _rows = "\n".join(
        f"N={2**i:<5d}| " + "  ".join(f"{val:14.9f}" for val in row)
        for i, row in enumerate(_R)
    )

    _N = 2 ** np.arange(_n)
    _err_trap = np.abs(np.array([_R[i][0] for i in range(_n)]) - _true)
    _err_simp = np.abs(np.array([_R[i][1] if i >= 1 else np.nan for i in range(_n)]) - _true)
    _err_best = np.abs(np.array([_R[i][i] for i in range(_n)]) - _true)

    _fig, _ax = plt.subplots()
    _ax.semilogy(_N, np.maximum(_err_trap, 1e-17), "o-", label="Trapezoidal (column 1)")
    _ax.semilogy(_N[1:], np.maximum(_err_simp[1:], 1e-17), "s-", label="1 extrapolation (= Simpson)")
    _ax.semilogy(_N, np.maximum(_err_best, 1e-17), "^-", label="Full Romberg, diagonal")
    _ax.set(xlabel="$N$ (trapezoidal slices already spent)", ylabel="Absolute error")
    _ax.set_xscale("log", base=2)
    _ax.legend()
    _ax.grid(alpha=0.3, which="both")

    mo.vstack(
        [
            md_romberg_title,
            md_romberg_text,
            mo.hstack([romberg_target_choice, romberg_levels]),
            mo.md(f"```\n{_rows}\n```"),
            _fig,
            mo.stat(
                label="Best Romberg estimate",
                value=f"{_R[-1][-1]:.10f}",
                caption=f"error {abs(_R[-1][-1] - _true):.2e}",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Romberg's extrapolation assumes the error really is a smooth power
    series in $h$ — true for a well-behaved integrand, but not for one
    with a kink, a jump, or a singularity, where a handful of samples
    can no longer see the true shape of the function. Switch to the
    Runge function above: extrapolation still helps, but far less
    dramatically, and adding rows can briefly make the estimate
    *worse* before it improves, because it keeps confidently
    amplifying an error trend that no longer holds. The next section
    looks at integrands that break the smoothness assumption outright.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    md_diff_title = mo.md(
        """
        ## 8. Difficult integrands
        ---
        """
    )
    md_diff_text = mo.md(
        r"""
        Every rule above assumes $f$ is smooth across the whole interval.
        Two common ways that assumption fails: the integrand has a **jump
        discontinuity** somewhere inside $(a,b)$, or the integral is
        **improper** — the integrand diverges at an endpoint, or the
        interval itself is unbounded.
        """
    )
    return md_diff_text, md_diff_title


@app.cell(hide_code=True)
def _(mo):
    md_disc_text = mo.md(
        r"""
        ### Discontinuities

        Consider a piecewise integrand with a jump at $x=0.7$,
        $$
        f(x) = \begin{cases} 3x^2+x+3, & x<0.7\\ 2x^3-3x^2+x+3, & x\geq0.7\end{cases},
        $$
        on $[0,2]$. Applying a composite rule blindly across the jump
        treats it as if $f$ were smooth there, contaminating every slice
        that straddles $x=0.7$ with an error that shrinks only slowly as
        $N$ grows — for Simpson's rule this can even be *worse* than the
        plain trapezoidal rule, since the spurious jump defeats the
        cancellation Simpson relies on. The fix costs nothing extra:
        **split the integral at the discontinuity**,
        $$
        \int_0^2 f(x)\,dx = \int_0^{0.7} f(x)\,dx + \int_{0.7}^2 f(x)\,dx,
        $$
        and apply the rule separately to each smooth piece — here each
        piece is a polynomial of degree $\leq3$, so split Simpson's rule
        recovers the exact answer to machine precision.
        """
    )
    return (md_disc_text,)


@app.cell
def _(np):
    disc_a, disc_b, disc_point = 0.0, 2.0, 0.7

    def f_disc1(x):
        return 3.0 * x**2 + x + 3.0

    def f_disc2(x):
        return 2.0 * x**3 - 3.0 * x**2 + x + 3.0

    def f_disc(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < disc_point, f_disc1(x), f_disc2(x))

    def _antideriv1(x):
        return x**3 + 0.5 * x**2 + 3.0 * x

    def _antideriv2(x):
        return 0.5 * x**4 - x**3 + 0.5 * x**2 + 3.0 * x

    I_disc_exact = (_antideriv1(disc_point) - _antideriv1(disc_a)) + (
        _antideriv2(disc_b) - _antideriv2(disc_point)
    )
    return I_disc_exact, disc_a, disc_b, disc_point, f_disc, f_disc1, f_disc2


@app.cell(hide_code=True)
def _(mo):
    disc_n = mo.ui.slider(4, 60, value=12, step=4, label="Number of slices $N$ (split evenly across the two halves)")
    return (disc_n,)


@app.cell(hide_code=True)
def _(
    I_disc_exact,
    disc_a,
    disc_b,
    disc_n,
    disc_point,
    f_disc,
    f_disc1,
    f_disc2,
    md_diff_text,
    md_diff_title,
    md_disc_text,
    mo,
    np,
    plt,
    simpson_rule,
    trapezoidal_rule,
):
    _n = disc_n.value

    _naive_trap = trapezoidal_rule(f_disc, disc_a, disc_b, _n)
    _naive_simp = simpson_rule(f_disc, disc_a, disc_b, _n)

    _n_half = _n // 2
    _split_trap = trapezoidal_rule(f_disc1, disc_a, disc_point, _n_half) + trapezoidal_rule(
        f_disc2, disc_point, disc_b, _n_half
    )
    _split_simp = simpson_rule(f_disc1, disc_a, disc_point, _n_half) + simpson_rule(
        f_disc2, disc_point, disc_b, _n_half
    )

    _xplot = np.linspace(disc_a, disc_b, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f_disc(_xplot), color="crimson", lw=2)
    _ax.fill_between(_xplot, f_disc(_xplot), alpha=0.15, color="crimson")
    _ax.axvline(disc_point, color="black", lw=0.8, ls="--", label="discontinuity")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    mo.vstack(
        [
            md_diff_title,
            md_diff_text,
            md_disc_text,
            disc_n,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.md("**Applied blindly across the jump:**"),
                            mo.hstack(
                                [
                                    mo.stat(label="Trapezoidal error", value=f"{abs(I_disc_exact - _naive_trap):.3e}"),
                                    mo.stat(label="Simpson error", value=f"{abs(I_disc_exact - _naive_simp):.3e}"),
                                ]
                            ),
                            mo.md("**Split at the discontinuity:**"),
                            mo.hstack(
                                [
                                    mo.stat(label="Trapezoidal error", value=f"{abs(I_disc_exact - _split_trap):.3e}"),
                                    mo.stat(label="Simpson error", value=f"{abs(I_disc_exact - _split_simp):.3e}"),
                                ]
                            ),
                        ]
                    ),
                ],
                widths=[0.5, 0.5],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_imp_text = mo.md(
        r"""
        ### Improper integrals

        Two more ways an integral can misbehave: an **integrable
        singularity**, where $f$ diverges but the area underneath stays
        finite, e.g.
        $$
        \int_0^1 \frac{dx}{\sqrt{x}} = 2,
        $$
        and an **unbounded domain**, e.g. $\int_0^\infty e^{-x}\,dx=1$ or
        $\int_{-\infty}^\infty e^{-x^2}\,dx=\sqrt{\pi}$.

        For the singularity, any rule that evaluates $f$ *at* the bad
        endpoint — trapezoidal, Simpson — divides by zero there. The
        rectangle rule never samples an endpoint, so it survives, though
        it still converges slowly since $f$ stays large nearby.

        For an unbounded domain, a change of variable maps it onto a
        finite one before integrating. A semi-infinite range $(a,\infty)$
        maps onto $(0,1)$ via
        $$
        x = a + \frac{t}{1-t}, \qquad dx = \frac{dt}{(1-t)^2},
        $$
        and the full real line $(-\infty,\infty)$ maps onto $(-1,1)$ via
        $$
        x = \frac{t}{1-t^2}, \qquad dx = \frac{1+t^2}{(1-t^2)^2}\,dt.
        $$
        Either way the new integrand typically still misbehaves right at
        the mapped-in endpoint ($t\to1$ or $t\to\pm1$), so once again the
        rectangle rule — which never touches an endpoint — is the natural
        choice.
        """
    )
    return (md_imp_text,)


@app.cell
def _(np):
    def f_singular(x):
        return 1.0 / np.sqrt(x)

    I_singular_exact = 2.0
    return I_singular_exact, f_singular


@app.cell(hide_code=True)
def _(mo):
    singular_n = mo.ui.slider(2, 200, value=20, step=2, label="Number of slices $N$")
    return (singular_n,)


@app.cell(hide_code=True)
def _(
    I_singular_exact,
    f_singular,
    md_imp_text,
    mo,
    np,
    rectangle_rule,
    singular_n,
    trapezoidal_rule,
):
    _n = singular_n.value
    _rect = rectangle_rule(f_singular, 0.0, 1.0, _n)
    with np.errstate(divide="ignore"):
        _trap = trapezoidal_rule(f_singular, 0.0, 1.0, _n)

    mo.vstack(
        [
            md_imp_text,
            singular_n,
            mo.hstack(
                [
                    mo.stat(
                        label="Rectangle estimate",
                        value=f"{_rect:.6f}",
                        caption=f"error {abs(I_singular_exact - _rect):.3e}",
                    ),
                    mo.stat(
                        label="Trapezoidal estimate",
                        value=f"{_trap}",
                        caption="samples f(0), which diverges",
                    ),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell
def _():
    def semi_infinite_map(g, a):
        def mapped(t):
            return g(a + t / (1.0 - t)) / (1.0 - t) ** 2

        return mapped

    def full_line_map(g):
        def mapped(t):
            return g(t / (1.0 - t**2)) * (1.0 + t**2) / (1.0 - t**2) ** 2

        return mapped

    return full_line_map, semi_infinite_map


@app.cell(hide_code=True)
def _(mo):
    improper_choice = mo.ui.dropdown(
        options={
            "Semi-infinite: ∫₀^∞ e^(−x) dx = 1": "semi",
            "Full line: ∫₋∞^∞ e^(−x²) dx = √π": "full",
        },
        value="Semi-infinite: ∫₀^∞ e^(−x) dx = 1",
        label="Integral:",
    )
    improper_n = mo.ui.slider(2, 200, value=40, step=2, label="Number of slices $N$ (in the mapped variable $t$)")
    return improper_choice, improper_n


@app.cell(hide_code=True)
def _(
    full_line_map,
    improper_choice,
    improper_n,
    mo,
    np,
    rectangle_rule,
    semi_infinite_map,
):
    if improper_choice.value == "semi":
        _g = lambda x: np.exp(-x)
        _mapped = semi_infinite_map(_g, 0.0)
        _t_a, _t_b = 0.0, 1.0
        _exact = 1.0
    else:
        _g = lambda x: np.exp(-(x**2))
        _mapped = full_line_map(_g)
        _t_a, _t_b = -1.0, 1.0
        _exact = np.sqrt(np.pi)

    _n = improper_n.value
    with np.errstate(over="ignore", divide="ignore"):
        _estimate = rectangle_rule(_mapped, _t_a, _t_b, _n)

    mo.vstack(
        [
            mo.hstack([improper_choice, improper_n]),
            mo.hstack(
                [
                    mo.stat(label="Rectangle estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Exact", value=f"{_exact:.6f}"),
                    mo.stat(label="Absolute error", value=f"{abs(_exact - _estimate):.3e}"),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_hq_title = mo.md(
        """
        ## 9. High-order quadratures
        ---
        """
    )
    md_hq_text = mo.md(
        r"""
        Every rule so far — rectangle, trapezoidal, Simpson — has the same
        shape: a weighted sum of function values,
        $$
        \int_a^b f(x)\,dx \approx \sum_{k=0}^{N} w_k\,f(x_k).
        $$
        There is a systematic way to build such a rule to any order: fit
        the unique degree-$N$ polynomial through $N+1$ nodes $x_k$ using
        the Lagrange basis functions
        $L_k(x)=\prod_{j\neq k}\frac{x-x_j}{x_k-x_j}$, then integrate that
        polynomial exactly. Since integration is linear, this hands back
        exactly the weighted-sum form above, with
        $$
        w_k = \int_a^b L_k(x)\,dx.
        $$
        The rule is then automatically **exact for any polynomial of
        degree $\leq N$** — the interpolating polynomial *is* $f$ in that
        case. Rectangle, trapezoidal and Simpson's rule are the $N=0,1,2$
        members of exactly this family, built on equally spaced nodes;
        this is called a **Newton–Cotes** rule. Since each weight is just
        an integral, the Romberg integrator from Section 7 computes it to
        machine precision without any extra machinery.
        """
    )
    return md_hq_text, md_hq_title


@app.cell
def _(romberg_table, trapezoidal_doublings):
    def lagrange_basis(x, j, nodes):
        result = 1.0
        for k, xk in enumerate(nodes):
            if k != j:
                result = result * (x - xk) / (nodes[j] - xk)
        return result

    def romberg_integrate(f, a, b, levels=11):
        return romberg_table(list(trapezoidal_doublings(f, a, b, levels)))[-1][-1]

    return lagrange_basis, romberg_integrate


@app.cell
def _(lagrange_basis, np, romberg_integrate):
    def newton_cotes(n, a=-1.0, b=1.0, open_rule=False, levels=11):
        if open_rule:
            h = (b - a) / (n + 2.0)
            nodes = [a + (i + 1) * h for i in range(n + 1)]
        else:
            h = (b - a) / n if n > 0 else (b - a)
            nodes = [a + i * h for i in range(n + 1)]
        weights = [
            romberg_integrate(lambda x, j=j: lagrange_basis(x, j, nodes), a, b, levels)
            for j in range(len(nodes))
        ]
        return np.array(nodes), np.array(weights)

    return (newton_cotes,)


@app.cell(hide_code=True)
def _(mo):
    nc_open_choice = mo.ui.dropdown(
        options={"Closed (includes endpoints)": False, "Open (excludes endpoints)": True},
        value="Closed (includes endpoints)",
        label="Newton–Cotes variant:",
    )
    nc_n_slider = mo.ui.slider(0, 14, value=2, step=1, label="Polynomial degree $N$ (number of nodes − 1)")
    return nc_n_slider, nc_open_choice


@app.cell(hide_code=True)
def _(
    md_hq_text,
    md_hq_title,
    mo,
    nc_n_slider,
    nc_open_choice,
    newton_cotes,
    plt,
):
    _n = nc_n_slider.value
    _nodes, _weights = newton_cotes(_n, -1.0, 1.0, nc_open_choice.value)

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _colors = ["crimson" if w < 0 else "steelblue" for w in _weights]
    _ax.vlines(_nodes, 0, _weights, color=_colors, lw=2)
    _ax.scatter(_nodes, _weights, color=_colors, zorder=3)
    _ax.axhline(0.0, color="black", lw=0.8)
    _ax.set(xlabel="node $x_k$", ylabel="weight $w_k$", xlim=(-1.15, 1.15))

    if _n == 0 and nc_open_choice.value:
        _note = "This is exactly the **rectangle rule**: one node, one weight $w_0=2$."
    elif _n == 1 and not nc_open_choice.value:
        _note = "This is exactly the **trapezoidal rule**: two endpoint nodes, equal weights."
    elif _n == 2 and not nc_open_choice.value:
        _note = "This is exactly **Simpson's rule**: the 1-4-1 weight pattern."
    elif any(w < 0 for w in _weights):
        _note = "Some weights have turned **negative** (red) — a warning sign that this rule is becoming numerically unstable."
    else:
        _note = ""

    mo.vstack(
        [
            md_hq_title,
            md_hq_text,
            mo.hstack([nc_open_choice, nc_n_slider]),
            _fig,
            mo.md(_note),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_runge_text = mo.md(
        r"""
        ### Exactness — and where it breaks down

        By construction, a Newton–Cotes rule with $N+1$ nodes integrates
        any polynomial of degree $\leq N$ exactly. But equally spaced
        nodes are a poor choice for interpolation in general: as $N$
        grows, high-degree interpolating polynomials tend to oscillate
        wildly near the ends of the interval (the **Runge phenomenon**),
        and the weights above already showed the symptom — large weights
        of alternating sign. Applied to the classic example of the Runge
        phenomenon (Section 7's harder test integrand),
        $$
        \int_{-1}^1 \frac{dx}{1+25x^2} = \frac{2}{5}\arctan(5) \approx 0.5494,
        $$
        increasing $N$ eventually makes a Newton–Cotes rule **worse**,
        not better.
        """
    )
    return (md_runge_text,)


@app.cell(hide_code=True)
def _(mo):
    hq_nmax_slider = mo.ui.slider(6, 24, value=16, step=2, label="Maximum $N$ in the comparison below")
    return (hq_nmax_slider,)


@app.cell(hide_code=True)
def _(
    hq_nmax_slider,
    newton_cotes,
    np,
    runge_a,
    runge_b,
    runge_exact,
    runge_fn,
):
    hq_N_sweep = np.arange(2, hq_nmax_slider.value + 1, 2)

    def _nc_error(n):
        nodes, weights = newton_cotes(n, runge_a, runge_b, False)
        estimate = np.sum(weights * runge_fn(nodes))
        return abs(estimate - runge_exact)

    hq_err_nc = np.array([_nc_error(n) for n in hq_N_sweep])
    return hq_N_sweep, hq_err_nc


@app.cell(hide_code=True)
def _(hq_N_sweep, hq_err_nc, hq_nmax_slider, md_runge_text, mo, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(hq_N_sweep, hq_err_nc, "o-", color="crimson", label="Newton–Cotes")
    _ax.set(xlabel="$N$ (polynomial degree)", ylabel="Absolute error", title="Integrating 1/(1+25x²) on [-1,1]")
    _ax.legend()

    mo.vstack([md_runge_text, hq_nmax_slider, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    md_cc_text = mo.md(
        r"""
        ### Taming the Runge phenomenon: Clenshaw–Curtis quadrature

        The weighted-sum construction never actually required the nodes
        to be equally spaced — only that they be distinct. Choosing them
        instead as the **Chebyshev nodes**,
        $$
        x_k = \frac{a+b}{2} + \frac{b-a}{2}\cos\!\left(\frac{(2k+1)\pi}{2N+2}\right), \qquad k=0,\ldots,N,
        $$
        which cluster near the endpoints of $[a,b]$, is known to control
        the interpolation error and suppress the Runge phenomenon. A
        Newton–Cotes-style rule built on these nodes instead of equally
        spaced ones is called **Clenshaw–Curtis quadrature**; the weights
        are obtained exactly the same way, by integrating the Lagrange
        basis functions — in production code they are instead computed
        in $\mathcal{O}(N\log N)$ via a discrete cosine transform, but
        plain Romberg integration is simpler and fast enough here.
        """
    )
    return (md_cc_text,)


@app.cell
def _(np):
    def chebyshev_nodes(n, a, b):
        k = np.arange(n + 1)
        return (a + b) / 2.0 + (b - a) / 2.0 * np.cos((2.0 * k + 1.0) / (2.0 * n + 2.0) * np.pi)

    return (chebyshev_nodes,)


@app.cell
def _(chebyshev_nodes, lagrange_basis, np, romberg_integrate):
    def clenshaw_curtis(n, a=-1.0, b=1.0, levels=11):
        nodes = chebyshev_nodes(n, a, b)
        weights = np.array(
            [
                romberg_integrate(lambda x, j=j: lagrange_basis(x, j, nodes), a, b, levels)
                for j in range(len(nodes))
            ]
        )
        return nodes, weights

    return (clenshaw_curtis,)


@app.cell(hide_code=True)
def _(clenshaw_curtis, md_cc_text, mo, plt):
    _nodes, _weights = clenshaw_curtis(10, -1.0, 1.0)

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.vlines(_nodes, 0, _weights, color="seagreen", lw=2)
    _ax.scatter(_nodes, _weights, color="seagreen", zorder=3)
    _ax.axhline(0.0, color="black", lw=0.8)
    _ax.set(xlabel="node $x_k$", ylabel="weight $w_k$", xlim=(-1.15, 1.15), title="10-point Clenshaw–Curtis weights")

    mo.vstack(
        [
            md_cc_text,
            _fig,
            mo.md(
                "Unlike Newton–Cotes, Clenshaw–Curtis weights stay "
                "**positive** at every order, another reason it avoids "
                "the numerical instability seen above."
            ),
        ]
    )
    return


@app.cell
def _(
    clenshaw_curtis,
    hq_N_sweep,
    np,
    runge_a,
    runge_b,
    runge_exact,
    runge_fn,
):
    def _cc_error(n):
        nodes, weights = clenshaw_curtis(n, runge_a, runge_b)
        estimate = np.sum(weights * runge_fn(nodes))
        return abs(estimate - runge_exact)

    hq_err_cc = np.array([_cc_error(n) for n in hq_N_sweep])
    return (hq_err_cc,)


@app.cell(hide_code=True)
def _(hq_N_sweep, hq_err_cc, hq_err_nc, mo, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(hq_N_sweep, hq_err_nc, "o-", color="crimson", label="Newton–Cotes")
    _ax.semilogy(hq_N_sweep, hq_err_cc, "s-", color="seagreen", label="Clenshaw–Curtis")
    _ax.set(xlabel="$N$", ylabel="Absolute error", title="Integrating 1/(1+25x²) on [-1,1]")
    _ax.legend()

    mo.vstack(
        [
            mo.md(
                "Applied to the same Runge-function test, Clenshaw–Curtis "
                "keeps converging where Newton–Cotes diverges — the "
                "Chebyshev nodes tame exactly the instability the weight "
                "plot above warned about."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_gauss_title = mo.md(
        """
        ## 10. Gaussian quadrature
        ---
        """
    )
    md_gauss_text = mo.md(
        r"""
        Every rule so far fixes the nodes first (equally spaced, or
        Chebyshev) and only *solves for* the weights. An $(N+1)$-point
        rule has $2(N+1)$ free numbers — $N+1$ nodes and $N+1$ weights —
        so fixing the nodes throws away half of that freedom. Letting the
        nodes float too, and choosing *both* nodes and weights to
        maximize the exactness degree, is **Gaussian quadrature**: an
        $n$-point Gauss rule is exact for every polynomial up to degree
        $2n-1$ — twice what an $n$-point Newton–Cotes rule achieves.

        The optimal nodes turn out to be the roots of the $n$th Legendre
        polynomial $P_n(x)$ on $[-1,1]$, with weights
        $$
        w_k = \frac{2}{(1-x_k^2)\,[P_n'(x_k)]^2}.
        $$
        Finding polynomial roots and weights this accurately by hand is
        impractical, so the code below reuses a dedicated root-finder
        (`gaussxw`, from `IntegrateGauss.py`) — the same routine already
        used for the Dawson-function example in the derivatives notebook.
        """
    )
    return md_gauss_text, md_gauss_title


@app.cell
def _():
    from IntegrateGauss import gaussxw, integrate_quadrature

    def gauss_legendre(n, a, b):
        x, w = gaussxw(n)
        return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

    return gauss_legendre, integrate_quadrature


@app.cell(hide_code=True)
def _(mo):
    gauss_n_slider = mo.ui.slider(1, 6, value=1, step=1, label="Number of Gauss–Legendre points $n$")
    return (gauss_n_slider,)


@app.cell(hide_code=True)
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    gauss_legendre,
    gauss_n_slider,
    integrate_quadrature,
    md_gauss_text,
    md_gauss_title,
    mo,
):
    _n = gauss_n_slider.value
    _estimate = integrate_quadrature(f, gauss_legendre(_n, a_ref, b_ref))

    mo.vstack(
        [
            md_gauss_title,
            md_gauss_text,
            gauss_n_slider,
            mo.hstack(
                [
                    mo.stat(label="Estimate", value=f"{_estimate:.10f}"),
                    mo.stat(label="Exact", value=f"{I_exact:.10f}"),
                    mo.stat(label="Absolute error", value=f"{abs(I_exact - _estimate):.3e}"),
                ],
                justify="center",
            ),
            mo.md(
                "The reference quartic has degree 4, so it takes "
                "$2n-1\\geq4$, i.e. $n\\geq3$ points, to integrate exactly "
                "— watch the error collapse to rounding error exactly "
                "there."
                if _n < 3
                else "**Exact to machine precision** — $n=3$ already satisfies $2n-1\\geq4$."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(clenshaw_curtis, gauss_legendre, integrate_quadrature, mo, newton_cotes):
    def _f2(x):
        return 7.0 * x**3 - 8.0 * x**2 - 3.0 * x + 3.0

    _exact_cubic = 2.0 / 3.0
    _trap_est = integrate_quadrature(_f2, newton_cotes(1, -1.0, 1.0, False))
    _cc_est = integrate_quadrature(_f2, clenshaw_curtis(1, -1.0, 1.0))
    _gauss_est = integrate_quadrature(_f2, gauss_legendre(2, -1.0, 1.0))

    mo.vstack(
        [
            mo.md(
                r"""
                It is not only the weights that matter — the node
                *positions* do too. All three rules below use exactly
                **two** function evaluations to integrate the same cubic,
                $\int_{-1}^1(7x^3-8x^2-3x+3)\,dx=\tfrac23$: two equally
                spaced nodes (trapezoidal), two Chebyshev nodes
                (Clenshaw–Curtis), and two *optimally placed* Gauss–Legendre
                nodes. Only the last is exact — $2n-1=3$ with $n=2$ — since
                it is the only one that also chose *where* to sample.
                """
            ),
            mo.hstack(
                [
                    mo.stat(label="Trapezoidal (2 pts)", value=f"{_trap_est:.6f}", caption=f"error {abs(_exact_cubic - _trap_est):.2e}"),
                    mo.stat(label="Clenshaw–Curtis (2 pts)", value=f"{_cc_est:.6f}", caption=f"error {abs(_exact_cubic - _cc_est):.2e}"),
                    mo.stat(label="Gauss–Legendre (2 pts)", value=f"{_gauss_est:.6f}", caption=f"error {abs(_exact_cubic - _gauss_est):.2e}"),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    gauss_legendre,
    hq_N_sweep,
    integrate_quadrature,
    np,
    runge_a,
    runge_b,
    runge_exact,
    runge_fn,
):
    hq_err_gauss = np.array(
        [
            abs(integrate_quadrature(runge_fn, gauss_legendre(n + 1, runge_a, runge_b)) - runge_exact)
            for n in hq_N_sweep
        ]
    )
    return (hq_err_gauss,)


@app.cell(hide_code=True)
def _(hq_N_sweep, hq_err_cc, hq_err_gauss, hq_err_nc, mo, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(hq_N_sweep, hq_err_nc, "o-", color="crimson", label="Newton–Cotes")
    _ax.semilogy(hq_N_sweep, hq_err_cc, "s-", color="seagreen", label="Clenshaw–Curtis")
    _ax.semilogy(hq_N_sweep, hq_err_gauss, "^-", color="steelblue", label="Gauss–Legendre")
    _ax.set(
        xlabel="$N$ (nodes − 1)",
        ylabel="Absolute error",
        title="Integrating 1/(1+25x²) on [-1,1]: all three methods",
    )
    _ax.legend()

    mo.vstack(
        [
            mo.md(
                "Putting all three side by side: Newton–Cotes eventually "
                "diverges, Clenshaw–Curtis and Gauss–Legendre both keep "
                "converging, and Gauss–Legendre needs the fewest points "
                "for a given accuracy — the payoff of also optimizing the "
                "node positions."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_genquad_text = mo.md(
        r"""
        ### Generalized Gaussian quadrature

        Gauss–Legendre is the $\omega(x)=1$ member of a larger family.
        For an integral with an explicit **weight function** $\omega(x)$,
        $$
        \int_a^b \omega(x)\,f(x)\,dx \approx \sum_{k=1}^n w_k\,f(x_k),
        $$
        the same idea — choose nodes and weights to maximize the
        exactness degree — works for any $\omega(x)$, provided the nodes
        are taken as the roots of the polynomial family orthogonal with
        respect to that weight:

        | | interval | weight $\omega(x)$ | polynomial family |
        |---|---|---|---|
        | Gauss–Legendre | $(-1,1)$ | $1$ | Legendre $P_n$ |
        | Gauss–Laguerre | $[0,\infty)$ | $e^{-x}$ | Laguerre $L_n$ |
        | Gauss–Hermite | $(-\infty,\infty)$ | $e^{-x^2}$ | Hermite $H_n$ |

        Nodes and weights for either family are obtained the same way as
        for Gauss–Legendre — as roots of the corresponding orthogonal
        polynomial — just swapping in a different, efficient root-finder
        for each family.
        """
    )
    return (md_genquad_text,)


@app.cell(hide_code=True)
def _(mo):
    md_laguerre_text = mo.md(
        r"""
        ### Gauss–Laguerre quadrature

        Gauss–Laguerre is built for exactly the situation Section 8 had
        to handle with a change of variables: a semi-infinite integral
        whose integrand decays like $e^{-x}$. Factor that decay out
        explicitly, $f(x) = e^{-x}g(x)$, and Gauss–Laguerre integrates
        $g(x)$ directly — no substitution needed, as long as $g$ itself
        is well-behaved.
        """
    )
    return (md_laguerre_text,)


@app.cell(hide_code=True)
def _(md_genquad_text, md_laguerre_text, mo, np, pd):
    import math
    from numpy.polynomial.laguerre import laggauss

    _xk, _wk = laggauss(8)
    _rows = [
        {
            "k": _k,
            "Gauss–Laguerre (n=8)": float(np.sum(_wk * _xk**_k)),
            "exact (k!)": float(math.factorial(_k)),
        }
        for _k in range(6)
    ]
    _df = pd.DataFrame(_rows)

    mo.vstack(
        [
            md_genquad_text,
            md_laguerre_text,
            mo.md("A quick sanity check: $\\int_0^\\infty x^k e^{-x}\\,dx = k!$ for every $k$ shown."),
            mo.ui.table(_df, page_size=6),
        ]
    )
    return laggauss, math


@app.cell(hide_code=True)
def _(mo):
    md_hermite_text = mo.md(
        r"""
        ### Gauss–Hermite quadrature

        Gauss–Hermite is the full-line counterpart: it targets integrals
        over $(-\infty,\infty)$ whose integrand carries a Gaussian tail
        $e^{-x^2}$ — exactly the kind that shows up when averaging over a
        normally distributed variable, or in matrix elements of the
        quantum harmonic oscillator. Writing $f(x)=e^{-x^2}g(x)$ and
        handing $g$ to a Gauss–Hermite rule integrates it directly on the
        whole real line, with no truncation and no change of variables.
        """
    )
    return (md_hermite_text,)


@app.cell(hide_code=True)
def _(math, md_hermite_text, mo, np, pd):
    from numpy.polynomial.hermite import hermgauss

    _xh, _wh = hermgauss(8)
    _rows = [
        {
            "k": 2 * _k,
            "Gauss–Hermite (n=8)": float(np.sum(_wh * _xh ** (2 * _k))),
            "exact (Γ(k+1/2))": float(math.gamma(_k + 0.5)),
        }
        for _k in range(4)
    ]
    _df = pd.DataFrame(_rows)

    mo.vstack(
        [
            md_hermite_text,
            mo.md(
                "A quick sanity check: $\\int_{-\\infty}^{\\infty} x^{2k} e^{-x^2}\\,dx = \\Gamma(k+\\tfrac12)$ "
                "for every even moment shown (odd moments vanish by symmetry)."
            ),
            mo.ui.table(_df, page_size=6),
        ]
    )
    return (hermgauss,)


@app.cell(hide_code=True)
def _(mo):
    weight_family_choice = mo.ui.dropdown(
        options={"Gauss–Laguerre, weight e⁻ˣ on [0,∞)": "laguerre", "Gauss–Hermite, weight e⁻ˣ² on (−∞,∞)": "hermite"},
        value="Gauss–Laguerre, weight e⁻ˣ on [0,∞)",
        label="Family:",
    )
    weight_family_n = mo.ui.slider(4, 24, value=12, step=2, label="Number of points $n$")
    return weight_family_choice, weight_family_n


@app.cell(hide_code=True)
def _(hermgauss, laggauss, mo, plt, weight_family_choice, weight_family_n):
    _n = weight_family_n.value
    if weight_family_choice.value == "laguerre":
        _nodes, _weights = laggauss(_n)
        _title = f"{_n}-point Gauss–Laguerre weights (log scale)"
        _color = "darkorange"
    else:
        _nodes, _weights = hermgauss(_n)
        _title = f"{_n}-point Gauss–Hermite weights (log scale)"
        _color = "purple"

    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.vlines(_nodes, 1e-20, _weights, color=_color, lw=2)
    _ax.scatter(_nodes, _weights, color=_color, zorder=3)
    _ax.set_yscale("log")
    _ax.set(xlabel="node $x_k$", ylabel="weight $w_k$ (log scale)", title=_title)

    mo.vstack(
        [
            mo.hstack([weight_family_choice, weight_family_n]),
            _fig,
            mo.md(
                "On a log scale the weights fall off by many orders of "
                "magnitude away from the origin — nodes far out in the "
                "tail barely matter, since the weight function itself has "
                "already suppressed the integrand there."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_capstone_title = mo.md(
        """
        ## 11. A physics capstone: density of a relativistic quantum gas
        ---
        """
    )
    md_capstone_text = mo.md(
        r"""
        The number density of a relativistic ideal gas of particles with
        mass $m$, spin degeneracy $d$, temperature $T$ and chemical
        potential $\mu$ is a momentum-space integral,
        $$
        n = \frac{d}{2\pi^2}\int_0^\infty dk\;k^2
        \left[\exp\!\left(\frac{\sqrt{m^2+k^2}-\mu}{T}\right) + \eta\right]^{-1},
        $$
        where $\eta=0,+1,-1$ selects Maxwell–Boltzmann, Fermi–Dirac or
        Bose–Einstein statistics. Rescaling $x=k/T$, $\tilde m=m/T$,
        $\tilde\mu=\mu/T$ makes everything dimensionless,
        $$
        \tilde n \equiv n/T^3 = \int_0^\infty dx\; x^2
        \left[\exp\!\left(\sqrt{\tilde m^2+x^2}-\tilde\mu\right)+\eta\right]^{-1}.
        $$
        At large $x$ the bracket behaves like
        $e^{-(\sqrt{\tilde m^2+x^2}-\tilde\mu)}$, so the integrand decays
        like $e^{-x}$ — exactly the tail Gauss–Laguerre quadrature was
        built for. Writing the integrand as $e^{-x}g(x)$ with
        $$
        g(x) = x^2\,e^{x}\left[\exp\!\left(\sqrt{\tilde m^2+x^2}-\tilde\mu\right)+\eta\right]^{-1}
        $$
        and handing $g$ to a Gauss–Laguerre rule integrates it directly on
        $[0,\infty)$ — no change of variables at all, unlike the
        rectangle-rule approach of Section 8. In the Maxwell–Boltzmann
        limit ($\eta=0$) the result can be checked against a closed form
        in terms of the modified Bessel function $K_2$,
        $$
        \tilde n_{\rm MB} = \frac{d\,\tilde m^2}{2\pi^2}\,K_2(\tilde m)\,e^{\tilde\mu}.
        $$
        """
    )
    return md_capstone_text, md_capstone_title


@app.cell
def _(np):
    def g_thermal(x, T, mu, m, d, eta):
        return d * x**2 * np.exp(x) / (2.0 * np.pi**2) / (np.exp(np.sqrt((m / T) ** 2 + x**2) - mu / T) + eta)

    return (g_thermal,)


@app.cell
def _(kn, np):
    def nT3_analytic(T, mu, m, d=1.0):
        return d * m**2 / (2.0 * np.pi**2 * T**2) * kn(2, m / T) * np.exp(mu / T)

    return (nT3_analytic,)


@app.cell
def _(laggauss):
    capstone_laguerre_nodes, capstone_laguerre_weights = laggauss(32)
    return capstone_laguerre_nodes, capstone_laguerre_weights


@app.cell(hide_code=True)
def _():
    capstone_eta_options = {
        "Maxwell–Boltzmann (η=0)": 0.0,
        "Bose–Einstein (η=−1)": -1.0,
        "Fermi–Dirac (η=+1)": 1.0,
    }
    capstone_eta_labels = {v: k for k, v in capstone_eta_options.items()}
    return capstone_eta_labels, capstone_eta_options


@app.cell(hide_code=True)
def _(capstone_eta_options, mo):
    capstone_eta_choice = mo.ui.dropdown(
        options=capstone_eta_options,
        value="Maxwell–Boltzmann (η=0)",
        label="Statistics:",
    )
    capstone_T_slider = mo.ui.slider(50.0, 400.0, value=150.0, step=5.0, label="Temperature $T$ [MeV]")
    capstone_mu_slider = mo.ui.slider(-100.0, 100.0, value=0.0, step=5.0, label="Chemical potential $\\mu$ [MeV]")
    return capstone_T_slider, capstone_eta_choice, capstone_mu_slider


@app.cell
def _(
    capstone_T_slider,
    capstone_eta_choice,
    capstone_laguerre_nodes,
    capstone_laguerre_weights,
    capstone_mu_slider,
    g_thermal,
    nT3_analytic,
    np,
):
    _T = capstone_T_slider.value
    _mu = capstone_mu_slider.value
    _eta = capstone_eta_choice.value
    _m = 138.0
    _d = 1.0

    capstone_nT3_numeric = float(
        np.sum(capstone_laguerre_weights * g_thermal(capstone_laguerre_nodes, _T, _mu, _m, _d, _eta))
    )
    capstone_nT3_mb = nT3_analytic(_T, _mu, _m, _d)
    return capstone_nT3_mb, capstone_nT3_numeric


@app.cell(hide_code=True)
def _(
    capstone_T_slider,
    capstone_eta_choice,
    capstone_eta_labels,
    capstone_laguerre_nodes,
    capstone_laguerre_weights,
    capstone_mu_slider,
    capstone_nT3_mb,
    capstone_nT3_numeric,
    g_thermal,
    md_capstone_text,
    md_capstone_title,
    mo,
    nT3_analytic,
    np,
    plt,
):
    _mu = capstone_mu_slider.value
    _eta = capstone_eta_choice.value
    _m = 138.0
    _d = 1.0
    _T_range = np.linspace(60.0, 400.0, 40)

    _n_numeric = np.array(
        [
            np.sum(capstone_laguerre_weights * g_thermal(capstone_laguerre_nodes, T, _mu, _m, _d, _eta))
            for T in _T_range
        ]
    )
    _n_mb = nT3_analytic(_T_range, _mu, _m, _d)

    _fig, _ax = plt.subplots()
    _ax.plot(_T_range, _n_mb, "k--", lw=1.5, label="Maxwell–Boltzmann, analytic ($K_2$)")
    _ax.plot(_T_range, _n_numeric, color="crimson", lw=2, label=f"{capstone_eta_labels[_eta]}, Gauss–Laguerre")
    _ax.axvline(capstone_T_slider.value, color="0.6", lw=0.8, ls=":")
    _ax.set(xlabel="$T$ [MeV]", ylabel=r"$n/T^3$")
    _ax.legend()

    mo.vstack(
        [
            md_capstone_title,
            md_capstone_text,
            mo.hstack([capstone_eta_choice, capstone_T_slider, capstone_mu_slider]),
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Numeric $n/T^3$ (32-pt Gauss–Laguerre)", value=f"{capstone_nT3_numeric:.8f}"),
                    mo.stat(label="Analytic $n/T^3$ (Maxwell–Boltzmann)", value=f"{capstone_nT3_mb:.8f}"),
                    mo.stat(
                        label="Deviation from Maxwell–Boltzmann",
                        value=f"{abs(capstone_nT3_numeric - capstone_nT3_mb):.3e}"
                        if _eta == 0.0
                        else "quantum correction",
                    ),
                ],
                justify="center",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
