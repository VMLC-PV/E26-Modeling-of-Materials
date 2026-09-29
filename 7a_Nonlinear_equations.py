import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from wigglystuff import FormulaAnimation, PlaySlider

    # try:
    #     from utils import plot_settings_screen
    # except ImportError:       
    #     pass
    return FormulaAnimation, PlaySlider, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Non-linear equations and root-finding

    Often we encounter a problem where we need to find a root of an equation, i.e.:
    \[
    f(x) = 0
    \]
    that we cannot solve explicitly.
    The solution then has to be obtained numerically, by an **iterative** procedure that produces a
    sequence $x_0, x_1, x_2, \ldots$ converging (hopefully) to the root $x^*$.

    The most common methods fall into two families, the **local** and the **non-local** methods. The local methods need a single starting point, and converge fast *when* they converge. The non-local methods need two points that bracket the root, and converge slowly but reliably.

    | Section | Method | Family | Needs | Guaranteed? |
    |---|---|---|---|---|
    | 1 | Relaxation (fixed-point iteration) | local | a rewrite $x=\varphi(x)$ | no |
    | 2 | Bisection | non-local (two-point) | a bracket $[a,b]$ with a sign change | yes |
    | 3 | Secant | non-local (two-point) | two starting points | no |
    | 4 | False position | non-local (two-point) | a bracket $[a,b]$ with a sign change | yes |
    | 5 | Newton–Raphson | local | one starting point and $f'(x)$ | no |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test problems
    ---

    Throughout this notebook we use three test equations:

    - $f_1(x) = x + e^{-x} - 2 = 0$, with a root at $x^*\approx1.8414$;
    - $f_2(x) = x^3 - x - 1 = 0$, a polynomial with a single real root at
      $x^*\approx1.3247$ and a local extremum at $x=1/\sqrt3\approx0.577$;
    - $f_3(x) = x^3 - 2x + 2 = 0$, with a single real root at
      $x^*\approx-1.7693$ — a classic trap for Newton's method (Section 5).
    """)
    return


@app.cell(hide_code=True)
def _(np):
    root_problems = {
        "f1(x) = x + exp(-x) - 2": dict(
            tex=r"x+e^{-x}-2",
            f=lambda x: x + np.exp(-x) - 2.0,
            df=lambda x: 1.0 - np.exp(-x),
            view=(-0.5, 3.5),
            ylim=(-1.5, 2.0),
            bracket=(0.0, 3.0),
            x0=0.5,
            x0_range=(-0.4, 3.4),
        ),
        "f2(x) = x^3 - x - 1": dict(
            tex=r"x^3-x-1",
            f=lambda x: x**3 - x - 1.0,
            df=lambda x: 3.0 * x**2 - 1.0,
            view=(-1.5, 3.0),
            ylim=(-4.0, 8.0),
            bracket=(0.0, 3.0),
            x0=0.5,
            x0_range=(-1.4, 2.2),
        ),
        "f3(x) = x^3 - 2x + 2": dict(
            tex=r"x^3-2x+2",
            f=lambda x: x**3 - 2.0 * x + 2.0,
            df=lambda x: 3.0 * x**2 - 2.0,
            view=(-2.5, 2.0),
            ylim=(-10.0, 8.0),
            bracket=(-3.0, 3.0),
            x0=0.0,
            x0_range=(-2.4, 1.9),
        ),
    }
    # Reference roots, converged to machine precision
    for _p in root_problems.values():
        _p["root"] = bisection_method(_p["f"], *_p["bracket"], tolerance=1e-15)[0]
    return (root_problems,)


@app.cell(hide_code=True)
def _(np, plt, root_problems):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for _ax, (_name, _p) in zip(_axes, root_problems.items()):
        _x = np.linspace(*_p["view"], 400)
        _ax.plot(_x, _p["f"](_x), color="crimson", lw=2)
        _ax.axhline(0.0, color="black", ls="--", lw=0.8)
        _ax.plot([_p["root"]], [0.0], "o", color="black", zorder=3)
        _ax.annotate(f"$x^*={_p['root']:.4f}$", (_p["root"], 0.0), xytext=(8, -18), textcoords="offset points")
        _ax.set(xlabel="$x$", ylim=_p["ylim"], title=f"${_p['tex']}$")
    _axes[0].set_ylabel("$f(x)$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(np):
    def view_window(problem, points, zoom_center=None, zoom_half=None):
        """x- and y-limits: the problem window widened to the visited points, or a zoom."""
        f = problem["f"]
        v0, v1 = problem["view"]
        y0, y1 = problem["ylim"]
        if zoom_center is not None:
            half = max(zoom_half, 1e-14)
            xlo, xhi = zoom_center - half, zoom_center + half
            fs = f(np.linspace(xlo, xhi, 600))
            ylo, yhi = min(fs.min(), 0.0), max(fs.max(), 0.0)  # keep y = 0 in view
            ypad = 0.15 * (yhi - ylo + 1e-300)
            return (xlo, xhi), (ylo - ypad, yhi + ypad)

        # Widen to the visited points, but never beyond 3 window widths on each side
        width = v1 - v0
        pts = [p for p in points if np.isfinite(p)] + [v0, v1]
        xlo = max(min(pts), v0 - 3.0 * width)
        xhi = min(max(pts), v1 + 3.0 * width)
        pad = 0.05 * (xhi - xlo)
        xlo, xhi = xlo - pad, xhi + pad
        fs = f(np.linspace(xlo, xhi, 600))
        # Grow the y-range with the view, but never beyond 3x the problem's window
        return (xlo, xhi), (max(min(y0, fs.min()), 3.0 * y0), min(max(y1, fs.max()), 3.0 * y1))

    return (view_window,)


@app.cell(hide_code=True)
def _(plt):
    def method_figure():
        """Common layout for every method: a large construction panel and a small error-history panel."""
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[2.6, 1]))
        ax2.tick_params(labelsize=8)
        ax2.xaxis.label.set_size(9)
        ax2.title.set_size(10)
        return fig, ax, ax2

    def finish_error_panel(ax2, n):
        ax2.axvline(n, color="#1d4ed8", lw=1.0, ls=":")
        ax2.set_title("Error history", fontsize=10)
        ax2.set_xlabel("Iteration $n$", fontsize=9)
        ax2.legend(fontsize=8)

    return finish_error_panel, method_figure


@app.cell(hide_code=True)
def _(mo):
    def failure_callout(method, status):
        """A red callout when an iteration did not converge, otherwise nothing."""
        reasons = {
            "not applicable": "$f(a)$ and $f(b)$ have the same sign: the interval is not a bracket.",
            "diverged": "the iterates ran away to infinity (overflow or a pole).",
            "max. iterations": "no convergence within the maximum number of iterations.",
            "flat secant": "two points with equal $f$ give a horizontal secant that never crosses zero.",
            "f'(x) = 0": "the tangent is horizontal, $f'(x_n)=0$, and never crosses zero.",
        }
        if status == "converged":
            return mo.md("")
        return mo.md(f"**{method} fails:** {reasons[status]}").callout(kind="danger")

    return (failure_callout,)


@app.cell(hide_code=True)
def _(mo):
    def bracket_slider(problem, label):
        lo = min(problem["view"][0], problem["bracket"][0]) - 0.5
        hi = max(problem["view"][1], problem["bracket"][1]) + 0.5
        return mo.ui.range_slider(start=lo, stop=hi, step=0.05, value=list(problem["bracket"]), label=label)

    return (bracket_slider,)


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    relax_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="When does x = φ(x) converge?",
            steps=[
                {
                    "tex": r"\varphi(x_{n}) = \varphi(x^*) + (x_{n}-x^*)\varphi'(x^*) + \cdots",
                    "note": "Taylor expansion.",
                },
                {
                    "tex": r"\varphi(x_{n}) = x^* + (x_{n}-x^*)\varphi'(x^*) + \cdots",
                    "note": "x* is a solution so φ(x*) = x* .",
                },
                {
                    "tex": r"x_{n+1} = x^* + (x_{n}-x^*)\varphi'(x^*) + \cdots",
                    "note": "φ(xn) = xn+1 as we iterate.",
                },
                {
                    "tex": r"x_{n+1} - x^* = (x_{n}-x^*)\varphi'(x^*) ",
                    "note": "The distance as we iterate change by φ'(x) .",
                },
                {
                    "tex": r"|\varphi'(x^*)| < 1 \;\Longrightarrow\; \text{convergence}",
                    "note": "A geometric progression: linear convergence with ratio |φ'(x*)|.",
                },
            ],
            height=280,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 1. Relaxation method (fixed-point iteration)
                ---

                The simplest idea is to rewrite the equation $f(x)=0$ in the
                form $x = \varphi(x)$, and iterate $x_{n+1}=\varphi(x_n)$ from an initial guess
                $x_0$.  
                Such a rewrite always exists, e.g. $\varphi(x)=x+f(x)$, but it is far from unique, and the choice of $\varphi$ decides everything.
                """
            ),
            relax_anim,
            mo.md(
                r"""
                The iteration converges if $|\varphi'(x)|<1$ over the range of
                values visited by the iterates, and the smaller $|\varphi'|$,
                the faster. This is a **local** method: nothing guarantees
                convergence from an arbitrary starting point.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def relaxation_method(phi, x0, tolerance=1e-10, max_iterations=100):
        xs = [np.float64(x0)]
        with np.errstate(all="ignore"):  # let divergence show up as inf/nan
            for _ in range(max_iterations):
                xs.append(phi(xs[-1]))  # the next iteration
                if not np.isfinite(xs[-1]):
                    return xs[-1], xs, "diverged"
                if abs(xs[-1] - xs[-2]) < tolerance:
                    return xs[-1], xs, "converged"
        return xs[-1], xs, "max. iterations"

    return (relaxation_method,)


@app.cell(hide_code=True)
def _(np):
    relax_choices = {
        "φ1(x) = 2 - exp(-x)   (from f1)": dict(
            tex=r"\varphi_1(x)=2-e^{-x}", phi=lambda x: 2.0 - np.exp(-x), dphi=lambda x: np.exp(-x), bounds=(-0.5, 3.0), x0=0.5, root=1.8414056604369606
        ),
        "φ2(x) = x^3 - 1   (from f2)": dict(
            tex=r"\varphi_2(x)=x^3-1", phi=lambda x: x**3 - 1.0, dphi=lambda x: 3.0 * x**2, bounds=(-1.5, 2.5), x0=0.0, root=1.324717957244746
        ),
        "φ3(x) = (1 + x)^(1/3)   (from f2)": dict(
            tex=r"\varphi_3(x)=(1+x)^{1/3}", phi=lambda x: np.cbrt(1.0 + x), dphi=lambda x: 1.0 / (3.0 * np.cbrt(1.0 + x) ** 2), bounds=(-1.0, 3.0), x0=0.0, root=1.324717957244746
        ),
        "φ4(x) = 1/(x^2 - 1)   (from f2)": dict(
            tex=r"\varphi_4(x)=\frac{1}{x^2-1}", phi=lambda x: 1.0 / (x**2 - 1.0), dphi=lambda x: -2.0 * x / (x**2 - 1.0) ** 2, bounds=(-2.5, 2.5), x0=1.2, root=1.324717957244746
        ),
    }
    return (relax_choices,)


@app.cell(hide_code=True)
def _(mo, relax_choices):
    relax_choice = mo.ui.dropdown(options=list(relax_choices), value="φ1(x) = 2 - exp(-x)   (from f1)", label="Rewrite $x=\\varphi(x)$:")
    relax_zoom = mo.ui.switch(label="Zoom on current step")
    return relax_choice, relax_zoom


@app.cell(hide_code=True)
def _(mo, relax_choice, relax_choices):
    _c = relax_choices[relax_choice.value]
    relax_x0 = mo.ui.slider(_c["bounds"][0] + 0.05, _c["bounds"][1] - 0.05, step=0.05, value=_c["x0"], label="Starting point $x_0$")
    return (relax_x0,)


@app.cell
def _(relax_choice, relax_choices, relax_x0, relaxation_method):
    relax_root, relax_xs, relax_status = relaxation_method(relax_choices[relax_choice.value]["phi"], round(relax_x0.value, 4))
    return relax_root, relax_status, relax_xs


@app.cell(hide_code=True)
def _(PlaySlider, mo, relax_xs):
    relax_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(len(relax_xs) - 1, 1), step=1, interval_ms=500))
    return (relax_step,)


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, np):
    def draw_cobweb_step(choice, xs, n, zoom=False):
        """Cobweb diagram of x_{k+1} = phi(x_k) up to step n, plus the error history."""
        lo, hi = choice["bounds"]
        finite = [x for x in xs if np.isfinite(x)]
        n = min(n, len(finite) - 1)
        if zoom and n > 0:
            # Square window framing the current step (x_{n-1}, x_{n-1}) -> (x_{n-1}, x_n) -> (x_n, x_n)
            center = 0.5 * (finite[n - 1] + finite[n])
            half = max(abs(finite[n] - finite[n - 1]), 1e-14)
            wlo, whi = center - half, center + half
        else:
            wlo, whi = lo, hi
        xgrid = np.linspace(wlo, whi, 800)
        with np.errstate(all="ignore"):
            phigrid = choice["phi"](xgrid)
        phigrid[np.abs(phigrid - xgrid) > 10 * (whi - wlo)] = np.nan  # hide poles

        def staircase(pts):
            px, py = [pts[0]], [pts[0]]
            for xa, xb in zip(pts[:-1], pts[1:]):
                px += [xa, xb]
                py += [xb, xb]
            return px, py

        fig, ax, ax2 = method_figure()

        # --- Left: the cobweb
        ax.plot(xgrid, phigrid, color="crimson", lw=2, label=f"${choice['tex']}$")
        ax.plot(xgrid, xgrid, color="black", lw=1.0, ls="--", label="$y=x$")
        ax.plot(*staircase(finite), color="0.8", lw=1.0, zorder=1)  # the whole path, faintly
        ax.plot(*staircase(finite[: n + 1]), color="steelblue", lw=1.6, label="Iterates up to $x_n$")
        ax.plot([choice["root"]], [choice["root"]], "o", color="black", ms=6, zorder=4, label="Fixed point $x^*$")
        ax.plot([finite[n]], [finite[n]], "o", color="#1d4ed8", ms=8, zorder=5, label=f"Current iterate $x_{{{n}}}$")
        ax.set(xlim=(wlo, whi), ylim=(wlo, whi), xlabel="$x$", ylabel="$y$", title=f"Relaxation: iteration {n}")
        ax.legend(loc="upper left", fontsize=9)

        # --- Right: error history, with the geometric prediction
        err = np.abs(np.array(finite) - choice["root"])
        its = np.arange(len(err))
        slope = abs(choice["dphi"](choice["root"]))
        ax2.semilogy(its, np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label="$|x_n-x^*|$")
        with np.errstate(over="ignore"):
            ax2.semilogy(its, err[0] * slope**its, "--", color="0.4", lw=1.0, label=r"$|\varphi'(x^*)|^n|x_0-x^*|$")
        ax2.set(ylim=(1e-17, 1e6), xlim=(-0.5, max(len(xs) - 1, 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_cobweb_step,)


@app.cell(hide_code=True)
def _(
    draw_cobweb_step,
    failure_callout,
    mo,
    relax_choice,
    relax_choices,
    relax_root,
    relax_status,
    relax_step,
    relax_x0,
    relax_xs,
    relax_zoom,
):
    _c = relax_choices[relax_choice.value]
    _n = min(int(relax_step.value["value"]), len(relax_xs) - 1)
    _slope = abs(_c["dphi"](_c["root"]))

    mo.vstack(
        [
            mo.md(
                "The **cobweb diagram** alternates a vertical step to the curve, $x_n\\to\\varphi(x_n)$, and a horizontal "
                "step back to the diagonal, $\\varphi(x_n)\\to x_{n+1}$. Choose $x_0$, then press ▶."
            ),
            mo.hstack([relax_choice, relax_x0, relax_zoom], justify="center", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), relax_step], justify="start", align="center"),
            failure_callout("Relaxation", relax_status),
            mo.hstack(
                [
                    draw_cobweb_step(_c, relax_xs, _n, zoom=relax_zoom.value),
                    mo.vstack(
                        [
                            mo.stat(
                                label="Slope at the fixed point",
                                value=f"{_slope:.3f}",
                                caption="|φ'(x*)| < 1: attracts" if _slope < 1 else "|φ'(x*)| > 1: repels",
                            ),
                            mo.stat(label="Current iterate $x_n$", value=f"{relax_xs[_n]:.10g}"),
                            mo.stat(label="Last iterate", value=f"{relax_root:.10g}", caption=relax_status),
                            mo.stat(label="Iterations", value=f"{len(relax_xs) - 1}"),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dropdown offers four rewrites. $\varphi_1$ comes from $f_1$; the other
    three are rewrites of the *same* equation $f_2$: $x^3-x-1=0$, and they
    give three very different iterations:

    | Rewrite | $\varphi'(x^*)$ | Behaviour |
    |---|---|---|
    | $\varphi_2(x)=x^3-1$ | $3x^{*2}\approx5.3$ | runs away to $-\infty$ (overflow) |
    | $\varphi_3(x)=(1+x)^{1/3}$ | $\approx0.19$ | converges geometrically |
    | $\varphi_4(x)=1/(x^2-1)$ | $\approx-4.6$ | jumps around chaotically, then hits the pole at $x=\pm1$ |

    For $f_1$, $\varphi_1(x)=2-e^{-x}$ has $\varphi_1'(x^*)=e^{-x^*}\approx0.16$
    and converges from any starting point; the measured error follows the
    geometric prediction (dashed line) almost exactly. Note that
    $(1+x)^{1/3}$ is evaluated with `np.cbrt`: `(1 + x)**(1/3)` returns
    `nan` for negative arguments.

    Choosing a good $\varphi$ requires insight into the problem. The next
    methods need no such rewrite, only $f$ itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Bisection method
    ---

    If $f$ is continuous and changes sign on an interval, $f(a)\,f(b)<0$,
    then it has at least one root $a<x^*<b$. The **bisection method**
    halves the bracket at every step:

    1. take the midpoint $c=(a+b)/2$;
    2. keep the half $[a,c]$ or $[c,b]$ on which $f$ still changes sign;
    3. repeat until $b-a$ is smaller than the desired tolerance $\varepsilon$.

    The method *cannot* fail once a bracket is found, and its cost is known in
    advance: after $n$ steps the bracket has width $(b-a)/2^n$, so reaching a
    tolerance $\varepsilon$ takes
    $$
    n = \left\lceil \log_2\frac{b-a}{\varepsilon}\right\rceil
    $$
    iterations — about 35 for $b-a=3$ and $\varepsilon=10^{-10}$. Each step
    gains exactly one binary digit: the convergence is **linear**, with
    ratio $1/2$.
    """)
    return


@app.function
def bisection_method(f, a, b, tolerance=1e-10, max_iterations=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return None, [], "not applicable"  # no sign change: no bracket

    history = []
    while (b - a) > tolerance and len(history) < max_iterations:
        c = 0.5 * (a + b)  # take the midpoint
        fc = f(c)
        history.append(dict(a=a, b=b, x=c, fx=fc))
        if fc * fa < 0.0:
            b, fb = c, fc  # the root is in the left half
        else:
            a, fa = c, fc  # the root is in the right half
    return 0.5 * (a + b), history, "converged"


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, np, view_window):
    def draw_bracket_step(problem, history, n, method, zoom=False):
        """Draw iteration n (1-based) of a two-point method, plus its convergence history."""
        f = problem["f"]
        step = history[n - 1]
        a, b, x_new = step["a"], step["b"], step["x"]

        visited = [v for s in history[:n] for v in (s["a"], s["b"], s["x"])]
        if zoom:
            (xlo, xhi), ylim = view_window(problem, visited, zoom_center=x_new, zoom_half=1.5 * abs(b - a))
        else:
            (xlo, xhi), ylim = view_window(problem, visited)
        xs = np.linspace(xlo, xhi, 600)

        fig, ax, ax2 = method_figure()

        # --- Left: the geometry of the current step
        ax.plot(xs, f(xs), color="crimson", lw=2, label="$f(x)$")
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        if method in ("bisection", "false position"):
            ax.axvspan(min(a, b), max(a, b), color="#fdf0d5", zorder=0, label="Current bracket")
        if method in ("false position", "secant"):
            pts = sorted([(a, f(a)), (b, f(b)), (x_new, 0.0)])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="steelblue", lw=1.5, label="Secant line")
        for s in history[: n - 1]:
            ax.plot([s["x"]], [0.0], "o", color="0.65", ms=5, zorder=3)
        ax.plot([a, b], [f(a), f(b)], "o", color="black", ms=7, zorder=4, label="Points used, $a$ and $b$")
        ax.plot([x_new], [0.0], "o", color="#1d4ed8", ms=8, zorder=5, label="New estimate")
        ax.vlines(x_new, 0.0, step["fx"], colors="#1d4ed8", linestyles=":", lw=1.2)
        ax.set(xlim=(xlo, xhi), ylim=ylim, xlabel="$x$", ylabel="$f(x)$", title=f"{method.capitalize()}: iteration {n}")
        ax.legend(loc="upper left", fontsize=9)

        # --- Right: error history (and bracket width for the bracketing methods)
        its = np.arange(1, len(history) + 1)
        err = np.abs(np.array([s["x"] for s in history]) - problem["root"])
        ax2.semilogy(its, np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label="$|x_n-x^*|$")
        if method in ("bisection", "false position"):
            width = np.array([abs(s["b"] - s["a"]) for s in history])
            ax2.semilogy(its, np.maximum(width, 1e-17), "-", color="0.4", lw=1.0, label="$|b-a|$")
        ax2.set(ylim=(1e-17, 1e3), xlim=(0.5, len(history) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_bracket_step,)


@app.cell(hide_code=True)
def _(pd):
    def history_table(history):
        return pd.DataFrame(
            [dict(n=i + 1, a=s["a"], b=s["b"], x_n=s["x"], f_x_n=s["fx"]) for i, s in enumerate(history)]
        )

    return (history_table,)


@app.cell(hide_code=True)
def _(mo, root_problems):
    bis_problem = mo.ui.dropdown(options=list(root_problems), value="f1(x) = x + exp(-x) - 2", label="Equation:")
    bis_zoom = mo.ui.switch(label="Zoom on bracket")
    return bis_problem, bis_zoom


@app.cell(hide_code=True)
def _(bis_problem, bracket_slider, root_problems):
    bis_range = bracket_slider(root_problems[bis_problem.value], "Initial bracket $[a,b]$")
    return (bis_range,)


@app.cell
def _(bis_problem, bis_range, root_problems):
    bis_root, bis_history, bis_status = bisection_method(root_problems[bis_problem.value]["f"], *bis_range.value)
    return bis_history, bis_root, bis_status


@app.cell(hide_code=True)
def _(PlaySlider, bis_history, mo):
    bis_step = mo.ui.anywidget(PlaySlider(value=1, min_value=1, max_value=max(len(bis_history), 1), step=1, interval_ms=400))
    return (bis_step,)


@app.cell(hide_code=True)
def _(
    bis_history,
    bis_problem,
    bis_range,
    bis_root,
    bis_status,
    bis_step,
    bis_zoom,
    draw_bracket_step,
    failure_callout,
    history_table,
    mo,
    np,
    root_problems,
):
    _p = root_problems[bis_problem.value]
    _controls = mo.hstack([bis_problem, bis_range, bis_zoom], justify="center", align="center", gap=2)

    if bis_status == "not applicable":
        _body = failure_callout("Bisection", bis_status)
    else:
        _n = min(int(bis_step.value["value"]), len(bis_history))
        _a, _b = bis_range.value
        _body = mo.vstack(
            [
                mo.hstack([mo.md("**Iteration:**"), bis_step], justify="start", align="center"),
                mo.hstack(
                    [
                        draw_bracket_step(_p, bis_history, _n, "bisection", zoom=bis_zoom.value),
                        mo.vstack(
                            [
                                mo.stat(label="Estimate $x_n$", value=f"{bis_history[_n - 1]['x']:.12f}"),
                                mo.stat(label="Final root", value=f"{bis_root:.12f}"),
                                mo.stat(
                                    label="Iterations",
                                    value=f"{len(bis_history)}",
                                    caption=f"predicted ⌈log₂((b−a)/ε)⌉ = {int(np.ceil(np.log2((_b - _a) / 1e-10)))}",
                                ),
                            ]
                        ),
                    ],
                    widths=[0.8, 0.2],
                    align="center",
                ),
                mo.accordion({"Iteration table": mo.ui.table(history_table(bis_history), page_size=10, selection=None)}),
            ]
        )

    mo.vstack([_controls, _body])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Press ▶ to animate, switch on **Zoom on bracket** to follow the bracket
    as it shrinks, and compare the equations: the number of iterations
    depends *only* on the initial bracket width and the tolerance, never on
    the shape of $f$. Bisection ignores everything about $f$ except its sign
    — robust, but wasteful. Try also a bracket without sign change, e.g.
    $[2,3]$ for $f_1$.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    secant_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Where does the secant cross zero?",
            steps=[
                {
                    "tex": r"y(x) = f(x_n) + \frac{f(x_n)-f(x_{n-1})}{x_n-x_{n-1}}\,(x-x_n)",
                    "note": "Straight line through the two most recent points.",
                },
                {
                    "tex": r"0 = f(x_n) + \frac{f(x_n)-f(x_{n-1})}{x_n-x_{n-1}}\,(x_{n+1}-x_n)",
                    "note": "Require that the line crosses y = 0 at the new point.",
                },
                {
                    "tex": r"x_{n+1} = x_n - f(x_n)\,\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}",
                    "note": "Solve for the new estimate of the root.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 3. Secant method
                ---

                Bisection throws away the *values* of $f$: if $|f(a)|$ is much
                smaller than $|f(b)|$, the root is probably closer to $a$. The
                **secant method** uses this information. It replaces $f$ by
                the straight line through the two most recent points and takes
                the point where that line crosses $y=0$ as the next estimate.
                """
            ),
            secant_anim,
            mo.md(
                r"""
                The method always keeps the two most recent points, whether or
                not they bracket the root. This makes it fast — its order of
                convergence is the golden ratio $p=(1+\sqrt5)/2\approx1.618$,
                i.e. the number of correct digits grows by about 60% per step —
                but there is no guarantee: the iterates may leave the initial
                interval, and nothing forces them to come back.
                """
            ),
        ]
    )
    return


@app.function
def secant_method(f, a, b, tolerance=1e-10, max_iterations=100):
    fa, fb = f(a), f(b)
    history = []
    x_prev = x_new = a
    for _ in range(max_iterations):
        if fb == fa:
            return x_new, history, "flat secant"  # the line never crosses zero
        x_prev = x_new
        x_new = a - fa * (b - a) / (fb - fa)
        f_new = f(x_new)
        history.append(dict(a=a, b=b, x=x_new, fx=f_new))
        b, fb = a, fa  # keep the two most recent points
        a, fa = x_new, f_new
        if abs(x_new - x_prev) < tolerance:
            return x_new, history, "converged"
    return x_new, history, "max. iterations"


@app.cell(hide_code=True)
def _(mo, root_problems):
    sec_problem = mo.ui.dropdown(options=list(root_problems), value="f2(x) = x^3 - x - 1", label="Equation:")
    sec_zoom = mo.ui.switch(label="Zoom on last points")
    return sec_problem, sec_zoom


@app.cell(hide_code=True)
def _(bracket_slider, root_problems, sec_problem):
    sec_range = bracket_slider(root_problems[sec_problem.value], "Starting points $x_0, x_1$")
    return (sec_range,)


@app.cell
def _(root_problems, sec_problem, sec_range):
    sec_root, sec_history, sec_status = secant_method(root_problems[sec_problem.value]["f"], *sec_range.value)
    return sec_history, sec_root, sec_status


@app.cell(hide_code=True)
def _(PlaySlider, mo, sec_history):
    sec_step = mo.ui.anywidget(PlaySlider(value=1, min_value=1, max_value=max(len(sec_history), 1), step=1, interval_ms=400))
    return (sec_step,)


@app.cell(hide_code=True)
def _(
    draw_bracket_step,
    failure_callout,
    history_table,
    mo,
    root_problems,
    sec_history,
    sec_problem,
    sec_range,
    sec_root,
    sec_status,
    sec_step,
    sec_zoom,
):
    _p = root_problems[sec_problem.value]
    _controls = mo.hstack([sec_problem, sec_range, sec_zoom], justify="center", align="center", gap=2)

    if not sec_history:
        _body = failure_callout("The secant method", sec_status)
    else:
        _n = min(int(sec_step.value["value"]), len(sec_history))
        _body = mo.vstack(
            [
                mo.hstack([mo.md("**Iteration:**"), sec_step], justify="start", align="center"),
                failure_callout("The secant method", sec_status),
                mo.hstack(
                    [
                        draw_bracket_step(_p, sec_history, _n, "secant", zoom=sec_zoom.value),
                        mo.vstack(
                            [
                                mo.stat(label="Estimate $x_n$", value=f"{sec_history[_n - 1]['x']:.12f}"),
                                mo.stat(label="Final estimate", value=f"{sec_root:.12f}", caption=sec_status),
                                mo.stat(label="Iterations", value=f"{len(sec_history)}"),
                            ]
                        ),
                    ],
                    widths=[0.8, 0.2],
                    align="center",
                ),
                mo.accordion({"Iteration table": mo.ui.table(history_table(sec_history), page_size=10, selection=None)}),
            ]
        )

    mo.vstack([_controls, _body])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $f_1$ on $[0,3]$ the secant method converges in 7 iterations. For
    $f_2$ on the same interval, however, it needs **31 iterations**. Why so
    many? Animate it to see: the interval $(0,3)$ contains the point
    $x=1/\sqrt3\approx0.577$ where $f_2'(x)=0$. A secant through two points
    near that local minimum is almost horizontal, crosses zero very far away,
    and the method wanders for a long time before it settles into its fast
    final phase. Narrowing the starting points to $[1,3]$, away from the
    extremum, brings the count down to 8.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. False position method (*regula falsi*)
    ---

    The **false position method** combines the secant step with the safety of
    bisection. It takes the point where the secant through the bracket edges
    $(a,f(a))$ and $(b,f(b))$ crosses zero,
    $$
    x = a - f(a)\,\frac{b-a}{f(b)-f(a)},
    $$
    and then keeps the sub-interval on which $f$ changes sign, exactly like
    bisection. The root therefore stays bracketed at every step, and the
    method cannot run away as the secant method did.

    It is usually much faster than bisection — but not always: if $f$ is
    convex (or concave) over the bracket, one edge of the bracket never moves,
    the bracket width does *not* shrink to zero, and convergence degrades to
    linear. We therefore stop when two successive estimates agree,
    $|x_{n}-x_{n-1}|<\varepsilon$.
    """)
    return


@app.function
def falseposition_method(f, a, b, tolerance=1e-10, max_iterations=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return None, [], "not applicable"

    history = []
    x_prev = x_new = 0.5 * (a + b)
    for _ in range(max_iterations):
        x_prev = x_new
        x_new = a - fa * (b - a) / (fb - fa)  # where the secant crosses y = 0
        f_new = f(x_new)
        history.append(dict(a=a, b=b, x=x_new, fx=f_new))
        if f_new * fa < 0.0:
            b, fb = x_new, f_new
        else:
            a, fa = x_new, f_new
        if abs(x_new - x_prev) < tolerance:
            return x_new, history, "converged"
    return x_new, history, "max. iterations"


@app.cell(hide_code=True)
def _(mo, root_problems):
    fp_problem = mo.ui.dropdown(options=list(root_problems), value="f2(x) = x^3 - x - 1", label="Equation:")
    fp_zoom = mo.ui.switch(label="Zoom on bracket")
    return fp_problem, fp_zoom


@app.cell(hide_code=True)
def _(bracket_slider, fp_problem, root_problems):
    fp_range = bracket_slider(root_problems[fp_problem.value], "Initial bracket $[a,b]$")
    return (fp_range,)


@app.cell
def _(fp_problem, fp_range, root_problems):
    fp_root, fp_history, fp_status = falseposition_method(root_problems[fp_problem.value]["f"], *fp_range.value)
    fp_bis_iterations = len(bisection_method(root_problems[fp_problem.value]["f"], *fp_range.value)[1])
    return fp_bis_iterations, fp_history, fp_root, fp_status


@app.cell(hide_code=True)
def _(PlaySlider, fp_history, mo):
    fp_step = mo.ui.anywidget(PlaySlider(value=1, min_value=1, max_value=max(len(fp_history), 1), step=1, interval_ms=400))
    return (fp_step,)


@app.cell(hide_code=True)
def _(
    draw_bracket_step,
    failure_callout,
    fp_bis_iterations,
    fp_history,
    fp_problem,
    fp_range,
    fp_root,
    fp_status,
    fp_step,
    fp_zoom,
    history_table,
    mo,
    root_problems,
):
    _p = root_problems[fp_problem.value]
    _controls = mo.hstack([fp_problem, fp_range, fp_zoom], justify="center", align="center", gap=2)

    if fp_status == "not applicable":
        _body = failure_callout("False position", fp_status)
    else:
        _n = min(int(fp_step.value["value"]), len(fp_history))
        _body = mo.vstack(
            [
                mo.hstack([mo.md("**Iteration:**"), fp_step], justify="start", align="center"),
                failure_callout("False position", fp_status),
                mo.hstack(
                    [
                        draw_bracket_step(_p, fp_history, _n, "false position", zoom=fp_zoom.value),
                        mo.vstack(
                            [
                                mo.stat(label="Estimate $x_n$", value=f"{fp_history[_n - 1]['x']:.12f}"),
                                mo.stat(label="Final root", value=f"{fp_root:.12f}", caption=fp_status),
                                mo.stat(label="Iterations", value=f"{len(fp_history)}", caption=f"bisection: {fp_bis_iterations}"),
                            ]
                        ),
                    ],
                    widths=[0.8, 0.2],
                    align="center",
                ),
                mo.accordion({"Iteration table": mo.ui.table(history_table(fp_history), page_size=10, selection=None)}),
            ]
        )

    mo.vstack([_controls, _body])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $f_2$ on $[0,3]$ the function is convex near the root, so the right
    edge $b=3$ never moves (look at the grey $|b-a|$ curve: it levels off
    instead of going to zero). Each new secant still pivots about the same far
    point $(3, f(3))$, and progress toward the root is only linear: false
    position needs **66 iterations**, almost twice as many as bisection — and
    twice as many as the unbracketed secant method. For $f_1$ the same method
    converges in 11 iterations, three times faster than bisection.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    newton_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Newton–Raphson from a Taylor expansion",
            steps=[
                {
                    "tex": r"f(x^*) = f(x) + f'(x)\,(x^*-x) + \frac{f''(x)}{2}(x^*-x)^2 + \cdots",
                    "note": "Taylor expand f around a point x close to the root x*.",
                },
                {
                    "tex": r"f(x^*) \approx f(x) + f'(x)\,(x^*-x)",
                    "note": "Keep only the linear term.",
                },
                {
                    "tex": r"0 \approx f(x) + f'(x)\,(x^*-x)",
                    "note": "By definition of the root, f(x*) = 0.",
                },
                {
                    "tex": r"x^* \approx x - \frac{f(x)}{f'(x)}",
                    "note": "Solve for x*: accurate if x is close to x*.",
                },
                {
                    "tex": r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}",
                    "note": "Turn it into an iteration.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 5. Newton–Raphson method
                ---

                Newton's method is a **local** method: it needs a single
                starting point $x_0$, assumed to be reasonably close to the
                root $x^*$, and the derivative $f'(x)$. It can be seen as the
                limit of the secant method when the two points merge: the
                secant becomes the **tangent**.
                """
            ),
            newton_anim,
            mo.md(
                r"""
                Geometrically, $x_{n+1}$ is where the tangent to $f$ at $x_n$
                crosses zero. Neglecting the quadratic term makes an error
                proportional to $(x^*-x_n)^2$, so the convergence is
                **quadratic**: close to the root, the number of correct digits
                doubles at every step. The price is one evaluation of $f'$ per
                iteration, and the method misbehaves near points where
                $f'(x)=0$, where the tangent is nearly horizontal.

                Newton's method is also a relaxation method (Section 1), with
                $\varphi(x)=x-f(x)/f'(x)$. One checks that $\varphi'(x^*)=0$:
                the error ratio $|\varphi'(x^*)|$ is zero, which is why the
                convergence is faster than any geometric progression.
                """
            ),
        ]
    )
    return


@app.function
def newton_method(f, df, x0, tolerance=1e-10, max_iterations=100):
    xs = [x0]
    for _ in range(max_iterations):
        dfval = df(xs[-1])
        if dfval == 0.0:
            return xs[-1], xs, "f'(x) = 0"  # horizontal tangent
        xs.append(xs[-1] - f(xs[-1]) / dfval)  # the next iteration
        if abs(xs[-1] - xs[-2]) < tolerance:
            return xs[-1], xs, "converged"
    return xs[-1], xs, "max. iterations"


@app.cell(hide_code=True)
def _(mo, root_problems):
    newton_problem = mo.ui.dropdown(options=list(root_problems), value="f2(x) = x^3 - x - 1", label="Equation:")
    newton_zoom = mo.ui.switch(label="Zoom on current step")
    return newton_problem, newton_zoom


@app.cell(hide_code=True)
def _(mo, newton_problem, root_problems):
    _p = root_problems[newton_problem.value]
    newton_x0 = mo.ui.slider(*_p["x0_range"], step=0.01, value=_p["x0"], label="Starting point $x_0$")
    return (newton_x0,)


@app.cell
def _(newton_problem, newton_x0, root_problems):
    _p = root_problems[newton_problem.value]
    newton_root, newton_xs, newton_status = newton_method(_p["f"], _p["df"], round(newton_x0.value, 4))
    return newton_root, newton_status, newton_xs


@app.cell(hide_code=True)
def _(PlaySlider, mo, newton_xs):
    # A cycle never converges: animate its first 40 steps only
    newton_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(min(len(newton_xs) - 1, 40), 1), step=1, interval_ms=600))
    return (newton_step,)


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, np, view_window):
    def draw_newton_step(problem, xs, n, zoom=False):
        """Tangent steps of Newton's method up to x_n, plus the error history."""
        f, df = problem["f"], problem["df"]
        if zoom and n > 0:
            # Frame the current step: from (x_{n-1}, f(x_{n-1})) down the tangent to (x_n, 0)
            (xlo, xhi), ylim = view_window(problem, xs[: n + 1], zoom_center=0.5 * (xs[n - 1] + xs[n]), zoom_half=abs(xs[n] - xs[n - 1]))
        else:
            (xlo, xhi), ylim = view_window(problem, xs[: n + 1])
        xgrid = np.linspace(xlo, xhi, 600)

        fig, ax, ax2 = method_figure()

        # --- Left: tangent steps
        ax.plot(xgrid, f(xgrid), color="crimson", lw=2, label="$f(x)$")
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        for xa, xb in zip(xs[: n - 1], xs[1:n]):  # completed steps, faintly
            ax.plot([xa, xb], [f(xa), 0.0], color="0.7", lw=1.0)
            ax.plot([xb, xb], [0.0, f(xb)], color="0.7", lw=0.8, ls=":")
            ax.plot([xb], [0.0], "o", color="0.65", ms=5, zorder=3)
        ax.plot([xs[n - 1] if n > 0 else xs[0]], [f(xs[n - 1] if n > 0 else xs[0])], "o", color="black", ms=7, zorder=4, label="Current point")
        if n > 0:
            xa, xb = xs[n - 1], xs[n]
            ax.plot(xgrid, f(xa) + df(xa) * (xgrid - xa), color="steelblue", lw=1.6, label=f"Tangent at $x_{{{n - 1}}}$")
            ax.plot([xb], [0.0], "o", color="#1d4ed8", ms=8, zorder=5, label=f"New estimate $x_{{{n}}}$")
            ax.vlines(xb, 0.0, f(xb), colors="#1d4ed8", linestyles=":", lw=1.2)
        # The iterates may land outside the window: list them
        first = max(0, n - 6)
        listing = "\n".join(f"$x_{{{i}}}$ = {x:.6f}" for i, x in enumerate(xs[first : n + 1], start=first))
        ax.text(0.98, 0.03, listing, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))
        ax.set(xlim=(xlo, xhi), ylim=ylim, xlabel="$x$", ylabel="$f(x)$", title=f"Newton–Raphson: iteration {n}")
        ax.legend(loc="upper left", fontsize=9)

        # --- Right: error history
        err = np.abs(np.array(xs, dtype=float) - problem["root"])
        ax2.semilogy(np.arange(len(err)), np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label="$|x_n-x^*|$")
        ax2.set(ylim=(1e-17, 1e3), xlim=(-0.5, max(min(len(xs) - 1, 40), 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_newton_step,)


@app.cell(hide_code=True)
def _(
    draw_newton_step,
    failure_callout,
    mo,
    newton_problem,
    newton_root,
    newton_status,
    newton_step,
    newton_x0,
    newton_xs,
    newton_zoom,
    np,
    pd,
    root_problems,
):
    _p = root_problems[newton_problem.value]
    _n = min(int(newton_step.value["value"]), len(newton_xs) - 1)
    _x0 = newton_xs[0]

    _err = np.abs(np.array(newton_xs, dtype=float) - _p["root"])
    _table = pd.DataFrame(dict(n=np.arange(len(newton_xs)), x_n=newton_xs, f_x_n=_p["f"](np.array(newton_xs, dtype=float)), error=_err))

    mo.vstack(
        [
            mo.hstack([newton_problem, newton_x0, newton_zoom], justify="center", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), newton_step], justify="start", align="center"),
            failure_callout("Newton–Raphson", newton_status),
            mo.hstack(
                [
                    draw_newton_step(_p, newton_xs, _n, zoom=newton_zoom.value),
                    mo.vstack(
                        [
                            mo.stat(label="Initial guess $x_0$", value=f"{_x0:.4f}", caption=f"f'(x₀) = {_p['df'](_x0):.3f}"),
                            mo.stat(label="Final estimate", value=f"{newton_root:.12f}", caption=newton_status),
                            mo.stat(label="Iterations", value=f"{len(newton_xs) - 1}"),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.accordion({"Iteration table (watch the error column)": mo.ui.table(_table, page_size=10, selection=None)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - **$f_1$, $x_0=0.5$:** converges in 6 iterations; open the table and
      watch the number of zeros in the error column double — quadratic
      convergence. At $x_0=0$ the tangent is horizontal and the method stops
      immediately.
    - **$f_2$, $x_0=0.5$:** $x_0$ sits next to the minimum at $0.577$, the
      tangent is almost flat, and $x_1=-5$ is flung far away. The method
      still finds its way back, but it takes 21 iterations. Move $x_0$ past
      the minimum, e.g. to $x_0=1.5$, and it needs only 5.
    - **$f_3$, $x_0=0$:** for an unfortunate choice of initial guess, Newton's
      method enters a **cycle**: $x_1=1$, $x_2=0$, $x_3=1$, ... and never
      converges. Move $x_0$ around $[-0.5, 2]$: starting points only $0.1$
      apart alternately converge (in 10–40 steps) or get trapped in the
      cycle. Starting left of $-1$, on the other hand, converges in 4–8 steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Comparing the rates of convergence
    ---

    A method has **order of convergence** $p$ if, close to the root, the
    error $e_n=|x_n-x^*|$ satisfies
    $$
    e_{n+1} \approx C\,e_n^{\,p}.
    $$
    $p=1$ (with $C<1$) is *linear* convergence: a fixed number of digits is
    gained per step. $p>1$ is *superlinear*: the number of correct digits is
    multiplied by $p$ per step. On a semi-log plot, linear convergence is a
    straight line while superlinear convergence curves downward ever more
    steeply.
    """)
    return


@app.cell(hide_code=True)
def _(mo, root_problems):
    cmp_problem = mo.ui.dropdown(options=list(root_problems)[:2], value="f1(x) = x + exp(-x) - 2", label="Equation:")
    return (cmp_problem,)


@app.cell
def _(cmp_problem, np, relax_choices, relaxation_method, root_problems):
    _p = root_problems[cmp_problem.value]
    _a, _b = 1.0, 3.0  # a bracket away from the extremum of f2
    _phi = relax_choices["φ1(x) = 2 - exp(-x)   (from f1)" if cmp_problem.value.startswith("f1") else "φ3(x) = (1 + x)^(1/3)   (from f2)"]["phi"]

    _runs = {
        "Relaxation": relaxation_method(_phi, _b)[1][1:],
        "Bisection": [s["x"] for s in bisection_method(_p["f"], _a, _b)[1]],
        "Secant": [s["x"] for s in secant_method(_p["f"], _a, _b)[1]],
        "False position": [s["x"] for s in falseposition_method(_p["f"], _a, _b)[1]],
        "Newton": newton_method(_p["f"], _p["df"], _b)[1][1:],
    }
    cmp_errors = {name: np.abs(np.array(xs, dtype=float) - _p["root"]) for name, xs in _runs.items()}
    cmp_evals_per_step = {"Relaxation": 1, "Bisection": 1, "Secant": 1, "False position": 1, "Newton": 2}
    return cmp_errors, cmp_evals_per_step


@app.cell(hide_code=True)
def _(cmp_errors, cmp_evals_per_step, cmp_problem, mo, np, pd, plt):
    _colors = {"Relaxation": "purple", "Bisection": "0.4", "Secant": "seagreen", "False position": "darkorange", "Newton": "#1d4ed8"}
    _orders = {"Relaxation": "1 (ratio |φ'(x*)|)", "Bisection": "1 (ratio 1/2)", "Secant": "1.618", "False position": "1", "Newton": "2"}

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _name, _e in cmp_errors.items():
        _ax.semilogy(np.arange(1, len(_e) + 1), np.maximum(_e, 1e-17), "o-", ms=4, lw=1.2, color=_colors[_name], label=_name)
    _ax.axhline(1e-10, color="black", lw=0.8, ls=":")
    _ax.set(xlabel="Iteration $n$", ylabel="$|x_n-x^*|$", ylim=(1e-17, 10), xlim=(0, 40))
    _ax.legend()

    _rows = []
    for _name, _e in cmp_errors.items():
        _hit = np.nonzero(_e < 1e-10)[0]
        _n = int(_hit[0]) + 1 if len(_hit) else None
        _rows.append(
            {
                "Method": _name,
                "Order p": _orders[_name],
                "Iterations to 1e-10": _n,
                "f evaluations to 1e-10": _n * cmp_evals_per_step[_name] if _n else None,
            }
        )

    mo.vstack(
        [
            mo.hstack([cmp_problem, mo.md("Bracket $[1,3]$; Newton and relaxation start from $x_0=3$.")], justify="start", align="center", gap=2),
            _fig, 
            mo.ui.table(pd.DataFrame(_rows), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: single non-linear equations

    | Method | Order | Evaluations / step | Strengths | Weaknesses |
    |---|---|---|---|---|
    | Relaxation | 1 (ratio $\lvert\varphi'(x^*)\rvert$) | 1 × $\varphi$ | trivial to implement | converges only if $\lvert\varphi'\rvert<1$ |
    | Bisection | 1 (ratio ½) | 1 × $f$ | cannot fail, predictable cost | slow; needs a bracket |
    | Secant | 1.618 | 1 × $f$ | fast, no derivative needed | may diverge near $f'=0$ |
    | False position | 1 (often faster) | 1 × $f$ | cannot fail, uses $f$ values | can stall with one fixed end |
    | Newton–Raphson | 2 | 1 × $f$ + 1 × $f'$ | fastest near the root | needs $f'$; cycles or diverges from a bad $x_0$ |

    In practice, robust library routines such as `scipy.optimize.brentq`
    combine the two families: they keep a bracket for safety, and take
    secant or inverse-interpolation steps for speed whenever those steps stay
    inside the bracket.
    """)
    return


if __name__ == "__main__":
    app.run()
