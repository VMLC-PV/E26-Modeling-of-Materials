import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from wigglystuff import FormulaAnimation, PlaySlider

    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return FormulaAnimation, PlaySlider, colors, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Search for extrema

    Finding the minimum of a function is at least as common as finding a
    root: the ground-state structure of a crystal minimizes the energy, a fit
    minimizes the squared residuals, a trained neural network minimizes a
    loss. A maximum of $f$ is a minimum of $-f$, so we only ever need to
    *minimize*.

    At a smooth minimum $x^*$ the derivative vanishes, $f'(x^*)=0$, so a
    minimization problem is also a root-finding problem for $f'$, and the
    methods of notebook 7a come back in a new guise:

    | Section | Method | Root-finding analogue | Needs |
    |---|---|---|---|
    | 1 | Golden section search | bisection | a bracket of the minimum, $f$ only |
    | 2 | Newton's method | Newton–Raphson on $f'$ | $f'$ and $f''$ |
    | 3 | Gradient descent | relaxation, $\varphi(x)=x-\gamma f'(x)$ | $f'$ |
    | 4 | Gradient descent in 2D | — | $\nabla f$ |

    A new difficulty appears: a function can have many **local** minima,
    and all of these methods find *a* minimum, not necessarily the
    **global** one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test functions
    ---

    - $g_1(x) = \sin x$, minima at
      $x = 3\pi/2 + 2\pi k$, maxima at $x = \pi/2 + 2\pi k$, all with the same
      value $\mp1$.
    - $g_2(x) = x^4 - 3x^2 + x$: a double well with a **global** minimum at
      $x\approx-1.3008$ ($g_2=-3.514$), a **local** minimum at $x\approx1.1309$
      ($g_2=-1.070$) and a maximum in between at $x\approx0.1699$.
    """)
    return


@app.cell
def _(np):
    _g2_stationary = np.sort(np.roots([4.0, 0.0, -6.0, 1.0]).real)  # g2'(x) = 4x^3 - 6x + 1 = 0

    extrema_functions = {
        "g1(x) = sin(x)": dict(
            tex=r"g_1(x)=\sin x",
            f=np.sin,
            df=np.cos,
            d2f=lambda x: -np.sin(x),
            view=(-0.5, 7.0),
            ylim=(-1.4, 1.4),
            bracket=(0.0, 2.0 * np.pi),
            x0=5.0,
            gamma=0.1,
            minima=np.array([1.5 * np.pi + 2.0 * np.pi * k for k in range(-3, 4)]),
            maxima=np.array([0.5 * np.pi + 2.0 * np.pi * k for k in range(-3, 4)]),
        ),
        "g2(x) = x^4 - 3x^2 + x": dict(
            tex=r"g_2(x)=x^4-3x^2+x",
            f=lambda x: x**4 - 3.0 * x**2 + x,
            df=lambda x: 4.0 * x**3 - 6.0 * x + 1.0,
            d2f=lambda x: 12.0 * x**2 - 6.0,
            view=(-2.2, 2.2),
            ylim=(-4.5, 8.0),
            bracket=(-2.0, 2.0),
            x0=1.5,
            gamma=0.05,
            minima=_g2_stationary[[0, 2]],
            maxima=_g2_stationary[[1]],
        ),
    }
    return (extrema_functions,)


@app.cell(hide_code=True)
def _(extrema_functions, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for _ax, _p in zip(_axes, extrema_functions.values()):
        _x = np.linspace(*_p["view"], 500)
        _ax.plot(_x, _p["f"](_x), color="crimson", lw=2)
        _mins = _p["minima"][(_p["minima"] > _p["view"][0]) & (_p["minima"] < _p["view"][1])]
        _maxs = _p["maxima"][(_p["maxima"] > _p["view"][0]) & (_p["maxima"] < _p["view"][1])]
        _ax.plot(_mins, _p["f"](_mins), "o", color="black", ms=7, label="Minima")
        _ax.plot(_maxs, _p["f"](_maxs), "^", color="0.5", ms=7, label="Maxima")
        _ax.set(xlabel="$x$", ylim=_p["ylim"], title=f"${_p['tex']}$")
        _ax.legend(loc="upper center", fontsize=14)
    _fig.tight_layout()
    _fig
    return


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
        ax2.set_title("Error history", fontsize=14)
        ax2.set_xlabel("Iteration $n$", fontsize=14)
        ax2.legend(fontsize=12)

    return finish_error_panel, method_figure


@app.cell(hide_code=True)
def _(mo, np):
    def failure_callout(method, status):
        """A red callout when an iteration did not find a minimum, otherwise nothing."""
        reasons = {
            "diverged": "the iterates ran away to infinity.",
            "max. iterations": "no convergence within the maximum number of iterations.",
            "f''(x) = 0": "the second derivative vanishes, $f''(x_n)=0$, and the quadratic model has no vertex.",
            "maximum": "the iteration converged to a stationary point with $f''(x^*)<0$ — a **maximum**, not a minimum.",
        }
        if status == "converged":
            return mo.md("")
        return mo.md(f"**{method} fails:** {reasons[status]}").callout(kind="danger")

    def local_minimum_callout(problem, x_found):
        """A yellow callout when the minimum found is not the lowest one in the plotted window."""
        xg = np.linspace(*problem["view"], 4001)
        f_low = problem["f"](xg).min()
        if problem["f"](x_found) > f_low + 1e-6:
            return mo.md(
                f"**Local minimum only:** the method converged to $x^*\\approx{x_found:.4f}$, "
                f"$f(x^*)={problem['f'](x_found):.4f}$, but lower values ($f\\approx{f_low:.4f}$) exist elsewhere in the window."
            ).callout(kind="warn")
        return mo.md("")

    return failure_callout, local_minimum_callout


@app.cell(hide_code=True)
def _(np):
    def view_1d(problem, points, zoom_points=None):
        """x- and y-limits: the problem window widened to the visited points, or a zoom on some points."""
        f = problem["f"]
        v0, v1 = problem["view"]
        if zoom_points is not None and np.all(np.isfinite(zoom_points)):
            lo, hi = min(zoom_points), max(zoom_points)
            half = max(0.75 * (hi - lo), 1e-10)
            xlo, xhi = 0.5 * (lo + hi) - half, 0.5 * (lo + hi) + half
            with np.errstate(all="ignore"):
                fs = f(np.linspace(xlo, xhi, 600))
            if np.all(np.isfinite(fs)):  # otherwise (far-away divergent iterates) fall back to the full view
                ypad = 0.15 * (fs.max() - fs.min() + 1e-300)
                return (xlo, xhi), (fs.min() - ypad, fs.max() + ypad)
        width = v1 - v0
        pts = [p for p in points if np.isfinite(p)] + [v0, v1]
        xlo = max(min(pts), v0 - 2.0 * width)
        xhi = min(max(pts), v1 + 2.0 * width)
        pad = 0.03 * (xhi - xlo)
        xlo, xhi = xlo - pad, xhi + pad
        fs = f(np.linspace(xlo, xhi, 600))
        y0, y1 = problem["ylim"]
        return (xlo, xhi), (max(min(y0, fs.min()), 3.0 * y0), min(max(y1, fs.max()), 3.0 * y1))

    def nearest(targets, xs):
        """Distance of every iterate to the nearest point of `targets`."""
        xs = np.asarray(xs, dtype=float)
        return np.min(np.abs(xs[:, None] - np.asarray(targets)[None, :]), axis=1)

    return nearest, view_1d


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    gss_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Why the golden ratio?",
            steps=[
                {
                    "tex": r"a < c < d < b, \qquad c = b - \frac{b-a}{\phi}, \qquad d = a + \frac{b-a}{\phi}",
                    "note": "Two interior points, placed symmetrically in the bracket [a, b].",
                },
                {
                    "tex": r"f(c) < f(d) \;\Rightarrow\; [a,b] \to [a,d], \qquad \text{else}\;\; [a,b] \to [c,b]",
                    "note": "Keep the sub-interval that must contain the minimum.",
                },
                {
                    "tex": r"d - a = \frac{b-a}{\phi}, \qquad c - a = (b-a)\left(1-\frac{1}{\phi}\right)",
                    "note": "The new bracket [a, d] is shorter by a factor φ; the old point c lies inside it.",
                },
                {
                    "tex": r"c - a = \frac{d-a}{\phi} \;\Longleftrightarrow\; 1-\frac{1}{\phi} = \frac{1}{\phi^2} \;\Longleftrightarrow\; \phi^2 = \phi + 1",
                    "note": "Demand that c is exactly the new 'd': then f(c) can be reused.",
                },
                {
                    "tex": r"\phi = \frac{1+\sqrt5}{2} \approx 1.618, \qquad |b_n-a_n| = \frac{|b_0-a_0|}{\phi^{\,n}}",
                    "note": "One new function evaluation per step, and the bracket shrinks by 1/φ ≈ 0.618.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 1. Golden section search
                ---

                The analogue of bisection. A sign change brackets a root; a
                minimum is bracketed by an interval $[a,b]$ on which $f$ is
                **unimodal** (decreasing, then increasing). A single interior
                point cannot tell on which side of it the minimum lies, but two
                can: if $f(c)<f(d)$ the minimum cannot be in $(d,b]$, otherwise
                it cannot be in $[a,c)$.
                """
            ),
            gss_anim,
            mo.md(
                r"""
                The golden ratio is exactly the choice that lets one of the two
                interior points be reused at the next step, so every iteration
                costs a single evaluation of $f$. The convergence is linear, with
                ratio $1/\phi\approx0.618$ — a bit slower per step than bisection
                ($1/2$), which needs a sign and not a comparison.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    phi = (np.sqrt(5.0) + 1.0) / 2.0

    def golden_section_search(f, a, b, tolerance=1e-10, max_iterations=200):
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        fc, fd = f(c), f(d)
        history = []
        while abs(b - a) > tolerance and len(history) < max_iterations:
            history.append(dict(a=a, b=b, c=c, d=d, fc=fc, fd=fd))
            if fc < fd:  # the minimum is in [a, d]
                b, d, fd = d, c, fc  # the old c becomes the new d: f(c) is reused
                c = b - (b - a) / phi
                fc = f(c)
            else:  # the minimum is in [c, b]
                a, c, fc = c, d, fd  # the old d becomes the new c: f(d) is reused
                d = a + (b - a) / phi
                fd = f(d)
        return 0.5 * (a + b), history, "converged"

    # Recursive implementation: same bracket, but two evaluations per step
    def gss_recursive(f, a, b, tolerance=1e-10):
        if abs(b - a) <= tolerance:
            return 0.5 * (a + b)
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        if f(c) < f(d):
            return gss_recursive(f, a, d, tolerance)
        return gss_recursive(f, c, b, tolerance)

    return golden_section_search, gss_recursive


@app.cell(hide_code=True)
def _(extrema_functions, mo):
    gss_problem = mo.ui.dropdown(options=list(extrema_functions), value="g1(x) = sin(x)", label="Function:")
    gss_zoom = mo.ui.switch(label="Zoom on bracket")
    return gss_problem, gss_zoom


@app.cell(hide_code=True)
def _(extrema_functions, gss_problem, mo):
    _p = extrema_functions[gss_problem.value]
    gss_range = mo.ui.range_slider(
        start=min(_p["view"][0], _p["bracket"][0]), stop=max(_p["view"][1], _p["bracket"][1]), step=0.05,
        value=list(_p["bracket"]), label="Initial bracket $[a,b]$",
    )
    return (gss_range,)


@app.cell
def _(
    extrema_functions,
    golden_section_search,
    gss_problem,
    gss_range,
    gss_recursive,
):
    _f = extrema_functions[gss_problem.value]["f"]
    gss_root, gss_history, gss_status = golden_section_search(_f, *gss_range.value)
    gss_root_recursive = gss_recursive(_f, *gss_range.value)
    return gss_history, gss_root


@app.cell(hide_code=True)
def _(PlaySlider, gss_history, mo):
    gss_step = mo.ui.anywidget(PlaySlider(value=1, min_value=1, max_value=max(len(gss_history), 1), step=1, interval_ms=400))
    return (gss_step,)


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, nearest, np, view_1d):
    def draw_gss_step(problem, history, n, zoom=False):
        """Iteration n (1-based) of golden section search, plus the error history."""
        f = problem["f"]
        s = history[n - 1]
        a, b, c, d = s["a"], s["b"], s["c"], s["d"]
        (xlo, xhi), ylim = view_1d(problem, [a, b], zoom_points=[a, b] if zoom else None)
        xg = np.linspace(xlo, xhi, 600)

        fig, ax, ax2 = method_figure()

        # --- Left: bracket, interior points and the part that is thrown away
        ax.plot(xg, f(xg), color="crimson", lw=2, label=f"${problem['tex']}$")
        ax.axvspan(a, b, color="#fdf0d5", zorder=0, label="Current bracket")
        drop = (d, b) if s["fc"] < s["fd"] else (a, c)
        ax.axvspan(*drop, color="#fdecea", zorder=0, label="Discarded")
        ax.plot([c, d], [s["fc"], s["fd"]], "o", color="#1d4ed8", ms=8, zorder=5, label="Interior points $c$, $d$")
        ax.vlines([c, d], ylim[0], [s["fc"], s["fd"]], colors="#1d4ed8", linestyles=":", lw=1.0)
        ax.plot([a, b], [f(a), f(b)], "o", color="black", ms=6, zorder=4, label="Bracket edges $a$, $b$")
        ax.set(xlim=(xlo, xhi), ylim=ylim, xlabel="$x$", ylabel="$f(x)$", title=f"Golden section search: iteration {n}")
        ax.legend(loc="upper left", fontsize=12)

        # --- Right: error and bracket width
        mids = [0.5 * (h["a"] + h["b"]) for h in history]
        err = nearest(problem["minima"], mids)
        width = np.array([abs(h["b"] - h["a"]) for h in history])
        its = np.arange(1, len(history) + 1)
        ax2.semilogy(its, np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label="$|x_n-x^*|$")
        ax2.semilogy(its, width, "-", color="0.4", lw=1.0, label=r"$|b-a|\propto\phi^{-n}$")
        ax2.axhline(np.sqrt(np.finfo(float).eps), color="0.6", lw=0.8, ls="--")
        ax2.text(1, np.sqrt(np.finfo(float).eps) * 3, r"$\sqrt{\epsilon_m}$", fontsize=12, color="0.4")
        ax2.set(ylim=(1e-17, 1e2), xlim=(0.5, len(history) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_gss_step,)


@app.cell(hide_code=True)
def _(
    draw_gss_step,
    extrema_functions,
    gss_history,
    gss_problem,
    gss_range,
    gss_root,
    gss_step,
    gss_zoom,
    local_minimum_callout,
    mo,
    nearest,
):
    _p = extrema_functions[gss_problem.value]
    _n = min(int(gss_step.value["value"]), len(gss_history))

    mo.vstack(
        [
            mo.hstack([gss_problem, gss_range], justify="start", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), gss_step, gss_zoom], justify="start", align="center"),
            local_minimum_callout(_p, gss_root),
            mo.hstack(
                [
                    draw_gss_step(_p, gss_history, _n, zoom=gss_zoom.value),
                    mo.vstack(
                        [
                            # mo.stat(label="Estimate $x_n$", value=f"{0.5 * (gss_history[_n - 1]['a'] + gss_history[_n - 1]['b']):.12f}"),
                            # mo.stat(label="Final estimate", value=f"{gss_root:.12f}", caption=f"recursive version: {gss_root_recursive:.12f}"),
                            mo.stat(label="Iterations", value=f"{len(gss_history)}"),
                            mo.stat(label="Final error", value=f"{nearest(_p['minima'], [gss_root])[0]:.1e}", caption="≈ √ε: see below"),
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
    Two things to notice:

    - **The error stalls at $\sim10^{-8}$** even though the bracket keeps
      shrinking to $10^{-10}$. Near a minimum,
      $f(x)\approx f(x^*)+\tfrac12f''(x^*)(x-x^*)^2$: two points closer to
      $x^*$ than about $\sqrt{\epsilon_m}\,|x^*|\approx10^{-8}$ give values of
      $f$ that are identical in floating point, and the comparison
      $f(c)<f(d)$ becomes a coin toss. A minimum can only be located to about
      the *square root* of machine precision from values of $f$ alone.
    - **$g_2$ with a bracket containing both wells**, e.g. $[-1,2]$: the
      function is not unimodal there, and the search ends in the *local*
      minimum at $1.13$. With $[-2,2]$ it happens to find the global one.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    newton_ext_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Deriving Newton's Method for Optimization",
            steps=[
                {
                    "tex": r"f(x) \approx f(x_n) + f'(x_n)(x - x_n) + \frac{1}{2}f''(x_n)(x - x_n)^2",
                    "note": "Write the second-order Taylor series expansion of f(x) around the current guess x_n.",
                },
                {
                    "tex": r"\frac{d}{dx} \left[ f(x) \right] \approx f'(x_n) + f''(x_n)(x - x_n)",
                    "note": "Differentiate this quadratic approximation with respect to x.",
                },
                {
                    "tex": r"f'(x_n) + f''(x_n)(x - x_n) = 0",
                    "note": "Set the derivative to zero to find the extremum (vertex) of the parabola.",
                },
                {
                    "tex": r"f''(x_n)(x - x_n) = -f'(x_n)",
                    "note": "Rearrange the terms to isolate the step vector (x - x_n).",
                },
                {
                    "tex": r"x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}",
                    "note": "Solve for x and set it as the next point, x_{n+1}, giving the standard update rule.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 2. Newton's method
                ---

                If the first two derivatives are available, the minimum can be
                found by solving $f'(x)=0$ with the Newton–Raphson method of
                notebook 7a. Geometrically, each step replaces $f$ by the
                **parabola** that matches $f$, $f'$ and $f''$ at $x_n$, and jumps
                to its vertex.
                """
            ),
            newton_ext_anim,
            mo.md(
                r"""
                The convergence is quadratic, and because the method works with
                $f'$ rather than with values of $f$, it is *not* limited to
                $\sqrt{\epsilon_m}$: it locates the minimum to full precision. But
                it finds any stationary point — a starting point where $f''<0$
                sends it to a maximum.
                """
            ),
        ]
    )
    return


@app.function
def newton_extremum(df, d2f, x0, tolerance=1e-10, max_iterations=100):
    xs = [x0]
    for _ in range(max_iterations):
        curvature = d2f(xs[-1])
        if curvature == 0.0:
            return xs[-1], xs, "f''(x) = 0"
        xs.append(xs[-1] - df(xs[-1]) / curvature)  # vertex of the local parabola
        if abs(xs[-1] - xs[-2]) < tolerance:
            # f' = 0 there, but is it a minimum?
            return xs[-1], xs, "converged" if d2f(xs[-1]) > 0.0 else "maximum"
    return xs[-1], xs, "max. iterations"


@app.cell(hide_code=True)
def _(extrema_functions, mo):
    ne_problem = mo.ui.dropdown(options=list(extrema_functions), value="g1(x) = sin(x)", label="Function:")
    ne_zoom = mo.ui.switch(label="Zoom on current step")
    return ne_problem, ne_zoom


@app.cell(hide_code=True)
def _(extrema_functions, mo, ne_problem):
    _p = extrema_functions[ne_problem.value]
    ne_x0 = mo.ui.slider(_p["view"][0] + 0.05, _p["view"][1] - 0.05, step=0.05, value=_p["x0"], label="Starting point $x_0$")
    return (ne_x0,)


@app.cell
def _(extrema_functions, ne_problem, ne_x0):
    _p = extrema_functions[ne_problem.value]
    ne_root, ne_xs, ne_status = newton_extremum(_p["df"], _p["d2f"], round(ne_x0.value, 4))
    return ne_root, ne_status, ne_xs


@app.cell(hide_code=True)
def _(PlaySlider, mo, ne_xs):
    ne_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(min(len(ne_xs) - 1, 40), 1), step=1, interval_ms=700))
    return (ne_step,)


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, nearest, np, view_1d):
    def draw_newton_extremum_step(problem, xs, n, zoom=False):
        """Newton step n: the quadratic model at x_{n-1} and its vertex x_n, plus the error history."""
        f, df, d2f = problem["f"], problem["df"], problem["d2f"]
        (xlo, xhi), ylim = view_1d(problem, xs[: n + 1], zoom_points=xs[n - 1 : n + 1] if (zoom and n > 0) else None)
        xg = np.linspace(xlo, xhi, 600)

        fig, ax, ax2 = method_figure()

        # --- Left: the function and the local parabola
        ax.plot(xg, f(xg), color="crimson", lw=2, label=f"${problem['tex']}$")
        for xa in xs[: n - 1]:
            ax.plot([xa], [f(xa)], "o", color="0.65", ms=5, zorder=3)
        if n > 0:
            xp = xs[n - 1]
            model = f(xp) + df(xp) * (xg - xp) + 0.5 * d2f(xp) * (xg - xp) ** 2
            ax.plot(xg, model, color="steelblue", lw=1.6, label=f"Quadratic model at $x_{{{n - 1}}}$")
            ax.plot([xp], [f(xp)], "o", color="black", ms=7, zorder=4, label=f"$x_{{{n - 1}}}$")
            x_new = xs[n]
            y_vertex = f(xp) + df(xp) * (x_new - xp) + 0.5 * d2f(xp) * (x_new - xp) ** 2
            ax.plot([x_new], [y_vertex], "s", color="steelblue", ms=6, zorder=4, label="Vertex of the model")
            ax.vlines(x_new, y_vertex, f(x_new), colors="#1d4ed8", linestyles=":", lw=1.2)
        ax.plot([xs[n]], [f(xs[n])], "o", color="#1d4ed8", ms=8, zorder=5, label=f"$x_{{{n}}}$")
        ax.set(xlim=(xlo, xhi), ylim=ylim, xlabel="$x$", ylabel="$f(x)$", title=f"Newton's method: iteration {n}")
        ax.legend(loc="upper left", fontsize=12)

        # --- Right: distance to the nearest stationary point
        err = nearest(np.concatenate([problem["minima"], problem["maxima"]]), xs)
        ax2.semilogy(np.arange(len(err)), np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label="$|x_n-x^*|$, nearest $f'=0$")
        ax2.set(ylim=(1e-17, 1e2), xlim=(-0.5, max(min(len(xs) - 1, 40), 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_newton_extremum_step,)


@app.cell(hide_code=True)
def _(
    draw_newton_extremum_step,
    extrema_functions,
    failure_callout,
    local_minimum_callout,
    mo,
    ne_problem,
    ne_root,
    ne_status,
    ne_step,
    ne_x0,
    ne_xs,
    ne_zoom,
):
    _p = extrema_functions[ne_problem.value]
    _n = min(int(ne_step.value["value"]), len(ne_xs) - 1)

    mo.vstack(
        [
            mo.hstack([ne_problem, ne_x0], justify="start", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), ne_step, ne_zoom], justify="start", align="center"),
            failure_callout("Newton's method", ne_status),
            local_minimum_callout(_p, ne_root) if ne_status == "converged" else mo.md(""),
            mo.hstack(
                [
                    draw_newton_extremum_step(_p, ne_xs, _n, zoom=ne_zoom.value),
                    mo.vstack(
                        [
                            # mo.stat(label="Initial guess $x_0$", value=f"{ne_xs[0]:.4f}", caption=f"f''(x₀) = {_p['d2f'](ne_xs[0]):.3f}"),
                            # mo.stat(label="Final estimate", value=f"{ne_root:.12f}", caption=ne_status),
                            mo.stat(label="Iterations", value=f"{len(ne_xs) - 1}"),
                            mo.stat(label="f''(x*)", value=f"{_p['d2f'](ne_root):.3f}", caption="> 0: minimum, < 0: maximum"),
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
    Things to try:

    - **$g_1$, $x_0=5$** (the original example): 4 iterations to
      $3\pi/2=4.71238898038469$, correct to the last digit — compare with the
      $10^{-8}$ of golden section search.
    - **$g_1$, $x_0=2$:** $g_1''(2)=-\sin2<0$, the parabola opens downward,
      and Newton's method climbs to the **maximum** at $\pi/2$.
    - **$g_2$, $x_0$ between $-0.7$ and $0.7$:** here $g_2''<0$ and the
      first parabola opens downward. From $x_0=0.3$ the method converges to
      the maximum at $0.17$; from other points it is thrown into one of the
      wells — close to the inflection points $\pm1/\sqrt2$, where $g_2''=0$,
      $x_0=0.70$ and $x_0=0.71$ end in *different* minima.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    gd_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Gradient descent as a relaxation method",
            steps=[
                {
                    "tex": r"x_{n+1} = x_n - \gamma\, f'(x_n)",
                    "note": "Step downhill, proportionally to the slope; γ > 0 is the step size (learning rate).",
                },
                {
                    "tex": r"x_{n+1} = \varphi(x_n), \qquad \varphi(x) = x - \gamma\, f'(x)",
                    "note": "A fixed-point iteration: its fixed points are the zeros of f'.",
                },
                {
                    "tex": r"\varphi'(x^*) = 1 - \gamma\, f''(x^*)",
                    "note": "The error is multiplied by φ'(x*) at every step near the minimum.",
                },
                {
                    "tex": r"|1-\gamma f''(x^*)| < 1 \;\Longleftrightarrow\; 0 < \gamma < \frac{2}{f''(x^*)}",
                    "note": "Convergence condition. γ = 1/f''(x*) gives φ' = 0: the Newton step.",
                },
            ],
            height=280,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 3. Gradient descent
                ---

                The simplest idea needs only the first derivative: move against
                the slope,
                $$
                x_{n+1} = x_n - \gamma\, f'(x_n),
                $$
                with a fixed **step size** (or *learning rate*) $\gamma$. In
                machine learning, where $f$ depends on millions of parameters and
                second derivatives are out of reach, this is the workhorse.
                """
            ),
            gd_anim,
            mo.md(
                r"""
                So gradient descent converges linearly, with ratio
                $|1-\gamma f''(x^*)|$: too small a $\gamma$ creeps, $\gamma$
                between $1/f''$ and $2/f''$ overshoots and oscillates around the
                minimum, and $\gamma>2/f''$ diverges from it. Unlike Newton's
                method, it always goes *downhill*, so it never converges to a
                maximum.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def gradient_descent(df, x0, gamma=0.01, tolerance=1e-8, max_iterations=1000):
        xs = [np.float64(x0)]
        with np.errstate(all="ignore"):
            for _ in range(max_iterations):
                xs.append(xs[-1] - gamma * df(xs[-1]))  # step against the slope
                if not np.isfinite(xs[-1]):
                    return xs[-1], xs, "diverged"
                if abs(xs[-1] - xs[-2]) < tolerance:
                    return xs[-1], xs, "converged"
        return xs[-1], xs, "max. iterations"

    return (gradient_descent,)


@app.cell(hide_code=True)
def _(extrema_functions, mo):
    gd_problem = mo.ui.dropdown(options=list(extrema_functions), value="g1(x) = sin(x)", label="Function:")
    gd_zoom = mo.ui.switch(label="Zoom on current step")
    return gd_problem, gd_zoom


@app.cell(hide_code=True)
def _(extrema_functions, gd_problem, mo, np):
    _p = extrema_functions[gd_problem.value]
    gd_x0 = mo.ui.slider(_p["view"][0] + 0.05, _p["view"][1] - 0.05, step=0.05, value=_p["x0"], label="Starting point $x_0$")
    gd_log_gamma = mo.ui.slider(-2.5, 0.5, step=0.05, value=round(float(np.log10(_p["gamma"])), 2), label="Step size, $\\log_{10}\\gamma$")
    return gd_log_gamma, gd_x0


@app.cell
def _(extrema_functions, gd_log_gamma, gd_problem, gd_x0, gradient_descent):
    _p = extrema_functions[gd_problem.value]
    gd_gamma = 10.0 ** gd_log_gamma.value
    gd_root, gd_xs, gd_status = gradient_descent(_p["df"], round(gd_x0.value, 4), gd_gamma)
    return gd_gamma, gd_root, gd_status, gd_xs


@app.cell(hide_code=True)
def _(PlaySlider, gd_xs, mo):
    # Slow runs can take hundreds of steps: animate the first 150
    gd_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(min(len(gd_xs) - 1, 150), 1), step=1, interval_ms=300))
    return (gd_step,)


@app.cell(hide_code=True)
def _(finish_error_panel, method_figure, nearest, np, view_1d):
    def draw_gd_step(problem, xs, n, gamma, zoom=False):
        """Gradient descent up to step n, plus the error history with the predicted rate."""
        f, df = problem["f"], problem["df"]
        finite = [x for x in xs if np.isfinite(x)]
        n = min(n, len(finite) - 1)
        (xlo, xhi), ylim = view_1d(problem, finite[: n + 1], zoom_points=finite[n - 1 : n + 1] if (zoom and n > 0) else None)
        xg = np.linspace(xlo, xhi, 600)

        fig, ax, ax2 = method_figure()

        # --- Left: the path down the curve
        ax.plot(xg, f(xg), color="crimson", lw=2, label=f"${problem['tex']}$")
        path = np.array(finite[: n + 1])
        ax.plot(path, f(path), "-", color="steelblue", lw=1.0, alpha=0.7)
        ax.plot(path[:-1], f(path[:-1]), "o", color="0.6", ms=4, zorder=3)
        if n > 0:
            xp = finite[n - 1]
            ax.plot(xg, f(xp) + df(xp) * (xg - xp), color="0.3", lw=1.0, ls="--", label=f"Slope at $x_{{{n - 1}}}$")
            ax.annotate("", xy=(finite[n], f(xp)), xytext=(xp, f(xp)), arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.5))
            ax.plot([xp], [f(xp)], "o", color="black", ms=7, zorder=4, label=f"$x_{{{n - 1}}}$")
        ax.plot([finite[n]], [f(finite[n])], "o", color="#1d4ed8", ms=8, zorder=5, label=f"$x_{{{n}}}$")
        ax.set(xlim=(xlo, xhi), ylim=ylim, xlabel="$x$", ylabel="$f(x)$", title=f"Gradient descent ($\\gamma={gamma:.3g}$): iteration {n}")
        ax.legend(loc="upper left", fontsize=12)

        # --- Right: error, and the rate predicted at the minimum that is reached
        err = nearest(problem["minima"], finite)
        its = np.arange(len(err))
        ax2.semilogy(its, np.maximum(err, 1e-17), "o-", color="steelblue", ms=2, lw=1.0, label="$|x_n-x^*|$")
        x_star = problem["minima"][np.argmin(np.abs(problem["minima"] - finite[-1]))]
        rate = abs(1.0 - gamma * problem["d2f"](x_star))
        with np.errstate(all="ignore"):
            ax2.semilogy(its, err[-1] * rate ** (its - its[-1]), "--", color="0.5", lw=1.0, label=rf"$\propto|1-\gamma f''(x^*)|^n$ = {rate:.2f}$^n$")
        ax2.set(ylim=(1e-17, 1e2), xlim=(-0.5, max(len(finite) - 1, 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_gd_step,)


@app.cell(hide_code=True)
def _(
    draw_gd_step,
    extrema_functions,
    failure_callout,
    gd_gamma,
    gd_log_gamma,
    gd_problem,
    gd_root,
    gd_status,
    gd_step,
    gd_x0,
    gd_xs,
    gd_zoom,
    local_minimum_callout,
    mo,
    np,
):
    _p = extrema_functions[gd_problem.value]
    _n = min(int(gd_step.value["value"]), len(gd_xs) - 1)
    _x_star = _p["minima"][np.argmin(np.abs(_p["minima"] - gd_root))] if np.isfinite(gd_root) else np.nan

    mo.vstack(
        [
            mo.hstack([gd_problem, gd_x0, gd_log_gamma, gd_zoom], justify="center", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), gd_step], justify="start", align="center"),
            failure_callout("Gradient descent", gd_status),
            local_minimum_callout(_p, gd_root) if gd_status == "converged" else mo.md(""),
            mo.hstack(
                [
                    draw_gd_step(_p, gd_xs, _n, gd_gamma, zoom=gd_zoom.value),
                    mo.vstack(
                        [
                            mo.stat(label="Step size γ", value=f"{gd_gamma:.4g}", caption=f"stable if γ < 2/f''(x*) = {2.0 / _p['d2f'](_x_star):.3g}" if np.isfinite(_x_star) else ""),
                            mo.stat(label="Final estimate", value=f"{gd_root:.10g}", caption=gd_status),
                            mo.stat(label="Iterations", value=f"{len(gd_xs) - 1}"),
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
    For $g_1$, $f''(x^*)=1$ at every minimum, so the stability limit is
    $\gamma<2$:

    | $\gamma$ | Iterations from $x_0=5$ | Behaviour |
    |---|---|---|
    | 0.1 (original) | 143 | creeps, ratio 0.9 |
    | 0.5 | 25 | ratio 0.5 |
    | 1.0 | 4 | ratio 0: this *is* Newton's step at the minimum |
    | 1.9 | 169 | overshoots, oscillates with ratio $-0.9$ |
    | 2.5 | — | no convergence |

    For $g_2$ from $x_0=-2$: $\gamma=0.05$ reaches the global minimum, but
    $\gamma\approx0.13$ overshoots so far on the first step that the method
    lands in the *other* well and converges to the local minimum at $1.13$ —
    the step size also decides *which* minimum is found.

    Note also that the stopping test $|x_{n+1}-x_n|<\varepsilon$ is weak for
    a slow method: when the ratio is close to 1, the steps are tiny long
    before the error is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Gradient descent in two dimensions
    ---

    In $N$ dimensions the derivative becomes the gradient,
    $$
    \mathbf{x}_{n+1} = \mathbf{x}_n - \gamma\,\nabla f(\mathbf{x}_n),
    $$
    and the role of $f''$ is played by the eigenvalues $\lambda_i$ of the
    Hessian matrix at the minimum. Every direction must be stable,
    $\gamma<2/\lambda_{\max}$, while the slowest direction converges with ratio
    $1-\gamma\lambda_{\min}$. When $\lambda_{\max}\gg\lambda_{\min}$ — a long,
    narrow valley — gradient descent is forced to take tiny steps and
    zig-zags across the valley.

    Two test surfaces:

    - **An elongated bowl**, $q(x,y)=\tfrac12(x^2+10y^2)$, with Hessian
      eigenvalues $1$ and $10$: the step size must satisfy $\gamma<0.2$.
    - **The objective function of System A**,
      $\tilde f=|\mathbf{F}|^2/2$, with gradient $\nabla\tilde f=J^{T}\mathbf{F}$.
      Its global minimum $\tilde f=0$ is the root $(1.6500,-0.1580)$ — but it
      also has a **local** minimum at $(-0.4072,-1.7825)$ where
      $\tilde f=0.77\neq0$.
    """)
    return


@app.cell
def _(np):
    _FA = lambda x: np.array([x[0] + np.exp(-x[0]) - 2.0 - x[1], x[0] ** 3 - x[0] - 3.0 - x[1]])
    _JA = lambda x: np.array([[1.0 - np.exp(-x[0]), -1.0], [3.0 * x[0] ** 2 - 1.0, -1.0]])

    surfaces = {
        "Elongated bowl q(x, y)": dict(
            tex=r"q(x,y)=\frac{1}{2}(x^2+10y^2)",
            f=lambda x: 0.5 * (x[0] ** 2 + 10.0 * x[1] ** 2),
            grad=lambda x: np.array([x[0], 10.0 * x[1]]),
            extent=(-3.0, 3.0, -2.0, 2.0),
            x0=(-2.5, 1.5),
            gamma=0.15,
            minima=np.array([[0.0, 0.0]]),
            global_min=0.0,
            hessian_eigs=np.array([1.0, 10.0]),
            log_levels=False,
        ),
        "System A objective |F|^2/2": dict(
            tex=r"\tilde f=|\mathbf{F}|^2/2",
            f=lambda x: 0.5 * (_FA(x)[0] ** 2 + _FA(x)[1] ** 2),
            grad=lambda x: _JA(x).T @ _FA(x),
            extent=(-1.0, 3.0, -2.0, 2.0),
            x0=(2.5, 1.5),
            gamma=0.01,
            minima=np.array([[1.6499881922373312, -0.15795963144878492], [-0.40718989, -1.78246197]]),
            global_min=0.0,
            hessian_eigs=None,
            log_levels=True,
        ),
    }
    return (surfaces,)


@app.cell
def _(np):
    def gradient_descent_2d(grad, x0, gamma=0.01, tolerance=1e-8, max_iterations=5000):
        xs = [np.asarray(x0, dtype=float)]
        with np.errstate(all="ignore"):
            for _ in range(max_iterations):
                xs.append(xs[-1] - gamma * grad(xs[-1]))
                if not np.all(np.isfinite(xs[-1])):
                    return xs[-1], xs, "diverged"
                if np.linalg.norm(xs[-1] - xs[-2]) < tolerance:
                    return xs[-1], xs, "converged"
        return xs[-1], xs, "max. iterations"

    return (gradient_descent_2d,)


@app.cell(hide_code=True)
def _(mo, surfaces):
    gd2_surface = mo.ui.dropdown(options=list(surfaces), value="Elongated bowl q(x, y)", label="Surface:")
    gd2_zoom = mo.ui.switch(label="Zoom on current step")
    return gd2_surface, gd2_zoom


@app.cell(hide_code=True)
def _(gd2_surface, mo, np, surfaces):
    _s = surfaces[gd2_surface.value]
    gd2_x0 = mo.ui.slider(_s["extent"][0], _s["extent"][1], step=0.05, value=_s["x0"][0], label="$x_0$")
    gd2_y0 = mo.ui.slider(_s["extent"][2], _s["extent"][3], step=0.05, value=_s["x0"][1], label="$y_0$")
    gd2_log_gamma = mo.ui.slider(-3.0, -0.5, step=0.02, value=round(float(np.log10(_s["gamma"])), 2), label="Step size, $\\log_{10}\\gamma$")
    return gd2_log_gamma, gd2_x0, gd2_y0


@app.cell
def _(
    gd2_log_gamma,
    gd2_surface,
    gd2_x0,
    gd2_y0,
    gradient_descent_2d,
    surfaces,
):
    _s = surfaces[gd2_surface.value]
    gd2_gamma = 10.0 ** gd2_log_gamma.value
    gd2_root, gd2_xs, gd2_status = gradient_descent_2d(_s["grad"], (round(gd2_x0.value, 4), round(gd2_y0.value, 4)), gd2_gamma)
    return gd2_gamma, gd2_root, gd2_status, gd2_xs


@app.cell(hide_code=True)
def _(PlaySlider, gd2_xs, mo):
    # Runs can take thousands of steps: animate the first 300
    gd2_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(min(len(gd2_xs) - 1, 300), 1), step=1, interval_ms=120))
    return (gd2_step,)


@app.cell(hide_code=True)
def _(colors, finish_error_panel, method_figure, np):
    def draw_gd2_step(surface, xs, n, gamma, zoom=False):
        """2D gradient descent up to step n on a contour map, plus the error history."""
        finite = [x for x in xs if np.all(np.isfinite(x))]
        n = min(n, len(finite) - 1)
        x0, x1, y0, y1 = surface["extent"]
        if zoom and n > 0:
            a, b = finite[n - 1], finite[n]
            c, half = 0.5 * (a + b), max(np.abs(b - a).max(), 1e-12)
            x0, x1, y0, y1 = c[0] - half, c[0] + half, c[1] - half, c[1] + half

        fig, ax, ax2 = method_figure()

        # --- Left: contour map and path
        X, Y = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
        with np.errstate(all="ignore"):
            Z = surface["f"](np.array([X, Y]))
        if surface["log_levels"]:
            ax.contourf(X, Y, Z, levels=np.logspace(-3, 2, 21), norm=colors.LogNorm(), cmap="Greys_r", alpha=0.6, extend="both")
        else:
            ax.contourf(X, Y, Z, levels=20, cmap="Greys_r", alpha=0.6)
        ax.contour(X, Y, Z, levels=10, colors="0.5", linewidths=0.5)
        path = np.array(finite)
        ax.plot(path[:, 0], path[:, 1], "-", color="0.75", lw=0.8)  # the whole path, faintly
        ax.plot(path[: n + 1, 0], path[: n + 1, 1], "-", color="steelblue", lw=1.4, label="Path up to $\\mathbf{x}_n$")
        ax.plot(surface["minima"][:, 0], surface["minima"][:, 1], "*", color="crimson", ms=13, zorder=5, label="Minima")
        if n > 0:
            ax.annotate("", xy=finite[n], xytext=finite[n - 1], arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.8))
        ax.plot([finite[n][0]], [finite[n][1]], "o", color="#1d4ed8", ms=7, zorder=6, label=f"$\\mathbf{{x}}_{{{n}}}$")
        ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="$x$", ylabel="$y$", title=f"${surface['tex']}$, $\\gamma={gamma:.3g}$: iteration {n}")
        ax.legend(loc="upper left", fontsize=12)

        # --- Right: distance to the nearest minimum
        err = np.array([np.min(np.linalg.norm(surface["minima"] - x, axis=1)) for x in finite])
        its = np.arange(len(err))
        ax2.semilogy(its, np.maximum(err, 1e-17), "-", color="steelblue", lw=1.2, label=r"$|\mathbf{x}_n-\mathbf{x}^*|$")
        if surface["hessian_eigs"] is not None:
            rate = np.max(np.abs(1.0 - gamma * surface["hessian_eigs"]))
            with np.errstate(all="ignore"):
                ax2.semilogy(its, err[0] * rate**its, "--", color="0.5", lw=1.0, label=rf"$\max_i|1-\gamma\lambda_i|^n$ = {rate:.3f}$^n$")
        ax2.set(ylim=(1e-17, 1e3), xlim=(-0.5, max(len(finite) - 1, 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_gd2_step,)


@app.cell(hide_code=True)
def _(
    draw_gd2_step,
    failure_callout,
    gd2_gamma,
    gd2_log_gamma,
    gd2_root,
    gd2_status,
    gd2_step,
    gd2_surface,
    gd2_x0,
    gd2_xs,
    gd2_y0,
    gd2_zoom,
    mo,
    np,
    surfaces,
):
    _s = surfaces[gd2_surface.value]
    _n = min(int(gd2_step.value["value"]), len(gd2_xs) - 1)
    _f_final = _s["f"](gd2_root) if np.all(np.isfinite(gd2_root)) else np.nan

    if gd2_status == "converged" and _f_final > _s["global_min"] + 1e-6:
        _local = mo.md(
            f"**Local minimum only:** gradient descent stopped at $({gd2_root[0]:.4f}, {gd2_root[1]:.4f})$ where "
            f"$\\tilde f={_f_final:.4f}>0$. The gradient vanishes, but $\\mathbf{{F}}\\neq\\mathbf{{0}}$: this is **not** a root of "
            "System A. Minimizing $|\\mathbf{F}|^2$ is not a safe way to solve $\\mathbf{F}=\\mathbf{0}$."
        ).callout(kind="danger")
    else:
        _local = mo.md("")

    _stable = f"stable if γ < 2/λ_max = {2.0 / _s['hessian_eigs'].max():.3g}" if _s["hessian_eigs"] is not None else ""

    mo.vstack(
        [
            mo.hstack([gd2_surface, gd2_x0, gd2_y0,], justify="start", align="center", gap=1.5),
             gd2_log_gamma,
            mo.hstack([mo.md("**Iteration:**"), gd2_step, gd2_zoom], justify="start", align="center"),
            failure_callout("Gradient descent", gd2_status),
            _local,
            mo.hstack(
                [
                    draw_gd2_step(_s, gd2_xs, _n, gd2_gamma, zoom=gd2_zoom.value),
                    mo.vstack(
                        [
                            mo.stat(label="Step size γ", value=f"{gd2_gamma:.4g}", caption=_stable),
                            # mo.stat(label="Final estimate", value=f"({gd2_root[0]:.5f}, {gd2_root[1]:.5f})", caption=gd2_status),
                            mo.stat(label="Iterations", value=f"{len(gd2_xs) - 1}"),
                            mo.stat(label="f at final estimate", value=f"{_f_final:.2e}"),
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
    Things to try:

    - **Bowl, $\gamma=0.15$:** the $y$-direction ($\lambda=10$) overshoots
      back and forth while the $x$-direction ($\lambda=1$) creeps: the classic
      zig-zag. Push $\gamma$ above $0.2$ and the $y$-direction blows up; lower
      it to $0.05$ and the zig-zag disappears, but the $x$-direction becomes
      even slower.
    - **System A objective from $(2.5, 1.5)$, $\gamma=0.01$:** about 1800
      iterations to reach the root that Newton's method found in
      a handful — the Hessian eigenvalues at the root are $0.76$ and $53$, a
      narrow valley.
    - **System A objective from $(0,0)$:** the path slides into the spurious
      local minimum at $(-0.41,-1.78)$ and stops there. The same happens for
      $\gamma=0.03$ even from $(2.5,1.5)$, and $\gamma=0.04$ diverges from
      there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: search for extrema

    | Method | Order | Needs | Accuracy | Weaknesses |
    |---|---|---|---|---|
    | Golden section search | 1 (ratio 0.618) | $f$, a bracket | $\sim\sqrt{\epsilon_m}$ | only 1D; needs a unimodal bracket |
    | Newton's method | 2 | $f'$, $f''$ | machine precision | may converge to a maximum |
    | Gradient descent | 1 (ratio $\lvert1-\gamma f''\rvert$) | $f'$ (or $\nabla f$) | set by the stopping test | step size; slow in narrow valleys |

    All three methods are *local*: they find a minimum, and nothing tells
    them whether it is the global one. In practice one restarts from many
    starting points, or turns to global strategies (simulated annealing,
    basin hopping, ...). Library routines: `scipy.optimize.minimize_scalar`
    (Brent's method: golden section combined with parabolic steps) in 1D, and
    `scipy.optimize.minimize` (quasi-Newton BFGS, conjugate gradients, ...)
    in $N$ dimensions — the minimization cousins of Broyden's method.
    """)
    return


if __name__ == "__main__":
    app.run()
