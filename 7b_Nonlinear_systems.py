import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time

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
    return FormulaAnimation, PlaySlider, colors, mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Systems of non-linear equations

    Sometimes we need to solve a *system* of non-linear equations,
    \[
    \begin{aligned}
    F_1(x_1,\ldots,x_N) &= 0,\\
    F_2(x_1,\ldots,x_N) &= 0,\\
    &\;\;\vdots\\
    F_N(x_1,\ldots,x_N) &= 0.
    \end{aligned}
    \]
    Denoting $\mathbf{F}=(F_1,\ldots,F_N)$ and $\mathbf{x}=(x_1,\ldots,x_N)$,
    this is written compactly as
    \[
    \mathbf{F}(\mathbf{x}) = \mathbf{0}.
    \]
    Equilibrium geometries (all forces zero), self-consistent field
    equations and implicit time steps of a simulation all take this form.

    Bisection has no counterpart in several dimensions: there is no notion of
    a bracket. The local methods of notebook 7a, on the other hand, generalize
    directly:

    | Section | Method | 1D analogue (notebook 7a) | Needs |
    |---|---|---|---|
    | 2 | Newton–Raphson | Newton–Raphson | $\mathbf{F}$ and its Jacobian $J$ |
    | 3 | Broyden | secant | $\mathbf{F}$ only |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test systems
    ---

    We use two systems of two equations in two unknowns $\mathbf{x}=(x,y)$,
    so that everything can be drawn in the plane. Each equation
    $F_i(x,y)=0$ defines a curve, and the roots are the points where the two
    curves cross.

    - **System A** — the intersection of $y=x+e^{-x}-2$ and $y=x^3-x-3$:
      $$
      F_1(x,y) = x + e^{-x} - 2 - y, \qquad F_2(x,y) = x^3 - x - 3 - y,
      $$
      with a single root at $(1.6500, -0.1580)$.
    - **System B** — the intersection of a circle and a hyperbola:
      $$
      F_1(x,y) = x^2 + y^2 - 4, \qquad F_2(x,y) = xy - 1,
      $$
      with four roots, $\pm(1.9319, 0.5176)$ and $\pm(0.5176, 1.9319)$.
    """)
    return


@app.cell
def _(np):
    nl_systems = {
        "System A: curve intersection": dict(
            F=lambda x: np.array([x[0] + np.exp(-x[0]) - 2.0 - x[1], x[0] ** 3 - x[0] - 3.0 - x[1]]),
            J=lambda x: np.array([[1.0 - np.exp(-x[0]), -1.0], [3.0 * x[0] ** 2 - 1.0, -1.0]]),
            extent=(-1.0, 3.0, -2.0, 2.0),
            x0=(0.0, 0.0),
            roots=np.array([[1.6499881922373312, -0.15795963144878492]]),
        ),
        "System B: circle and hyperbola": dict(
            F=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0, x[0] * x[1] - 1.0]),
            J=lambda x: np.array([[2.0 * x[0], 2.0 * x[1]], [x[1], x[0]]]),
            extent=(-3.0, 3.0, -3.0, 3.0),
            x0=(1.0, 0.25),
            roots=np.array(
                [[s * np.sqrt(2.0 + t * np.sqrt(3.0)), s / np.sqrt(2.0 + t * np.sqrt(3.0))] for s in (1.0, -1.0) for t in (1.0, -1.0)]
            ),
        ),
    }
    return (nl_systems,)


@app.cell(hide_code=True)
def _(colors, np):
    def draw_landscape(ax, system, extent, n_grid=250):
        """Objective function |F|^2/2 on a log scale, with the curves F1 = 0 and F2 = 0."""
        xg = np.linspace(extent[0], extent[1], n_grid)
        yg = np.linspace(extent[2], extent[3], n_grid)
        X, Y = np.meshgrid(xg, yg)
        with np.errstate(all="ignore"):
            F1, F2 = system["F"](np.array([X, Y]))
        ftil = 0.5 * (F1**2 + F2**2)
        im = ax.imshow(
            ftil, origin="lower", extent=extent, aspect="auto", cmap="Greys_r",
            norm=colors.LogNorm(1e-2, 1e2, clip=True), alpha=0.55,
        )
        ax.contour(X, Y, F1, levels=[0.0], colors="crimson", linewidths=2)
        ax.contour(X, Y, F2, levels=[0.0], colors="darkorange", linewidths=2)
        ax.plot([], [], color="crimson", lw=2, label="$F_1=0$")
        ax.plot([], [], color="darkorange", lw=2, label="$F_2=0$")
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="$x$", ylabel="$y$")
        return im

    return (draw_landscape,)


@app.cell(hide_code=True)
def _(draw_landscape, nl_systems, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for _ax, (_name, _s) in zip(_axes, nl_systems.items()):
        _im = draw_landscape(_ax, _s, _s["extent"])
        _ax.plot(_s["roots"][:, 0], _s["roots"][:, 1], "o", color="black", ms=7, zorder=5, label="Roots")
        _ax.set_title(_name)
        _ax.legend(loc="upper left", fontsize=9)
    _fig.colorbar(_im, ax=_axes, label=r"$\tilde f(\mathbf{x})=|\mathbf{F}|^2/2$", shrink=0.9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Roots as minima of an objective function
    ---

    The grey background above is the scalar **objective function**
    $$
    \tilde f(\mathbf{x}) = \frac{\mathbf{F}(\mathbf{x})\cdot\mathbf{F}(\mathbf{x})}{2} \;\geq\; 0,
    $$
    shown on a log scale (dark = small). It vanishes exactly at the roots,
    so finding a root is equivalent to finding a point where $\tilde f$
    reaches its *global* minimum, zero. This connects root-finding to the
    search for extrema with a caveat: $\tilde f$ may also
    have *local* minima where $\mathbf{F} \neq \mathbf{0}$, and a method that
    only makes $|\mathbf{F}|$ smaller can get stuck there.

    In what follows, the two coloured curves $F_1=0$ and $F_2=0$ are the
    objects to keep an eye on: a root-finder in two dimensions is looking
    for the point where they cross.
    """)
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
        ax2.set_title("Error history", fontsize=10)
        ax2.set_xlabel("Iteration $n$", fontsize=9)
        ax2.legend(fontsize=12)

    return finish_error_panel, method_figure


@app.cell(hide_code=True)
def _(mo):
    def failure_callout(method, status):
        """A red callout when an iteration did not converge, otherwise nothing."""
        reasons = {
            "diverged": "the iterates ran away to infinity.",
            "max. iterations": "no convergence within the maximum number of iterations.",
            "singular Jacobian": r"the (approximate) Jacobian is singular, $\det J=0$: the linear system for the step has no unique solution.",
            "stalled": r"the steps became tiny, but $\mathbf{F}(\mathbf{x})\neq\mathbf{0}$: the iteration stalled away from a root.",
        }
        if status == "converged":
            return mo.md("")
        return mo.md(f"**{method} fails:** {reasons[status]}").callout(kind="danger")

    return (failure_callout,)


@app.cell(hide_code=True)
def _(np):
    def plot_window(system, points, zoom_step=None):
        """Axis extent: the system's window widened to the visited points, or a zoom on one step."""
        x0, x1, y0, y1 = system["extent"]
        if zoom_step is not None and np.all(np.isfinite(zoom_step)):
            a, b = np.asarray(zoom_step[0]), np.asarray(zoom_step[1])
            center = 0.5 * (a + b)
            half = max(np.abs(b - a).max(), 1e-12)
            return (center[0] - half, center[0] + half, center[1] - half, center[1] + half)
        pts = np.array([p for p in points if np.all(np.isfinite(p))] + [(x0, y0), (x1, y1)])
        wx, wy = x1 - x0, y1 - y0
        lo = np.maximum(pts.min(axis=0), (x0 - wx, y0 - wy))
        hi = np.minimum(pts.max(axis=0), (x1 + wx, y1 + wy))
        pad = 0.05 * (hi - lo)
        return (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])

    def root_error(system, xs):
        """Distance of every iterate to the nearest root."""
        return np.array([np.min(np.linalg.norm(system["roots"] - x, axis=1)) for x in xs])

    return plot_window, root_error


@app.cell
def _(
    draw_landscape,
    finish_error_panel,
    method_figure,
    np,
    plot_window,
    root_error,
):
    def draw_system_step(system, xs, n, model_J=None, zoom=False, reference=None, title="Newton"):
        """Iteration n of a 2D root-finder: the linear model at x_{n-1}, whose zero lines cross at x_n."""
        extent = plot_window(system, xs[: n + 1], zoom_step=(xs[n - 1], xs[n]) if (zoom and n > 0) else None)
        fig, ax, ax2 = method_figure()

        # --- Left: landscape, path and the linear model of the current step
        draw_landscape(ax, system, extent)
        path = np.array(xs[: n + 1])
        ax.plot(path[:, 0], path[:, 1], "-", color="steelblue", lw=1.4, zorder=4)
        ax.plot(path[:-1, 0], path[:-1, 1], "o", color="0.55", ms=4, zorder=4)
        if n > 0 and model_J is not None and np.all(np.isfinite(xs[n - 1])) and np.all(np.isfinite(model_J)):
            # Zero lines of the linear model F(x_{n-1}) + J (x - x_{n-1}) = 0, one per component
            xp = np.asarray(xs[n - 1])
            fx = system["F"](xp)
            gx = np.linspace(extent[0], extent[1], 200)
            gy = np.linspace(extent[2], extent[3], 200)
            X, Y = np.meshgrid(gx, gy)
            for i, col in enumerate(("crimson", "darkorange")):
                L = fx[i] + model_J[i, 0] * (X - xp[0]) + model_J[i, 1] * (Y - xp[1])
                ax.contour(X, Y, L, levels=[0.0], colors=col, linewidths=1.3, linestyles="--")
            ax.plot([], [], color="0.3", lw=1.3, ls="--", label="Linear model $=0$")
            ax.plot([xp[0]], [xp[1]], "o", color="black", ms=7, zorder=5, label=f"$\\mathbf{{x}}_{{{n - 1}}}$")
        ax.plot([xs[n][0]], [xs[n][1]], "o", color="#1d4ed8", ms=8, zorder=6, label=f"$\\mathbf{{x}}_{{{n}}}$")
        ax.set(xlim=extent[:2], ylim=extent[2:], title=f"{title}: iteration {n}")
        ax.legend(loc="upper left", fontsize=12)

        # --- Right: distance to the nearest root
        err = root_error(system, xs)
        ax2.semilogy(np.arange(len(err)), np.maximum(err, 1e-17), "o-", color="steelblue", ms=3, lw=1.0, label=r"$|\mathbf{x}_n-\mathbf{x}^*|$")
        if reference is not None:
            ref = root_error(system, reference)
            ax2.semilogy(np.arange(len(ref)), np.maximum(ref, 1e-17), "-", color="0.7", lw=1.0, label="Newton, same $\\mathbf{x}_0$")
        ax2.set(ylim=(1e-17, 1e3), xlim=(-0.5, max(len(xs) - 1, 1) + 0.5))
        finish_error_panel(ax2, n)
        fig.tight_layout()
        return fig

    return (draw_system_step,)


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    newton_multi_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Newton–Raphson in N dimensions",
            steps=[
                {
                    "tex": r"f(x^*) \approx f(x) + f'(x)\,(x^*-x)",
                    "note": "Recall the one-dimensional Taylor expansion around x.",
                },
                {
                    "tex": r"\mathbf{F}(\mathbf{x}^*) \approx \mathbf{F}(\mathbf{x}) + J(\mathbf{x})\,(\mathbf{x}^*-\mathbf{x})",
                    "note": "In N dimensions the derivative becomes the N × N Jacobian matrix.",
                },
                {
                    "tex": r"J_{ij} = \frac{\partial F_i}{\partial x_j}",
                    "note": "The Jacobian is the matrix of all first partial derivatives of F."},
                {
                    "tex": r"J(\mathbf{x})\,(\mathbf{x}^*-\mathbf{x}) \approx -\mathbf{F}(\mathbf{x})",
                    "note": "Use F(x*) = 0: ",
                },
                {
                    "tex": r"J(\mathbf{x}) \mathbf{\Delta x} = \mathbf{F}(\mathbf{x} )",
                    "note": "System of linear equations (Ax = v) for the step Δx = x* − x.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 2. Newton–Raphson method
                ---
                """),
            mo.hstack([
            mo.md(r"""
                Newton's method generalizes directly to $N$ dimensions: the
                derivative $f'(x)$ becomes the Jacobian matrix.
                \[       
                \mathbf{J} = \begin{bmatrix}
                \dfrac{\partial f_1}{\partial x} & \dfrac{\partial f_1}{\partial y} & \dfrac{\partial f_1}{\partial z} \\[2ex]
                \dfrac{\partial f_2}{\partial x} & \dfrac{\partial f_2}{\partial y} & \dfrac{\partial f_2}{\partial z} \\[2ex]
                \dfrac{\partial f_3}{\partial x} & \dfrac{\partial f_3}{\partial y} & \dfrac{\partial f_3}{\partial z}
                \end{bmatrix}
                \]
                The multi-dimensional Newton’s method is an iterative procedure where:
                \[
                \mathbf{x_{n+1}} = \mathbf{x_{n}} - J^{-1}(\mathbf{x_{n}})\mathbf{f}(\mathbf{x_{n}}).
                \]
                """
            ),
            newton_multi_anim,],widths=[0.4,0.6],align="start"),
            mo.md(
                r"""
                Where we stop when the step $|\mathbf{x_{n+1}} - \mathbf{x_{n}}|$ is small enough, or the residual $\Vert{}\mathbf{f}(\mathbf{x_n})\Vert{}$ is sufficiently close to zero.  
            
                Geometrically, in two dimensions each component $F_i$ is
                replaced by its **tangent plane** at $\mathbf{x}_n$. Each tangent
                plane crosses zero along a straight line — the linear
                approximation of the curve $F_i=0$ — and $\mathbf{x}_{n+1}$ is
                the point where the two lines cross. This is the exact analogue
                of the tangent crossing zero in one dimension.  
                As in 1D, the convergence is quadratic near the root. The cost of
                each iteration is the evaluation of the $N^2$ derivatives of the
                Jacobian and one $\mathcal{O}(N^3)$ linear solve, and the method
                breaks down where the Jacobian is singular.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def newton_method_multi(F, jacobian, x0, tolerance=1e-8, max_iterations=100):
        xs = [np.asarray(x0, dtype=float)]
        for _ in range(max_iterations):
            try:
                delta = np.linalg.solve(jacobian(xs[-1]), -F(xs[-1]))  # J delta = -F
            except np.linalg.LinAlgError:
                return xs[-1], xs, "singular Jacobian"
            xs.append(xs[-1] + delta)
            if not np.all(np.isfinite(xs[-1])):
                return xs[-1], xs, "diverged"
            if np.linalg.norm(delta) < tolerance:
                # A tiny step is not enough: check that we really are at a root
                status = "converged" if np.linalg.norm(F(xs[-1])) < 1e-6 else "stalled"
                return xs[-1], xs, status
        return xs[-1], xs, "max. iterations"

    return (newton_method_multi,)


@app.cell(hide_code=True)
def _(mo, nl_systems):
    nm_system = mo.ui.dropdown(options=list(nl_systems), value="System A: curve intersection", label="System:")
    nm_zoom = mo.ui.switch(label="Zoom on current step")
    return nm_system, nm_zoom


@app.cell(hide_code=True)
def _(mo, nl_systems, nm_system):
    _s = nl_systems[nm_system.value]
    nm_x0 = mo.ui.slider(_s["extent"][0], _s["extent"][1], step=0.05, value=_s["x0"][0], label="$x_0$")
    nm_y0 = mo.ui.slider(_s["extent"][2], _s["extent"][3], step=0.05, value=_s["x0"][1], label="$y_0$")
    return nm_x0, nm_y0


@app.cell
def _(newton_method_multi, nl_systems, nm_system, nm_x0, nm_y0):
    _s = nl_systems[nm_system.value]
    nm_root, nm_xs, nm_status = newton_method_multi(_s["F"], _s["J"], (round(nm_x0.value, 4), round(nm_y0.value, 4)))
    return nm_root, nm_status, nm_xs


@app.cell(hide_code=True)
def _(PlaySlider, mo, nm_xs):
    nm_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(len(nm_xs) - 1, 1), step=1, interval_ms=800))
    return (nm_step,)


@app.cell(hide_code=True)
def _(
    draw_system_step,
    failure_callout,
    mo,
    nl_systems,
    nm_root,
    nm_status,
    nm_step,
    nm_system,
    nm_x0,
    nm_xs,
    nm_y0,
    nm_zoom,
    np,
):
    _s = nl_systems[nm_system.value]
    _n = min(int(nm_step.value["value"]), len(nm_xs) - 1)
    _J = _s["J"](nm_xs[_n - 1]) if _n > 0 else None

    mo.vstack(
        [
            mo.hstack([nm_system, nm_x0, nm_y0, nm_zoom], justify="center", align="center", gap=2),
            mo.hstack([mo.md("**Iteration:**"), nm_step], justify="start", align="center"),
            failure_callout("Newton–Raphson", nm_status),
            mo.hstack(
                [
                    draw_system_step(_s, nm_xs, _n, model_J=_J, zoom=nm_zoom.value),
                    mo.vstack(
                        [
                            # mo.stat(label="Current iterate", value=f"({nm_xs[_n][0]:.6f}, {nm_xs[_n][1]:.6f})"),
                            # mo.stat(label="Final estimate", value=f"({nm_root[0]:.6f}, {nm_root[1]:.6f})", caption=nm_status),
                            mo.stat(label="Iterations", value=f"{len(nm_xs) - 1}"),
                            mo.stat(label="|F| at final estimate", value=f"{np.linalg.norm(_s['F'](nm_root)):.1e}"),
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

    - **System A from $(0,0)$:** 12 iterations. The first step jumps to
      $(-2,-1)$ and step 5 is thrown out to $(3.5, 0.9)$ before the fast final
      phase — follow the two dashed lines of the linear model at each step,
      and switch on the zoom to see them cross.
    - **System B from $(0,0)$ or $(1,1)$:** the Jacobian
      $\begin{pmatrix}2x & 2y\\ y & x\end{pmatrix}$ has determinant
      $2(x^2-y^2)$, which vanishes on the diagonals $y=\pm x$. There the two
      tangent lines are parallel, and Newton's method stops immediately.
    - **System B from other points:** four roots compete. The root that is
      reached is not always the nearest one — e.g. from $(-1.5, 1)$ Newton
      converges to $(-1.93,-0.52)$.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    broyden_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Broyden's update of the Jacobian",
            steps=[
                {
                    "tex": r"f'(x_{n+1}) \approx \frac{f(x_{n+1})-f(x_n)}{x_{n+1}-x_n}",
                    "note": "The secant method replaces f' by a finite difference of the last two points.",
                },
                {
                    "tex": r"J_{n+1}\,\Delta\mathbf{x} = \Delta\mathbf{F}, \qquad \Delta\mathbf{x} = \mathbf{x}_{n+1}-\mathbf{x}_n,\quad \Delta\mathbf{F} = \mathbf{F}(\mathbf{x}_{n+1})-\mathbf{F}(\mathbf{x}_n)",
                    "note": "Secant condition in N dimensions: N equations for N² unknowns — not enough to fix J.",
                },
                {
                    "tex": r"J_{n+1} = J_n + \frac{(\Delta\mathbf{F} - J_n\Delta\mathbf{x})\,\Delta\mathbf{x}^{T}}{\Delta\mathbf{x}^{T}\Delta\mathbf{x}}",
                    "note": "Broyden: the smallest change of J_n that satisfies the secant condition (a rank-one update).",
                },
                {
                    "tex": r"J_{n+1}^{-1} = J_n^{-1} + \frac{(\Delta\mathbf{x} - J_n^{-1}\Delta\mathbf{F})\,\Delta\mathbf{x}^{T}J_n^{-1}}{\Delta\mathbf{x}^{T}J_n^{-1}\Delta\mathbf{F}}",
                    "note": "Sherman–Morrison: the inverse of a rank-one update is itself a rank-one update, in O(N²) operations.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 3. Broyden's method
                ---

                The Jacobian may be unavailable, or too expensive to evaluate at
                every step. **Broyden's method** is the multi-dimensional
                generalization of the secant method: it starts from a guess
                $J_0$ (often simply the identity matrix) and *improves* it after
                every step using only the values of $\mathbf{F}$ already computed.
                """
            ),
            broyden_anim,
            mo.md(
                r"""
                Since $J_n$ is only an approximation, the convergence is
                superlinear rather than quadratic, and more iterations are
                needed than with Newton's method — but each one costs a single
                evaluation of $\mathbf{F}$ and no derivatives at all.

                Below are two implementations: a direct one that solves a linear
                system with $J_n$ at every step, and one that updates the
                *inverse* $J_n^{-1}$ with the Sherman–Morrison formula, so that
                each step is a matrix–vector product.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def broyden_method_direct(F, x0, J0, tolerance=1e-8, max_iterations=200):
        """Broyden's method, solving a linear system with the approximate Jacobian at each step."""
        xs = [np.asarray(x0, dtype=float)]
        Js = [np.array(J0, dtype=float)]
        f_val = F(xs[0])
        for _ in range(max_iterations):
            try:
                delta = np.linalg.solve(Js[-1], -f_val)
            except np.linalg.LinAlgError:
                return xs[-1], xs, Js, "singular Jacobian"
            xs.append(xs[-1] + delta)
            if not np.all(np.isfinite(xs[-1])):
                return xs[-1], xs, Js, "diverged"
            f_new = F(xs[-1])
            if np.linalg.norm(delta) < tolerance:
                status = "converged" if np.linalg.norm(f_new) < 1e-6 else "stalled"
                return xs[-1], xs, Js, status
            u = f_new - f_val
            Js.append(Js[-1] + np.outer(u - Js[-1] @ delta, delta) / (delta @ delta))  # rank-one update
            f_val = f_new
        return xs[-1], xs, Js, "max. iterations"

    def broyden_method(F, x0, Jinv0, tolerance=1e-8, max_iterations=200):
        """Broyden's method, updating the inverse Jacobian with the Sherman-Morrison formula."""
        x = np.asarray(x0, dtype=float)
        Jinv = np.array(Jinv0, dtype=float)
        f_val = F(x)
        for n in range(max_iterations):
            delta = -Jinv @ f_val  # no linear solve: a matrix-vector product
            x = x + delta
            f_new = F(x)
            if np.linalg.norm(delta) < tolerance:
                return x, n + 1
            df = f_new - f_val
            Jinv = Jinv + np.outer(delta - Jinv @ df, delta @ Jinv) / (delta @ Jinv @ df)
            f_val = f_new
        return x, max_iterations

    return broyden_method, broyden_method_direct


@app.cell(hide_code=True)
def _(mo, nl_systems):
    br_system = mo.ui.dropdown(options=list(nl_systems), value="System A: curve intersection", label="System:")
    br_J0 = mo.ui.dropdown(options=["Identity matrix", "Exact Jacobian at x0"], value="Identity matrix", label="Initial $J_0$:")
    br_zoom = mo.ui.switch(label="Zoom on current step")
    return br_J0, br_system, br_zoom


@app.cell(hide_code=True)
def _(br_system, mo, nl_systems):
    _s = nl_systems[br_system.value]
    br_x0 = mo.ui.slider(_s["extent"][0], _s["extent"][1], step=0.05, value=_s["x0"][0], label="$x_0$")
    br_y0 = mo.ui.slider(_s["extent"][2], _s["extent"][3], step=0.05, value=_s["x0"][1], label="$y_0$")
    return br_x0, br_y0


@app.cell
def _(
    br_J0,
    br_system,
    br_x0,
    br_y0,
    broyden_method,
    broyden_method_direct,
    newton_method_multi,
    nl_systems,
    np,
):
    _s = nl_systems[br_system.value]
    br_start = np.array([round(br_x0.value, 4), round(br_y0.value, 4)])
    _J0 = np.eye(2) if br_J0.value == "Identity matrix" else _s["J"](br_start)

    br_root, br_xs, br_Js, br_status = broyden_method_direct(_s["F"], br_start, _J0)
    br_newton_xs = newton_method_multi(_s["F"], _s["J"], br_start)[1]

    # The Sherman-Morrison version must follow the same path
    if abs(np.linalg.det(_J0)) > 1e-14:
        with np.errstate(all="ignore"):
            br_sm_root, br_sm_iterations = broyden_method(_s["F"], br_start, np.linalg.inv(_J0))
    else:
        br_sm_root, br_sm_iterations = np.array([np.nan, np.nan]), 0
    return (
        br_Js,
        br_newton_xs,
        br_root,
        br_sm_iterations,
        br_sm_root,
        br_status,
        br_xs,
    )


@app.cell(hide_code=True)
def _(PlaySlider, br_xs, mo):
    br_step = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(len(br_xs) - 1, 1), step=1, interval_ms=600))
    return (br_step,)


@app.cell(hide_code=True)
def _(
    br_J0,
    br_Js,
    br_newton_xs,
    br_root,
    br_sm_iterations,
    br_sm_root,
    br_status,
    br_step,
    br_system,
    br_x0,
    br_xs,
    br_y0,
    br_zoom,
    draw_system_step,
    failure_callout,
    mo,
    nl_systems,
    np,
):
    _s = nl_systems[br_system.value]
    _n = min(int(br_step.value["value"]), len(br_xs) - 1)
    _J = br_Js[min(_n - 1, len(br_Js) - 1)] if _n > 0 else None

    mo.vstack(
        [
            mo.hstack([br_system, br_J0, br_x0, br_y0], justify="center", align="center", gap=1.5),
            mo.hstack([mo.md("**Iteration:**"), br_step, br_zoom], justify="start", align="center"),
            failure_callout("Broyden's method", br_status),
            mo.hstack(
                [
                    draw_system_step(_s, br_xs, _n, model_J=_J, zoom=br_zoom.value, reference=br_newton_xs, title="Broyden"),
                    mo.vstack(
                        [
                            # mo.stat(label="Current iterate", value=f"({br_xs[_n][0]:.6f}, {br_xs[_n][1]:.6f})"),
                            # mo.stat(label="Final estimate", value=f"({br_root[0]:.6f}, {br_root[1]:.6f})", caption=br_status),
                            mo.stat(label="Iterations", value=f"{len(br_xs) - 1}", caption=f"Newton: {len(br_newton_xs) - 1}"),
                            mo.stat(
                                label="Sherman–Morrison version",
                                value=f"{br_sm_iterations} iterations",
                                caption=f"differs by {np.max(np.abs(br_sm_root - br_root)):.1e}",
                            ),
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
    The dashed lines are now the zero lines of the *approximate* linear model
    $\mathbf{F}(\mathbf{x}_n)+J_n(\mathbf{x}-\mathbf{x}_n)$. With $J_0$ the
    identity they have nothing to do with the true curves at first, and the
    first steps are poor; as the updates accumulate information, the lines
    align with the curves near the root. Things to try:

    - **System A from $(0,0)$, $J_0=I$:** 55 iterations instead of Newton's 12
      (grey curve in the error panel). Starting from the exact Jacobian at
      $\mathbf{x}_0$ cuts this to 16 — a good $J_0$ matters.
    - **System A from $(1,0)$, $J_0=I$:** the steps shrink to nothing near
      $(-1.71,-6.29)$, which is **not** a root. Stopping on a small step alone
      would report a false convergence; checking $|\mathbf{F}|$ catches it.
    - **System B from $(0,0)$:** Newton fails there (singular Jacobian), but
      Broyden with $J_0=I$ never uses the true Jacobian and converges in 11
      iterations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The cost of an iteration
    ---

    For two unknowns every method is instantaneous. What matters for large
    $N$ is how the cost of one iteration grows:

    | Method | Work per iteration |
    |---|---|
    | Newton | $N^2$ Jacobian entries + one linear solve, $\mathcal{O}(N^3)$ |
    | Broyden, direct | one linear solve with $J_n$, $\mathcal{O}(N^3)$ |
    | Broyden, Sherman–Morrison | a few matrix–vector and outer products, $\mathcal{O}(N^2)$ |

    Below, one iteration step of each Broyden variant is timed for a dense
    $N\times N$ matrix (the evaluation of $\mathbf{F}$ itself is left out).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cost_N = mo.ui.slider(100, 2000, step=100, value=800, label="Number of unknowns $N$")
    return (cost_N,)


@app.cell
def _(cost_N, np, time):
    _rng = np.random.default_rng(1)
    _N = cost_N.value
    _J = _rng.uniform(-1.0, 1.0, (_N, _N)) + _N * np.eye(_N)
    _Jinv = np.linalg.inv(_J)
    _f, _df = _rng.uniform(-1.0, 1.0, _N), _rng.uniform(-1.0, 1.0, _N)

    def _best_of(fn, repeats=5):
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    def _direct_step():
        delta = np.linalg.solve(_J, -_f)
        return _J + np.outer(_df - _J @ delta, delta) / (delta @ delta)

    def _sherman_morrison_step():
        delta = -_Jinv @ _f
        return _Jinv + np.outer(delta - _Jinv @ _df, delta @ _Jinv) / (delta @ _Jinv @ _df)

    cost_t_direct = _best_of(_direct_step)
    cost_t_sm = _best_of(_sherman_morrison_step)
    return cost_t_direct, cost_t_sm


@app.cell(hide_code=True)
def _(cost_N, cost_t_direct, cost_t_sm, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
    _bars = _ax.bar(["Broyden, direct\n(linear solve)", "Broyden,\nSherman–Morrison"], [cost_t_direct, cost_t_sm], color=["#eb6834", "#2a78d6"], width=0.55, edgecolor="black", lw=0.8)
    for _bar, _t in zip(_bars, [cost_t_direct, cost_t_sm]):
        _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height(), f"{_t * 1e3:.2f} ms", ha="center", va="bottom", fontsize=11)
    _ax.set(ylabel="Time for one iteration (s)", title=f"$N={cost_N.value}$", ylim=(0, 1.2 * max(cost_t_direct, cost_t_sm)))
    _fig.tight_layout()

    mo.vstack(
        [
            cost_N,
            mo.hstack(
                [
                    _fig,
                    mo.stat(label="Speed-up per iteration", value=f"{cost_t_direct / cost_t_sm:.1f}×", caption="grows roughly like N"),
                ],
                widths=[0.7, 0.3],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: systems of non-linear equations

    | Method | Convergence | Needs | Cost per iteration | Weaknesses |
    |---|---|---|---|---|
    | Newton–Raphson | quadratic | $\mathbf{F}$ and $J$ | $N^2$ derivatives + $\mathcal{O}(N^3)$ solve | singular $J$; bad starting points |
    | Broyden (direct) | superlinear | $\mathbf{F}$ and a guess $J_0$ | $\mathcal{O}(N^3)$ solve | more iterations; can stall |
    | Broyden (Sherman–Morrison) | superlinear | $\mathbf{F}$ and a guess $J_0^{-1}$ | $\mathcal{O}(N^2)$ | idem |

    Both methods are local: a good starting point matters even more in
    several dimensions, where there is no bracket to fall back on. Always
    check $|\mathbf{F}(\mathbf{x})|$ at the end — a small step does not prove
    that a root was found. Library routines such as
    `scipy.optimize.root` (with `method="hybr"`, `"broyden1"`, ...) combine
    these steps with safeguards that make the objective function
    $\tilde f$ decrease at every iteration.
    """)
    return


if __name__ == "__main__":
    app.run()
