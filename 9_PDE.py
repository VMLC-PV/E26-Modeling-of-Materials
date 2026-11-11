import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.linalg import solve_banded
    from wigglystuff import PlaySlider
    from wigglystuff import FormulaAnimation as _FormulaAnimation


    class FormulaAnimation(_FormulaAnimation):
        """FormulaAnimation with smaller formulas, so that long equations fit in the box."""

        _css = str(_FormulaAnimation._css) + """
        .formula-animation { width: min(100%, 900px); }
        .formula-animation__formula { font-size: clamp(0.8rem, 1.5vw, 1.1rem); }
        .formula-animation__formula--framed { padding: 10px 16px; }
        """

    # try:
    #     from utils import plot_settings_screen
    # except ImportError:
    #     pass
    return FormulaAnimation, PlaySlider, mo, np, pd, plt, solve_banded


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Partial differential equations

    Partial differential equations (PDEs) occur when we are dealing with functions of more than one variable, for instance *fields*.

    Examples:

    - Electrostatic potential $\phi (x,y,z)$ (Poisson's equation)
    $$
    \Delta\phi(x,y,z) = -\frac{\rho(x,y,z)}{\epsilon_0}
    $$
    - Density or temperature profiles (diffusion/heat equation)
    $$
    \frac{\partial u(\mathbf{x},t)}{\partial t} = D \Delta u(\mathbf{x},t)
    $$
    - Displacement (amplitude) profile (wave equation)
    $$
    \frac{\partial^2 u(\mathbf{x},t)}{\partial t^2} = c^2\Delta u(\mathbf{x},t)
    $$
    - Fluid dynamical fields (flow velocity), e.g. the Navier–Stokes equations.

    In this notebook we solve two kinds of problems with the **finite difference method**:

    | Section | Problem | Equation | Method | Type |
    |---|---|---|---|---|
    | 1 | Boundary value | Laplace, 2D | Jacobi relaxation | iterative |
    | 2 | Boundary value | Laplace, 2D | Gauss–Seidel with overrelaxation | iterative |
    | 3 | Initial value | Heat, 1D | FTCS | explicit |
    | 4 | Initial value | Heat, 1D | Implicit scheme | implicit |
    | 5 | Initial value | Heat, 1D | Crank–Nicolson | implicit |
    | 6 | Initial value | Heat, 2D | FTCS | explicit |
    | 7 | Initial value | Wave, 1D: standing wave and piano string | FTCS, implicit, Crank–Nicolson | both |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## General methods for solving PDEs
    ---

    | Method | Idea | Best for |
    |---|---|---|
    | **Finite difference** | approximate the derivatives by finite differences on a grid | regular (rectangular) shapes; the easiest to implement |
    | Finite element | subdivide the system into smaller parts, the finite elements | boundary value problems in 2/3 dimensions with irregular shapes |
    | Finite volume | convert the volume integral around each mesh point into surface integrals (fluxes) | conserves mass by design: fluid dynamics |

    We use finite differences throughout: every derivative becomes one of the difference formulas from the notebook on numerical derivatives.
    """)
    return


@app.cell(hide_code=True)
def _():
    method_colors = {
        "Jacobi": "darkorange",
        "Gauss–Seidel": "seagreen",
        "FTCS": "darkorange",
        "Implicit": "0.4",
        "Crank–Nicolson": "#1d4ed8",
    }
    return (method_colors,)


@app.cell(hide_code=True)
def _(np, plt):
    def time_colors(n):
        """n colors from a sequential colormap: early times light, late times dark."""
        return plt.cm.viridis(np.linspace(0.9, 0.0, n))

    def plot_profiles(ax, x, profiles, times, unit="s", exact=None):
        """Profiles u(x) at several times, optionally with a reference drawn as thin dashed lines."""
        for i, (u, t, c) in enumerate(zip(profiles, times, time_colors(len(times)))):
            ax.plot(x, u, color=c, lw=2, label=f"$t={t:g}$ {unit}")
            if exact is not None:
                ax.plot(x, exact[i], "--", color="black", lw=0.8, label="exact" if i == 0 else None)

    return (plot_profiles,)


@app.cell(hide_code=True)
def _(np, plt):
    # Schematic of the electrostatics problem: a grounded box whose top wall is held at voltage V
    _M = 8
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _g = np.arange(_M + 1)
    _X, _Y = np.meshgrid(_g, _g)
    _ax.plot(_X, _Y, ".", color="0.75", ms=5)
    _ax.plot([0, _M, _M, 0], [_M, _M, 0, 0], color="black", lw=2.5)
    _ax.plot([0, 0], [0, _M], color="black", lw=2.5)
    _ax.plot([0, _M], [_M, _M], color="crimson", lw=4)
    _ax.text(_M / 2, _M + 0.35, r"$\phi = V$", color="crimson", ha="center", va="bottom", fontsize=13)
    _ax.text(_M / 2, -0.35, r"$\phi = 0$", ha="center", va="top", fontsize=12)
    _ax.text(-0.35, _M / 2, r"$\phi = 0$", ha="right", va="center", fontsize=12, rotation=90)
    _ax.text(_M + 0.35, _M / 2, r"$\phi = 0$", ha="left", va="center", fontsize=12, rotation=90)
    # The five-point stencil
    _i, _j = 5, 3
    for _di, _dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        _ax.plot([_i, _i + _di], [_j, _j + _dj], color="darkorange", lw=1.5)
        _ax.plot([_i + _di], [_j + _dj], "o", color="darkorange", ms=8)
    _ax.plot([_i], [_j], "o", color="#1d4ed8", ms=10, zorder=3)
    _ax.annotate("", xy=(2, 0.5), xytext=(1, 0.5), arrowprops=dict(arrowstyle="<->", lw=1))
    _ax.text(1.5, 0.65, "$a$", ha="center", fontsize=11)
    _ax.set(xlim=(-1.2, _M + 1.2), ylim=(-1.2, _M + 1.2), aspect="equal")
    _ax.axis("off")
    _fig.tight_layout()
    plt.close(_fig)
    laplace_sketch = _fig
    return (laplace_sketch,)


@app.cell(hide_code=True)
def _(FormulaAnimation, laplace_sketch, mo):
    laplace_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="From Laplace's equation to the Jacobi iteration",
            steps=[
                {
                    "tex": r"\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = 0",
                    "note": "Laplace's equation inside the box.",
                },
                {
                    "tex": r"\frac{\partial^2 \phi}{\partial x^2} \approx \frac{\phi(x+a,y) - 2\phi(x,y) + \phi(x-a,y)}{a^2}",
                    "note": "Central difference on a grid with spacing a = L/M; the same in y.",
                },
                {
                    "tex": r"\phi(x+a,y) + \phi(x-a,y) + \phi(x,y+a) + \phi(x,y-a) - 4 \phi(x,y) = 0",
                    "note": "Insert both differences and multiply by a².",
                },
                {
                    "tex": r"\phi(x,y) = \frac{\phi(x+a,y) + \phi(x-a,y) + \phi(x,y+a) + \phi(x,y-a)}{4}",
                    "note": "Each value is the average of its four neighbours.",
                },
                {
                    "tex": r"\phi_{n+1}(x,y) = \frac{\phi_n(x+a,y) + \phi_n(x-a,y) + \phi_n(x,y+a) + \phi_n(x,y-a)}{4}",
                    "note": "Turn it into an iteration: the Jacobi method.",
                },
            ],
            height=400,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 1. Boundary value problems: the Jacobi method
                ---

                Boundary value problems typically deal with static solutions (no time variable).
                Consider a square box of side $L$: one of the walls has voltage $V$ applied to it, the rest are grounded.
                Inside there are no charges, so the electrostatic potential obeys Laplace's equation in two dimensions,
                $$
                \frac{\partial^2 \phi(x,y)}{\partial x^2} + \frac{\partial^2 \phi(x,y)}{\partial y^2} = 0,
                $$
                with the boundary conditions
                $$
                \phi(x,L) = V, \qquad \phi(x,0) = \phi(0,y) = \phi(L,y) = 0.
                $$
                """
            ),
            mo.hstack(
                [
                    laplace_sketch,
                    mo.md(
                        r"""
                        We apply the finite difference method: discretize the box into a grid
                        with steps $a = L/M$ in each direction, and approximate the derivatives by
                        central differences. The equation then links each grid point (blue) to its
                        four neighbours (orange): the **five-point stencil**.
                        """
                    ),
                ],
                widths=[0.4, 0.6],
                align="center",
            ),
            laplace_anim,
            mo.md(
                r"""
                In principle, this is a system of $N \sim M^2$ linear equations that can be solved exactly.
                Gaussian elimination of a dense matrix requires $O(N^3) = O(M^6)$ operations, which becomes
                unfeasible already for $M \sim 100$ or so.

                The **Jacobi** (or relaxation) method instead starts with some initial guess $\phi_0(x,y)$ and
                repeatedly replaces every interior value by the average of its four neighbours, while the
                boundary values are kept fixed at each iteration. It is exactly the relaxation method
                $x_{n+1} = \varphi(x_n)$ from the notebook on non-linear equations, applied to a whole vector of
                unknowns at once. For Laplace's equation it always converges.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def box_potential(M, V=1.0):
        """Initial grid of (M+1) x (M+1) points, phi[i, j] = phi(x = a i, y = a j).

        phi = V on the top wall y = L, and 0 on the other walls and inside the box.
        """
        phi = np.zeros((M + 1, M + 1))
        phi[:, M] = V
        return phi

    def jacobi_iteration(phi):
        """One Jacobi sweep. Returns the new field and the largest change."""
        phinew = phi.copy()  # keeps the boundary conditions
        # The slices do the double loop over i, j = 1 ... M-1 at once:
        # phinew[i, j] = (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]) / 4
        phinew[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2]) / 4
        delta = np.max(np.abs(phinew - phi))
        return phinew, delta

    def relaxation_solve(iteration, phi0, target_accuracy=1e-6, max_iterations=100, monitor=None):
        """Iterate phi -> iteration(phi) until the largest change is below target_accuracy.

        monitor(n, phi, delta), if given, is called after every iteration.
        Returns the field, the list of changes delta_n, and a status.
        """
        phi = phi0.copy()
        deltas = []
        for n in range(1, max_iterations + 1):
            phi, delta = iteration(phi)
            deltas.append(delta)
            if monitor is not None:
                monitor(n, phi, delta)
            if delta <= target_accuracy:
                return phi, deltas, "converged"
        return phi, deltas, "max. iterations"

    return box_potential, jacobi_iteration, relaxation_solve


@app.cell(hide_code=True)
def _(plt):
    def field_figure():
        """Common layout: a square field map and a convergence-history panel."""
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
        return fig, ax, ax2

    def draw_potential(fig, ax, phi, title):
        """The potential in the box, with a few equipotential lines."""
        im = ax.imshow(phi.T, origin="lower", extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0, cmap="viridis")
        cs = ax.contour(phi.T, levels=[0.02, 0.05, 0.1, 0.2, 0.4], origin="lower", extent=[0, 1, 0, 1], colors="white", linewidths=0.8)
        ax.clabel(cs, fontsize=7)
        ax.set(xlabel="$x/L$", ylabel="$y/L$", title=title)
        fig.colorbar(im, ax=ax, label=r"$\phi/V$", shrink=0.9)

    return draw_potential, field_figure


@app.cell(hide_code=True)
def _(mo):
    lap_M = mo.ui.dropdown(options={"20": 20, "50": 50, "100": 100}, value="100", label="Grid $M$:")
    lap_target = mo.ui.dropdown(
        options={"1e-3": 1e-3, "1e-4": 1e-4, "1e-5": 1e-5, "1e-6": 1e-6}, value="1e-4", label="Target accuracy:"
    )
    return lap_M, lap_target


@app.cell
def _(box_potential, jacobi_iteration, lap_M, lap_target, relaxation_solve):
    lap_phi0 = box_potential(lap_M.value)
    lap_phi, lap_deltas, lap_status = relaxation_solve(jacobi_iteration, lap_phi0, lap_target.value, max_iterations=50000)
    return lap_deltas, lap_phi0, lap_status


@app.cell(hide_code=True)
def _(jacobi_iteration, lap_phi0, lap_target, np, relaxation_solve):
    # Run again, recording some snapshots for the animation
    lap_frames = [(0, lap_phi0)]
    _keep = set(np.unique(np.geomspace(1, 50000, 90).astype(int)))

    def _monitor(n, phi, delta):
        if n in _keep:
            lap_frames.append((n, phi.copy()))

    _phi, _deltas, _ = relaxation_solve(jacobi_iteration, lap_phi0, lap_target.value, 50000, _monitor)
    if lap_frames[-1][0] != len(_deltas):
        lap_frames.append((len(_deltas), _phi))
    return (lap_frames,)


@app.cell(hide_code=True)
def _(PlaySlider, lap_frames, mo):
    lap_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(len(lap_frames) - 1, 1), step=1, interval_ms=150))
    return (lap_play,)


@app.cell(hide_code=True)
def _(
    draw_potential,
    field_figure,
    lap_M,
    lap_deltas,
    lap_frames,
    lap_play,
    lap_status,
    lap_target,
    method_colors,
    mo,
    np,
    plt,
):
    _k = min(int(lap_play.value["value"]), len(lap_frames) - 1)
    _n, _phi = lap_frames[_k]
    _its = np.arange(1, len(lap_deltas) + 1)

    _fig, _ax, _ax2 = field_figure()
    draw_potential(_fig, _ax, _phi, f"Jacobi: iteration {_n}")
    _ax2.semilogy(_its, lap_deltas, color=method_colors["Jacobi"], lw=1.5, label=r"largest change $\Delta_n$")
    _ax2.axhline(lap_target.value, color="black", lw=0.8, ls=":", label="target accuracy")
    _ax2.axvline(max(_n, 1), color="#1d4ed8", lw=1.0, ls=":")
    _ax2.set(xscale="log", xlabel="Iteration $n$", title="Convergence history", ylim=(min(lap_deltas) / 5, 2.0))
    _ax2.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md("Choose the grid and the target accuracy, then press ▶ to watch the potential spread from the top wall into the box."),
            mo.hstack([lap_M, lap_target, mo.md("**Snapshot:**"), lap_play], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Iterations", value=f"{len(lap_deltas)}", caption=lap_status),
                            mo.stat(label="Last change Δₙ", value=f"{lap_deltas[-1]:.1e}"),
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
    Look at the convergence history: the stopping criterion only checks the largest *change* per iteration,
    $\Delta_n = \max|\phi_{n}-\phi_{n-1}|$, and for $M=100$ the Jacobi method changes the field very slowly.
    With a target of $10^{-4}$ it stops after about 1900 iterations, but this does not mean that the potential is accurate to $10^{-4}$.

    The number of iterations needed grows like $M^2$: a finer grid is much more expensive, which motivates the next section.
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    sor_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Overrelaxation",
            steps=[
                {
                    "tex": r"\phi_{n+1}(x,y) = \phi_n(x,y) + \Delta_n \phi(x,y)",
                    "note": "Any relaxation method: the old value plus a correction.",
                },
                {
                    "tex": r"\phi_{n+1}(x,y) = \phi_n(x,y) + (1+\omega)\, \Delta_n \phi(x,y)",
                    "note": "Overrelax: move a bit further than the correction suggests, with ω > 0.",
                },
                {
                    "tex": r"\phi_{n+1} = (1+\omega) \frac{\phi_n(x+a,y) + \phi_n(x-a,y) + \phi_n(x,y+a) + \phi_n(x,y-a)}{4} - \omega \, \phi_n(x,y)",
                    "note": "With the Jacobi correction Δφ = (average of the neighbours) − φn. Unstable for any ω > 0!",
                },
                {
                    "tex": r"\phi_{n+1} = (1+\omega) \frac{\phi_n(x+a,y) + \phi_{n+1}(x-a,y) + \phi_n(x,y+a) + \phi_{n+1}(x,y-a)}{4} - \omega \, \phi_n(x,y)",
                    "note": "The same for Gauss–Seidel: stable for ω < 1.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 2. Gauss–Seidel method with overrelaxation
                ---

                The Jacobi method computes the whole new field $\phi_{n+1}$ from the old one, $\phi_n$:
                $$
                \phi_{n+1}(x,y) = \frac{\phi_n(x+a,y) + \phi_n(x-a,y) + \phi_n(x,y+a) + \phi_n(x,y-a)}{4}.
                $$
                This requires two arrays. In the **Gauss–Seidel method** one instead updates the field in place,
                so that the already computed values of $\phi_{n+1}$ are used where available. Sweeping through the grid
                in increasing $x$ and $y$, the neighbours to the left and below have already been updated:
                $$
                \phi_{n+1}(x,y) = \frac{\phi_n(x+a,y) + \phi_{n+1}(x-a,y) + \phi_n(x,y+a) + \phi_{n+1}(x,y-a)}{4}.
                $$
                This uses less memory and converges about twice as fast.

                A much bigger speed-up comes from **overrelaxation**:
                """
            ),
            sor_anim,
            mo.md(
                r"""
                Applied to the Jacobi method, overrelaxation is unstable. Applied to Gauss–Seidel,
                it is stable for $0\le\omega<1$, and the best choice is
                $$
                \omega_{\rm opt} = \frac{2}{1+\sin(\pi/M)} - 1 \approx 1 - \frac{2\pi}{M},
                $$
                i.e. $\omega_{\rm opt}\approx0.94$ for $M=100$. Then the number of iterations grows only like $M$ instead of $M^2$.
                (This method is often called *successive overrelaxation*, SOR, with the factor $1+\omega$ written as a single parameter between 1 and 2.)
                """
            ),
        ]
    )
    return


@app.function
def gaussseidel_iteration(phi, omega=0.0):
    """One Gauss-Seidel sweep with overrelaxation parameter omega >= 0, done in place.

    Returns the field and the largest change.
    """
    M = len(phi) - 1
    delta = 0.0
    for i in range(1, M):
        for j in range(1, M):
            phiold = phi[i, j]
            phi[i, j] = (1.0 + omega) * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]) / 4 - omega * phiold
            delta = max(delta, abs(phi[i, j] - phiold))
    return phi, delta


@app.cell(hide_code=True)
def _(mo):
    gs_M = mo.ui.dropdown(options={"20": 20, "50": 50, "100": 100}, value="50", label="Grid $M$:")
    gs_omega = mo.ui.slider(0.0, 0.99, step=0.01, value=0.9, show_value=True, label=r"Overrelaxation $\omega$")
    return gs_M, gs_omega


@app.cell
def _(box_potential, gs_M, gs_omega, relaxation_solve):
    gs_target = 1e-4
    gs_phi, gs_deltas, gs_status = relaxation_solve(
        lambda phi: gaussseidel_iteration(phi, gs_omega.value), box_potential(gs_M.value), gs_target, max_iterations=20000
    )
    return gs_deltas, gs_phi, gs_status, gs_target


@app.cell(hide_code=True)
def _(
    box_potential,
    draw_potential,
    field_figure,
    gs_M,
    gs_deltas,
    gs_omega,
    gs_phi,
    gs_status,
    gs_target,
    jacobi_iteration,
    method_colors,
    mo,
    np,
    plt,
    relaxation_solve,
):
    _phi0 = box_potential(gs_M.value)
    _jac_phi, _jac_deltas, _ = relaxation_solve(jacobi_iteration, _phi0, gs_target, max_iterations=50000)
    _omega_opt = 2.0 / (1.0 + np.sin(np.pi / gs_M.value)) - 1.0

    _fig, _ax, _ax2 = field_figure()
    draw_potential(_fig, _ax, gs_phi, rf"Gauss–Seidel, $\omega={gs_omega.value:.2f}$")
    _ax2.semilogy(np.arange(1, len(_jac_deltas) + 1), _jac_deltas, color=method_colors["Jacobi"], lw=1.5, label="Jacobi")
    _ax2.semilogy(np.arange(1, len(gs_deltas) + 1), gs_deltas, color=method_colors["Gauss–Seidel"], lw=1.5, label=rf"Gauss–Seidel, $\omega={gs_omega.value:.2f}$")
    _ax2.axhline(gs_target, color="black", lw=0.8, ls=":", label="target accuracy")
    _ax2.set(xscale="log", xlabel="Iteration $n$", ylabel=r"largest change $\Delta_n$", title="Convergence history")
    _ax2.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md("Change $\\omega$ and compare the number of iterations with the Jacobi method on the same grid (target accuracy $10^{-4}$). The Gauss–Seidel sweep is a plain Python double loop: $M=100$ with a small $\\omega$ takes several seconds."),
            mo.hstack([gs_M, gs_omega], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Gauss–Seidel iterations", value=f"{len(gs_deltas)}", caption=gs_status),
                            mo.stat(label="Jacobi iterations", value=f"{len(_jac_deltas)}"),
                            mo.stat(label="Optimal ω", value=f"{_omega_opt:.3f}", caption="2/(1+sin(π/M)) − 1"),
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
def _(box_potential, method_colors, mo, np, plt, relaxation_solve):
    # Number of iterations against omega, on a small grid where the Python loops are fast
    _M, _target = 20, 1e-6
    _omegas = np.round(np.linspace(0.0, 0.95, 20), 3)
    _counts = [
        len(relaxation_solve(lambda phi, w=_w: gaussseidel_iteration(phi, w), box_potential(_M), _target, 20000)[1])
        for _w in _omegas
    ]
    _omega_opt = 2.0 / (1.0 + np.sin(np.pi / _M)) - 1.0

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_omegas, _counts, "o-", color=method_colors["Gauss–Seidel"], ms=4, lw=1.2)
    _ax.axvline(_omega_opt, color="black", lw=0.8, ls=":")
    _ax.annotate(rf"$\omega_{{\rm opt}}={_omega_opt:.2f}$", (_omega_opt, max(_counts) * 0.8), xytext=(-8, 0), textcoords="offset points", ha="right")
    _ax.set(xlabel=r"$\omega$", ylabel="Iterations", yscale="log", title=f"Gauss–Seidel with overrelaxation, $M={_M}$, target $10^{{-6}}$")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            _fig,
            mo.md(
                rf"""
                Without overrelaxation ($\omega=0$) Gauss–Seidel needs {_counts[0]} iterations on this small grid; near
                $\omega_{{\rm opt}}$ it needs only {min(_counts)}. Beyond the optimum the iterations start to overshoot and
                oscillate, and the count rises steeply again; for $\omega\ge1$ the method diverges.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial value problems
    ---

    In most cases we study the time evolution of a certain profile $u(t,\mathbf{x})$.
    Then the PDE describes the time evolution of the field(s) starting from some initial conditions and obeying boundary conditions.

    Let us take the heat equation as an example. The field $u$ is the temperature $T$.
    In one dimension the heat equation reads
    $$
    \frac{\partial u}{\partial t} = D \, \frac{\partial^2 u}{\partial x^2},
    $$
    where $D$ is the thermal diffusivity.
    This equation describes the time evolution of $u(t,x)$ given the initial profile
    $$
    u(t=0,x) = u_0(x),
    $$
    and the boundary conditions
    $$
    u(t,x=0) = u_{\rm left}(t), \qquad u(t,x=L) = u_{\rm right}(t).
    $$
    If $u_{\rm left}$ and $u_{\rm right}$ are time-independent, the solution approaches a stationary profile as $t \to \infty$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The FTCS scheme
    ---

    The FTCS (Forward Time, Centered Space) scheme is the simplest method for solving the heat equation.

    First we discretize the spatial coordinate into a grid with $N + 1$ points,
    $$
    x_k = a k, \qquad k = 0\ldots N, \qquad a = L/N,
    $$
    and approximate the derivative $\partial^2 u / \partial x^2$ by the lowest-order central difference
    $$
    \frac{\partial^2 u(t,x)}{\partial x^2} \approx \frac{u(t,x+a) - 2u(t,x) + u(t,x-a)}{a^2}.
    $$
    To evaluate the time evolution we work in small time steps of size $h$, and approximate the time derivative by the forward difference
    $$
    \frac{\partial u(t,x)}{\partial t} \approx \frac{u(t+h,x) - u(t,x)}{h}.
    $$
    This gives the discretized equation
    $$
    \frac{u(t+h,x) - u(t,x)}{h} = D \frac{u(t,x+a) - 2u(t,x) + u(t,x-a)}{a^2}.
    $$
    The method is **explicit**: to evaluate $u(t+h,x)$ at the next time step we only need the profile $u(t,x)$ at the present time step.
    Denoting the time step by a superscript $n$ ($t_n = hn$) and the grid point by a subscript $k$ ($x_k = ak$) we get the iteration
    $$
    u^{n+1}_k = u^n_k + r \, (u^n_{k+1} - 2u^n_k + u^n_{k-1}), \qquad k = 1 \ldots N-1,
    $$
    with the dimensionless parameter
    $$
    r \equiv \frac{Dh}{a^2}.
    $$
    For every grid point this is Euler's method from the notebook on ODEs.
    """)
    return


@app.cell
def _():
    def heat_FTCS_iteration(u, r):
        """Single FTCS time step for the heat equation, with r = D h / a^2."""
        unew = u.copy()  # keeps the boundary values u[0] and u[N]
        # Same as the loop over k = 1 ... N-1 of u[k] + r * (u[k+1] - 2 * u[k] + u[k-1])
        unew[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        return unew

    def heat_solve(iteration, u0, h, nsteps, a, D=1.0):
        """Perform nsteps time steps of size h with the given scheme.

        u0: the initial profile, a: the grid spacing, D: the diffusion constant.
        """
        u = u0.copy()
        r = h * D / a**2
        for _ in range(nsteps):
            u = iteration(u, r)
        return u

    def heat_profiles(iteration, u0, h, a, D, times):
        """The profiles at the given times."""
        profiles = []
        u, current_time = u0, 0.0
        for time in times:
            nsteps = round((time - current_time) / h)
            u = heat_solve(iteration, u, h, nsteps, a, D)
            profiles.append(u)
            current_time += nsteps * h
        return profiles

    return heat_FTCS_iteration, heat_profiles


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                Let us consider Example 9.3 from M. Newman, *Computational Physics*:

                We have a 1 cm thick steel container, initially at a temperature of 20 °C.
                It is placed in a bath of cold water at 0 °C and filled on top with hot water at 50 °C.
                Our goal is to calculate the temperature profile across the steel as a function of time.
                The thermal diffusivity of stainless steel is $D = 4.25 \cdot 10^{-6}$ m$^2$ s$^{-1}$.
                """
            ),
            mo.image(src=mo.notebook_dir() / "./images/fig9-i.png", width=520).center(),
            mo.md("Source: M. Newman, *Computational Physics*").center(),
            mo.md(
                r"""
                The bottom of the container is a 1D problem across its thickness: $x=0$ is the side of the hot water
                ($u=50$ °C) and $x=L=1$ cm the side of the cold bath ($u=0$ °C).
                We calculate the profile at times $t = 0.01$ s, $0.1$ s, $0.4$ s, $1$ s, and $10$ s.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    steel_L = 0.01  # thickness of the steel in m
    steel_D = 4.25e-6  # thermal diffusivity in m^2/s
    T_lo, T_mid, T_hi = 0.0, 20.0, 50.0  # temperatures in Celsius
    steel_times = [0.01, 0.1, 0.4, 1.0, 10.0]  # output times in s

    def steel_initial(N):
        """N+1 grid points: hot water at x = 0, cold bath at x = L, steel at T_mid in between."""
        u = np.full(N + 1, T_mid)
        u[0], u[N] = T_hi, T_lo
        return u

    return T_hi, T_lo, steel_D, steel_L, steel_initial, steel_times


@app.cell(hide_code=True)
def _(np):
    def heat_exact(u0, a, D, times):
        """Exact time evolution of the heat equation discretized in space only,
        du_k/dt = D (u_{k+1} - 2 u_k + u_{k-1}) / a^2, from its eigenmodes.

        The difference with a scheme on the same grid is the error of the time stepping alone.
        """
        N = len(u0) - 1
        A = D / a**2 * (np.diag(-2.0 * np.ones(N - 1)) + np.diag(np.ones(N - 2), 1) + np.diag(np.ones(N - 2), -1))
        b = np.zeros(N - 1)
        b[0], b[-1] = D / a**2 * u0[0], D / a**2 * u0[N]
        w, V = np.linalg.eigh(A)
        steady = -np.linalg.solve(A, b)
        c = V.T @ (u0[1:N] - steady)
        out = []
        for t in times:
            u = u0.copy()
            u[1:N] = steady + V @ (c * np.exp(w * t))
            out.append(u)
        return out

    return (heat_exact,)


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    ftcs_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Von Neumann stability analysis of FTCS",
            steps=[
                {
                    "tex": r"u^n_k = A^n\, e^{iqak}",
                    "note": "Try a single Fourier mode with wavenumber q; A is its amplification factor per step.",
                },
                {
                    "tex": r"A = 1 + r\left(e^{iqa} - 2 + e^{-iqa}\right)",
                    "note": "Insert it into the FTCS update and divide by the mode itself.",
                },
                {
                    "tex": r"A = 1 - 4r\sin^2\frac{qa}{2}",
                    "note": "Use e^{iθ} + e^{−iθ} = 2cos θ and 1 − cos θ = 2 sin²(θ/2).",
                },
                {
                    "tex": r"|A| \le 1 \;\Longleftrightarrow\; 1-4r \ge -1 \;\Longleftrightarrow\; r \le \frac{1}{2}",
                    "note": "The worst mode, sin² = 1, is a zig-zag between neighbouring grid points.",
                },
                {
                    "tex": r"h \le \frac{a^2}{2D}",
                    "note": "Halving the grid spacing forces a time step four times smaller.",
                },
            ],
            height=280,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### Stability

                Like Euler's method for a stiff ODE, FTCS is only conditionally stable. To see why, follow a single
                Fourier mode of the profile through one time step:
                """
            ),
            ftcs_anim,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    ftcs_h = mo.ui.dropdown(
        options={"1e-4 s": 1e-4, "5e-4 s": 5e-4, "1e-3 s": 1e-3, "1.1e-3 s": 1.1e-3, "1.2e-3 s": 1.2e-3},
        value="1e-4 s",
        label="Time step $h$:",
    )
    ftcs_N = mo.ui.dropdown(options={"50": 50, "90": 90, "100": 100}, value="100", label="Grid divisions $N$:")
    return ftcs_N, ftcs_h


@app.cell(hide_code=True)
def _(
    T_hi,
    T_lo,
    ftcs_N,
    ftcs_h,
    heat_FTCS_iteration,
    heat_profiles,
    mo,
    np,
    plot_profiles,
    plt,
    steel_D,
    steel_L,
    steel_initial,
    steel_times,
):
    _N, _h = ftcs_N.value, ftcs_h.value
    _a = steel_L / _N
    _r = _h * steel_D / _a**2
    with np.errstate(all="ignore"):  # let the instability show up as overflow
        _profiles = heat_profiles(heat_FTCS_iteration, steel_initial(_N), _h, _a, steel_D, steel_times)
    _x = np.linspace(0.0, steel_L, _N + 1) * 100  # in cm

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    plot_profiles(_ax, _x, _profiles, steel_times)
    _ax.set(xlabel="$x$ [cm]", ylabel="$T$ [°C]", ylim=(T_lo - 10, T_hi + 10), title=f"Heat equation, FTCS: $N={_N}$, $h={_h:g}$ s")
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    plt.close(_fig)

    _stable = _r <= 0.5
    mo.vstack(
        [
            mo.md("Choose the time step and the grid. The original example uses $h=10^{-4}$ s and $N=100$."),
            mo.hstack([ftcs_h, ftcs_N], justify="center", align="center", gap=2),
            mo.md("") if _stable else mo.md(f"**FTCS is unstable:** $r={_r:.3f} > 1/2$, the zig-zag mode grows by a factor $|1-4r|={abs(1 - 4 * _r):.2f}$ every step.").callout(kind="danger"),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="r = Dh/a²", value=f"{_r:.3f}", caption="stable: r ≤ 1/2" if _stable else "unstable: r > 1/2"),
                            mo.stat(label="Time steps to t = 10 s", value=f"{round(10.0 / _h)}"),
                            mo.stat(label="Largest |T| at t = 10 s", value=f"{np.nanmax(np.abs(_profiles[-1])):.3g} °C"),
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

    - **$h=10^{-4}$ s, $N=100$:** $r=0.0425$. The heat from the hot side diffuses into the steel, and the profile
      relaxes towards the stationary straight line from 50 °C to 0 °C.
    - **$h=1.2\cdot10^{-3}$ s:** $r=0.51$, just above $1/2$. The profile looks fine at first, but a zig-zag between
      neighbouring points grows exponentially and destroys the solution.
    - **$h=1.2\cdot10^{-3}$ s and $N=90$:** a *coarser* grid brings $r$ back below $1/2$ ($r=0.41$), and the method is stable
      again. FTCS is caught between two constraints: an accurate solution needs a small $a$, and stability then needs $h\propto a^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Implicit scheme
    ---

    In the implicit scheme one uses a backward difference for the time derivative,
    $$
    \frac{\partial u(t+h,x)}{\partial t} \approx \frac{u(t+h,x) - u(t,x)}{h},
    $$
    thus
    $$
    \frac{u(t+h,x) - u(t,x)}{h} = D \frac{u(t+h,x+a) - 2u(t+h,x) + u(t+h,x-a)}{a^2}.
    $$
    In discretized notation
    $$
    u^{n+1}_k = u^n_k + r \, (u^{n+1}_{k+1} - 2u^{n+1}_k + u^{n+1}_{k-1}), \qquad k = 1 \ldots N-1.
    $$
    This is the implicit Euler method from the notebook on ODEs. In other words, we have a system of linear equations for $u^{n+1}_k$:
    $$
    -r u^{n+1}_{k-1} + (1+2r) u^{n+1}_{k} - r u^{n+1}_{k+1} = u^n_k, \qquad k = 1 \ldots N-1.
    $$
    The boundary values $u^{n+1}_0$ and $u^{n+1}_N$ are known, and move to the right-hand side of the first and last equations.
    The system has to be solved at every time step. Luckily, it is **tridiagonal**, so Gaussian elimination
    without pivoting (the *Thomas algorithm*) solves it in $O(N)$ operations instead of $O(N^3)$.
    We use `scipy.linalg.solve_banded`, which only stores the three diagonals, as the rows of a $3\times(N-1)$ array:
    the upper diagonal (its first entry is not used), the diagonal, and the lower diagonal (its last entry is not used).

    The amplification factor of a Fourier mode is now $A = 1/(1+4r\sin^2(qa/2))$, between 0 and 1 for any $r$:
    the scheme is **unconditionally stable**. We compare it with the other schemes at the end of the next section.
    """)
    return


@app.cell
def _(np, solve_banded):
    def heat_implicit_iteration(u, r):
        """Single implicit time step for the heat equation, with r = D h / a^2."""
        N = len(u) - 1
        unew = u.copy()  # keeps the boundary values u[0] and u[N]
        # Rows: upper diagonal, diagonal, lower diagonal
        ab = np.array([np.full(N - 1, -r), np.full(N - 1, 1 + 2 * r), np.full(N - 1, -r)])
        v = u[1:N].copy()
        v[0] += r * u[0]  # the known boundary values
        v[N - 2] += r * u[N]
        unew[1:N] = solve_banded((1, 1), ab, v)
        return unew

    return (heat_implicit_iteration,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Method of lines: the heat equation as a stiff ODE system

    Discretizing only the spatial derivative turns the PDE into a system of $N-1$ coupled ODEs,
    $$
    \frac{du_k}{dt} = \frac{D}{a^2}\,(u_{k+1} - 2u_k + u_{k-1}),
    $$
    one for each grid point. FTCS is Euler's method and the implicit scheme is implicit Euler applied to
    this system. Its decay rates range from about $D\pi^2/L^2$ (the smooth profile) to $4D/a^2$ (the zig-zag),
    a ratio of $\sim 4N^2/\pi^2 \approx 4000$ for $N=100$: the system is **stiff**, exactly like
    $dx/dt=-15x$ in the notebook on ODEs. The fastest mode sets the stability limit of the explicit method,
    even though it is gone after a few steps and contributes nothing to the solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Crank–Nicolson scheme
    ---

    The Crank–Nicolson method is a combination of the FTCS and implicit schemes.
    It approximates the time derivative by the *average* of the explicit and implicit right-hand sides,
    $$
    \frac{\partial u}{\partial t} \approx \frac{1}{2} \left[ D \, \frac{\partial^2 u(t+h,x)}{\partial x^2} + D \, \frac{\partial^2 u(t,x)}{\partial x^2} \right],
    $$
    like the trapezoidal rule for integrals. Applying central differences to the spatial derivatives we obtain
    $$
    \frac{u(t+h,x) - u(t,x)}{h} = \frac{D}{2} \frac{u(t+h,x+a) - 2u(t+h,x) + u(t+h,x-a)}{a^2} + \frac{D}{2} \frac{u(t,x+a) - 2u(t,x) + u(t,x-a)}{a^2}.
    $$
    This corresponds to the following tridiagonal system of linear equations:
    $$
    -r u^{n+1}_{k-1} + 2(1+r) u^{n+1}_{k} - r u^{n+1}_{k+1} = ru^n_{k-1} + 2(1-r)u^n_k + ru^n_{k+1}, \qquad k = 1 \ldots N-1.
    $$
    The errors of the forward and backward differences cancel to first order, so the method is **second-order**
    accurate in time (FTCS and the implicit scheme are first-order), and it is unconditionally stable:
    $|A| = |1-2r\sin^2(qa/2)|\,/\,(1+2r\sin^2(qa/2)) \le 1$ for any $r$.
    """)
    return


@app.cell
def _(np, solve_banded):
    def heat_crank_nicolson_iteration(u, r):
        """Single Crank-Nicolson time step for the heat equation, with r = D h / a^2."""
        N = len(u) - 1
        unew = u.copy()  # keeps the boundary values u[0] and u[N]
        ab = np.array([np.full(N - 1, -r), np.full(N - 1, 2 * (1 + r)), np.full(N - 1, -r)])
        v = 2 * (1 - r) * u[1:N] + r * u[:-2] + r * u[2:]  # the explicit half of the step
        v[0] += r * u[0]  # the known boundary values
        v[N - 2] += r * u[N]
        unew[1:N] = solve_banded((1, 1), ab, v)
        return unew

    return (heat_crank_nicolson_iteration,)


@app.cell(hide_code=True)
def _(mo):
    cn_scheme = mo.ui.dropdown(options=["FTCS", "Implicit", "Crank–Nicolson"], value="Crank–Nicolson", label="Scheme:")
    cn_h = mo.ui.dropdown(
        options={"1e-4 s": 1e-4, "1e-3 s": 1e-3, "1e-2 s": 1e-2, "0.1 s": 0.1, "1 s": 1.0}, value="1e-2 s", label="Time step $h$:"
    )
    return cn_h, cn_scheme


@app.cell(hide_code=True)
def _(
    cn_h,
    cn_scheme,
    heat_FTCS_iteration,
    heat_crank_nicolson_iteration,
    heat_exact,
    heat_implicit_iteration,
    heat_profiles,
    np,
    steel_D,
    steel_L,
    steel_initial,
):
    # Frames every 0.02 s up to t = 2 s, or every step when the step is longer
    _schemes = {"FTCS": heat_FTCS_iteration, "Implicit": heat_implicit_iteration, "Crank–Nicolson": heat_crank_nicolson_iteration}
    cn_N = 100
    cn_r = cn_h.value * steel_D / (steel_L / cn_N) ** 2
    cn_times = np.arange(0.0, 2.0 + 1e-9, max(cn_h.value, 0.02))
    cn_x = np.linspace(0.0, steel_L, cn_N + 1) * 100
    with np.errstate(all="ignore"):  # let the FTCS instability show up as overflow
        cn_frames = heat_profiles(_schemes[cn_scheme.value], steel_initial(cn_N), cn_h.value, steel_L / cn_N, steel_D, cn_times)
        cn_exact = heat_exact(steel_initial(cn_N), steel_L / cn_N, steel_D, cn_times)
        cn_errors = np.array([np.max(np.abs(_u - _e)) for _u, _e in zip(cn_frames, cn_exact)])
    return cn_errors, cn_exact, cn_frames, cn_r, cn_times, cn_x


@app.cell(hide_code=True)
def _(PlaySlider, cn_times, mo):
    cn_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=max(len(cn_times) - 1, 1), step=1, interval_ms=100))
    return (cn_play,)


@app.cell(hide_code=True)
def _(
    T_hi,
    T_lo,
    cn_errors,
    cn_exact,
    cn_frames,
    cn_h,
    cn_play,
    cn_r,
    cn_scheme,
    cn_times,
    cn_x,
    method_colors,
    mo,
    np,
    plt,
):
    _k = min(int(cn_play.value["value"]), len(cn_times) - 1)
    _stable = cn_scheme.value != "FTCS" or cn_r <= 0.5
    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    _ax.plot(cn_x, cn_frames[0], color="0.75", lw=1.0, label="$t=0$")
    _ax.plot(cn_x, T_hi * (1 - cn_x / cn_x[-1]), ":", color="0.4", lw=1.0, label=r"stationary profile, $t\to\infty$")
    _ax.plot(cn_x, cn_exact[_k], "--", color="black", lw=1.0, label="exact")
    _ax.plot(cn_x, cn_frames[_k], color=method_colors[cn_scheme.value], lw=2.2, label=f"{cn_scheme.value}, $t={cn_times[_k]:.2f}$ s")
    _ax.set(xlabel="$x$ [cm]", ylabel="$T$ [°C]", ylim=(T_lo - 10, T_hi + 10), title=f"Heat equation, {cn_scheme.value}: $N=100$, $h={cn_h.value:g}$ s")
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    plt.close(_fig)


    def _fmt(err):
        if not np.isfinite(err) or err > 1e6:
            return "blown up"
        return "0 °C" if err < 1e-10 else f"{err:.2g} °C"


    mo.vstack(
        [
            mo.hstack([cn_scheme, cn_h, mo.md("**Animate:**"), cn_play], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(
                                label="r = Dh/a²",
                                value=f"{cn_r:g}",
                                caption=("stable: r ≤ 1/2" if _stable else "unstable: r > 1/2") if cn_scheme.value == "FTCS" else "stable for any r",
                            ),
                            mo.stat(label="Error now", value=_fmt(cn_errors[_k]), caption="vs. exact time evolution"),
                            mo.stat(label="Largest error", value=_fmt(np.max(cn_errors)), caption="over the whole animation"),
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
    The dashed line is the exact time evolution on the same grid, so the difference is the error of the time stepping alone. Things to try:

    - **FTCS:** stable for $h=10^{-4}$ s and $10^{-3}$ s ($r=0.425$), but at $h=10^{-2}$ s ($r=4.25$) it blows up within a few steps.
    - **Implicit, $h=10^{-2}$ s:** stable, far beyond the FTCS limit. A step this long cannot follow the fast initial transient
      (at $t=0.02$ s the error is about 2 °C), but from $t=0.4$ s on the error is a fraction of a degree. With $h=1$ s ($r=425$)
      it is still stable but inaccurate: the heat diffuses too slowly, and the animation has only three frames, because that is
      all the scheme computes up to $t=2$ s. As for implicit Euler, large steps are *safe*, not *precise*.
    - **Crank–Nicolson, $h=10^{-2}$ s:** the most accurate at late times, but look at the first frames near $x=0$: a zig-zag.
      For large $r$ the amplification factor of the zig-zag mode, $(1-2r)/(1+2r)\approx-0.79$ for $r=4.25$, is close to $-1$:
      the mode is barely damped and flips sign at every step. The sharp jumps of the initial profile at the walls excite it,
      and it dies out only slowly. The implicit scheme damps it immediately ($A\approx0.06$). At $h=0.1$ s ($A\approx-0.98$)
      the zig-zag hardly decays at all.

    The order of each scheme shows up in the error at a fixed time as a function of $h$:
    """)
    return


@app.cell(hide_code=True)
def _(
    heat_FTCS_iteration,
    heat_crank_nicolson_iteration,
    heat_exact,
    heat_implicit_iteration,
    heat_profiles,
    method_colors,
    mo,
    np,
    pd,
    plt,
    steel_D,
    steel_L,
    steel_initial,
):
    _N, _T = 100, 1.0
    _a = steel_L / _N
    _u0 = steel_initial(_N)
    _exact = heat_exact(_u0, _a, steel_D, [_T])[0]
    _hs = np.array([1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4])
    _schemes = {"FTCS": heat_FTCS_iteration, "Implicit": heat_implicit_iteration, "Crank–Nicolson": heat_crank_nicolson_iteration}
    _errors = {}
    for _name, _it in _schemes.items():
        _hh = _hs[_hs * steel_D / _a**2 <= 0.5] if _name == "FTCS" else _hs  # FTCS: stable steps only
        _errors[_name] = (_hh, np.array([np.max(np.abs(heat_profiles(_it, _u0, _h, _a, steel_D, [_T])[0] - _exact)) for _h in _hh]))

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _styles = {"FTCS": dict(marker="s", ms=8, ls="--", mfc="none"), "Implicit": dict(marker="o", ms=4), "Crank–Nicolson": dict(marker="o", ms=4)}
    for _name, (_hh, _e) in _errors.items():
        _ax.loglog(_hh, _e, lw=1.2, color=method_colors[_name], label=_name, **_styles[_name])
    _ax.axvline(0.5 * _a**2 / steel_D, color="black", lw=0.8, ls=":")
    _ax.text(0.5 * _a**2 / steel_D, 1e-7, " FTCS limit\n $r=1/2$", fontsize=9, va="center")
    _ax.set(xlabel="Time step $h$ [s]", ylabel=r"$\max_k |u_k - u_k^{\rm exact}|$ [°C]", title=f"Time-stepping error at $t={_T:g}$ s, $N={_N}$")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    plt.close(_fig)

    _fit = {_n: np.polyfit(np.log(_hh), np.log(_e), 1)[0] for _n, (_hh, _e) in _errors.items()}
    _rows = [{"h [s]": _h, **{_n: (_e[list(_hh).index(_h)] if _h in _hh else None) for _n, (_hh, _e) in _errors.items()}} for _h in _hs]

    mo.vstack(
        [
            _fig,
            mo.md("Fitted slopes: " + ", ".join(f"**{_n}** {_s:.2f}" for _n, _s in _fit.items()) + "."),
            mo.accordion({"Error table [°C]": mo.ui.table(pd.DataFrame(_rows), selection=None)}),
            mo.md(
                r"""
                FTCS and the implicit scheme are first-order and almost equally accurate, but FTCS is limited to the left of the
                dotted line. Crank–Nicolson is second-order: at $h=10^{-2}$ s it is as accurate as the first-order schemes at
                $h\approx3\cdot10^{-5}$ s, which need some 300 times more steps.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Heat equation in two dimensions
    ---

    In two dimensions the heat equation reads
    $$
    \frac{\partial u}{\partial t} = D \, \left[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right].
    $$
    This equation describes the time evolution of $u(t,x,y)$ given the initial profile $u(t=0,x,y) = u_0(x,y)$,
    and boundary conditions on the four walls:
    $$
    \begin{aligned}
    u(t,0,y) &= u_{\rm left}(t;y), & u(t,L_x,y) &= u_{\rm right}(t;y), \\
    u(t,x,0) &= u_{\rm bottom}(t;x), & u(t,x,L_y) &= u_{\rm top}(t;x).
    \end{aligned}
    $$
    Now we discretize in both the $x$ and $y$ directions.
    Taking the same step size $a$ in both directions, we obtain the discretized FTCS scheme
    $$
    \begin{aligned}
    u^{n+1}_{i,j} = u^n_{i,j} &+ r \, (u^n_{i+1,j} - 2u^n_{i,j} + u^n_{i-1,j}) \\
    &+ r \, (u^n_{i,j+1} - 2u^n_{i,j} + u^n_{i,j-1}), \qquad i = 1 \ldots N-1, \quad j = 1 \ldots M-1.
    \end{aligned}
    $$
    Here, as before, $r \equiv Dh/a^2$, $N = L_x/a$, $M = L_y/a$, and $u^n_{i,j} = u(t = hn, x = ai, y = aj)$.

    The worst mode is now a checkerboard, with amplification factor $1-8r$: the 2D scheme is stable for $r\le1/4$.
    In $d$ dimensions the condition is $r\le1/(2d)$.

    As an example, take a square steel plate of side 1 cm, initially at 20 °C, with the top edge in contact with hot water
    at 50 °C and the other three edges in a cold bath at 0 °C.
    """)
    return


@app.function
def heat_FTCS_iteration_2D(u, r):
    """Single 2D FTCS time step for the heat equation, with r = D h / a^2."""
    unew = u.copy()  # keeps the boundary values on the four edges
    unew[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + r * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        + r * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
    )
    return unew


@app.cell
def _(T_hi, T_lo, np):
    def plate_initial(N, M):
        """u[i, j] = u(x = a i, y = a j): the plate at 20 C, top edge hot, other edges cold."""
        u = np.full((N + 1, M + 1), 20.0)
        u[0, :] = u[N, :] = u[:, 0] = T_lo
        u[:, M] = T_hi
        return u

    return (plate_initial,)


@app.cell(hide_code=True)
def _(mo):
    h2d_h = mo.ui.dropdown(
        options={"1e-4 s": 1e-4, "5e-4 s": 5e-4, "6e-4 s": 6e-4}, value="1e-4 s", label="Time step $h$:"
    )
    return (h2d_h,)


@app.cell(hide_code=True)
def _(h2d_h, heat_profiles, np, plate_initial, steel_D, steel_L):
    h2d_N = 100
    h2d_r = h2d_h.value * steel_D / (steel_L / h2d_N) ** 2
    h2d_anim_times = np.linspace(0.0, 2.0, 101)
    with np.errstate(all="ignore"):
        h2d_frames = heat_profiles(heat_FTCS_iteration_2D, plate_initial(h2d_N, h2d_N), h2d_h.value, steel_L / h2d_N, steel_D, h2d_anim_times)
    return h2d_N, h2d_anim_times, h2d_frames, h2d_r


@app.cell(hide_code=True)
def _(PlaySlider, h2d_anim_times, mo):
    h2d_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=len(h2d_anim_times) - 1, step=1, interval_ms=120))
    return (h2d_play,)


@app.cell(hide_code=True)
def _(
    T_hi,
    T_lo,
    h2d_N,
    h2d_anim_times,
    h2d_frames,
    h2d_h,
    h2d_play,
    h2d_r,
    mo,
    np,
    plt,
    steel_L,
):
    _k = int(h2d_play.value["value"])
    _u = h2d_frames[_k]
    _y = np.linspace(0.0, steel_L, h2d_N + 1) * 100

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    _im = _ax.imshow(_u.T, vmin=T_lo, vmax=T_hi, origin="lower", extent=[0, steel_L * 100, 0, steel_L * 100], cmap="inferno")
    _ax.axvline(steel_L * 50, color="white", lw=0.8, ls="--")
    _ax.set(xlabel="$x$ [cm]", ylabel="$y$ [cm]", title=f"$t={h2d_anim_times[_k]:.2f}$ s")
    _fig.colorbar(_im, ax=_ax, label="$T$ [°C]")
    _ax2.plot(_y, h2d_frames[0][h2d_N // 2, :], color="0.75", lw=1.0, label="$t=0$")
    _ax2.plot(_y, _u[h2d_N // 2, :], color="crimson", lw=2, label=f"$t={h2d_anim_times[_k]:.2f}$ s")
    _ax2.set(xlabel="$y$ [cm]", ylabel="$T$ [°C]", ylim=(T_lo - 2, T_hi + 2), title="Cut along the middle, $x=L/2$")
    _ax2.legend(loc="upper left")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack(
                [
                    h2d_h,
                    mo.md("**Animate:**"),
                    h2d_play,
                    mo.stat(label="r = Dh/a²", value=f"{h2d_r:.4f}", caption="stable: r ≤ 1/4" if h2d_r <= 0.25 else "unstable: r > 1/4"),
                ],
                justify="center",
                align="center",
                gap=2,
            ),
            _fig,
            mo.md(
                r"""
                With $h=6\cdot10^{-4}$ s, $r=0.255$ is only slightly above the 2D limit $1/4$, yet a checkerboard pattern appears
                after a few tenths of a second and grows without bound. In 1D the same $r$ would be perfectly stable.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Wave equation
    ---

    The wave equation is an example of a second-order linear PDE describing waves and standing wave fields. In one dimension it reads
    $$
    \frac{\partial^2 \phi}{\partial t^2} = v^2 \frac{\partial^2 \phi}{\partial x^2}.
    $$
    Since it is second-order in time, it is supplemented by initial conditions for both $\phi(t=0,x)$ and $\partial_t\phi(t=0,x)$:
    $$
    \phi(t=0,x) = \phi_0(x), \qquad \partial_t\phi(t=0,x) = \psi_0(x).
    $$
    The boundary conditions can be of either Dirichlet form,
    $$
    \phi(t,x=0) = \phi_{\rm left}(t), \qquad \phi(t,x=L) = \phi_{\rm right}(t),
    $$
    or Neumann form,
    $$
    \partial_x\phi(t,x=0) = \phi'_{\rm left}(t), \qquad \partial_x\phi(t,x=L) = \phi'_{\rm right}(t).
    $$
    We shall focus on the Dirichlet form, with the ends of the string held fixed.

    ### Finite difference approach

    As for second-order ODEs, we deal with the second-order time derivative by introducing
    $$
    \psi(t,x) \equiv \frac{\partial \phi}{\partial t},
    $$
    so that we have a system of PDEs that are first-order in $t$:
    $$
    \frac{\partial \phi}{\partial t} = \psi, \qquad \frac{\partial \psi}{\partial t} = v^2 \frac{\partial^2 \phi}{\partial x^2}.
    $$
    We approximate $\partial^2 \phi / \partial x^2$ by the lowest-order central difference, just like for the heat equation,
    and apply the same three time-stepping schemes to $\phi$ and $\psi$ simultaneously.
    Denoting $\phi^n_k = \phi(t = nh,x = ka)$, $\psi^n_k = \psi(t = nh,x = ka)$ and now $r \equiv hv^2/a^2$, we get, for $k = 1 \ldots N-1$:

    | Scheme | $\phi^{n+1}_k =$ | $\psi^{n+1}_k =$ |
    |---|---|---|
    | FTCS | $\phi^{n}_k + h \psi^n_k$ | $\psi^n_k + r \, (\phi^n_{k+1} - 2\phi^n_k + \phi^n_{k-1})$ |
    | Implicit | $\phi^{n}_k + h \psi^{n+1}_k$ | $\psi^n_k + r \, (\phi^{n+1}_{k+1} - 2\phi^{n+1}_k + \phi^{n+1}_{k-1})$ |
    | Crank–Nicolson | $\phi^{n}_k + \frac{h}{2} \left( \psi^{n+1}_k + \psi^{n}_k\right)$ | $\psi^n_k + \frac{r}{2} (\phi^{n+1}_{k+1} - 2\phi^{n+1}_k + \phi^{n+1}_{k-1}) + \frac{r}{2} (\phi^{n}_{k+1} - 2\phi^{n}_k + \phi^{n}_{k-1})$ |

    For the implicit scheme, substituting the first equation into the second one gives a tridiagonal system of linear equations for $\psi^{n+1}_k$:
    $$
    -rh \psi^{n+1}_{k-1} + (1+2rh) \psi^{n+1}_k - rh \psi^{n+1}_{k+1} = \psi^n_k + r \, (\phi^{n}_{k+1} - 2\phi^{n}_k + \phi^{n}_{k-1}),
    $$
    and similarly for Crank–Nicolson:
    $$
    -\frac{rh}{2} \psi^{n+1}_{k-1} + (2+rh) \psi^{n+1}_k - \frac{rh}{2} \psi^{n+1}_{k+1} = 2 \psi^n_k + 2r \, (\phi^{n}_{k+1} - 2\phi^{n}_k + \phi^{n}_{k-1}) + \frac{rh}{2} \, (\psi^{n}_{k+1} - 2\psi^{n}_k + \psi^{n}_{k-1}).
    $$
    Then $\phi^{n+1}_k$ follows from the first equation.
    """)
    return


@app.cell
def _(np, solve_banded):
    def wave_FTCS_iteration(phi, psi, h, r):
        """Single FTCS time step for the wave equation, with r = h v^2 / a^2."""
        phinew, psinew = phi.copy(), psi.copy()  # static Dirichlet boundaries: phi fixed, psi = 0
        phinew[1:-1] = phi[1:-1] + h * psi[1:-1]
        psinew[1:-1] = psi[1:-1] + r * (phi[2:] - 2 * phi[1:-1] + phi[:-2])
        return phinew, psinew

    def wave_implicit_iteration(phi, psi, h, r):
        """Single implicit time step for the wave equation, with r = h v^2 / a^2."""
        N = len(phi) - 1
        phinew, psinew = phi.copy(), psi.copy()
        # Tridiagonal system for psi (psi = 0 at both ends, so no boundary terms)
        ab = np.array([np.full(N - 1, -r * h), np.full(N - 1, 1 + 2 * r * h), np.full(N - 1, -r * h)])
        v = psi[1:N] + r * (phi[2:] - 2 * phi[1:N] + phi[:-2])
        psinew[1:N] = solve_banded((1, 1), ab, v)
        phinew[1:N] = phi[1:N] + h * psinew[1:N]
        return phinew, psinew

    def wave_crank_nicolson_iteration(phi, psi, h, r):
        """Single Crank-Nicolson time step for the wave equation, with r = h v^2 / a^2."""
        N = len(phi) - 1
        phinew, psinew = phi.copy(), psi.copy()
        ab = np.array([np.full(N - 1, -r * h / 2), np.full(N - 1, 2 + r * h), np.full(N - 1, -r * h / 2)])
        v = 2 * psi[1:N] + 2 * r * (phi[2:] - 2 * phi[1:N] + phi[:-2]) + r * h / 2 * (psi[2:] - 2 * psi[1:N] + psi[:-2])
        psinew[1:N] = solve_banded((1, 1), ab, v)
        phinew[1:N] = phi[1:N] + h * (psinew[1:N] + psi[1:N]) / 2
        return phinew, psinew

    def wave_solve(iteration, phi0, psi0, h, nsteps, a, v=1.0):
        """Perform nsteps time steps of size h for the wave equation with speed v."""
        phi, psi = phi0.copy(), psi0.copy()
        r = h * v**2 / a**2
        for _ in range(nsteps):
            phi, psi = iteration(phi, psi, h, r)
        return phi, psi

    def wave_profiles(iteration, phi0, psi0, h, a, v, times):
        """The fields (phi, psi) at the given times."""
        out = []
        phi, psi, current_time = phi0, psi0, 0.0
        for time in times:
            nsteps = round((time - current_time) / h)
            phi, psi = wave_solve(iteration, phi, psi, h, nsteps, a, v)
            out.append((phi, psi))
            current_time += nsteps * h
        return out

    return (
        wave_FTCS_iteration,
        wave_crank_nicolson_iteration,
        wave_implicit_iteration,
        wave_profiles,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We test the three schemes on a string of length $L=1$ with $N=100$ grid divisions and two initial conditions:

    1. **Standing wave:** wave speed $v=0.1$ and $h=10^{-2}$; the string is released at rest from the profile of its
       fundamental mode, $\phi_0(x)=\sin(\pi x/L)$. The exact solution is $\phi(t,x) = \sin(\pi x/L)\cos(\pi v t/L)$, with period $2L/v=20$.
    2. **Struck piano string** *(Exercise 9.5 from M. Newman, Computational Physics)*: $v=100$ m/s; the string is initially
       flat, $\phi_0(x)=0$, and struck by the piano hammer at a distance $d=10$ cm from the end, which gives it the initial velocity profile
       $$
       \psi_0(x) = C\,\frac{x(L-x)}{L^2}\,\exp\!\left[-\frac{(x-d)^2}{2\sigma^2}\right],
       $$
       with $C=1$ m/s and $\sigma=0.3$ m. The exercise asks for $h=10^{-6}$ s; to keep the notebook responsive we use $h=10^{-5}$ s.

    Besides the displacement, we follow the energy of the string,
    $$
    E = \sum_k a\left[\frac{\psi_k^2}{2} + \frac{v^2}{2}\left(\frac{\phi_{k+1}-\phi_k}{a}\right)^2\right],
    $$
    which is conserved by the exact solution.
    """)
    return


@app.cell
def _(np):
    string_L, string_N = 1.0, 100
    string_x = np.linspace(0.0, string_L, string_N + 1)

    # Standing wave, released at rest from the fundamental mode
    _v = 0.1
    _standing = dict(
        v=_v, h=1e-2, t_end=200.0, t_unit=("", 1.0),
        phi0=np.sin(np.pi * string_x / string_L), psi0=np.zeros(string_N + 1),
        exact=lambda t: np.sin(np.pi * string_x / string_L) * np.cos(np.pi * _v * t / string_L),
    )

    # Piano string struck by the hammer at distance d from the end
    _d, _C, _sigma = 0.1, 1.0, 0.3
    _piano = dict(
        v=100.0, h=1e-5, t_end=0.1, t_unit=(" ms", 1e3),
        phi0=np.zeros(string_N + 1),
        psi0=_C * string_x * (string_L - string_x) / string_L**2 * np.exp(-((string_x - _d) ** 2) / (2 * _sigma**2)),
        exact=None,
    )
    string_cases = {"Standing wave": _standing, "Struck piano string": _piano}
    return string_L, string_N, string_cases, string_x


@app.cell(hide_code=True)
def _(np):
    def wave_energy(phi, psi, a, v):
        """Kinetic plus elastic energy of the string, per unit mass density."""
        return a * (0.5 * np.sum(psi**2) + 0.5 * v**2 * np.sum(np.diff(phi) ** 2) / a**2)

    return (wave_energy,)


@app.cell(hide_code=True)
def _(mo, string_cases):
    wave_case = mo.ui.dropdown(options=list(string_cases), value="Standing wave", label="Initial condition:")
    wave_method = mo.ui.dropdown(options=["FTCS", "Implicit", "Crank–Nicolson"], value="Crank–Nicolson", label="Scheme:")
    return wave_case, wave_method


@app.cell(hide_code=True)
def _(
    np,
    string_L,
    string_N,
    string_cases,
    wave_FTCS_iteration,
    wave_case,
    wave_crank_nicolson_iteration,
    wave_energy,
    wave_implicit_iteration,
    wave_method,
    wave_profiles,
):
    _c = string_cases[wave_case.value]
    _schemes = {"FTCS": wave_FTCS_iteration, "Implicit": wave_implicit_iteration, "Crank–Nicolson": wave_crank_nicolson_iteration}
    wave_times = np.linspace(0.0, _c["t_end"], 201)
    with np.errstate(all="ignore"):  # let the FTCS instability show up as overflow
        _out = wave_profiles(_schemes[wave_method.value], _c["phi0"], _c["psi0"], _c["h"], string_L / string_N, _c["v"], wave_times)
        _E = np.array([wave_energy(_phi, _psi, string_L / string_N, _c["v"]) for _phi, _psi in _out])
    wave_frames = [_phi for _phi, _ in _out]
    wave_E = _E / _E[0]
    return wave_E, wave_frames, wave_times


@app.cell(hide_code=True)
def _(PlaySlider, mo, wave_times):
    wave_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=len(wave_times) - 1, step=1, interval_ms=80))
    return (wave_play,)


@app.cell(hide_code=True)
def _(
    method_colors,
    mo,
    np,
    plt,
    string_cases,
    string_x,
    wave_E,
    wave_case,
    wave_frames,
    wave_method,
    wave_play,
    wave_times,
):
    _c = string_cases[wave_case.value]
    _unit, _scale_t = _c["t_unit"]
    _k = int(wave_play.value["value"])
    _t = wave_times[_k]
    _blowup = np.nonzero(~(wave_E < 2.0))[0]  # energy doubled (or overflow)
    # Displacement scale: the largest displacement before any blow-up
    _stable = wave_frames[: _blowup[0]] if len(_blowup) else wave_frames
    _ymax = 1.3 * max(np.max(np.abs(_f)) for _f in _stable)

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    if _c["exact"] is not None:
        _ax.plot(string_x, _c["exact"](_t), "--", color="black", lw=1.0, label="exact")
    _ax.plot(string_x, wave_frames[_k], color=method_colors[wave_method.value], lw=2, label=wave_method.value)
    _ax2.plot(wave_times[: _k + 1] * _scale_t, wave_E[: _k + 1], color=method_colors[wave_method.value], lw=1.5)
    _ax.set(xlim=(0, 1), ylim=(-_ymax, _ymax), xlabel="$x$", ylabel=r"Displacement $\phi$", title=f"$t={_t * _scale_t:.1f}${_unit}")
    _ax.legend(loc="upper right", fontsize=9)
    _ax2.axhline(1.0, color="black", lw=0.8, ls=":")
    _ax2.set(xlim=(0, wave_times[-1] * _scale_t), ylim=(0.0, 2.0), xlabel=f"$t${_unit}", ylabel="$E(t)/E(0)$", title="Energy of the string")
    _fig.tight_layout()
    plt.close(_fig)

    if len(_blowup):
        _stat = mo.stat(label="Blows up at", value=f"t = {wave_times[_blowup[0]] * _scale_t:.3g}{_unit}", caption="energy doubled")
    else:
        _stat = mo.stat(label="Energy left", value=f"{wave_E[-1]:.0%}", caption=f"at t = {wave_times[-1] * _scale_t:g}{_unit}")

    mo.vstack(
        [
            mo.hstack([wave_case, wave_method, mo.md("**Animate:**"), wave_play], justify="center", align="center", gap=2),
            mo.hstack(
                [_fig, mo.vstack([mo.stat(label="Energy now", value=f"{wave_E[_k]:.3g}", caption="E(t)/E(0)"), _stat])],
                widths=[0.8, 0.2],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Standing wave.** At $t=10$ and $t=20$ (half and one full period) the exact solution is $\mp\sin(\pi x/L)$. FTCS develops
    a zig-zag around $t=18$ that then explodes; the implicit scheme is stable but the amplitude slowly decays, losing 18% of the
    energy by $t=200$; Crank–Nicolson conserves the energy exactly, and for this smooth mode it also keeps the phase:
    after 10 periods it still agrees with the exact solution to $10^{-5}$.

    The reason is the same as for the ODE of a harmonic oscillator. Each Fourier mode of the string oscillates with
    angular frequency $\omega_q = (2v/a)\sin(qa/2)$, and in one step its amplitude is multiplied by
    $$
    |A|_{\rm FTCS} = \sqrt{1+h^2\omega_q^2} > 1, \qquad |A|_{\rm implicit} = \frac{1}{\sqrt{1+h^2\omega_q^2}} < 1, \qquad |A|_{\rm CN} = 1.
    $$
    FTCS is **unconditionally unstable** for the wave equation, whatever $h$. For the smooth fundamental mode the growth is
    slow, $h\omega\approx 0.003$, but for the zig-zag mode, created by round-off errors, $h\omega_{\max}=2vh/a=0.2$:
    it grows by 2% per step, and needs about 1900 steps (19 time units) to grow from $10^{-16}$ to order 1.

    **Piano string.** The pulse splits into two halves travelling in opposite directions, which are reflected upside down at the
    fixed ends; after one period $2L/v=20$ ms the string is back to its initial (flat) shape. Here too $h\omega_{\max}=2vh/a=0.2$,
    and since the pulse already contains some short wavelengths FTCS blows up even sooner, after less than 10 ms; a smaller $h$
    delays the blow-up but never prevents it. The implicit and
    Crank–Nicolson schemes remain stable, but both distort the pulse, whose sharp features are made of short wavelengths:
    the implicit scheme damps them, while in Crank–Nicolson they travel too slowly (numerical **dispersion**) and lag behind as ripples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: finite difference methods for PDEs

    **Boundary value problems** (Laplace/Poisson equation): the discretized equation is a large sparse linear system.

    | Method | Iterations for an $M\times M$ grid | Strengths | Weaknesses |
    |---|---|---|---|
    | Jacobi | $\propto M^2$ | trivial, vectorizes easily | very slow on fine grids; small changes $\neq$ small error |
    | Gauss–Seidel | $\propto M^2$ (half of Jacobi) | in place, one array | sequential sweep (Python loops) |
    | Gauss–Seidel with overrelaxation | $\propto M$ at $\omega_{\rm opt}$ | much faster | needs a good $\omega$; diverges for $\omega\ge1$ |

    **Initial value problems** (heat and wave equations): discretizing space turns the PDE into a large system of ODEs (*method of lines*).

    | Scheme | Order in $h$ | Heat equation | Wave equation | Cost per step |
    |---|---|---|---|---|
    | FTCS | 1 | stable for $r=Dh/a^2\le1/(2d)$ in $d$ dimensions | always unstable | cheapest, explicit |
    | Implicit | 1 | always stable, damps the zig-zag | stable, but damps the waves | tridiagonal solve, $O(N)$ |
    | Crank–Nicolson | 2 | always stable, slowly damped zig-zag for large $r$ | stable, conserves energy | tridiagonal solve, $O(N)$ |

    In practice, the linear systems of boundary value problems are solved with multigrid and Krylov methods, the method of lines hands the ODE system to a stiff solver (`scipy.integrate.solve_ivp` with `BDF` or `Radau`),
    and irregular geometries call for the finite element method (e.g. FEniCS).
    """)
    return


if __name__ == "__main__":
    app.run()
