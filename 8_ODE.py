import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve
    from wigglystuff import FormulaAnimation, PlaySlider

    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return FormulaAnimation, PlaySlider, fsolve, mo, np, pd, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ordinary differential equations

    An ordinary differential equation is an equation of the type
    $$
    \frac{dx}{dt} = f(x,t),
    $$
    where $f(x,t)$ is some function.
    Together with the initial condition $x(t=0) = x_0$ it defines
    the $x(t)$ dependence at $t>0$.

    In many physical applications $t$ plays the role of the time variable (classical mechanics problems),
    although this is not always the case.

    Some differential equations can be solved explicitly, the solution to
    $$
    \frac{dx}{dt} = \sin t,
    $$
    reads
    $$
    x(t) = x_0 + \int_0^t \sin t' \, dt' = x_0 + 1 - \cos t.
    $$

    In the general case, however, the solution cannot be obtained analytically.
    Let us consider for example
    $$
    \frac{dx}{dt} = -x^3 + \sin t.
    $$

    In this case one resorts to **numerical methods**: starting from $x_0$ they advance the solution
    in small steps $h$ along $t$.
    """)
    return


@app.cell(hide_code=True)
def _():
    method_colors = {
        "Euler": "darkorange",
        "RK2": "seagreen",
        "RK4": "#1d4ed8",
        "Implicit Euler": "0.4",
        "Leapfrog": "purple",
    }
    return (method_colors,)


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    euler_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Euler's method from a Taylor expansion",
            steps=[
                {
                    "tex": r"x(t+h) = x(t) + h\,\frac{dx}{dt} + \mathcal{O}(h^2)",
                    "note": "Taylor expand x(t+h) around t.",
                },
                {
                    "tex": r"x(t+h) = x(t) + h\,f[x(t),t] + \mathcal{O}(h^2)",
                    "note": "Use the ODE: dx/dt = f(x,t).",
                },
                {
                    "tex": r"x(t+h) \approx x(t) + h\,f[x(t),t]",
                    "note": "Neglect the high-order terms in h.",
                },
                {
                    "tex": r"x_{n+1} = x_n + h\,f(x_n,t_n)",
                    "note": "Turn it into an iteration, starting from t = 0.",
                },
            ],
            height=280,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 1. Euler's method
                ---

                Suppose we have an ODE:
                $$
                \frac{dx}{dt} = f(x,t).
                $$
                The Taylor expansion expresses $x(t+h)$ in terms of $x(t)$.
                """
            ),
            euler_anim,
            mo.md(
                r"""
                We can iteratively apply this relation starting from $t = 0$ to evaluate $x(t)$ at $t > 0$.
                This is the essence of *Euler's method* — the simplest method for solving ODEs numerically.
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def euler_step(f, x, t, h):
        """Perform a single step h using Euler's scheme."""
        return x + h * f(x, t)

    def ode_solve(step, f, x0, t0, h, nsteps):
        """Solve dx/dt = f(x,t) from t0 to t0 + nsteps*h with a fixed-step method.

        step: the one-step scheme, step(f, x, t, h) -> x(t+h).
        x0: initial value, either a number or a vector (system of ODEs).
        Returns the arrays t and x (one row per time for a vector x).
        """
        t = t0 + h * np.arange(nsteps + 1)
        x = np.zeros((nsteps + 1,) + np.shape(x0))
        x[0] = x0
        for i in range(nsteps):
            x[i + 1] = step(f, x[i], t[i], h)
        return t, x

    return euler_step, ode_solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us apply Euler's method to the differential equation
    $$
    \frac{dx}{dt} = -x^3 + \sin t
    $$
    with the initial condition $x_0 = 0$ at $t_0 = 0$, from $t = 0$ to $t = 10$.
    """)
    return


@app.cell
def _(np):
    def f_test(x, t):
        return -(x**3) + np.sin(t)

    test_t0, test_tend, test_x0 = 0.0, 10.0, 0.0
    return f_test, test_t0, test_tend, test_x0


@app.cell(hide_code=True)
def _(mo):
    euler_N = mo.ui.slider(5, 100, step=1, value=10, label="Number of steps $N$")
    return (euler_N,)


@app.cell(hide_code=True)
def _(euler_N, euler_step, f_test, ode_solve, test_t0, test_tend, test_x0):
    euler_h = (test_tend - test_t0) / euler_N.value
    euler_t, euler_x = ode_solve(euler_step, f_test, test_x0, test_t0, euler_h, euler_N.value)
    return euler_h, euler_t, euler_x


@app.cell(hide_code=True)
def _(euler_step, f_test, ode_solve, test_t0, test_tend, test_x0):
    euler_fine_t, euler_fine_x = ode_solve(euler_step, f_test, test_x0, test_t0, (test_tend - test_t0) / 1000, 1000)
    return euler_fine_t, euler_fine_x


@app.cell(hide_code=True)
def _(PlaySlider, euler_N, mo):
    euler_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=euler_N.value, step=1, interval_ms=500))
    return (euler_play,)


@app.cell(hide_code=True)
def _(
    euler_N,
    euler_fine_t,
    euler_fine_x,
    euler_h,
    euler_play,
    euler_t,
    euler_x,
    f_test,
    mo,
    np,
    plt,
):
    _n = min(int(euler_play.value["value"]), euler_N.value)

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    _ax.plot(euler_fine_t, euler_fine_x, color="crimson", lw=2, label="Euler, $N=1000$")
    _ax.plot(euler_t[: _n + 1], euler_x[: _n + 1], "o-", color="steelblue", ms=5, lw=1.2, label=f"Euler, $N={euler_N.value}$")
    if _n < euler_N.value:
        # the tangent that produces the next point
        _tt = np.array([euler_t[_n], euler_t[_n + 1]])
        _ax.plot(_tt, euler_x[_n] + f_test(euler_x[_n], euler_t[_n]) * (_tt - euler_t[_n]), "--", color="#1d4ed8", lw=1.6, label="Next step: the tangent")
    _ax.plot([euler_t[_n]], [euler_x[_n]], "o", color="#1d4ed8", ms=8, zorder=5, label="Current point")
    _ax.set(xlabel="$t$", ylabel="$x$", title=f"Euler's method: step {_n}")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md("Choose the number of steps, then press ▶ to follow the tangents one step at a time."),
            mo.hstack([euler_N, mo.md("**Step:**"), euler_play], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Step size h", value=f"{euler_h:.4f}"),
                            mo.stat(label="Current xₙ", value=f"{euler_x[_n]:.6f}"),
                            mo.stat(label="x(10) with N = 1000", value=f"{euler_fine_x[-1]:.6f}", caption=f"Euler N={euler_N.value}: {euler_x[-1]:.6f}"),
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
    The error of Euler's scheme is proportional to $h^2$ at each step. Since the total number of steps is $N = (t_{\rm end} - t_0)/h$, the total accumulated error is linear in $h$, i.e. of order $\mathcal{O}(h)$.
    This is checked quantitatively in Section 3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Midpoint (second-order Runge–Kutta) method
    ---

    Better schemes with a smaller error can be achieved with little extra work.

    Euler's method essentially corresponds to approximating the derivative $dx/dt$ with a forward difference
    $$
    \frac{dx}{dt} = f(x,t) \approx \frac{x(t+h) - x(t)}{h} + \mathcal{O}(h).
    $$

    Recall that better accuracy in numerical differentiation can be obtained if one uses a central (midpoint) difference,
    $$
    f(x,t+h/2) \approx \frac{x(t+h) - x(t)}{h} + \mathcal{O}(h^2).
    $$

    Let us exploit this idea for ODEs.
    Consider the Taylor expansions of $x(t+h)$ and $x(t)$ around $x(t+h/2)$:

    \begin{align}
    x(t+h) = x(t + h/2) + \frac{1}{2} h \left( \frac{dx}{dt} \right)_{t+h/2} + \frac{1}{8} h^2 \left( \frac{d^2x}{dt^2} \right)_{t+h/2} + \mathcal{O}(h^3),\\
    x(t) = x(t + h/2) - \frac{1}{2} h \left( \frac{dx}{dt} \right)_{t+h/2} + \frac{1}{8} h^2 \left( \frac{d^2x}{dt^2} \right)_{t+h/2} + \mathcal{O}(h^3).
    \end{align}

    Subtracting the second equation from the first one we get

    \begin{align}
    x(t+h) & = x(t) + h \left( \frac{dx}{dt} \right)_{t+h/2} + \mathcal{O}(h^3),\\
    & = x(t) + h f[x(t+h/2),t+h/2] + \mathcal{O}(h^3).
    \end{align}

    How to calculate $x(t+h/2)$? Use Euler's scheme
    $$
    x(t+h/2) = x(t) + \frac{1}{2}h f(x,t) + \mathcal{O}(h^2).
    $$

    It can be shown using a Taylor expansion that
    $$
    f[x(t+h/2),t+h/2] = f\left[x(t) + \frac{1}{2}h f(x,t), t + \frac{1}{2}h\right] + \mathcal{O}(h^2).
    $$

    Therefore,
    $$
    x(t+h) = x(t) + h f\left[x(t) + \frac{1}{2}h f(x,t), t + \frac{1}{2}h\right]  + \mathcal{O}(h^3),
    $$
    which can iteratively be written as

    \begin{align}
    k_1 & = h \, f(x,t), \\
    k_2 & = h \, f(x+k_1/2,t+h/2), \\
    x(t+h) & = x(t) + k_2~.
    \end{align}

    This is the midpoint (or second-order Runge–Kutta) method, which has a local (per step) error of order $\mathcal{O}(h^3)$.
    """)
    return


@app.function
def rk2_step(f, x, t, h):
    """Perform a single step h using the 2nd order Runge-Kutta (midpoint) scheme."""
    k1 = h * f(x, t)
    k2 = h * f(x + k1 / 2.0, t + h / 2.0)
    return x + k2


@app.cell(hide_code=True)
def _(f_test, method_colors, ode_solve, plt, test_t0, test_tend, test_x0):
    def compare_methods(methods, N, ref_name, ref_step):
        """Fixed-step methods with N steps, against a fine solution with 1000 steps."""
        h = (test_tend - test_t0) / N
        t_ref, x_ref = ode_solve(ref_step, f_test, test_x0, test_t0, (test_tend - test_t0) / 1000, 1000)
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.plot(t_ref, x_ref, color="crimson", lw=2, label=f"{ref_name}, $N=1000$")
        for name, step in methods.items():
            t, x = ode_solve(step, f_test, test_x0, test_t0, h, N)
            ax.plot(t, x, "o", ms=4, color=method_colors[name], label=f"{name}, $N={N}$")
        ax.set(xlabel="$t$", ylabel="$x$", title=f"Fixed-step methods with $h={h:.3f}$")
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.close(fig)
        return fig

    return (compare_methods,)


@app.cell(hide_code=True)
def _(mo):
    rk2_N = mo.ui.slider(5, 100, step=1, value=20, label="Number of steps $N$")
    return (rk2_N,)


@app.cell(hide_code=True)
def _(compare_methods, euler_step, mo, rk2_N):
    mo.vstack(
        [
            rk2_N,
            compare_methods({"Euler": euler_step, "RK2": rk2_step}, rk2_N.value, "RK2", rk2_step),
            mo.md(
                r"""
                With the same number of steps, RK2 follows the fine solution far more closely than Euler's method.
                The price is two evaluations of $f$ per step instead of one.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The fourth-order Runge–Kutta method
    ---

    The above logic can be generalized to cancel higher-order error terms in various powers of $h$.
    This requires more evaluations of the function $f(x,t)$ at intermediate points.
    The following classical 4th order Runge–Kutta method is often considered to be a sweet spot.
    It corresponds to the following scheme:

    \begin{align}
    k_1 & = h \, f(x,t), \\
    k_2 & = h \, f(x+k_1/2,t+h/2), \\
    k_3 & = h \, f(x+k_2/2,t+h/2), \\
    k_4 & = h \, f(x+k_3,t+h), \\
    x(t+h) & = x(t) + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)~.
    \end{align}
    """)
    return


@app.function
def rk4_step(f, x, t, h):
    """Perform a single step h using the 4th order Runge-Kutta scheme."""
    k1 = h * f(x, t)
    k2 = h * f(x + k1 / 2.0, t + h / 2.0)
    k3 = h * f(x + k2 / 2.0, t + h / 2.0)
    k4 = h * f(x + k3, t + h)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


@app.cell(hide_code=True)
def _(mo):
    rk4_N = mo.ui.slider(5, 100, step=1, value=20, label="Number of steps $N$")
    return (rk4_N,)


@app.cell(hide_code=True)
def _(compare_methods, euler_step, mo, rk4_N):
    mo.vstack(
        [
            rk4_N,
            compare_methods({"Euler": euler_step, "RK2": rk2_step, "RK4": rk4_step}, rk4_N.value, "RK4", rk4_step),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    euler_step,
    f_test,
    mo,
    np,
    ode_solve,
    pd,
    plt,
    solve_ivp,
    test_t0,
    test_tend,
    test_x0,
):
    # A reference value for x(t_end), converged to machine precision
    _ref = solve_ivp(lambda t, x: f_test(x, t), (test_t0, test_tend), [test_x0], method="DOP853", rtol=1e-13, atol=1e-14)
    _x_exact = _ref.y[0, -1]

    _Ns = np.array([20, 50, 100, 200, 500, 1000, 2000])
    _hs = (test_tend - test_t0) / _Ns
    _steps = {"Euler": euler_step, "RK2": rk2_step, "RK4": rk4_step}
    _errors = {
        _name: np.array([abs(ode_solve(_step, f_test, test_x0, test_t0, _h, _N)[1][-1] - _x_exact) for _N, _h in zip(_Ns, _hs)])
        for _name, _step in _steps.items()
    }
    _orders = {"Euler": 1, "RK2": 2, "RK4": 4}
    _colors = {"Euler": "darkorange", "RK2": "seagreen", "RK4": "#1d4ed8"}

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _name, _e in _errors.items():
        _ax.loglog(_hs, _e, "o-", ms=4, lw=1.2, color=_colors[_name], label=_name)
        _p = _orders[_name]
        _ax.loglog(_hs, _e[2] * (_hs / _hs[2]) ** _p, "--", lw=0.9, color="0.5")  # guide of slope p
        _ax.annotate(f"$h^{_p}$", (_hs[0], _e[2] * (_hs[0] / _hs[2]) ** _p), xytext=(4, 4), textcoords="offset points", color="0.4")
    _ax.set(xlabel="Step size $h$", ylabel="$|x(10) - x_{\\rm exact}(10)|$", title="Global error at $t=10$")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    plt.close(_fig)

    _rows = [
        {"N": int(_N), "h": _h, **{f"{_name} error": _errors[_name][_i] for _name in _steps}}
        for _i, (_N, _h) in enumerate(zip(_Ns, _hs))
    ]
    _fit = {_name: np.polyfit(np.log(_hs), np.log(_e), 1)[0] for _name, _e in _errors.items()}

    mo.vstack(
        [
            mo.md(
                r"""
                ### Order of convergence

                A method of order $p$ has a global error $\propto h^p$, a straight line of slope $p$ on a log–log plot.
                Euler, RK2 and RK4 have $p=1, 2, 4$: halving $h$ reduces the error by a factor $2$, $4$, and $16$ respectively.
                """
            ),
            _fig,
            mo.md("Fitted slopes: " + ", ".join(f"**{_n}** {_s:.2f}" for _n, _s in _fit.items()) + "."),
            mo.accordion({"Error table": mo.ui.table(pd.DataFrame(_rows), selection=None)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    adaptive_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="An error estimate for RK4 from step doubling",
            steps=[
                {
                    "tex": r"x(t+2h) = x_1 + 2ch^5",
                    "note": "Two RK4 steps of size h give x1; each step has error c h^5.",
                },
                {
                    "tex": r"x(t+2h) = x_2 + 32ch^5",
                    "note": "One RK4 step of size 2h gives x2; its error is c (2h)^5.",
                },
                {
                    "tex": r"x_1 - x_2 = 30\,c h^5",
                    "note": "Subtract the two expressions.",
                },
                {
                    "tex": r"\epsilon = |ch^5| = \frac{|x_1 - x_2|}{30}",
                    "note": "The error of a single RK4 step of size h.",
                },
                {
                    "tex": r"h' = h \left( \frac{30\,h\,\delta}{|x_1 - x_2|} \right)^{1/4}",
                    "note": "A step h' has error ε (h'/h)^5; demand that it equals h' δ, the target accuracy per step.",
                },
            ],
            height=300,
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 4. Adaptive time step
                ---

                It can be useful to control the time step $h$ adaptively, such that it corresponds to the desired accuracy of the calculation.
                If $h$ is too small, we waste computing resources by making too many calculations.
                On the other hand, if $h$ is too large, we will not achieve the desired accuracy.

                To solve this problem we can introduce an error estimate at each step, and adjust the time step accordingly.
                One way to do so is to use a combination of two Runge–Kutta methods of different order.
                The difference between the two gives an error estimate for the present step size, which can then be adjusted as necessary.
                The [Runge–Kutta–Fehlberg method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method) is one popular example combining 4th and 5th order Runge–Kutta methods.

                Here we consider another approach applied to the 4th order Runge–Kutta method.
                Recall that the error for one time step $h$ is of order $ch^5$.
                """
            ),
            adaptive_anim,
            mo.md(
                r"""
                If the desired accuracy per unit time is $\delta$, the desired accuracy per time step is $h' \delta$.
                If the estimated step $h'$ is smaller than $h$ the step is repeated with $h'$; otherwise it is accepted
                and the next step is made with a larger $h$ (at most doubled).
                """
            ),
        ]
    )
    return


@app.cell
def _(np):
    def distance_norm(x1, x2):
        """Default error between two state vectors: the length of the difference."""
        return np.linalg.norm(np.atleast_1d(x1 - x2))

    def ode_rk4_adaptive(f, x0, t0, h0, tmax, delta=1e-6, distance=distance_norm):
        """Solve dx/dt = f(x,t) from t0 to tmax using RK4 with an adaptive time step.

        h0: the initial time step.
        delta: the desired accuracy per unit time.
        distance: how to measure the error between two state vectors.
        Returns the arrays t and x.
        """
        ts = [t0]
        xs = [np.asarray(x0, dtype=float)]
        h, t, i = h0, t0, 0
        while t < tmax:
            if t + h >= tmax:  # the last step lands exactly on tmax
                ts.append(tmax)
                xs.append(rk4_step(f, xs[i], ts[i], tmax - t))
                break
            x1 = rk4_step(f, xs[i], ts[i], h)
            x1 = rk4_step(f, x1, ts[i] + h, h)  # two steps of size h
            x2 = rk4_step(f, xs[i], ts[i], 2 * h)  # one step of size 2h
            error = distance(x1, x2)
            rho = 2.0**4 if error == 0.0 else 30.0 * h * delta / error
            if rho < 1.0:
                h = h * rho ** (1 / 4.0)  # too inaccurate: redo with a smaller step
            else:
                if t + 2.0 * h < tmax:
                    xs.append(x1)
                    ts.append(t + 2 * h)
                    t = t + 2 * h
                else:
                    xs.append(rk4_step(f, xs[i], ts[i], h))
                    ts.append(t + h)
                    t = t + h
                i = i + 1
                h = min(2.0 * h, h * rho ** (1 / 4.0))
        return np.array(ts), np.array(xs)

    return distance_norm, ode_rk4_adaptive


@app.cell(hide_code=True)
def _(mo):
    adapt_delta = mo.ui.slider(
        steps=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], value=1e-3, show_value=True, label=r"Accuracy per unit time $\delta$"
    )
    return (adapt_delta,)


@app.cell(hide_code=True)
def _(
    adapt_delta,
    f_test,
    mo,
    ode_rk4_adaptive,
    ode_solve,
    plt,
    test_t0,
    test_tend,
    test_x0,
):
    _t, _x = ode_rk4_adaptive(f_test, test_x0, test_t0, (test_tend - test_t0) / 10, test_tend, adapt_delta.value)
    _t_ref, _x_ref = ode_solve(rk4_step, f_test, test_x0, test_t0, (test_tend - test_t0) / 1000, 1000)

    _fig, (_ax, _ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    _ax.plot(_t_ref, _x_ref, color="crimson", lw=2, label="RK4, $N=1000$")
    _ax.plot(_t, _x, "o", color="steelblue", ms=5, label="Adaptive RK4")
    _ax.set(ylabel="$x$", title=r"Adaptive RK4, $\delta=$" + f"{adapt_delta.value:g}")
    _ax.legend(loc="lower right")
    _dt = _t[1:] - _t[:-1]
    _ax2.plot(0.5 * (_t[:-1] + _t[1:]), _dt, "o-", color="steelblue", ms=4, lw=1)
    _ax2.set(xlabel="$t$", ylabel="step size $h$")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            adapt_delta,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Adaptive steps", value=f"{len(_t) - 1}", caption="fixed-step reference: 1000"),
                            mo.stat(label="Smallest step", value=f"{_dt.min():.4f}"),
                            mo.stat(label="Largest step", value=f"{_dt.max():.4f}"),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.md(
                r"""
                The steps are shortest around the maxima and minima of $x(t)$, where the curve bends most sharply, and
                longest on the steeper, nearly straight stretches in between: the error of an RK4 step depends on the
                higher derivatives of the solution, not on the slope itself. Lowering $\delta$ increases the number of steps,
                roughly as $\delta^{-1/4}$.

                The error estimate assumes that $h$ is small. With the loosest values of $\delta$ the first, long steps break
                this assumption: at $\delta=0.01$ the first accepted point ($t=2$) is off by 0.08, four times the accuracy
                it was supposed to reach (0.02), while for $\delta\le10^{-3}$ the error stays within about its budget.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Implicit methods and stiff equations
    ---

    Consider the following ODE
    $$
    \frac{dx}{dt} = -15x,
    $$
    with $x(0) = 1$.
    The solution is
    $$
    x(t) = e^{-15t},
    $$
    which tends to zero as $t \to \infty$.

    Apply Euler's scheme:
    $$
    x_{n+1} = x_n - 15 h x_n = (1 - 15 h) x_n = (1-15h)^n x_0,
    $$
    where $x_n \equiv x(nh)$.
    For $h > 2/15$ we have $|x_{n+1}| > |x_n|$, so $|x_n| \to \infty$ as $n \to \infty$, although the true solution tends to zero.

    This is an example of **instability** of an ODE solver, which is typical for explicit solvers. The same happens for
    the 4th-order Runge–Kutta method, for $h > 2.785/15 \approx 0.186$.
    Once the fast transient $e^{-15t}$ has died out the solution is essentially zero and would be easy to follow
    accurately with large steps, yet stability still forces $h<2/15$: the step size is limited by stability,
    not by the desired accuracy. Equations with this property are called **stiff**.
    Implicit methods, introduced next, remove the restriction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implicit methods

    In implicit methods, the value of the variable at the next time step itself enters the r.h.s.
    For example, implicit Euler's method reads
    $$
    x(t+h) = x(t) + h f[x(t+h),t+h].
    $$

    Applied to the stiff equation above we have
    $$
    x_{n+1} = x_n - 15 h x_{n+1},
    $$
    thus
    $$
    x_{n+1} = \frac{x_n}{1+15h} = \frac{x_0}{(1+15h)^n}.
    $$
    Here $x_{n+1} \to 0$ as $n \to \infty$ for **any** $h>0$: the method is unconditionally stable.

    Stability is not accuracy, though. Implicit Euler is still only first-order accurate: for $h=1/2$ every step
    multiplies $x$ by $1/(1+7.5)\approx0.12$ instead of $e^{-7.5}\approx5\times10^{-4}$, so the numerical solution
    decays, but far too slowly. Large steps are *safe*, not *precise*.

    The drawback of implicit methods is that one has to solve an algebraic equation at each time step to determine
    $x_{n+1}$ [or $x(t+h)$]. For this linear equation it was done by hand above; for a non-linear $f$ it needs a
    root finder, such as Newton's method from the notebook on non-linear equations. The code below simply hands the equation
    $y = x_n + h f(y, t_{n+1})$ to `scipy.optimize.fsolve`; since the equation is linear here, it reproduces the
    formula above exactly.
    """)
    return


@app.cell
def _(fsolve, np):
    def f_stiff(x, t):
        return -15.0 * x

    def implicit_euler_step(f, x, t, h):
        """Perform a single implicit Euler step: solve y = x + h f(y, t+h) for y = x(t+h)."""
        y = fsolve(lambda u: u - x - h * f(u, t + h), x + h * f(x, t))  # start from the explicit Euler guess
        return y.reshape(np.shape(x))

    return f_stiff, implicit_euler_step


@app.cell(hide_code=True)
def _(mo):
    stiff_method = mo.ui.dropdown(options=["Euler", "RK4", "Implicit Euler"], value="Euler", label="Method:")
    stiff_h = mo.ui.multiselect(
        options={"1/2": 0.5, "1/4": 0.25, "1/8": 0.125, "1/16": 1 / 16, "1/32": 1 / 32},
        value=["1/4", "1/8", "1/16"],
        label="Step sizes $h$:",
    )
    return stiff_h, stiff_method


@app.cell(hide_code=True)
def _(
    euler_step,
    f_stiff,
    implicit_euler_step,
    method_colors,
    mo,
    np,
    ode_solve,
    plt,
    stiff_h,
    stiff_method,
):
    _steps = {"Euler": euler_step, "RK4": rk4_step, "Implicit Euler": implicit_euler_step}
    _limits = {"Euler": "h < 2/15 ≈ 0.133", "RK4": "h < 2.785/15 ≈ 0.186", "Implicit Euler": "none: stable for any h > 0"}

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    _tref = np.linspace(0.0, 1.0, 200)
    _ax.plot(_tref, np.exp(-15.0 * _tref), color="crimson", lw=2, label="exact")
    for _h, _alpha in zip(sorted(stiff_h.value, reverse=True), [1.0, 0.75, 0.55, 0.4, 0.3]):
        _N = round(1.0 / _h)
        _t, _x = ode_solve(_steps[stiff_method.value], f_stiff, 1.0, 0.0, _h, _N)
        _ax.plot(_t, _x, "o-", ms=5, lw=1, alpha=_alpha, color=method_colors[stiff_method.value], label=f"{stiff_method.value}, $h={_h:.4g}$")
    _ax.set_ylim((-1.0, 3.0) if stiff_method.value == "Implicit Euler" else (-3.0, 8.0))
    _ax.set(xlabel="$t$", ylabel="$x$", title=f"Stiff ODE $dx/dt=-15x$: {stiff_method.value}")
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                "Choose a method and a few step sizes. The explicit methods blow up once $h$ exceeds their "
                "stability limit; implicit Euler never does, but for large $h$ it is stable and inaccurate."
            ),
            mo.hstack([stiff_method, stiff_h], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.stat(label="Stability condition", value=_limits[stiff_method.value]),
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
    ## 6. Systems of ODEs
    ---

    A system of $N$ first-order ODEs
    $$
    \frac{d x_1}{dt} = f_1(x_1,\ldots,x_N,t),\\
    \frac{d x_2}{dt} = f_2(x_1,\ldots,x_N,t),\\
    \ldots \\
    \frac{d x_N}{dt} = f_N(x_1,\ldots,x_N,t),
    $$
    can be written in vector notation as
    $$
    \frac{d \mathbf{x}}{dt} = \mathbf{f}(\mathbf{x},t).
    $$

    All the methods we covered have exactly the same structure for systems of ODEs, applied to vectors:

    **Euler method**
    $$
    \mathbf{x}(t+h) = \mathbf{x}(t) + h \, \mathbf{f}[\mathbf{x}(t),t]~.
    $$

    **RK2**

    \begin{align}
    \mathbf {k_1} & = h \, \mathbf{f}(\mathbf{x},t), \\
    \mathbf {k_2} & = h \, \mathbf {f}(\mathbf {x}+\mathbf {k_1}/2,t+h/2), \\
    \mathbf{x}(t+h) & = \mathbf {x}(t) + \mathbf {k_2}~.
    \end{align}

    **RK4**

    \begin{align}
    \mathbf{k_1} & = h \, \mathbf{f}(\mathbf{x},t), \\
    \mathbf{k_2} & = h \, \mathbf{f}(\mathbf{x}+\mathbf{k_1}/2,t+h/2), \\
    \mathbf{k_3} & = h \, \mathbf{f}(\mathbf{x}+\mathbf{k_2}/2,t+h/2), \\
    \mathbf{k_4} & = h \, \mathbf{f}(\mathbf{x}+\mathbf{k_3},t+h), \\
    \mathbf{x}(t+h) & = \mathbf{x}(t) + \frac{1}{6} (\mathbf{k_1} + 2\mathbf{k_2} + 2\mathbf{k_3} + \mathbf{k_4})~.
    \end{align}

    Because `euler_step`, `rk2_step`, `rk4_step` and `ode_solve` only use arithmetic on `x`, they already work
    unchanged when `x` is a NumPy array: no separate "multi-dimensional" versions are needed.

    As an example, take the system

    \begin{align}
    \frac{dx}{dt} & = xy - x,\\
    \frac{dy}{dt} & = y - xy + (\sin t)^2,
    \end{align}

    with $x(0)=y(0)=1$.
    """)
    return


@app.cell
def _(np):
    def f_system(xin, t):
        x, y = xin
        return np.array([x * y - x, y - x * y + np.sin(t) ** 2])

    return (f_system,)


@app.cell(hide_code=True)
def _(mo):
    system_method = mo.ui.dropdown(options=["Euler", "RK2", "RK4"], value="Euler", label="Method:")
    system_N = mo.ui.slider(20, 400, step=10, value=100, label="Number of steps $N$")
    return system_N, system_method


@app.cell(hide_code=True)
def _(
    euler_step,
    f_system,
    method_colors,
    mo,
    ode_solve,
    plt,
    system_N,
    system_method,
):
    _steps = {"Euler": euler_step, "RK2": rk2_step, "RK4": rk4_step}
    _tend = 10.0
    _t, _x = ode_solve(_steps[system_method.value], f_system, [1.0, 1.0], 0.0, _tend / system_N.value, system_N.value)
    _t_ref, _x_ref = ode_solve(rk4_step, f_system, [1.0, 1.0], 0.0, _tend / 5000, 5000)

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    _ax.plot(_t_ref, _x_ref[:, 0], color="crimson", lw=2, label="$x$ (RK4, $N=5000$)")
    _ax.plot(_t_ref, _x_ref[:, 1], color="black", lw=1.2, label="$y$ (RK4, $N=5000$)")
    _ax.plot(_t, _x[:, 0], "o", ms=3, color=method_colors[system_method.value], label=f"$x$ ({system_method.value})")
    _ax.plot(_t, _x[:, 1], "s", ms=3, color="steelblue", label=f"$y$ ({system_method.value})")
    _ax.set(xlabel="$t$", ylabel="$x,\\ y$", title=f"ODE system solved with {system_method.value}, $N={system_N.value}$")
    _ax.legend(loc="upper right", ncol=2)
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([system_method, system_N], justify="center", align="center", gap=2),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Second-order ODEs: the non-linear pendulum
    ---

    A system of $N$ 2nd-order ODEs
    $$
    \frac{d^2 \mathbf{x}}{dt^2} = \mathbf{f}(\mathbf{x}, d\mathbf{x}/dt,t),
    $$
    can be written as a system of $2N$ 1st-order ODEs by denoting $d\mathbf{x}/dt = \mathbf{v}$:

    \begin{align*}
    \frac{d \mathbf{x}}{dt} &= \mathbf{v},\\
    \frac{d \mathbf{v}}{dt} &= \mathbf{f}(\mathbf{x}, \mathbf{v},t),
    \end{align*}

    and solved using standard methods.

    This is particularly relevant for classical mechanics problems since Newton/Lagrange equations of motion correspond to a system of 2nd-order ODEs.

    #### Simple pendulum

    The equation of motion for the non-linear pendulum reads
    $$
    m L \frac{d^2 \theta}{dt^2} = - m g \sin \theta.
    $$

    Denoting $d\theta / d t = \omega$, one can rewrite this 2nd-order differential equation as a system of two 1st-order equations

    \begin{align}
    \frac{d \theta}{dt} & = \omega, \\
    \frac{d \omega}{dt} & = -\frac{g}{L} \sin \theta~,
    \end{align}

    which can be solved using standard first-order ODE methods.

    For small angles $\theta$ one can approximate $\sin \theta \approx \theta$ and solve the equation
    $$
    \frac{d^2 \theta}{dt^2} \approx - \frac{g}{L} \theta.
    $$
    For a pendulum initially at angle $\theta_0$ and at rest, the analytic solution for small angles reads
    $$
    \theta(t) \approx \theta_0 \cos\left( \sqrt{\frac{g}{L}} t \right).
    $$
    """)
    return


@app.cell
def _(np):
    g_earth = 9.81  # m/s^2
    L_pend = 0.1  # m

    def f_pendulum(xin, t):
        theta, omega = xin
        return np.array([omega, -g_earth / L_pend * np.sin(theta)])

    return L_pend, f_pendulum, g_earth


@app.cell(hide_code=True)
def _(mo):
    pend_theta0 = mo.ui.slider(5, 179, step=1, value=20, label=r"Initial angle $\theta_0$ [deg]")
    return (pend_theta0,)


@app.cell(hide_code=True)
def _(f_pendulum, np, ode_solve, pend_theta0):
    pend_T, pend_N = 5.0, 5000
    pend_t, pend_x = ode_solve(rk4_step, f_pendulum, [np.radians(pend_theta0.value), 0.0], 0.0, pend_T / pend_N, pend_N)
    return pend_N, pend_t, pend_x


@app.cell(hide_code=True)
def _(PlaySlider, mo):
    pend_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=200, step=1, interval_ms=60))
    return (pend_play,)


@app.cell(hide_code=True)
def _(
    L_pend,
    g_earth,
    mo,
    np,
    pend_N,
    pend_play,
    pend_t,
    pend_theta0,
    pend_x,
    plt,
):
    _k = int(pend_play.value["value"]) * (pend_N // 200)
    _theta0 = np.radians(pend_theta0.value)
    _small = _theta0 * np.cos(np.sqrt(g_earth / L_pend) * pend_t)

    _fig, (_axp, _ax) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1, 2.2]))
    # Left: the pendulum
    _phi = np.linspace(0, 2 * np.pi, 200)
    _axp.plot(L_pend * np.sin(_phi), -L_pend * np.cos(_phi), color="0.85", lw=1)
    _axp.plot([0.0, L_pend * np.sin(pend_x[_k, 0])], [0.0, -L_pend * np.cos(pend_x[_k, 0])], color="black", lw=2)
    _axp.plot([L_pend * np.sin(pend_x[_k, 0])], [-L_pend * np.cos(pend_x[_k, 0])], "o", color="crimson", ms=16, zorder=3)
    _axp.plot([0.0], [0.0], "ks", ms=5)
    _axp.set(xlim=(-1.3 * L_pend, 1.3 * L_pend), ylim=(-1.3 * L_pend, 1.3 * L_pend), aspect="equal", xticks=[], yticks=[], title=f"$t={pend_t[_k]:.2f}$ s")
    # Right: the angle against time
    _ax.plot(pend_t, pend_x[:, 0], color="crimson", lw=2, label="RK4")
    _ax.plot(pend_t, _small, "--", color="0.4", lw=1.4, label=r"Small angles: $\theta_0\cos(\sqrt{g/L}\,t)$")
    _ax.plot([pend_t[_k]], [pend_x[_k, 0]], "o", color="#1d4ed8", ms=8, zorder=5)
    _ax.set(xlabel="$t$ [s]", ylabel=r"$\theta$ [rad]", title=rf"Non-linear pendulum, $\theta_0={pend_theta0.value}^\circ$")
    _ax.legend(loc="upper right", fontsize=9)
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([pend_theta0, mo.md("**Time:**"), pend_play], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.stat(
                        label="Small-angle period",
                        value=f"{2 * np.pi * np.sqrt(L_pend / g_earth):.3f} s",
                        caption="2π√(L/g)",
                    ),
                ],
                widths=[0.85, 0.15],
                align="center",
            ),
            mo.md(
                r"""
                For $\theta_0=20^\circ$ the period is only 0.8% longer than the small-angle value $T_0$. As $\theta_0$ grows, the pendulum
                oscillates more slowly than the harmonic approximation predicts: $T\approx1.18\,T_0$ at $90^\circ$ and $T\approx3.9\,T_0$ at $179^\circ$,
                and the period diverges as $\theta_0\to180^\circ$.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Adaptive time step for systems of ODEs
    ---

    The adaptive 4th-order Runge–Kutta method for a single first-order ODE can be generalized to a system of ODEs.
    One has to generalize the notion of the difference $|x_1 - x_2|$ between two estimates of the dependent variable to the case of vectors $\mathbf{x}_1$ and $\mathbf{x}_2$.

    The simplest choice is the modulus of the difference vector, $|\mathbf{x}_1 - \mathbf{x}_2|$, which is the default (`distance_norm` above).
    Alternatively, in some cases we may only be interested in the accuracy of a single variable from the vector,
    for example of the coordinate $\theta$ but not of the velocity $\omega$ in the case of a second-order ODE cast as a system of two first-order ODEs.
    In that case one can use a distance appropriate for the problem at hand, which `ode_rk4_adaptive` accepts through its `distance` argument.
    """)
    return


@app.cell
def _(np):
    def distance_theta(x1, x2):
        """Only the accuracy of the angle matters."""
        return np.abs(x1[0] - x2[0])

    return (distance_theta,)


@app.cell(hide_code=True)
def _(mo):
    pend_eps = mo.ui.slider(steps=[1e-3, 1e-4, 1e-5, 1e-6], value=1e-5, show_value=True, label=r"Accuracy of $\theta$ per unit time")
    return (pend_eps,)


@app.cell(hide_code=True)
def _(f_pendulum, np, solve_ivp):
    # An accurate reference solution: SciPy's DOP853 with tight tolerances
    ref_pend_sol = solve_ivp(
        lambda t, x: f_pendulum(x, t), (0.0, 3.0), [np.radians(179.0), 0.0], method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True
    ).sol
    return (ref_pend_sol,)


@app.cell(hide_code=True)
def _(distance_theta, f_pendulum, np, ode_rk4_adaptive, pend_eps):
    adapt_pend_t, adapt_pend_x = ode_rk4_adaptive(
        f_pendulum, [np.radians(179.0), 0.0], 0.0, 3.0 / 500, 3.0, pend_eps.value, distance_theta
    )
    return adapt_pend_t, adapt_pend_x


@app.cell(hide_code=True)
def _(adapt_pend_t, adapt_pend_x, mo, np, pend_eps, plt, ref_pend_sol):
    _t_ref = np.linspace(0.0, 3.0, 600)
    _err = abs(adapt_pend_x[-1, 0] - ref_pend_sol(3.0)[0])

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _ax.plot(_t_ref, ref_pend_sol(_t_ref)[0], color="crimson", lw=2, label="Reference (SciPy DOP853)")
    _ax.plot(adapt_pend_t, adapt_pend_x[:, 0], ".", color="steelblue", ms=8, label="Adaptive RK4")
    _ax.set(xlabel="$t$ [s]", ylabel=r"$\theta$ [rad]", title=r"Pendulum, $\theta_0=179^\circ$")
    _ax.legend(loc="upper right")
    _ax2.plot(0.5 * (adapt_pend_t[:-1] + adapt_pend_t[1:]), adapt_pend_t[1:] - adapt_pend_t[:-1], ".", color="steelblue", ms=8)
    _ax2.set(xlabel="$t$ [s]", ylabel="$h$ [s]", title="Time step lengths")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            pend_eps,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Adaptive steps", value=f"{len(adapt_pend_t) - 1}", caption="fixed RK4 above: 5000"),
                            mo.stat(label="Error in θ at t = 3 s", value=f"{_err:.1e} rad", caption="vs the reference"),
                        ]
                    ),
                ],
                widths=[0.85, 0.15],
                align="center",
            ),
            mo.md(
                r"""
                The steps shrink while the pendulum moves fast and grow while it moves slowly. (The very first steps are
                small only because the method starts from $h_0=6$ ms and needs a few steps to grow.)

                A pendulum released at $179^\circ$ is almost balanced upright, an unstable equilibrium: a small error in $\theta$
                grows exponentially, at a rate of about $\sqrt{g/L}\approx10\ \mathrm{s}^{-1}$, every time the pendulum passes near the top.
                The accuracy $\delta$ controls the error *per unit time*, not the final error, so with a loose $\delta$ the curve has the
                right shape but the wrong phase: at $\delta=10^{-3}$ the dots peel away from the reference after about 1.5 s.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Comet motion
    ---

    *(Exercise 8.10 from M. Newman, Computational Physics)*

    Many comets travel in highly elongated orbits around the Sun. For
    much of their lives they are far out in the solar system, moving very
    slowly, but on rare occasions their orbit brings them close to the Sun for
    a fly-by and for a brief period of time they move very fast indeed.

    This is a classic example of a system for which an adaptive
    step size method is useful, because for the large periods of time when the
    comet is moving slowly we can use long time-steps, so that the program runs
    quickly, but short time-steps are crucial in the brief but fast-moving
    period close to the Sun.

    The differential equation obeyed by a comet is straightforward to
    derive. The force between the Sun, with mass $M$ at the origin, and a
    comet of mass $m$ with position vector $\vec{r}$ is $GMm/r^2$ in the direction
    $-\vec{r}/r$ (i.e., the direction towards the Sun), and hence Newton's
    second law tells us that
    $$
    m {d^2\vec{r}\over d t^2} = -\biggl({GMm\over r^2}\biggr)\,{\vec{r}\over r}.
    $$
    Canceling the mass $m$ and taking the $x$ component we have
    $$
    {d^2 x\over d t^2} = -GM {x\over r^3}\,,
    $$
    and similarly for the other two coordinates. We can, however, throw out
    one of the coordinates because the comet stays in a single plane as it
    orbits. If we orient our axes so that this plane is perpendicular to
    the $z$-axis, we can forget about the $z$ coordinate and we are left with
    just two second-order equations to solve:

    \begin{align*}
    {d^2 x\over d t^2} & = -GM {x\over r^3}\,, \\
    {d^2 y\over d t^2} & = -GM {y\over r^3}\,,
    \end{align*}

    where $r=\sqrt{x^2+y^2}$.

    We solve these equations with the fourth-order Runge–Kutta method with a *fixed* step size first.
    As an initial condition, we take a comet at coordinates $x=4$ billion kilometers
    and $y=0$ (which is somewhere out around the orbit of Neptune) with
    initial velocity $v_x=0$ and $v_y = 500\,\mathrm{m\,s}^{-1}$. The trajectory of the comet is a plot of $y$
    against $x$.
    """)
    return


@app.cell
def _(np):
    G_newton = 6.6743e-11  # m^3 kg^-1 s^-2
    M_sun = 1.9885e30  # kg
    year = 365.0 * 24.0 * 60.0 * 60.0  # s

    def f_comet(xin, t):
        x, y, vx, vy = xin
        r = np.sqrt(x * x + y * y)
        return np.array([vx, vy, -G_newton * M_sun * x / r**3, -G_newton * M_sun * y / r**3])

    def distance_position(x1, x2):
        """Only the accuracy of the position matters."""
        return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)

    comet_x0 = [4.0e12, 0.0, 0.0, 500.0]
    comet_tend = 50.0 * year
    return comet_tend, comet_x0, distance_position, f_comet, year


@app.cell(hide_code=True)
def _(comet_tend, comet_x0, f_comet, ode_solve):
    # Reference: RK4 with 100000 fixed steps
    comet_N = 100000
    comet_fixed_t, comet_fixed_x = ode_solve(rk4_step, f_comet, comet_x0, 0.0, comet_tend / comet_N, comet_N)
    return comet_N, comet_fixed_t, comet_fixed_x


@app.cell(hide_code=True)
def _(np, plt, year):
    def plot_comet(t_fix, x_fix, t_ad, x_ad, label, step_label):
        """Orbit, coordinates, speed and time steps: a fixed-step reference against an adaptive solution."""
        scale = 1.0e12  # plot lengths in 10^12 m
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.5, 8))

        ax1.plot(x_fix[:, 0] / scale, x_fix[:, 1] / scale, color="crimson", lw=2, label="RK4 fixed")
        ax1.plot(x_ad[:, 0] / scale, x_ad[:, 1] / scale, ".", color="steelblue", label=label)
        ax1.plot([0.0], [0.0], "o", color="gold", mec="black", ms=10, label="Sun")
        ax1.set(xlabel="$x$ [$10^{12}$ m]", ylabel="$y$ [$10^{12}$ m]", title="Trajectory")
        ax1.legend(fontsize=9)

        ax2.plot(t_fix / year, x_fix[:, 0] / scale, color="crimson", lw=2, label="$x$ (RK4 fixed)")
        ax2.plot(t_fix / year, x_fix[:, 1] / scale, color="black", lw=1.2, label="$y$ (RK4 fixed)")
        ax2.plot(t_ad / year, x_ad[:, 0] / scale, ".", color="steelblue", label=f"$x$ ({label})")
        ax2.plot(t_ad / year, x_ad[:, 1] / scale, ".", color="seagreen", label=f"$y$ ({label})")
        ax2.set(xlabel="$t$ [yr]", ylabel="$x,\\ y$ [$10^{12}$ m]", title="Coordinates")
        ax2.legend(fontsize=8)

        ax3.plot(t_fix / year, np.hypot(x_fix[:, 2], x_fix[:, 3]), color="crimson", lw=2, label="RK4 fixed")
        ax3.plot(t_ad / year, np.hypot(x_ad[:, 2], x_ad[:, 3]), ".", color="steelblue", label=label)
        ax3.set(xlabel="$t$ [yr]", ylabel="$v$ [m/s]", title="Speed")
        ax3.legend(fontsize=9)

        ax4.plot(0.5 * (t_ad[:-1] + t_ad[1:]) / year, (t_ad[1:] - t_ad[:-1]) / year, ".", color="steelblue")
        ax4.set(xlabel="$t$ [yr]", ylabel=f"{step_label} [yr]", title="Time steps")
        fig.tight_layout()
        plt.close(fig)
        return fig

    return (plot_comet,)


@app.cell(hide_code=True)
def _(comet_N, comet_fixed_x, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.plot(comet_fixed_x[:, 0] / 1e12, comet_fixed_x[:, 1] / 1e12, color="crimson", lw=2, label=f"RK4, $N={comet_N}$")
    _ax.plot([0.0], [0.0], "o", color="gold", mec="black", ms=10, label="Sun")
    _ax.set(xlabel="$x$ [$10^{12}$ m]", ylabel="$y$ [$10^{12}$ m]", title="Comet motion, fixed-step RK4")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    _r = np.hypot(comet_fixed_x[:, 0], comet_fixed_x[:, 1])
    mo.hstack(
        [
            _fig,
            mo.vstack(
                [
                    mo.stat(label="Aphelion", value=f"{_r.max() / 1e12:.2f}e12 m"),
                    mo.stat(label="Closest approach", value=f"{_r.min() / 1e12:.3f}e12 m"),
                    mo.stat(label="Fastest speed", value=f"{np.hypot(comet_fixed_x[:, 2], comet_fixed_x[:, 3]).max():.0f} m/s"),
                ]
            ),
        ],
        widths=[0.7, 0.3],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now use the RK4 method with an adaptive time step. We expect a smaller step near the Sun, where the comet moves faster, and a larger step far away from it.
    The accuracy is measured on the comet position only (`distance_position`).
    """)
    return


@app.cell(hide_code=True)
def _(
    comet_fixed_t,
    comet_fixed_x,
    comet_tend,
    comet_x0,
    distance_position,
    f_comet,
    mo,
    ode_rk4_adaptive,
    plot_comet,
    year,
):
    _delta = 1000.0 * 1000.0 / year  # 1000 km per year
    _t, _x = ode_rk4_adaptive(f_comet, comet_x0, 0.0, year, comet_tend, _delta, distance_position)

    mo.hstack(
        [
            plot_comet(comet_fixed_t, comet_fixed_x, _t, _x, "RK4 adaptive", "$h$"),
            mo.stat(label="Adaptive steps", value=f"{len(_t) - 1}", caption="fixed-step reference: 100000"),
        ],
        widths=[0.85, 0.15],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Leapfrog method
    ---

    Recall the 2nd order Runge–Kutta (midpoint) method.
    The full step proceeds by evaluating
    $$
    x(t+h) = x(t) + h f[x(t+h/2),t+h/2],
    $$
    where the midpoint value $x(t+h/2)$ is estimated using Euler's method
    $$
    x(t+h/2) = x(t) + \frac{1}{2} h f(x,t)~.
    $$

    The idea of the leapfrog method is to use the last equation (Euler half-step) only in the first iteration, and then use the first equation to estimate both $x(t+h)$ and $x(t+3h/2)$, i.e., given the values of $x(t)$ and $x(t+h/2)$ at each step the algorithm calculates a pair of values

    \begin{align*}
    x(t+h) &= x(t) + h f[x(t+h/2),t+h/2], \\
    x(t+3h/2) &= x(t+h/2) + h f[x(t+h),t+h].
    \end{align*}

    The positions $x(t+kh)$ and the midpoints $x(t+(k+\tfrac12)h)$ are thus two interleaved sequences, and each one
    uses the other to take its step: they *leap over* each other. The sketch below compares the two stencils (cf. Newman's Fig. 8.9).
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    _blue, _orange = "#1d4ed8", "darkorange"

    def _arc(ax, x1, x2, rad, color):
        ax.annotate(
            "",
            xy=(x2, 0),
            xytext=(x1, 0),
            arrowprops=dict(arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}", color=color, lw=1.6),
            zorder=2,
        )

    _K = 4
    _full = np.arange(_K + 1, dtype=float)
    _mid = np.arange(_K) + 0.5

    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for _ax, _kind in zip(_axes, ["rk2", "leap"]):
        _ax.scatter(_full, np.zeros_like(_full), s=90, color="black", zorder=4)
        _ax.scatter(_mid, np.zeros_like(_mid), s=90, facecolor="white", edgecolor="black", zorder=4)
        for _k in range(_K):
            if _kind == "rk2":
                _arc(_ax, _k, _k + 0.5, -0.9, _blue)  # Euler half step to the midpoint
                _arc(_ax, _k, _k + 1, 0.55, _orange)  # full step, with the slope at the midpoint
            else:
                if _k == 0:
                    _arc(_ax, 0, 0.5, -0.9, _blue)  # only the very first half step is Euler
                else:
                    _arc(_ax, _k - 0.5, _k + 0.5, -0.55, _blue)  # midpoint from the previous midpoint
                _arc(_ax, _k, _k + 1, 0.55, _orange)
        _ax.set_ylim(-0.75, 0.85)
        _ax.set_yticks([])
        for _side in ("left", "right", "top"):
            _ax.spines[_side].set_visible(False)
    _axes[0].set_title("(a) RK2: every step restarts from a filled point", fontsize=11, loc="left")
    _axes[1].set_title("(b) Leapfrog: after the first half step the midpoints chain onto each other", fontsize=11, loc="left")
    _axes[1].set_xticks(np.arange(_K + 1))
    _axes[1].set_xticklabels(["$t$", "$t+h$", "$t+2h$", "$t+3h$", "$t+4h$"])
    _axes[1].text(
        0.0, -0.65, "filled: $x$ at whole steps,  open: $x$ at half steps,  orange arc: full step using the slope at the point it passes over",
        fontsize=9, color="0.3",
    )
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An attractive feature of this algorithm is that it is **time reversible**. By changing $h \to -h$ one gets

    \begin{align*}
    x(t-h) &= x(t) - h f[x(t-h/2),t-h/2], \\
    x(t-3h/2) &= x(t-h/2) - h f[x(t-h),t-h],
    \end{align*}

    i.e. the same steps are performed in reverse.
    Changing $t \to t + 3h/2$ one gets

    \begin{align*}
    x(t+h/2) &= x(t+3h/2) - h f[x(t+h),t+h], \\
    x(t) &= x(t+h) - h f[x(t+h/2),t+h/2].
    \end{align*}

    These are *exactly* the forward equations with the sign of the $hf$ term flipped: every operation is undone.
    The backward run therefore retraces the forward steps and recovers the starting values, up to rounding error.
    RK2 does not have this property: putting $h\to-h$ in the midpoint scheme does not undo the forward step.

    The time-reversal symmetry is important since the microscopic equations are typically time-reversal invariant, so it is good to have this feature built into the numerical scheme.
    """)
    return


@app.cell
def _(euler_step, np):
    def leapfrog_step(f, x, x2, t, h):
        """Perform a single leapfrog step h.

        x: the value of x(t).
        x2: the value of x(t+h/2).
        Returns x(t+h) and x(t+3h/2).
        """
        xnew = x + h * f(x2, t + h / 2.0)
        xnew2 = x2 + h * f(xnew, t + h)
        return xnew, xnew2

    def ode_leapfrog(f, x0, t0, h, nsteps, return_half=False):
        """Solve dx/dt = f(x,t) with the leapfrog method (x0 is a vector).

        With return_half=True the values x(t+h/2) at the half steps are returned too:
        the pair (x, x_half) is the full state of a leapfrog calculation.
        """
        t = t0 + h * np.arange(nsteps + 1)
        x = np.zeros((nsteps + 1, len(x0)))
        x_half = np.zeros_like(x)
        x[0] = x0
        x_half[0] = euler_step(f, x[0], t0, h / 2.0)  # the first half-step is an Euler step
        for i in range(nsteps):
            x[i + 1], x_half[i + 1] = leapfrog_step(f, x[i], x_half[i], t[i], h)
        return (t, x, x_half) if return_half else (t, x)

    def ode_leapfrog_backward(f, x_end, x_half_end, t_end, h, nsteps):
        """Run the leapfrog algorithm with step -h, starting from x(t_end) and x(t_end+h/2).

        Returns the values x(t_end - k h), k = 0..nsteps.
        """
        x, x_half = np.array(x_end, dtype=float), np.array(x_half_end, dtype=float)
        path = np.zeros((nsteps + 1, len(x)))
        path[0] = x
        for k in range(nsteps):
            t = t_end - k * h
            x_half = x_half - h * f(x, t)  # undo the second line of the forward step
            x = x - h * f(x_half, t - h / 2.0)  # undo the first line
            path[k + 1] = x
        return path

    return leapfrog_step, ode_leapfrog, ode_leapfrog_backward


@app.cell(hide_code=True)
def _(mo):
    rev_N = mo.ui.slider(100, 1000, step=100, value=500, label="Number of steps $N$")
    rev_h = mo.ui.slider(0.002, 0.01, step=0.002, value=0.006, label="Step size $h$ [s]")
    rev_theta = mo.ui.slider(30, 179, step=1, value=120, label=r"Initial angle $\theta_0$ [deg]")
    return rev_N, rev_h, rev_theta


@app.cell(hide_code=True)
def _(
    f_pendulum,
    method_colors,
    mo,
    np,
    ode_leapfrog,
    ode_leapfrog_backward,
    ode_solve,
    plt,
    rev_N,
    rev_h,
    rev_theta,
):
    _N, _h = int(rev_N.value), float(rev_h.value)
    _T = _N * _h
    _x0 = np.array([np.radians(rev_theta.value), 0.0])

    # Forward, then backward from the final state with step -h
    _t, _lf, _lf_half = ode_leapfrog(f_pendulum, _x0, 0.0, _h, _N, return_half=True)
    _lf_back = ode_leapfrog_backward(f_pendulum, _lf[-1], _lf_half[-1], _T, _h, _N)
    _, _rk = ode_solve(rk2_step, f_pendulum, _x0, 0.0, _h, _N)
    _, _rk_back = ode_solve(rk2_step, f_pendulum, _rk[-1], _T, -_h, _N)

    _lf_err = np.max(np.abs(_lf_back[-1] - _x0))
    _rk_err = np.max(np.abs(_rk_back[-1] - _x0))
    _lf_dev = np.abs(_lf_back[::-1, 0] - _lf[:, 0]) + 1e-18  # tiny offset so that zeros show on the log scale
    _rk_dev = np.abs(_rk_back[::-1, 0] - _rk[:, 0]) + 1e-18

    _fig, _axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
    for _col, (_name, _fwd, _back, _dev) in enumerate(
        [("Leapfrog", _lf, _lf_back, _lf_dev), ("RK2", _rk, _rk_back, _rk_dev)]
    ):
        _c = method_colors[_name]
        _axes[0, _col].plot(_t, _fwd[:, 0], "k-", lw=2, label="forward")
        _axes[0, _col].plot(_T - _t, _back[:, 0], "o", ms=3, color=_c, markevery=max(1, _N // 40), label="backward (step $-h$)")
        _axes[0, _col].set(title=f"{_name}: " + ("the backward run retraces the forward one" if _col == 0 else "it does not"))
        _axes[0, _col].legend(loc="upper right", fontsize=9)
        _axes[1, _col].semilogy(_t, _dev, color=_c)
        _axes[1, _col].set(xlabel="$t$ [s]", ylim=(1e-17, 1e1))
    _axes[0, 0].set_ylabel(r"$\theta$ [rad]")
    _axes[1, 0].set_ylabel(r"$|\theta_{\rm backward}-\theta_{\rm forward}|$")
    for _ax in _axes.ravel():
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                "**Running the algorithm backwards.** The pendulum is integrated forward with each method, "
                "then backward from the final state with step $-h$."
            ),
            mo.hstack([rev_N, rev_h, rev_theta], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Leapfrog: error in recovered start", value=f"{_lf_err:.1e}"),
                            mo.stat(label="RK2: error in recovered start", value=f"{_rk_err:.1e}"),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.md(
                r"""
                The leapfrog backward run recovers the initial $(\theta,\omega)$ to rounding error, however coarse the step
                (about $10^{-15}$ for short runs; up to about $10^{-11}$ for the longest runs released near the top, where the dynamics amplify rounding errors).
                RK2 comes back to somewhere near the start, by an amount set by its own truncation error, which grows quickly with $h$.
                The upper panels can look alike because a small deviation is hard to see on a plot of $\theta$ itself;
                the lower panels show the deviation on a log scale.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Time reversibility has a consequence for the energy of a periodic system: it rules out a steady drift
    (a drift up in the forward run would be a drift down in the backward run, yet the motion is the same),
    although the energy may still fluctuate within a period.
    Let us check this on the non-linear pendulum with the leapfrog method.
    We can monitor the behavior of the energy
    $$
    E = m L^2 \dot{\theta}^2/2 - mgL \cos(\theta)
    $$
    as a function of time (below, per unit mass $m$).

    If $\theta_0 = \pi/2$ and $\dot{\theta}_0 = 0$ the total energy is zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    leap_N = mo.ui.slider(100, 2000, step=50, value=500, label="Number of steps $N$")
    return (leap_N,)


@app.cell(hide_code=True)
def _(
    L_pend,
    f_pendulum,
    g_earth,
    leap_N,
    method_colors,
    mo,
    np,
    ode_leapfrog,
    ode_solve,
    plt,
):
    def pendulum_energy(x):
        """Energy per unit mass."""
        return L_pend**2 * x[:, 1] ** 2 / 2.0 - g_earth * L_pend * np.cos(x[:, 0])

    _x0 = [np.pi / 2.0, 0.0]
    _h = 10.0 / leap_N.value
    _t, _x_leap = ode_leapfrog(f_pendulum, _x0, 0.0, _h, leap_N.value)
    _t_rk2, _x_rk2 = ode_solve(rk2_step, f_pendulum, _x0, 0.0, _h, leap_N.value)

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _ax.plot(_t, _x_leap[:, 0], color=method_colors["Leapfrog"], lw=1.6, label="Leapfrog")
    _ax.plot(_t, _x_rk2[:, 0], color=method_colors["RK2"], lw=1.6, label="RK2")
    _ax.set(xlabel="$t$ [s]", ylabel=r"$\theta$ [rad]", title=r"Non-linear pendulum, $\theta_0=\pi/2$")
    _ax.legend(loc="lower left")
    _ax2.plot(_t, pendulum_energy(_x_leap), color=method_colors["Leapfrog"], lw=1.6, label="Leapfrog")
    _ax2.plot(_t, pendulum_energy(_x_rk2), color=method_colors["RK2"], lw=1.6, label="RK2")
    _ax2.set(xlabel="$t$ [s]", ylabel="Energy per unit mass", ylim=(-0.05, 0.05), title="Total energy (exactly 0)")
    _ax2.legend(loc="upper left")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            leap_N,
            _fig,
            mo.md(
                r"""
                The *average* leapfrog energy stays at the true value (zero), while the average RK2 energy
                grows steadily, and the pendulum swings higher and higher. The leapfrog energy does oscillate
                around zero, with an amplitude that grows slowly here (the curve leaves the plotted window
                after a few periods); both effects shrink quickly when $N$ is increased.
                The leapfrog method is thus well suited to long simulations of periodic systems.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Verlet method
    ---

    Suppose we are solving Newton's equations of motion
    $$
    \frac{d^2 x}{dt^2} = f(x,t).
    $$
    These are rewritten as

    \begin{align*}
    \frac{dx}{dt} &= v, \\
    \frac{dv}{dt} &= f(x,t).
    \end{align*}

    If we apply the leapfrog scheme to this system of equations, it looks like

    \begin{align*}
    x(t+h) & = x(t) + h v(t + h/2), \\
    v(t+3h/2) &= v(t+h/2) + h f[x(t+h), t+h],
    \end{align*}

    i.e. the coordinates are evaluated at full steps using velocity estimates at half-steps,
    and vice versa.

    This is the essence of the Verlet method.
    Its feature is that we need to keep track only of the coordinates at full steps and of the velocities at half-steps.
    In case we are interested in the velocity values also at full steps, these can be estimated as $v(t+h) = v(t+h/2) + (h/2) f[x(t+h),t+h]$.

    The full algorithm is as follows. Given initial values $x(t)$ and $v(t)$ one first computes
    $$
    v(t+h/2) = v(t) + \frac{h}{2} f[x(t),t].
    $$
    Then each step of the Verlet algorithm corresponds to

    \begin{align*}
    x(t+h) & = x(t) + h v(t+h/2), \\
    k & = h f[x(t+h),t+h], \\
    v(t+h) & = v(t+h/2) + k/2, \\
    v(t+3h/2) & = v(t+h/2) + k.
    \end{align*}

    The algorithm straightforwardly generalizes to more than one variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Modified midpoint method
    ---

    Recall that steps in the leapfrog method are time-reversible ($h \to -h$).
    For this reason the local error (per step) contains only odd powers of $h$,
    while the global error integrating from $t$ to $t+H$ has only even powers of $h$.
    The only issue is the first Euler half-step $x(t+h/2) = x(t) + \frac{h}{2} f(x,t)$ which is not symmetric in $h$.
    This half-step ruins the desired property of the global error containing only even powers of $h$.

    To fix this issue one uses an extra step at the very end,
    which is called the **modified midpoint method**.
    Let $y_n = x(t+H-h/2)$ and $x_n = x(t+H)$ be the estimates of the solution resulting from the leapfrog method.
    One then re-evaluates $x(t+H)$ as an average of $x_n$ and the value of $x(t+H)$ evaluated using an Euler half-step from $y_n$:
    $$
    x(t+H) = \frac{1}{2} \left[x_n + y_n + \frac{h}{2} f(x_n,t+H)\right].
    $$

    This modification cancels all odd terms in the global error for $x(t+H)$.
    """)
    return


@app.cell
def _(euler_step, leapfrog_step, np):
    def modified_midpoint(f, x0, t0, H, nsteps):
        """Integrate from t0 to t0+H with nsteps leapfrog steps and the final correction."""
        h = H / nsteps
        t, x = t0, np.asarray(x0, dtype=float)
        y = euler_step(f, x, t, h / 2.0)
        for _ in range(nsteps):
            y_prev = y
            x, y = leapfrog_step(f, x, y, t, h)
            t = t + h
        return 0.5 * (x + y_prev + 0.5 * h * f(x, t))

    return (modified_midpoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Bulirsch–Stoer method
    ---

    The Bulirsch–Stoer method is an analogue of Romberg integration applied to ODEs.
    The method allows one to integrate a (system of) ODE(s) from $t$ to $t+H$ to the desired accuracy.
    To achieve that one uses the modified midpoint method with an increasing number of steps,
    and the differences between subsequent estimates to cancel out higher and higher-order error terms in $h$.

    The iterative procedure is as follows.
    Let $R_{n,1}$ be an estimate of $x(t+H)$ from the modified midpoint method using $n$ steps of size $h_n = H/n$.
    Then
    $$
    x(t+H) = R_{n,1} + O(h_n^2).
    $$
    One constructs higher-order approximations $R_{n,m}$ such that
    $$
    x(t+H) = R_{n,m} + O(h_n^{2m}),
    $$
    from the iterative procedure
    $$
    R_{n,m+1} = R_{n,m} + \frac{R_{n,m} - R_{n-1,m}}{[n/(n-1)]^{2m} - 1}~.
    $$

    The procedure goes on in incremental steps of $n$ until the distance between the solution estimates $R_{n,n}$ and $R_{n,n-1}$ reaches the desired accuracy.

    In practice, if the desired accuracy is not reached after $n \sim 10$ steps, it may indicate that $H$ is too large, and one has to split it into two subintervals of width $H/2$ and apply the Bulirsch–Stoer method recursively.
    """)
    return


@app.cell
def _(distance_norm, modified_midpoint, np):
    def bulirsch_stoer_step(f, x0, t0, H, delta=1e-6, distance=distance_norm, maxsteps=10):
        """Use the Bulirsch-Stoer method to integrate from t0 to t0+H.

        Returns a list of [t, x] pairs: just one, unless H had to be split.
        """
        n = 1
        R1 = np.empty([1, len(x0)], float)
        R1[0] = modified_midpoint(f, x0, t0, H, 1)
        error = 2.0 * H * delta
        while error > H * delta and n < maxsteps:
            n = n + 1
            R2 = R1
            R1 = np.empty([n, len(x0)], float)
            R1[0] = modified_midpoint(f, x0, t0, H, n)
            for m in range(1, n):
                epsilon = (R1[m - 1] - R2[m - 1]) / ((n / (n - 1)) ** (2 * m) - 1)
                R1[m] = R1[m - 1] + epsilon
            error = distance(R1[n - 2], R1[n - 1])
        if error > H * delta:
            # Reached the maximum number of substeps: halve H and apply the method recursively
            sol1 = bulirsch_stoer_step(f, x0, t0, H / 2.0, delta, distance, maxsteps)
            sol2 = bulirsch_stoer_step(f, sol1[-1][1], t0 + H / 2.0, H / 2.0, delta, distance, maxsteps)
            return sol1 + sol2
        return [[t0 + H, R1[n - 1]]]

    def bulirsch_stoer(f, x0, t0, nsteps, tmax, delta=1e-6, distance=distance_norm, maxsubsteps=10):
        """Use the Bulirsch-Stoer method to integrate from t0 to tmax with nsteps big steps H.

        Each big step is split further if needed to reach the accuracy delta per unit time.
        Returns the arrays t and x.
        """
        H = (tmax - t0) / nsteps
        t = [t0]
        x = [np.asarray(x0, dtype=float)]
        for _ in range(nsteps):
            for t_new, x_new in bulirsch_stoer_step(f, x[-1], t[-1], H, delta, distance, maxsubsteps):
                t.append(t_new)
                x.append(x_new)
        return np.array(t), np.array(x)

    return (bulirsch_stoer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us apply the Bulirsch–Stoer method to the motion of the non-linear pendulum.
    Because the method dynamically adjusts $H$ to reach the desired accuracy, we can start with just one big step covering the whole time interval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bs_eps = mo.ui.slider(steps=[1e-4, 1e-6, 1e-8, 1e-10], value=1e-8, show_value=True, label=r"Accuracy of $\theta$ per unit time")
    return (bs_eps,)


@app.cell(hide_code=True)
def _(f_pendulum, np, solve_ivp):
    # An accurate reference solution: SciPy's DOP853 with tight tolerances
    ref_bs_sol = solve_ivp(
        lambda t, x: f_pendulum(x, t), (0.0, 10.0), [np.radians(179.0), 0.0], method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True
    ).sol
    return (ref_bs_sol,)


@app.cell(hide_code=True)
def _(
    bs_eps,
    bulirsch_stoer,
    distance_theta,
    f_pendulum,
    mo,
    np,
    plt,
    ref_bs_sol,
):
    _t, _x = bulirsch_stoer(f_pendulum, [np.radians(179.0), 0.0], 0.0, 1, 10.0, bs_eps.value, distance_theta)
    _t_ref = np.linspace(0.0, 10.0, 1000)
    _err = abs(_x[-1, 0] - ref_bs_sol(10.0)[0])

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _ax.plot(_t_ref, ref_bs_sol(_t_ref)[0], color="crimson", lw=2, label="Reference (SciPy DOP853)")
    _ax.plot(_t, _x[:, 0], ".", color="steelblue", ms=8, label="Bulirsch–Stoer")
    _ax.set(xlabel="$t$ [s]", ylabel=r"$\theta$ [rad]", ylim=(-np.pi, np.pi), title=r"Pendulum, $\theta_0=179^\circ$")
    _ax.legend(loc="upper right")
    _ax2.plot(0.5 * (_t[:-1] + _t[1:]), _t[1:] - _t[:-1], ".", color="steelblue", ms=8)
    _ax2.set(xlabel="$t$ [s]", ylabel="$H$ [s]", title="Time steps $H$")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            bs_eps,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Big steps", value=f"{len(_t) - 1}", caption="started from a single step H = 10 s"),
                            mo.stat(label="Error in θ at t = 10 s", value=f"{_err:.1e} rad", caption="vs the reference"),
                        ]
                    ),
                ],
                widths=[0.85, 0.15],
                align="center",
            ),
            mo.md(
                r"""
                As for the adaptive RK4 above, the nearly upright pendulum amplifies errors exponentially, so a loose
                accuracy gives the wrong phase after a few swings: the error at $t=10$ s falls from about 2 rad at $10^{-4}$
                to about $4\times10^{-7}$ rad at $10^{-10}$.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now apply the Bulirsch–Stoer method to the comet motion.
    Again, we can use a single big step $H$ initially.
    """)
    return


@app.cell(hide_code=True)
def _(
    bulirsch_stoer,
    comet_fixed_t,
    comet_fixed_x,
    comet_tend,
    comet_x0,
    distance_position,
    f_comet,
    mo,
    plot_comet,
    year,
):
    _delta = 1.0 * 1000.0 / year  # 1 km per year
    _t, _x = bulirsch_stoer(f_comet, comet_x0, 0.0, 1, comet_tend, _delta, distance_position)

    mo.vstack(
        [
            mo.hstack(
                [
                    plot_comet(comet_fixed_t, comet_fixed_x, _t, _x, "Bulirsch–Stoer", "$H$"),
                    mo.stat(label="Big steps", value=f"{len(_t) - 1}", caption="fixed-step reference: 100000"),
                ],
                widths=[0.85, 0.15],
                align="center",
            ),
            mo.md(
                r"""
                Note that the "reference" is not exact: at the end of the run the 100000-step RK4 position is off by about
                $6\times10^{8}$ m (0.016% of the orbit size), invisible in these plots. Against an accurate SciPy solution,
                Bulirsch–Stoer with its 27 steps is about 100 times closer, off by only $6\times10^{6}$ m.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. SIR model
    ---

    The SIR model is the simplest model for infectious disease dynamics in a population.
    The population is split into susceptible ($S$), infected ($I$), and recovered/immune ($R$) parts.
    The SIR equations read

    \begin{align*}
    \frac{dS}{dt} & = -\beta S I, \\
    \frac{dI}{dt} & = \beta S I - \gamma I, \\
    \frac{dR}{dt} & = \gamma I.
    \end{align*}

    Here $\beta$ is the infection rate and $\gamma$ is the recovery rate.
    They are inverse to the average time an infected person spreads the disease (when all of the population is susceptible, $S \approx 1$) and the time it takes to recover. The ratio $R_0 = \beta / \gamma$ is the basic reproduction number.

    Given that $S + I + R = 1 = \rm const$ at all times, one only needs to solve two ODEs, e.g. for $S$ and $I$.

    What if the immunity does not last forever but is lost after some time (e.g. three months)?
    One can add this effect to the equations through a loss-of-immunity rate $\kappa$:

    \begin{align*}
    \frac{dS}{dt} & = -\beta S I + \kappa R, \\
    \frac{dI}{dt} & = \beta S I - \gamma I, \\
    \frac{dR}{dt} & = \gamma I - \kappa R.
    \end{align*}

    Setting $\kappa=0$ recovers the plain SIR model; $\kappa=1/90\approx0.011$ corresponds to three months of immunity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sir_beta = mo.ui.slider(0.05, 0.5, step=0.01, value=0.25, label=r"Infection rate $\beta$ [1/day]")
    sir_gamma = mo.ui.slider(0.02, 0.3, step=0.01, value=0.1, label=r"Recovery rate $\gamma$ [1/day]")
    sir_kappa = mo.ui.slider(0.0, 0.05, step=0.001, value=0.0, label=r"Loss of immunity $\kappa$ [1/day]")
    return sir_beta, sir_gamma, sir_kappa


@app.cell(hide_code=True)
def _(bulirsch_stoer, np, sir_beta, sir_gamma, sir_kappa):
    def f_sir(xin, t):
        S, I = xin  # R = 1 - S - I
        return np.array(
            [
                -sir_beta.value * S * I + sir_kappa.value * (1.0 - S - I),
                sir_beta.value * S * I - sir_gamma.value * I,
            ]
        )

    sir_I0 = 1e-5
    sir_t, sir_x = bulirsch_stoer(f_sir, [1.0 - sir_I0, sir_I0], 0.0, 50, 365.0, 1e-9)
    return sir_I0, sir_t, sir_x


@app.cell(hide_code=True)
def _(mo, np, plt, sir_I0, sir_beta, sir_gamma, sir_kappa, sir_t, sir_x):
    _R0 = sir_beta.value / sir_gamma.value
    _S, _I = sir_x[:, 0], sir_x[:, 1]
    _R = 1.0 - _S - _I
    _texp = np.linspace(0.0, 90.0, 100)
    _Iexp = sir_I0 * np.exp(sir_gamma.value * (_R0 - 1.0) * _texp)

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    _ax.plot(sir_t, _S, ".-", ms=4, lw=1, color="steelblue", label="Susceptible")
    _ax.plot(sir_t, _I, ".-", ms=4, lw=1, color="crimson", label="Infected")
    _ax.plot(sir_t, _R, ".-", ms=4, lw=1, color="seagreen", label="Immune")
    _ax.plot(_texp, _Iexp, "--", color="crimson", lw=1.5, label="Infected (exponential phase)")
    _ax.set(xlabel="$t$ [days]", ylabel="Population fraction", ylim=(0, 1), title="SIR model, Bulirsch–Stoer")
    _ax.legend(loc="center right")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([sir_beta, sir_gamma, sir_kappa], justify="center", align="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Reproduction number R₀", value=f"{_R0:.2f}", caption="epidemic grows if R₀ > 1" if _R0 > 1 else "the infection dies out"),
                            mo.stat(label="Peak of infected", value=f"{_I.max():.3f}", caption=f"day {sir_t[np.argmax(_I)]:.0f}"),
                            mo.stat(label="Infected after 1 year", value=f"{_I[-1]:.4f}"),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.md(
                r"""
                One can clearly see the initial exponential phase of the epidemic, with growth rate $\gamma(R_0-1)$,
                and its end once a sufficient fraction of the population has become immune.
                With the default $\beta$ and $\gamma$, set $\kappa\approx0.011$: the immunity wears off and a second wave of infection follows.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Shooting method
    ---

    So far we solved initial-value problems. In a **boundary-value problem** the conditions are given at
    both ends of the interval, and the missing initial condition has to be found.

    Let us solve the problem of a ball thrown vertically at $t = 0$ which lands back on the ground at $t = T$.
    We want to find the initial velocity. The equations of motion are

    \begin{align}
    \frac{dx}{dt} & = v, \\
    \frac{dv}{dt} & = -g,
    \end{align}

    with $x(0) = 0$ and $v(0) = v_0$, and we require $x(T)=0$.

    The idea of the **shooting method** is to "shoot" with a guess for $v_0$, integrate the ODEs to $t=T$,
    and look at how far $x(T)$ misses the target. This miss is a function of $v_0$ alone, and the right $v_0$ is its root.
    We find it with the bisection method (Section 2 of *Non-linear equations*), integrating the ODEs with the 4th order Runge–Kutta method.
    """)
    return


@app.function
def bisection_method(f, a, b, tolerance=1e-10):
    """Find a root of f in [a,b]; returns the root and the number of iterations."""
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        return None, 0  # no sign change: bisection is not applicable
    iterations = 0
    while b - a > tolerance:
        iterations += 1
        c = 0.5 * (a + b)  # the midpoint
        fc = f(c)
        if fc * fa < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b), iterations


@app.cell(hide_code=True)
def _(mo):
    shoot_T = mo.ui.slider(2.0, 20.0, step=0.5, value=10.0, label="Time of flight $T$ [s]")
    return (shoot_T,)


@app.cell(hide_code=True)
def _(g_earth, mo, np, ode_solve, plt, shoot_T):
    def f_ball(xin, t):
        x, v = xin
        return np.array([v, -g_earth])

    _N = 100
    _h = shoot_T.value / _N

    def height_at_T(v0):
        """The miss: x(T) for the initial velocity v0."""
        return ode_solve(rk4_step, f_ball, [0.0, v0], 0.0, _h, _N)[1][-1][0]

    _v0, _iters = bisection_method(height_at_T, 0.01, 1000.0, 1e-10)
    _t, _x = ode_solve(rk4_step, f_ball, [0.0, _v0], 0.0, _h, _N)
    _v0_exact = g_earth * shoot_T.value / 2.0
    _vs = np.linspace(0.0, 2.0 * _v0_exact, 25)

    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _ax.plot(_vs, [height_at_T(_v) for _v in _vs], "o-", ms=4, lw=1, color="steelblue")
    _ax.axhline(0.0, color="black", ls="--", lw=0.8)
    _ax.plot([_v0], [0.0], "o", color="black", zorder=3)
    _ax.annotate(f"$v_0={_v0:.4f}$ m/s", (_v0, 0.0), xytext=(8, 14), textcoords="offset points")
    _ax.set(xlabel="$v_0$ [m/s]", ylabel="$x(T)$ [m]", title="The miss as a function of $v_0$")
    _ax2.plot(_t, _x[:, 0], color="crimson", lw=2)
    _ax2.axhline(0.0, color="black", ls="--", lw=0.8)
    _ax2.set(xlabel="$t$ [s]", ylabel="$x$ [m]", title=f"Trajectory for $v_0={_v0:.2f}$ m/s")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            shoot_T,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Initial velocity v₀", value=f"{_v0:.6f} m/s", caption=f"exact gT/2 = {_v0_exact:.6f} m/s"),
                            mo.stat(label="Bisection iterations", value=f"{_iters}", caption="bracket [0.01, 1000] m/s"),
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
    ## Summary: methods for ordinary differential equations

    | Method | Order | Evaluations of $f$ / step | Strengths | Weaknesses |
    |---|---|---|---|---|
    | Euler | 1 | 1 | trivial to implement | inaccurate, unstable for stiff equations |
    | Midpoint (RK2) | 2 | 2 | simple, much better than Euler | energy drifts in oscillating systems |
    | Runge–Kutta (RK4) | 4 | 4 | the workhorse: accurate and robust | fixed $h$ wastes work where $x$ is smooth |
    | Adaptive RK4 | 4 | ≈ 3 × 4 (step doubling) | chooses $h$ for a target accuracy | extra cost for the error estimate |
    | Implicit Euler | 1 | 1 + an algebraic equation | stable for stiff equations | must solve for $x_{n+1}$ at every step |
    | Leapfrog | 2 | 2 | time-reversible, no energy drift on average | needs the half-step start; fixed $h$ |
    | Verlet | 2 | 1 (acceleration only) | leapfrog for Newton's equations of motion | only for $\ddot x=f(x,t)$ |
    | Modified midpoint | 2 (even powers of $h$) | $2n+2$ for $n$ substeps | prepares Richardson extrapolation | — |
    | Bulirsch–Stoer | high, adaptive | many | high accuracy with few big steps | for smooth problems only |

    Two further ideas appeared along the way. A **second-order** equation is recast as a system of two
    first-order ones, so that every method above applies. A **boundary-value problem** is solved by shooting:
    wrap the ODE solver in a function of the unknown initial condition and find its root.

    In practice, robust library routines such as `scipy.integrate.solve_ivp` implement adaptive
    Runge–Kutta pairs (`RK45`, `DOP853`) and implicit methods for stiff problems (`Radau`, `BDF`, `LSODA`).
    """)
    return


if __name__ == "__main__":
    app.run()
