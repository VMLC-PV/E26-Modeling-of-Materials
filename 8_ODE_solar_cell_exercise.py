import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from wigglystuff import FormulaAnimation

    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary as numpyro_summary
    from numpyro.infer import MCMC, NUTS
    from scipy.integrate import solve_ivp

    jax.config.update("jax_enable_x64", True)  # double precision, to compare JAX with NumPy/SciPy
    return FancyBboxPatch, Rectangle, mo, np, plt


@app.cell(hide_code=True)
def _():
    # Colours shared by all figures (same conventions as 8_ODE.py)
    colors = {
        "exact": "black",
        "Euler": "darkorange",
        "RK4": "#1d4ed8",
        "data": "0.45",
        "G": "seagreen",
        "mono": "#c2410c",
        "bi": "#7c3aed",
    }
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise: Transient carrier dynamics in a solar-cell absorber

    **Topic:** ordinary differential equations, home-made solvers (Euler / RK4), differentiable solvers in JAX.

    In photovoltaic (PV) materials, the way photo-excited carriers **recombine** after a light pulse tells us
    how good the material is: the longer the carriers survive, the more of them can be collected at the contacts.
    In a *time-resolved photoluminescence* (TRPL) experiment a short laser pulse creates electron–hole pairs in a thin absorber film, and a fast detector records the light emitted while they recombine.
    """)
    return


@app.cell(hide_code=True)
def _(FancyBboxPatch, Rectangle, colors, np, plt):
    def wavy(ax, x0, y0, x1, y1, color, amp=0.12, wavelength=0.3, lw=1.8):
        """A photon: a sine wave from (x0, y0) to (x1, y1) ending in an arrow head."""
        length = np.hypot(x1 - x0, y1 - y0)
        s = np.linspace(0.0, length * 0.9, 300)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        w = amp * np.sin(2 * np.pi * s / wavelength)
        ax.plot(x0 + s * ux - w * uy, y0 + s * uy + w * ux, color=color, lw=lw)
        ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.9 * length * ux, y0 + 0.9 * length * uy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    def draw_experiment():
        """Cartoon of a TRPL experiment, next to a qualitative sketch of what is measured."""
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})

        # --- Left: the set-up
        ax.set(xlim=(0, 12), ylim=(0, 5))
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.2, 1.9), 2.2, 1.2, boxstyle="round,pad=0.1", fc="#fee2e2", ec="crimson"))
        ax.text(1.3, 2.5, "Pulsed\nlaser", ha="center", va="center", fontsize=10)
        wavy(ax, 2.6, 2.5, 5.1, 2.5, "crimson", lw=2.2)
        ax.text(3.8, 3.0, "$h\\nu$, ~100 fs", ha="center", fontsize=9, color="crimson")

        ax.add_patch(Rectangle((5.2, 0.8), 0.9, 3.4, fc="0.88", ec="0.5"))
        ax.add_patch(Rectangle((6.1, 0.8), 0.35, 3.4, fc="#78350f", ec="0.3"))
        ax.text(5.65, 0.45, "glass", ha="center", fontsize=8.5)
        ax.text(6.3, 4.4, "absorber\n(~500 nm)", ha="center", fontsize=8.5)
        _rng = np.random.default_rng(3)
        for _y in np.linspace(1.1, 3.9, 7):
            ax.plot(6.2 + 0.05 * _rng.random(), _y, "o", color="k", ms=3.5)
            ax.plot(6.37, _y + 0.15, "o", mfc="white", mec="k", ms=3.5)

        for _dy in (-0.9, 0.0, 0.9):
            wavy(ax, 6.7, 2.5 + 0.3 * _dy, 9.2, 2.5 + _dy, colors["bi"], amp=0.08, wavelength=0.22, lw=1.3)
        ax.text(7.9, 3.85, "photoluminescence", ha="center", fontsize=9, color=colors["bi"])
        ax.add_patch(FancyBboxPatch((9.4, 1.6), 2.3, 1.8, boxstyle="round,pad=0.1", fc="#ede9fe", ec=colors["bi"]))
        ax.text(10.55, 2.5, "Fast\ndetector\n(TCSPC)", ha="center", va="center", fontsize=10)
        ax.set_title("Time-resolved photoluminescence (TRPL)", fontsize=11)

        # --- Right: what is measured
        t = np.linspace(-0.15, 1.0, 600)
        pulse = np.exp(-0.5 * (t / 0.012) ** 2)
        k1, k2n0 = 1.0, 10.0  # dimensionless sketch
        n = np.where(t < 0, 0.0, k1 / ((k1 + k2n0) * np.exp(k1 * np.clip(t, 0, None) * 1.0) - k2n0))
        ax2.fill_between(t, 0, pulse, color="crimson", alpha=0.25, label="laser pulse")
        ax2.plot(t, n, color=colors["exact"], lw=2, label="excess carriers $n(t)$")
        ax2.annotate("$G=0$: only\nrecombination", xy=(0.3, 0.2), xytext=(0.45, 0.6),
                     arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=9)
        ax2.set(xlabel="time (a.u.)", ylabel="signal (a.u.)", yticks=[], xticks=[], title="What we want to model")
        ax2.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        plt.close(fig)
        return fig

    return draw_experiment, wavy


@app.cell(hide_code=True)
def _(draw_experiment):
    draw_experiment(),
    return


@app.cell(hide_code=True)
def _(Rectangle, colors, np, plt, wavy):
    def draw_band_diagram(ax, w_mono=1.0, w_bi=1.0, n_carriers=7):
        """Band diagram of the AB model; w_mono and w_bi scale the width of the recombination arrows."""
        ax.set(xlim=(0, 10), ylim=(-0.2, 6.4))
        ax.axis("off")

        ax.add_patch(Rectangle((0, 4.5), 10, 1.7, fc="#dbeafe", ec="none"))
        ax.add_patch(Rectangle((0, -0.2), 10, 1.7, fc="#fee2e2", ec="none"))
        ax.hlines(4.5, 0, 10, color=colors["RK4"], lw=2.5)
        ax.hlines(1.5, 0, 10, color="crimson", lw=2.5)
        ax.text(0.15, 5.85, "Conduction band", fontsize=10, color=colors["RK4"])
        ax.text(0.15, 0.05, "Valence band", fontsize=10, color="crimson")
        ax.text(9.85, 4.6, "$E_C$", ha="right", fontsize=11)
        ax.text(9.85, 1.6, "$E_V$", ha="right", fontsize=11)

        # electrons (filled) and holes (open)
        xs = np.linspace(2.3, 9.2, n_carriers)
        ax.plot(xs, np.full_like(xs, 4.78), "o", color="black", ms=7)
        ax.plot(xs, np.full_like(xs, 1.22), "o", mfc="white", mec="black", ms=7)

        # generation
        wavy(ax, 0.1, 3.0, 1.15, 3.0, colors["G"], amp=0.1, wavelength=0.25)
        ax.text(0.15, 3.25, "laser $h\\nu$", fontsize=9, color=colors["G"])
        ax.annotate("", xy=(1.5, 4.55), xytext=(1.5, 1.45),
                    arrowprops=dict(arrowstyle="-|>", color=colors["G"], lw=2.5, mutation_scale=18))
        ax.text(1.65, 2.2, "$G$", fontsize=13, color=colors["G"])

        # trap level and SRH path
        ax.hlines(3.0, 3.6, 5.6, color="0.3", lw=2, ls="--")
        ax.text(3.65, 3.12, "trap $E_t$", fontsize=9, color="0.3")
        _lw = 0.6 + 5 * w_mono
        ax.annotate("", xy=(4.9, 3.05), xytext=(4.9, 4.45),
                    arrowprops=dict(arrowstyle="-|>", color=colors["mono"], lw=_lw, mutation_scale=12 + 4 * w_mono))
        ax.annotate("", xy=(4.9, 1.55), xytext=(4.9, 2.95),
                    arrowprops=dict(arrowstyle="-|>", color=colors["mono"], lw=_lw, mutation_scale=12 + 4 * w_mono))
        ax.plot([4.9], [3.0], "o", color="black", ms=6)
        ax.text(3.3, 3.75, "SRH", fontsize=12, color=colors["mono"], ha="center")

        # band-to-band path
        _lw = 0.6 + 5 * w_bi
        ax.annotate("", xy=(7.4, 1.55), xytext=(7.4, 4.45),
                    arrowprops=dict(arrowstyle="-|>", color=colors["bi"], lw=_lw, mutation_scale=12 + 4 * w_bi))
        wavy(ax, 7.6, 3.0, 9.6, 3.0, colors["bi"], amp=0.1, wavelength=0.25)
        ax.text(8.55, 3.3, "PL $h\\nu\\approx E_g$", fontsize=9, color=colors["bi"], ha="center")
        ax.text(7.15, 3.5, "radiative", fontsize=12, color=colors["bi"], ha="right")

    _fig, _ax = plt.subplots(figsize=(8.5, 4.6))
    draw_band_diagram(_ax)
    _ax.set_title("Generation and recombination paths in the AB model")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When the absorber is illuminated by a short laser pulse, excess carriers of density $n$ are generated
    (electrons in the conduction band and, in equal number, holes in the valence band).
    Once the light is off ($G=0$), the excess density decays back to equilibrium through two recombination channels:

    1. **Monomolecular recombination $k_1$:** trap-assisted (Shockley–Read–Hall, SRH) recombination through
       localised defect states inside the band gap.
    2. **Bimolecular recombination $k_2$:** direct band-to-band recombination, in which an electron of the
       conduction band meets a hole of the valence band and emits a photon.

    The continuity equation is written as:
    $$
    \frac{\partial n}{\partial t} = \frac{1}{q} \frac{\partial J_n}{\partial x} + G - R ,
    $$
    With $G$ the generation rate, $R$ the recombination rate, and $J_n$ the electron current density.

    The PL intensity recorded by the detector is proportional to $n^2$ so $\phi_{\rm PL}(t) \propto n(t)^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _md = mo.md(r"""
                **Task 1**: Considering the two recombination channels and the schematic above, express the ODE for the carrier density $n(t)$ as a function of time $t$. Assume a uniform carrier density in the sample.   

                **Task 2**: Solve the ODE using `scipy.integrate.solve_ivp`  

                **Task 3** Write your own solver for the ODE using first the Euler method and then the RK4 method. Compare the results with the `scipy` results.

                **Task 4** Rewrite the RK4 solver using `jax.lax.scan` 

                **Task 5** Differentiate through it to calculate the differential decay time $\tau_{\rm diff} = -2 \left[ \frac{d \ln(\phi_{\rm PL})}{dt} \right]^{-1}$  and plot this versus time. What do you notice? 

                **Task 6** Compare the results to differential decay time calculated from the exact solution (ask teacher) and from the non-differentiable RK4 solver.

                **Task 7** Which recombination process dominates the decay? Find 2 scenarios where the other process dominates.
                """
            ).callout(kind="info")

    _md
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the following parameters for the homework questions:

    | Parameter | Symbol | Value |
    |---|---|---|
    | Initial carrier density | $n_0$ | $10^{23}\ \text{m}^{-3}$ |
    | Monomolecular (trap) rate | $k_1$ | $10^{6}\ \text{s}^{-1}$ |
    | Bimolecular coefficient | $k_2$ | $10^{-16}\ \text{m}^3\,\text{s}^{-1}$ |
    | Simulated time | $T$ | $10^{-6}\ \text{s}$ |

    All quantities are in SI units ($10^{23}\ \text{m}^{-3} = 10^{17}\ \text{cm}^{-3}$, $10^{-16}\ \text{m}^3/\text{s} = 10^{-10}\ \text{cm}^3/\text{s}$).
    """)
    return


@app.cell
def _():
    n0 = 1e23  # m^-3
    k1 = 1e6  # s^-1
    k2 = 1e-16  # m^3 / s
    t_max = 1e-6  # s
    return


if __name__ == "__main__":
    app.run()
