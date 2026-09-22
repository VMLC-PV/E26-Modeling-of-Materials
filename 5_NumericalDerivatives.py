import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # use cuda (i.e. NVIDIA GPU) if available)
    # os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    from wigglystuff import TangleLatex, FormulaAnimation, ChartPuck

    #plt.rcParams["figure.dpi"] = 150
    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return ChartPuck, FormulaAnimation, TangleLatex, mo, np, pd, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Numerical derivatives

    Many problems in computational physics and materials science need the derivative of a function that is expensive to differentiate by hand, or that is only known through a black-box evaluation: a simulation output, an experimental fit, or a quantity defined through another numerical procedure such as an integral.

    This notebook builds the standard toolbox for obtaining derivatives
    numerically.

    Derivative definition:
    \[
    f'(x) = \frac{d f(x)}{dx} = \lim_{h\to0} \frac{f(x+h)-f(x)}{h}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(FormulaAnimation, mo):
    md_fd_title = mo.md(
        """
        ## 1. Forward difference
        ---
        """
    )

    mo.Html("""
    <style>
    #fd-animation-wrap .formula-animation__formula {
      font-size: clamp(0.7rem, 1.5vw, 1.1rem) !important;
    }
    </style>
    """)

    fd_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Error for the forward difference",
            steps=[
                {"tex": r"\small f(x+h) = f(x) + (x+h-x) f'(x) + \frac{(x+h-x)^2}{2}f''(x) + \frac{(x+h-x)^3}{3!}f'''(x) + \cdots", "note": "Taylor expand f around x."},
                {
                    "tex": r"\small f(x+h) = f(x) + h f'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \mathcal{O}(h^4)",
                    "note": "Simplify.",
                },
                {"tex": r"\small f'(x) = \frac{f(x+h)-f(x)}{h} - \frac{h}{2}f''(x) + \mathcal{O}(h^2)", "note": "Rearranging the terms."},
                {
                    "tex": r"\small f'(x) = \underbrace{\frac{f(x+h)-f(x)}{h}}_{\text{Forward difference}} \underbrace{- \frac{h}{2}f''(x) }_{\text{Approximation error}}+ \mathcal{O}(h^2)",
                    "note": "Final expression.",
                },
            ],
            height=300

        )
    )
    md_fd_text = mo.md(
        r"""
        Forward difference:  
        $$
        f'(x) \approx \frac{f(x+h)-f(x)}{h}
        $$
        Approximation error:
        $$
        \delta_{\rm forw} = -\tfrac12 h f''(x) + \mathcal{O}(h^2)
        $$

        """
    )
    return fd_anim, md_fd_title


@app.cell(hide_code=True)
def _(np, plt):
    # Textbook-style curve
    def fd_fig():
        _f = lambda x: 1 - np.exp(-0.9 * x)

        # Locations
        _x0 = 2.0
        _h = 1.5
        _xp = _x0 + _h

        # Curve
        _x = np.linspace(1, 5, 500)
        _y = _f(_x)

        # Values
        _y0 = _f(_x0)
        _yp = _f(_xp)

        # Exact derivative and tangent
        _df = lambda x: 0.9 * np.exp(-0.9 * x)
        _slope = _df(_x0)

        # Tangent line
        _xtan = np.linspace(_x0 - 0.3, _xp + 0.6, 100)
        _ytan = _y0 + _slope * (_xtan - _x0)

        # Tangent value at x+h
        _ytan_p = _y0 + _slope * (_xp - _x0)

        fd_fig, _ax = plt.subplots()

        # Curve
        _ax.plot(_x, _y, color="black", lw=0.8)

        # Secant through x and x+h
        _slope = (_yp - _y0) / _h

        _xsec = np.linspace(_x0 - 0.2, _xp + 0.5, 100)
        _ysec = _y0 + _slope * (_xsec - _x0)

        # Plot secant
        _ax.plot(
            _xsec,
            _ysec,
            color="black",
            lw=0.7,
            ls="--"
        )

        # Vertical guide lines
        for _xi, _yi in [(_x0, _y0), (_xp, _yp)]:
            _ax.vlines(
                _xi, 0, _yi,
                colors="0.7",
                linestyles=(0, (3, 3)),
                lw=0.9
            )

        # Points on curve
        _ax.scatter(
            [_x0, _xp],
            [_y0, _yp],
            color="black",
            s=45,
            zorder=4
        )

        # Midpoint of interval
        _xarrow = (_x0 + _xp) / 2

        # Secant values at midpoint
        _ysec_mid = _y0 + _slope * (_xarrow - _x0)

        # Forward-difference rise
        _ax.arrow(
            _xarrow,
            0.65,
            0,
            0.25,
            width=0.015,
            head_width=0.12,
            head_length=0.04,
            length_includes_head=True,
            color="black"
        )

        _ax.text(
            _xarrow + 0.12,
            (_y0 + _ysec_mid) / 4,
            "forward\n difference",
            rotation=90,
            ha="center",
            va="center",
            fontsize=20,
            family="serif"
        )

        # Axis formatting
        _ax.set_xticks([_x0, _xp])
        _ax.set_xticklabels(
            [r"$x$", r"$x+h$"],
            fontsize=24
        )

        _ax.set_yticks([])
        _ax.set_ylabel(r"$f(x)$", fontsize=20)

        _ax.spines["top"].set_visible(False)
        # remove top ticks
        _ax.tick_params(top=False)
        _ax.spines["right"].set_visible(False)

        _ax.set_xlim(0.9, max(_x))
        _ax.set_ylim(-0.05, 1.05)
        return fd_fig


    return (fd_fig,)


@app.cell(hide_code=True)
def _(fd_anim, fd_fig, md_fd_title, mo):
    mo.vstack([md_fd_title, mo.hstack([fd_fig(),fd_anim],widths=[0.4,0.8], align='center',justify='center'),mo.hstack([
        mo.md(
        r"""
        Forward difference:  
        $$
        f'(x) \approx \frac{f(x+h)-f(x)}{h}
        $$"""),
        mo.md(r"""
        Approximation error:
        $$
        \delta_{\rm forw} = -\tfrac12 h f''(x) + \mathcal{O}(h^2)
        $$

        """)                                                                                                        ],widths=[0.5,0.5])
              ])
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Textbook-style curve
    def bd_fig():
        _f = lambda x: 1 - np.exp(-0.9 * x)

        # Locations
        _x0 = 3.5
        _h = 1.5
        _xm = _x0 - _h

        # Curve
        _x = np.linspace(1, 5, 500)
        _y = _f(_x)

        # Values
        _y0 = _f(_x0)
        _ym = _f(_xm)

        # Secant through x-h and x
        _slope = (_y0 - _ym) / _h

        fig, _ax = plt.subplots()

        # Curve
        _ax.plot(_x, _y, color="black", lw=0.8)

        _xsec = np.linspace(_xm - 0.5, _x0 + 0.4, 100)
        _ysec = _y0 + _slope * (_xsec - _x0)

        # Plot secant
        _ax.plot(
            _xsec,
            _ysec,
            color="black",
            lw=0.7,
            ls="--"
        )

        # Vertical guide lines
        for _xi, _yi in [(_xm, _ym), (_x0, _y0)]:
            _ax.vlines(
                _xi, 0, _yi,
                colors="0.7",
                linestyles=(0, (3, 3)),
                lw=0.9
            )

        # Points on curve
        _ax.scatter(
            [_xm, _x0],
            [_ym, _y0],
            color="black",
            s=45,
            zorder=4
        )

        # Midpoint of interval
        _xarrow = (_xm + _x0) / 2
        _ysec_mid = _y0 + _slope * (_xarrow - _x0)

        # Backward-difference rise
        _ax.arrow(
            _xarrow,
            0.65,
            0,
            0.25,
            width=0.015,
            head_width=0.12,
            head_length=0.04,
            length_includes_head=True,
            color="black"
        )

        _ax.text(
            _xarrow + 0.12,
            (_ym + _ysec_mid) / 4,
            "backward\n difference",
            rotation=90,
            ha="center",
            va="center",
            fontsize=20,
            family="serif"
        )

        # Axis formatting
        _ax.set_xticks([_xm, _x0])
        _ax.set_xticklabels(
            [r"$x-h$", r"$x$"],
            fontsize=24
        )

        _ax.set_yticks([])
        _ax.set_ylabel(r"$f(x)$", fontsize=20)

        _ax.spines["top"].set_visible(False)
        _ax.tick_params(top=False)
        _ax.spines["right"].set_visible(False)

        _ax.set_xlim(0.9, max(_x))
        _ax.set_ylim(-0.05, 1.05)
        return fig

    return (bd_fig,)


@app.cell(hide_code=True)
def _(FormulaAnimation, bd_fig, mo):
    md_bd_title = mo.md(
        """
        ## 2. Backward difference
        ---
        """
    )

    mo.Html("""
    <style>
    #fd-animation-wrap .formula-animation__formula {
      font-size: clamp(0.7rem, 1.5vw, 1.1rem) !important;
    }
    </style>
    """)

    bd_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Error for the backward difference",
            steps=[
                {"tex": r"\small f(x-h) = f(x) + (x-h-x) f'(x) + \frac{(x-h-x)^2}{2}f''(x) + \frac{(x-h-x)^3}{3!}f'''(x) + \cdots", "note": "Taylor expand f around x."},
                {
                    "tex": r"\small f(x-h) = f(x) - h f'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + \mathcal{O}(h^4)",
                    "note": "Simplify.",
                },
                {"tex": r"\small f'(x) = \frac{f(x)-f(x-h)}{h} + \frac{h}{2}f''(x) + \mathcal{O}(h^2)", "note": "Rearranging the terms."},
                {
                    "tex": r"\small f'(x) = \underbrace{\frac{f(x)-f(x-h)}{h}}_{\text{Backward difference}} \underbrace{+ \frac{h}{2}f''(x)}_{\text{Approximation error}} + \mathcal{O}(h^2)",
                    "note": "Final expression.",
                },
            ],
            height=300

        )
    )

    mo.vstack(
        [
            md_bd_title,
            mo.hstack([bd_fig(), bd_anim], widths=[0.4, 0.8], align="center", justify="center"),
            mo.hstack([
                mo.md(
                r"""
                Backward difference:  
                $$
                f'(x) \approx \frac{f(x)-f(x-h)}{h}
                $$"""),
                mo.md(r"""
                Approximation error:
                $$
                 \delta_{\rm back} = \tfrac12 h f''(x) + \mathcal{O}(h^2)
                $$

                """)                                                                                                        ],widths=[0.5,0.5])
        ]
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Textbook-style curve
    def cd_fig():
        _f = lambda x: 1 - np.exp(-0.9 * x)

        # Locations
        _x0 = 2.75
        _h = 0.75
        _xm = _x0 - _h
        _xp = _x0 + _h

        # Curve
        _x = np.linspace(1, 5, 500)
        _y = _f(_x)

        # Values
        _y0 = _f(_x0)
        _ym = _f(_xm)
        _yp = _f(_xp)

        # Secant through x-h and x+h
        _slope = (_yp - _ym) / (2.0 * _h)
        _xsec = np.linspace(_xm - 0.4, _xp + 0.4, 100)
        _ysec = _ym + _slope * (_xsec - _xm)

        fig, _ax = plt.subplots()

        # Curve
        _ax.plot(_x, _y, color="black", lw=0.8)

        # Plot secant
        _ax.plot(
            _xsec,
            _ysec,
            color="black",
            lw=0.7,
            ls="--"
        )

        # Vertical guide lines
        for _xi, _yi in [(_xm, _ym), (_x0, _y0), (_xp, _yp)]:
            _ax.vlines(
                _xi, 0, _yi,
                colors="0.7",
                linestyles=(0, (3, 3)),
                lw=0.9
            )

        # Points used in the difference formula
        _ax.scatter(
            [_xm, _xp],
            [_ym, _yp],
            color="black",
            s=45,
            zorder=4
        )
        # Evaluation point x
        _ax.scatter([_x0], [_y0], color="0.4", s=25, zorder=4)

        # Central-difference rise
        _ax.arrow(
            _x0,
            0.65,
            0,
            0.2,
            width=0.015,
            head_width=0.12,
            head_length=0.04,
            length_includes_head=True,
            color="black"
        )

        _ax.text(
            _x0,
            (_ym + _yp) / 4,
            "central\n difference",
            rotation=90,
            ha="center",
            va="center",
            fontsize=20,
            family="serif"
        )

        # Axis formatting
        _ax.set_xticks([_xm, _x0, _xp])
        _ax.set_xticklabels(
            [r"$x-h$", r"$x$", r"$x+h$"],
            fontsize=20
        )

        _ax.set_yticks([])
        _ax.set_ylabel(r"$f(x)$", fontsize=20)

        _ax.spines["top"].set_visible(False)
        _ax.tick_params(top=False)
        _ax.spines["right"].set_visible(False)

        _ax.set_xlim(0.9, max(_x))
        _ax.set_ylim(-0.05, 1.05)
        return fig


    return (cd_fig,)


@app.cell(hide_code=True)
def _(FormulaAnimation, cd_fig, mo):
    md_cd_title = mo.md(
        """
        ## 3. Central difference
        ---
        """
    )

    mo.Html("""
    <style>
    #fd-animation-wrap .formula-animation__formula {
      font-size: clamp(0.7rem, 1.5vw, 1.1rem) !important;
    }
    </style>
    """)

    cd_anim = mo.ui.anywidget(
        FormulaAnimation(
            title="Error for the central difference",
            steps=[
                {
                    "tex": r"\small f(x+h)=f(x)+hf'(x)+\frac{h^2}{2}f''(x)+\frac{h^3}{6}f'''(x)+\frac{h^4}{24}f^{(4)}(x)+\cdots",
                    "note": "Taylor expansion of f(x+h) about x.",
                },
                {
                    "tex": r"\small f(x-h)=f(x)-hf'(x)+\frac{h^2}{2}f''(x)-\frac{h^3}{6}f'''(x)+\frac{h^4}{24}f^{(4)}(x)+\cdots",
                    "note": "Taylor expansion of f(x-h) about x.",
                },
                {
                    "tex": r"\small f(x+h)-f(x-h)=\Bigl(f(x)+hf'(x)+\frac{h^2}{2}f''(x)+\frac{h^3}{6}f'''(x)+\cdots\Bigr)",
                    "note": "Subtract the two expansions.",
                },
                {
                    "tex": r"\small \qquad\qquad\qquad\qquad-\Bigl(f(x)-hf'(x)+\frac{h^2}{2}f''(x)-\frac{h^3}{6}f'''(x)+\cdots\Bigr)",
                    "note": "Substitute the second expansion.",
                },
                {
                    "tex": r"\small f(x+h)-f(x-h)=2hf'(x)+\frac{2h^3}{6}f'''(x)+\frac{2h^5}{120}f^{(5)}(x)+\cdots",
                    "note": "Even-order terms cancel.",
                },
                {
                    "tex": r"\small f(x+h)-f(x-h)=2hf'(x)+\frac{h^3}{3}f'''(x)+\frac{h^5}{60}f^{(5)}(x)+\cdots",
                    "note": "Simplify coefficients.",
                },
                {
                    "tex": r"\small \frac{f(x+h)-f(x-h)}{2h}=f'(x)+\frac{h^2}{6}f'''(x)+\frac{h^4}{120}f^{(5)}(x)+\cdots",
                    "note": "Divide by 2h.",
                },
                {
                    "tex": r"\small f'(x)=\frac{f(x+h)-f(x-h)}{2h}-\frac{h^2}{6}f'''(x)+\mathcal{O}(h^3)",
                    "note": "Rearrange to isolate f'(x).",
                },
                {
                    "tex": r"\small f'(x)=\underbrace{\frac{f(x+h)-f(x-h)}{2h}}_{\text{Central difference}}\underbrace{-\frac{h^2}{6}f'''(x)}_{\text{Approximation error}}+\mathcal{O}(h^3)",
                    "note": "Final expression.",
                },
            ],
            height=300,
        )
    )

    mo.vstack(
        [
            md_cd_title,
            mo.hstack([cd_fig(), cd_anim], widths=[0.4, 0.8], align="center", justify="center"),
            mo.hstack([
                mo.md(
                r"""
                Central difference:  
                $$
                f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}
                $$"""),
                mo.md(r"""
                Approximation error:
                $$
                 \delta_{\rm cent} = \tfrac16 h^2 f'''(x) + \mathcal{O}(h^3)
                $$

                """)                                                                                                        ],widths=[0.5,0.5])
        ]
    )
    return


@app.cell(hide_code=True)
def _(TangleLatex, mo):
    formula_widget = mo.ui.anywidget(
        TangleLatex(
            latex=r"f(x) = \tangle{a}\, e^{\tangle{b}\, x}",
            parameters={
                "a": {
                    "value": 1.0,
                    "min_value": 0.3,
                    "max_value": 3.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "a",
                    "label": "Amplitude",
                },
                "b": {
                    "value": 1.0,
                    "min_value": 0.3,
                    "max_value": 2.5,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "b",
                    "label": "Rate",
                },
            },
            editor="inline",
            reveal_all_on_drag=True,
        )
    )
    return (formula_widget,)


@app.cell
def _(formula_widget, np):
    a_val = float(formula_widget.values["a"])
    b_val = float(formula_widget.values["b"])

    def f(x):
        return a_val * np.exp(b_val * x)

    def df1_exact(x):
        return a_val * b_val**1 * np.exp(b_val * x)

    def df2_exact(x):
        return a_val * b_val**2 * np.exp(b_val * x)

    def df3_exact(x):
        return a_val * b_val**3 * np.exp(b_val * x)

    def df4_exact(x):
        return a_val * b_val**4 * np.exp(b_val * x)

    def df5_exact(x):
        return a_val * b_val**5 * np.exp(b_val * x)

    return (
        a_val,
        b_val,
        df1_exact,
        df2_exact,
        df3_exact,
        df4_exact,
        df5_exact,
        f,
    )


@app.cell(hide_code=True)
def _(mo):
    x0_slider = mo.ui.slider(-1.0, 1.5, value=0.5, step=0.1, label="Eval. point $x_0$")
    return (x0_slider,)


@app.cell(hide_code=True)
def _(a_val, b_val, df1_exact, f, formula_widget, mo, np, plt, x0_slider):
    _x0 = x0_slider.value
    _xplot = np.linspace(-1.5, 2.0, 400)

    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=2, label="$f(x)$")
    _ax.axvline(_x0, color="black", lw=0.8, ls="--")
    _ax.scatter([_x0], [f(_x0)], color="black", zorder=3)
    _ax.set(xlabel="$x$", ylabel="$f(x)$",xlim=(-1.5, 1.5), ylim=(-0.5, 3))
    _ax.legend()

    md_ref_title = mo.md(
        """
        ### A reference example
        ---
        """
    )
    md_ref_text = mo.md(
        rf"""
        Consider $f(x) = {a_val:g}\,e^{{{b_val:g}x}}$. Because $f$ reproduces itself under differentiation, $f^{{(n)}}(x) = a\,b^n\,e^{{bx}}$, so every derivative is known in closed form, a convenient reference against which any numerical scheme can be checked exactly.  
        At the marked point $x_0={_x0:g}$, the exact derivative is:  
        $f'(x_0)={df1_exact(_x0):.6g}$.
        """
    )

    mo.vstack(
        [
            md_ref_title,
            md_ref_text,
            mo.hstack([mo.vstack([formula_widget, x0_slider]),_fig], align="center",justify="center"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    # md_fd_title = mo.md(
    #     """
    #     ## 2. Forward, backward and central differences
    #     ---
    #     """
    # )
    # md_fd_text = mo.md(
    #     r"""
    #     The most direct route to a derivative is Taylor's theorem. Expanding
    #     $f$ around $x$,
    #     $$
    #     f(x+h) = f(x) + h f'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \cdots,
    #     $$
    #     and rearranging gives three elementary finite-difference formulas.

    #     **Forward difference**
    #     $$
    #     f'(x) \approx \frac{f(x+h)-f(x)}{h}, \qquad
    #     R_{\rm forw} = -\tfrac12 h f''(x) + \mathcal{O}(h^2).
    #     $$

    #     **Backward difference** — expand around $x-h$ instead
    #     $$
    #     f'(x) \approx \frac{f(x)-f(x-h)}{h}, \qquad
    #     R_{\rm back} = \tfrac12 h f''(x) + \mathcal{O}(h^2).
    #     $$

    #     **Central difference** — average the two, which cancels the
    #     $\mathcal{O}(h)$ term
    #     $$
    #     f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}, \qquad
    #     R_{\rm cent} = -\tfrac16 h^2 f'''(x) + \mathcal{O}(h^3).
    #     $$

    #     Geometrically, each formula is the slope of a **secant line**
    #     through one or two nearby points, while the true derivative is the
    #     slope of the **tangent line**. Drag the point below to move $x_0$
    #     along the curve, and use the controls to change the method and the
    #     step size $h$: watch the secant rotate onto the tangent as
    #     $h\to0$ — and see the next section for why shrinking $h$ forever is
    #     *not* actually a good idea on a computer.
    #     """
    # )
    return


@app.cell
def _():
    def df_forward(f, x, h):
        return (f(x + h) - f(x)) / h

    def df_backward(f, x, h):
        return (f(x) - f(x - h)) / h

    def df_central(f, x, h):
        return (f(x + h) - f(x - h)) / (2.0 * h)

    return df_backward, df_central, df_forward


@app.cell(hide_code=True)
def _(mo):
    secant_method = mo.ui.dropdown(
        options={"Forward": "forward", "Backward": "backward", "Central": "central"},
        value="Central",
        label="Method:",
    )
    secant_h_exp = mo.ui.slider(-2.0, 0.0, value=-0.5, step=0.05, label="Step size, $\\log_{10}h$")
    return secant_h_exp, secant_method


@app.cell(hide_code=True)
def _(ChartPuck, df1_exact, f, np, secant_h_exp, secant_method):
    # Fixed axes: computed once from the reference function, independent of
    # dragging the point or changing the method/step-size controls below.
    _x_bounds = (-1.5, 2.0)
    secant_xgrid = np.linspace(*_x_bounds, 300)
    _yvals = f(secant_xgrid)
    _pad = 0.15 * (_yvals.max() - _yvals.min())
    # _y_bounds = (_yvals.min() - _pad, _yvals.max() + _pad)
    _y_bounds = (0, 6)

    def _secant_points(x0, method, h):
        if method == "forward":
            return np.array([x0, x0 + h])
        if method == "backward":
            return np.array([x0 - h, x0])
        return np.array([x0 - h, x0 + h])

    def _draw_secant(ax, widget):
        _x0 = float(widget.x[0])

        # The puck can be dragged anywhere within the axes, but its y is
        # always snapped back onto the curve so it visually rides f(x).
        # Setting widget.y triggers a nested call to this same function
        # (with the corrected y) that does the actual drawing, so this
        # call must return immediately instead of drawing a second time.
        _y_on_curve = float(f(_x0))
        if widget.y[0] != _y_on_curve:
            widget.y = [_y_on_curve]
            return

        _method = secant_method.value
        _h = 10.0 ** secant_h_exp.value
        _exact = df1_exact(_x0)
        _x_pts = _secant_points(_x0, _method, _h)
        _y_pts = f(_x_pts)

        ax.plot(secant_xgrid, f(secant_xgrid), color="crimson", lw=2, label="$f(x)$")
        ax.plot(secant_xgrid, f(_x0) + _exact * (secant_xgrid - _x0), "k--", lw=1.5, label="Tangent (exact)")
        ax.plot(_x_pts, _y_pts, color="steelblue", lw=2, marker="o", ms=6, label=f"{_method.capitalize()} secant")
        ax.axvline(_x0, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="$x$", ylabel="$f(x)$")
        ax.legend(loc="upper left", )#fontsize=8)

    secant_chart_puck = ChartPuck.from_callback(
        draw_fn=_draw_secant,
        x_bounds=_x_bounds,
        y_bounds=_y_bounds,
        figsize=(6.5, 4.5),
        x=0.5,
        y=float(f(0.5)),
        drag_x_bounds=(_x_bounds[0] + 0.1, _x_bounds[1] - 0.1),
        puck_color="#1d4ed8",
    )
    return (secant_chart_puck,)


@app.cell(hide_code=True)
def _(mo, secant_chart_puck):
    secant_puck = mo.ui.anywidget(secant_chart_puck)
    return (secant_puck,)


@app.cell(hide_code=True)
def _(
    df1_exact,
    df_backward,
    df_central,
    df_forward,
    f,
    mo,
    secant_h_exp,
    secant_method,
    secant_puck,
):
    _x0 = float(secant_puck.value["x"][0])
    _h = 10.0 ** secant_h_exp.value
    _method = secant_method.value

    if _method == "forward":
        _estimate = df_forward(f, _x0, _h)
    elif _method == "backward":
        _estimate = df_backward(f, _x0, _h)
    else:
        _estimate = df_central(f, _x0, _h)

    _exact = df1_exact(_x0)
    _viz_md_title = mo.md(
        r"""###Visualization of the different methods.
            ---""")
    mo.vstack(
        [
            _viz_md_title,
            mo.hstack([secant_method, secant_h_exp], align="center", justify="center"),
            mo.hstack([secant_puck,
            mo.vstack(
                [
                    mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Exact", value=f"{_exact:.6f}"),
                    mo.stat(label="Absolute error", value=f"{abs(_estimate - _exact):.3e}"),
                ]
            ),])
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_conv_title = mo.md(
        """
        ## 4. Convergence and the optimal step size
        ---
        """
    )
    return (md_conv_title,)


@app.cell(hide_code=True)
def _(np):
    epsm = np.finfo(float).eps

    def error_scan(diff_fn, f, x0, h_values, exact, higher_deriv, trunc_coeff, trunc_order, round_coeff, round_order=1):
        estimates = np.array([diff_fn(f, x0, h) for h in h_values])
        rel_err = np.abs((estimates - exact) / exact)
        trunc = np.abs(trunc_coeff * h_values**trunc_order * higher_deriv / exact)
        total = trunc + round_coeff * epsm / h_values**round_order
        return rel_err, trunc, total

    return (error_scan,)


@app.cell(hide_code=True)
def _(mo):
    h_exp_min = mo.ui.slider(-16, -4, value=-5, step=1, label="Smallest step explored, $\\log_{10}h_{\\min}$")
    return (h_exp_min,)


@app.cell(hide_code=True)
def _(h_exp_min, np):
    h_values = np.logspace(0.0, float(h_exp_min.value), 200)
    return (h_values,)


@app.cell(hide_code=True)
def _(
    df1_exact,
    df2_exact,
    df3_exact,
    df_backward,
    df_central,
    df_forward,
    error_scan,
    f,
    h_values,
    x0_slider,
):
    _x0 = x0_slider.value
    _exact1 = df1_exact(_x0)

    err_forw, trunc_forw, total_forw = error_scan(df_forward, f, _x0, h_values, _exact1, df2_exact(_x0), 0.5, 1, 2.0)
    err_back, trunc_back, total_back = error_scan(df_backward, f, _x0, h_values, _exact1, df2_exact(_x0), 0.5, 1, 2.0)
    err_cent, trunc_cent, total_cent = error_scan(df_central, f, _x0, h_values, _exact1, df3_exact(_x0), 1.0 / 6.0, 2, 1.0)
    return err_back, err_cent, err_forw, total_back, total_cent, total_forw


@app.cell(hide_code=True)
def _(
    err_back,
    err_cent,
    err_forw,
    h_exp_min,
    h_values,
    md_conv_title,
    mo,
    np,
    plt,
    total_back,
    total_cent,
    total_forw,
):
    _fig, _ax = plt.subplots()
    _ax.loglog(h_values, err_forw, ".", color="crimson", ms=4, label="Forward (measured)")
    _ax.loglog(h_values, total_forw, color="crimson", lw=1, ls="--", label="Forward (theory)")
    _ax.loglog(h_values, err_back, ".", color="steelblue", ms=4, label="Backward (measured)")
    _ax.loglog(h_values, total_back, color="steelblue", lw=1, ls="--", label="Backward (theory)")
    _ax.loglog(h_values, err_cent, ".", color="seagreen", ms=4, label="Central (measured)")
    _ax.loglog(h_values, total_cent, color="seagreen", lw=1, ls="--", label="Central (theory)")
    _ax.set(xlabel="h", ylabel="Relative error", ylim=(1e-16, 1e2))
    _ax.legend(fontsize=8, ncol=2)

    _h_opt_forw = h_values[np.argmin(total_forw)]
    _h_opt_back = h_values[np.argmin(total_back)]
    _h_opt_cent = h_values[np.argmin(total_cent)]


    if h_exp_min.value < -5:
        md_conv_text = mo.md(
                r"""
                The error formulas above suggest a simple recipe: make $h$ as small
                as possible. Truncation error indeed shrinks as $h\to0$, but every
                floating-point evaluation of $f$ carries a relative round-off error
                of order the machine epsilon, $\epsilon_m\approx2\times10^{-16}$ in
                double precision. Subtracting two nearby, nearly equal numbers
                amplifies that noise by $1/h$:  
                This is **catastrophic cancellation**.  
                The total error then becomes:
                $$
                \varepsilon(h) \approx \underbrace{C_{\mathrm{trunc}}\,h^{p}}_{\text{truncation}} + \underbrace{C_{\mathrm{round}}\,\frac{\epsilon_m}{h}}_{\text{round-off}},
                $$
                a competition between a term that shrinks and a term that grows as
                $h\to0$. Its minimum defines an **optimal step size** $h_{\mathrm{opt}}$
                from which using a smaller $h$ only makes the answer *worse*.
            """

        )
    else:
        md_conv_text = mo.md(
            r"""
            The error formulas derived above show that the finite-difference
            approximation is not exact.  
            Replacing the derivative by a finite difference neglects higher-order terms from the Taylor expansion. This introduces a **truncation error** whose magnitude depends on the step size $h$.

            | Method | Leading truncation error | Order of accuracy |
            |----------|----------|----------|
            | Forward Difference | $\varepsilon_{\mathrm{trunc}} \propto h$ | First order, $\mathcal{O}(h)$ |
            | Backward Difference | $\varepsilon_{\mathrm{trunc}} \propto h$ | First order, $\mathcal{O}(h)$ |
            | Central Difference | $\varepsilon_{\mathrm{trunc}} \propto h^2$ | Second order, $\mathcal{O}(h^2)$ |

            As $h$ decreases, the omitted Taylor-series terms become smaller and
            the numerical derivative becomes more accurate. This is why reducing
            the step size generally improves the approximation.

            ###***What happens if we keep decreasing h?***
            """
        )

    mo.vstack(
        [
            md_conv_title,
            md_conv_text,
            h_exp_min,
            mo.hstack(
                [_fig,
                mo.vstack([mo.stat(label="Optimal h, forward/backward", value=f"{_h_opt_forw:.2e}"),
                    mo.stat(label="Optimal h, central", value=f"{_h_opt_cent:.2e}"),
                ])]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_hi_title = mo.md(
        """
        ## 5. A higher-order formula
        ---
        """
    )
    md_hi_text = mo.md(
        r"""
        Using more points buys a higher order. Combining the Taylor
        expansions of $f(x\pm h)$ and $f(x\pm2h)$ so as to cancel both the
        $h$ and $h^2$ error terms gives the **5-point central formula**
        $$
        f'(x) \approx \frac{-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}{12h}, \qquad
        \delta_{\rm cent5}= \frac{h^4}{30}f^{(5)}(x) + \mathcal{O}(h^5).
        $$
        It is $\mathcal{O}(h^4)$ — two orders better than the plain central
        difference — at the cost of two extra function evaluations. The
        round-off penalty grows too, so the optimal step size shifts to a
        somewhat *larger* $h$; nevertheless the best achievable error drops
        substantially.
        """
    )
    return md_hi_text, md_hi_title


@app.function
def df_central5(f, x, h):
    return (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)


@app.cell(hide_code=True)
def _(df1_exact, df5_exact, error_scan, f, h_values, x0_slider):
    _x0 = x0_slider.value
    _exact1 = df1_exact(_x0)
    err_cent5, trunc_cent5, total_cent5 = error_scan(df_central5, f, _x0, h_values, _exact1, df5_exact(_x0), 1.0 / 30.0, 4, 1.5)
    return err_cent5, total_cent5


@app.cell(hide_code=True)
def _(
    err_cent,
    err_cent5,
    h_values,
    md_hi_text,
    md_hi_title,
    mo,
    np,
    plt,
    total_cent,
    total_cent5,
):
    _fig, _ax = plt.subplots()
    _ax.loglog(h_values, err_cent, ".", color="seagreen", ms=4, label=r"Central, $\mathcal{O}(h^2)$")
    _ax.loglog(h_values, total_cent, color="seagreen", lw=1, ls="--")
    _ax.loglog(h_values, err_cent5, ".", color="purple", ms=4, label=r"5-point central, $\mathcal{O}(h^4)$")
    _ax.loglog(h_values, total_cent5, color="purple", lw=1, ls="--")
    _ax.set(xlabel="$h$", ylabel="Relative error", ylim=(1e-16, 1e2))
    _ax.legend()

    _h_opt3 = h_values[np.argmin(total_cent)]
    _err_opt3 = np.min(total_cent)
    _h_opt5 = h_values[np.argmin(total_cent5)]
    _err_opt5 = np.min(total_cent5)

    mo.vstack(
        [
            md_hi_title,
            md_hi_text,
            mo.hstack(
                [_fig,
                mo.vstack([mo.stat(label="Central: best error", value=f"{_err_opt3:.2e}", caption=f"at h≈{_h_opt3:.1e}"),
                    mo.stat(label="5-point: best error", value=f"{_err_opt5:.2e}", caption=f"at h≈{_h_opt5:.1e}"),])
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_d2_title = mo.md(
        """
        ## 6. Second derivatives
        ---
        """
    )
    md_d2_text = mo.md(
        r"""
        Differentiating the central-difference formula again (or expanding
        $f(x\pm h)$ to fourth order and adding) gives the standard
        three-point formula for the second derivative:
        $$
        f''(x) \approx \frac{f(x+h)-2f(x)+f(x-h)}{h^2}, \qquad
        \delta = -\frac{h^2}{12}f^{(4)}(x) + \mathcal{O}(h^4).
        $$
        It is again $\mathcal{O}(h^2)$ in truncation error, but the
        round-off term is now divided by $h^2$ rather than $h$ — the
        cancellation is more severe, and the optimal step size is
        noticeably larger than for the first derivative.
        """
    )
    return md_d2_text, md_d2_title


@app.function
def d2f_central(f, x, h):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / h**2


@app.cell
def _(df2_exact, df4_exact, error_scan, f, h_values, x0_slider):
    _x0 = x0_slider.value
    _exact2 = df2_exact(_x0)
    err_d2, trunc_d2, total_d2 = error_scan(d2f_central, f, _x0, h_values, _exact2, df4_exact(_x0), 1.0 / 12.0, 2, 4.0, round_order=2)
    return err_d2, total_d2


@app.cell
def _(err_d2, h_values, md_d2_text, md_d2_title, mo, np, plt, total_d2):
    _fig, _ax = plt.subplots()
    _ax.loglog(h_values, err_d2, ".", color="darkorange", ms=4, label="Central 2nd derivative (measured)")
    _ax.loglog(h_values, total_d2, color="darkorange", lw=1, ls="--", label="Theory (truncation + round-off)")
    _ax.set(xlabel="$h$", ylabel="Relative error", ylim=(1e-16, 1e2))
    _ax.legend()

    _h_opt_d2 = h_values[np.argmin(total_d2)]

    mo.vstack(
        [
            md_d2_title,
            md_d2_text,
            _fig,
            mo.stat(label="Optimal $h$", value=f"{_h_opt_d2:.2e}"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_ad_title = mo.md(
        """
        ## 7. Automatic differentiation
        ---
        """
    )
    md_ad_text = mo.md(
        r"""
        Finite differences always face the truncation/round-off trade-off
        above, because they only ever *sample* $f$ near a point.
        **Automatic differentiation (AD)** takes a different approach: it
        walks through the sequence of elementary operations (`+`, `*`,
        `exp`, ...) that make up the code for $f$, and applies the chain
        rule to each one exactly, in floating point. There is no step size
        $h$ and no truncation error at all — only ordinary round-off, at
        the level of $\epsilon_m$.

        Two "modes" propagate the chain rule in different directions:

        - **Forward mode** carries a derivative alongside the value, from
          inputs to outputs — cheap when there are few inputs.
        - **Reverse mode** ("backpropagation") first runs forward, then
          propagates sensitivities backwards from the output — cheap when
          there are few outputs, however many inputs there are.

        Very nice Youtube video for more details: [What is Automatic Differentiation?](https://youtu.be/wG_nF1awSSY?si=CR3Qdq4N81JMSZ5m)

        Below, [`jax`](https://docs.jax.dev/) provides `jax.jvp` (forward)
        and `jax.grad` (reverse); [`mygrad`](https://mygrad.readthedocs.io/)
        gives a reverse-mode alternative with a NumPy-like interface. All
        three differentiate the *same* reference function from Section 1.
        """
    )
    return md_ad_text, md_ad_title


@app.cell
def _(a_val, b_val):
    import jax 
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax import grad, jvp
    import mygrad as mg

    def f_jax(x):
        return a_val * jnp.exp(b_val * x)

    def f_mygrad(x):
        return a_val * mg.exp(b_val * x)

    def d_forward_ad(func, x):
        _, dy = jvp(func, (x,), (1.0,))
        return dy

    def d_reverse_ad(func, x):
        return grad(func)(x)

    def d_reverse_mygrad(func, x):
        xx = mg.Tensor(x)
        y = func(xx)
        y.backward()
        return xx.grad

    return d_forward_ad, d_reverse_ad, d_reverse_mygrad, f_jax, f_mygrad, jnp


@app.cell
def _(
    d_forward_ad,
    d_reverse_ad,
    d_reverse_mygrad,
    df1_exact,
    f_jax,
    f_mygrad,
    np,
    pd,
):
    _xs = np.linspace(-1.0, 1.5, 6)
    _rows = [
        {
            "x": round(float(_x), 2),
            "Exact": df1_exact(_x),
            "AD forward (jax)": float(d_forward_ad(f_jax, _x)),
            "AD reverse (jax)": float(d_reverse_ad(f_jax, _x)),
            "AD reverse (mygrad)": float(d_reverse_mygrad(f_mygrad, _x)),
        }
        for _x in _xs
    ]
    ad_table_df = pd.DataFrame(_rows)
    return (ad_table_df,)


@app.cell
def _(ad_table_df, md_ad_text, md_ad_title, mo):
    mo.vstack([md_ad_title, md_ad_text, mo.ui.table(ad_table_df, page_size=6)])
    return


@app.cell(hide_code=True)
def _(mo):
    md_admode_title = mo.md(
        """
        ### Forward mode, worked example
        ---
        """
    )
    md_admode_text = mo.md(
        r"""
        To see forward mode concretely, take
        $$
        f(x_1,x_2) = \Bigl[\sin\!\bigl(\tfrac{x_1}{x_2}\bigr) + \tfrac{x_1}{x_2} - e^{x_2}\Bigr]\Bigl[\tfrac{x_1}{x_2} - e^{x_2}\Bigr].
        $$
        Following the usual AD convention, the two inputs are the "seed" nodes
        $v_{-1}=x_1$ and $v_0=x_2$, and every elementary operation that touches
        them becomes one more node $v_1, v_2, \dots$ of a **computational
        graph** — drawn below as a two-row network, since some intermediate
        values (like $v_1$ and $v_4$) are reused by *two* later operations.

        Forward mode carries a **value** and a **tangent** $\dot v_i$ — the
        derivative of $v_i$ along whichever input direction was "seeded" with
        a 1 — together through every node, applying the chain rule locally,
        one elementary operation at a time. Drag the slider to grow the graph
        node by node and watch the value and the local tangent rule appear
        together.
        """
    )
    return md_admode_text, md_admode_title


@app.cell(hide_code=True)
def _():
    ad_nodes = [
        dict(id="vm1", label=r"$v_{-1}$", op="input", formula=r"$v_{-1}=x_1$",
             tangent=r"$\dot v_{-1}=\dot x_1$", pos=(0.0, 1.4), parents=[]),
        dict(id="v0", label=r"$v_{0}$", op="input", formula=r"$v_0=x_2$",
             tangent=r"$\dot v_0=\dot x_2$", pos=(0.0, 0.0), parents=[]),
        dict(id="v1", label=r"$v_1$", op="divide", formula=r"$v_1=v_{-1}/v_0$",
             tangent=r"$\dot v_1=\dfrac{\dot v_{-1}\,v_0-v_{-1}\,\dot v_0}{v_0^{2}}$",
             pos=(1.0, 1.4), parents=["vm1", "v0"]),
        dict(id="v2", label=r"$v_2$", op="sin", formula=r"$v_2=\sin(v_1)$",
             tangent=r"$\dot v_2=\cos(v_1)\,\dot v_1$",
             pos=(2.0, 1.4), parents=["v1"]),
        dict(id="v3", label=r"$v_3$", op="exp", formula=r"$v_3=\exp(v_0)$",
             tangent=r"$\dot v_3=\exp(v_0)\,\dot v_0=v_3\,\dot v_0$",
             pos=(2.0, 0.0), parents=["v0"]),
        dict(id="v4", label=r"$v_4$", op="subtract", formula=r"$v_4=v_1-v_3$",
             tangent=r"$\dot v_4=\dot v_1-\dot v_3$",
             pos=(3.0, 0.0), parents=["v1", "v3"]),
        dict(id="v5", label=r"$v_5$", op="add", formula=r"$v_5=v_2+v_4$",
             tangent=r"$\dot v_5=\dot v_2+\dot v_4$",
             pos=(4.0, 1.4), parents=["v2", "v4"]),
        dict(id="v6", label=r"$v_6$", op="multiply", formula=r"$v_6=v_5\cdot v_4=f(x_1,x_2)$",
             tangent=r"$\dot v_6=\dot v_5\,v_4+v_5\,\dot v_4=\dot f$",
             pos=(5.0, 0.7), parents=["v5", "v4"]),
    ]
    return (ad_nodes,)


@app.cell(hide_code=True)
def _(mo):
    ad_step_slider = mo.ui.slider(1, 8, value=1, step=1, label="Build step (reveal next node)")
    return (ad_step_slider,)


@app.cell(hide_code=True)
def _(ad_nodes, plt):
    def ad_graph_fig(step):
        fig, ax = plt.subplots(figsize=(11, 3.4))
        id_to_node = {nd["id"]: nd for nd in ad_nodes}
        visible = ad_nodes[:step]
        visible_ids = {nd["id"] for nd in visible}

        for nd in visible:
            x, y = nd["pos"]
            for pid in nd["parents"]:
                if pid in visible_ids:
                    px, py = id_to_node[pid]["pos"]
                    rad = 0.18 if py != y else 0.0
                    ax.annotate(
                        "", xy=(x - 0.30, y), xytext=(px + 0.30, py),
                        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.3,
                                        connectionstyle=f"arc3,rad={rad}",
                                        shrinkA=0, shrinkB=0),
                    )

        new_id = visible[-1]["id"]
        for nd in visible:
            x, y = nd["pos"]
            is_new = nd["id"] == new_id
            edge = "crimson" if is_new else "black"
            face = "#fdecea" if is_new else "white"
            ax.add_patch(plt.Circle((x, y), 0.30, facecolor=face, edgecolor=edge, lw=2.0, zorder=3))
            ax.text(x, y, nd["label"], ha="center", va="center", fontsize=13, zorder=4)
            ax.text(x, y + 0.46, nd["op"].upper(), ha="center", va="bottom",
                    fontsize=9.5, color="0.35", style="italic", zorder=4)
            ax.text(x, y - 0.46, nd["formula"], ha="center", va="top", fontsize=10.5, zorder=4)

        ax.set_xlim(-0.7, 5.7)
        ax.set_ylim(-0.7, 2.05)
        ax.axis("off")
        fig.tight_layout()
        return fig

    return (ad_graph_fig,)


@app.cell(hide_code=True)
def _(ad_nodes, mo):
    def ad_trace_md(step):
        header = "| node | value | forward-mode tangent |\n|---|---|---|\n"
        body = "\n".join(
            f"| {nd['label']} | {nd['formula']} | {nd['tangent']} |"
            for nd in ad_nodes[:step]
        )
        return mo.md(header + body)

    return (ad_trace_md,)


@app.cell(hide_code=True)
def _(
    ad_graph_fig,
    ad_step_slider,
    ad_trace_md,
    md_admode_text,
    md_admode_title,
    mo,
):
    mo.vstack(
        [
            md_admode_title,
            md_admode_text,
            ad_step_slider,
            ad_graph_fig(ad_step_slider.value),
            ad_trace_md(ad_step_slider.value),
        ]
    )
    return


@app.cell
def _(mo):
    md_bwmode_title = mo.md(
        """
        ### Reverse mode, worked example
        ---
        """
    )
    md_bwmode_text = mo.md(
        r"""
        Reverse mode reuses the very same computational graph, but runs the
        chain rule in the opposite direction. It first completes the whole
        forward pass above (every value $v_i$ is known), then propagates an
        **adjoint** $\bar v_i = \partial f/\partial v_i$ backwards from the
        output, seeded with $\bar v_6=1$ since $f=v_6$.

        The key difference from forward mode shows up exactly at the nodes
        that had *two* children in the graph above, $v_1$ and $v_4$: reused
        values become **accumulation points** on the way back, where the
        incoming adjoints from every child are summed. Because the sweep
        starts at the single output and ends at the inputs, this *one*
        backward pass yields both $\partial f/\partial x_1$ and $\partial
        f/\partial x_2$ — the mirror image of forward mode, which needed a
        separate pass per input. Drag the slider to propagate the adjoint one
        node further back at a time.
        """
    )
    return md_bwmode_text, md_bwmode_title


@app.cell
def _(ad_nodes):
    bw_children = {nd["id"]: [] for nd in ad_nodes}
    for _nd in ad_nodes:
        for _pid in _nd["parents"]:
            bw_children[_pid].append(_nd["id"])

    bw_adjoint = {
        "v6": r"$\bar v_6=1$",
        "v5": r"$\bar v_5=\bar v_6\,v_4$",
        "v4": r"$\bar v_4=\bar v_6\,v_5+\bar v_5$",
        "v3": r"$\bar v_3=-\bar v_4$",
        "v2": r"$\bar v_2=\bar v_5$",
        "v1": r"$\bar v_1=\bar v_4+\bar v_2\cos(v_1)$",
        "v0": r"$\bar v_0=\bar v_3\,v_3-\bar v_1\,\dfrac{v_{-1}}{v_0^{2}}=\dfrac{\partial f}{\partial x_2}$",
        "vm1": r"$\bar v_{-1}=\dfrac{\bar v_1}{v_0}=\dfrac{\partial f}{\partial x_1}$",
    }
    bw_order = list(reversed([nd["id"] for nd in ad_nodes]))
    return bw_adjoint, bw_children, bw_order


@app.cell(hide_code=True)
def _(mo):
    bw_step_slider = mo.ui.slider(1, 8, value=1, step=1, label="Backward step (propagate adjoint one node further)")
    return (bw_step_slider,)


@app.cell(hide_code=True)
def _(ad_nodes, bw_children, bw_order, plt):
    def ad_graph_fig_reverse(step):
        fig, ax = plt.subplots(figsize=(11, 3.4))
        id_to_node = {nd["id"]: nd for nd in ad_nodes}

        # completed forward pass, drawn once and lightly as the backdrop
        for nd in ad_nodes:
            x, y = nd["pos"]
            for pid in nd["parents"]:
                px, py = id_to_node[pid]["pos"]
                rad = 0.18 if py != y else 0.0
                ax.annotate(
                    "", xy=(x - 0.30, y), xytext=(px + 0.30, py),
                    arrowprops=dict(arrowstyle="-|>", color="0.82", lw=1.1,
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=0, shrinkB=0),
                )

        processed = bw_order[:step]
        current = bw_order[step - 1]

        # adjoint edges: flow from each child back into its parent, revealed
        # as the backward sweep reaches that parent
        for node_id in processed:
            x, y = id_to_node[node_id]["pos"]
            for child_id in bw_children[node_id]:
                cx, cy = id_to_node[child_id]["pos"]
                rad = -0.24 if cy != y else 0.0
                is_latest = node_id == current
                ax.annotate(
                    "", xy=(x + 0.30, y), xytext=(cx - 0.30, cy),
                    arrowprops=dict(arrowstyle="-|>",
                                    color="#1d4ed8" if is_latest else "#a9c2ef",
                                    lw=1.9 if is_latest else 1.3, linestyle="--",
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=0, shrinkB=0),
                )

        for nd in ad_nodes:
            x, y = nd["pos"]
            is_current = nd["id"] == current
            is_done = nd["id"] in processed
            edge = "#1d4ed8" if is_current else ("black" if is_done else "0.75")
            face = "#eaf0fc" if is_current else "white"
            ax.add_patch(plt.Circle((x, y), 0.30, facecolor=face, edgecolor=edge,
                                     lw=2.2 if is_current else 1.4, zorder=3))
            ax.text(x, y, nd["label"], ha="center", va="center", fontsize=13, zorder=4,
                    color="black" if is_done else "0.65")
            ax.text(x, y + 0.42, nd["op"].upper(), ha="center", va="bottom",
                    fontsize=9, color="0.55", style="italic", zorder=4)
            ax.text(x, y - 0.42, nd["formula"], ha="center", va="top", fontsize=10,
                    color="0.35", zorder=4)

        ax.set_xlim(-0.7, 5.7)
        ax.set_ylim(-0.7, 2.05)
        ax.axis("off")
        fig.tight_layout()
        return fig

    return (ad_graph_fig_reverse,)


@app.cell(hide_code=True)
def _(ad_nodes, bw_adjoint, bw_order, mo):
    def bw_trace_md(step):
        id_to_node = {nd["id"]: nd for nd in ad_nodes}
        header = "| node | value | reverse-mode adjoint |\n|---|---|---|\n"
        body = "\n".join(
            f"| {id_to_node[node_id]['label']} | {id_to_node[node_id]['formula']} | {bw_adjoint[node_id]} |"
            for node_id in bw_order[:step]
        )
        return mo.md(header + body)

    return (bw_trace_md,)


@app.cell(hide_code=True)
def _(
    ad_graph_fig_reverse,
    bw_step_slider,
    bw_trace_md,
    md_bwmode_text,
    md_bwmode_title,
    mo,
):
    mo.vstack(
        [
            md_bwmode_title,
            md_bwmode_text,
            bw_step_slider,
            ad_graph_fig_reverse(bw_step_slider.value),
            bw_trace_md(bw_step_slider.value),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    md_dawson_title = mo.md(
        """
        ### A capstone: differentiating through an integral
        ---
        """
    )
    md_dawson_text = mo.md(
        r"""
        Automatic differentiation composes with *any* differentiable
        numerical procedure — including one that is itself a quadrature
        sum. Consider the Dawson function
        $$
        D_+(x) = e^{-x^2}\int_0^x e^{t^2}\,dt,
        $$
        evaluated here by 32-point Gauss–Legendre quadrature
        (`IntegrateGauss.py`). Its derivative also follows directly from
        the definition,
        $$
        D_+'(x) = 1 - 2x\,D_+(x),
        $$
        an independent formula that lets us check AD applied to the
        quadrature sum, and compare its cost against a finite difference
        applied to the same black-box function.
        """
    )
    return md_dawson_text, md_dawson_title


@app.cell
def _():
    from IntegrateGauss import gaussxw, integrate_quadrature

    gaussxw32 = gaussxw(32)

    def gaussxwab32(a, b):
        x, w = gaussxw32
        return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

    return gaussxwab32, integrate_quadrature


@app.cell
def _(gaussxwab32, integrate_quadrature, jnp):
    def dawson(x):
        def integrand(t):
            return jnp.exp(t**2)

        nodes, weights = gaussxwab32(0.0, x)
        return jnp.exp(-(x**2)) * integrate_quadrature(integrand, (nodes, weights))

    def dawson_deriv_exact(x):
        return 1.0 - 2.0 * x * dawson(x)

    return dawson, dawson_deriv_exact


@app.cell
def _(d_forward_ad, d_reverse_ad, dawson, dawson_deriv_exact, np, time):
    _xs = np.linspace(-2.0, 2.0, 40)

    _t0 = time.perf_counter()
    dawson_fd_vals = np.array([float(df_central5(dawson, x, 1e-3)) for x in _xs])
    dawson_t_fd = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    dawson_ad_forw_vals = np.array([float(d_forward_ad(dawson, x)) for x in _xs])
    dawson_t_ad_forw = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    dawson_ad_reve_vals = np.array([float(d_reverse_ad(dawson, x)) for x in _xs])
    dawson_t_ad_reve = time.perf_counter() - _t0

    dawson_exact_vals = np.array([float(dawson_deriv_exact(x)) for x in _xs])
    dawson_xs = _xs

    dawson_err_fd = np.max(np.abs(dawson_fd_vals - dawson_exact_vals))
    dawson_err_ad_forw = np.max(np.abs(dawson_ad_forw_vals - dawson_exact_vals))
    dawson_err_ad_reve = np.max(np.abs(dawson_ad_reve_vals - dawson_exact_vals))
    return (
        dawson_ad_forw_vals,
        dawson_err_ad_forw,
        dawson_err_ad_reve,
        dawson_err_fd,
        dawson_exact_vals,
        dawson_t_ad_forw,
        dawson_t_ad_reve,
        dawson_t_fd,
        dawson_xs,
    )


@app.cell(hide_code=True)
def _(
    dawson_ad_forw_vals,
    dawson_err_ad_forw,
    dawson_err_ad_reve,
    dawson_err_fd,
    dawson_exact_vals,
    dawson_t_ad_forw,
    dawson_t_ad_reve,
    dawson_t_fd,
    dawson_xs,
    md_dawson_text,
    md_dawson_title,
    mo,
    plt,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.plot(dawson_xs, dawson_exact_vals, "k-", lw=2, label="Analytic identity")
    _ax1.plot(dawson_xs, dawson_ad_forw_vals, "r--", lw=1.5, label="AD forward")
    _ax1.set(xlabel="$x$", ylabel="$D_+'(x)$",ylim=(-0.5,1.5))
    _ax1.legend(fontsize=15,loc = 'upper left')

    _methods = ["Finite diff.\n(5-point)", "AD forw.", "AD rev."]
    _times = [dawson_t_fd, dawson_t_ad_forw, dawson_t_ad_reve]
    _ax2.bar(_methods, _times, color=["steelblue", "darkorange", "seagreen"])
    _ax2.set_ylabel("Wall time (s)")
    _ax2.yaxis.tick_right()
    _ax2.yaxis.set_label_position("right")
    # rotate ticks 90 degrees x axis
    _ax2.tick_params(axis="x", rotation=90)


    mo.vstack(
        [
            md_dawson_title,
            md_dawson_text,
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Max error, finite diff.", value=f"{dawson_err_fd:.2e}"),
                    mo.stat(label="Max error, AD forward", value=f"{dawson_err_ad_forw:.2e}"),
                    mo.stat(label="Max error, AD reverse", value=f"{dawson_err_ad_reve:.2e}"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    | Method | Order | Evaluations | Notes |
    |---|---|---|---|
    | Forward / backward difference | $\mathcal{O}(h)$ | 2 | Simplest; the only option at a domain boundary |
    | Central difference | $\mathcal{O}(h^2)$ | 2 | Cancels the $\mathcal{O}(h)$ term; the standard default |
    | 5-point central | $\mathcal{O}(h^4)$ | 4 | Higher accuracy at fixed $h$; larger optimal $h$ |
    | Second derivative (central) | $\mathcal{O}(h^2)$ | 3 | Round-off scales as $1/h^2$ — more sensitive |
    | Automatic differentiation | machine precision | one pass through the code | No step size, no truncation error |

    Every finite-difference formula above is a truncated Taylor series in
    disguise, and every one of them fights the same battle between
    truncation and round-off — a battle with no winner, only an optimal
    compromise $h_{\rm opt}$. Automatic differentiation sidesteps that
    battle entirely by working with the exact chain rule, at the cost of
    needing differentiable code rather than a black-box function.
    """)
    return


if __name__ == "__main__":
    app.run()
