import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from wigglystuff import FormulaAnimation

    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return Patch, mo, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Systems of linear equations

    Many problems in computational materials science reduce to solving a
    system of $N$ linear equations in $N$ unknowns,
    \[
    \sum_{j=1}^N a_{ij}\,x_j = v_i, \qquad i = 1,\ldots,N,
    \]
    or, in matrix form, $\mathbf{A}\mathbf{x} = \mathbf{v}$. Circuit
    networks, discretized differential equations, normal modes of coupled
    oscillators, and least-squares fits all reduce to exactly this problem.

    This notebook builds, from first principles, the standard numerical
    toolbox for solving such systems and for the closely related eigenvalue
    problem. A unique solution exists whenever $\det\mathbf{A}\neq0$; we
    start with the oldest and most direct route to it: **Gaussian
    elimination**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Gaussian elimination
    ---

    Two operations leave a system of equations unchanged, except for
    how it is written:

    1. multiplying any single equation (row) by a nonzero constant, and
    2. replacing any equation by a linear combination of itself and
       another equation.

    Gaussian elimination uses only these two operations, applied
    systematically to drive the matrix to **upper triangular** form.
    Consider a general system of 4 equations,
    $$
    \begin{pmatrix} a_{11}&a_{12}&a_{13}&a_{14}\\ a_{21}&a_{22}&a_{23}&a_{24}\\
    a_{31}&a_{32}&a_{33}&a_{34}\\ a_{41}&a_{42}&a_{43}&a_{44}\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}v_1\\v_2\\v_3\\v_4\end{pmatrix}.
    $$

    **Step 1 — normalize the pivot row.** Divide row 1 by $a_{11}$ so
    that its diagonal entry becomes 1 (operation 1 above):
    $$
    \begin{pmatrix}1&a_{12}/a_{11}&a_{13}/a_{11}&a_{14}/a_{11}\\
    a_{21}&a_{22}&a_{23}&a_{24}\\ a_{31}&a_{32}&a_{33}&a_{34}\\
    a_{41}&a_{42}&a_{43}&a_{44}\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}v_1/a_{11}\\v_2\\v_3\\v_4\end{pmatrix}.
    $$

    **Step 2 — eliminate the column below it.** Subtract $a_{j1}$ times
    the normalized first row from every row $j>1$ (operation 2 above),
    cancelling their $x_1$ coefficient:
    $$
    \begin{pmatrix}1&a_{12}'&a_{13}'&a_{14}'\\0&a_{22}'&a_{23}'&a_{24}'\\
    0&a_{32}'&a_{33}'&a_{34}'\\0&a_{42}'&a_{43}'&a_{44}'\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}v_1'\\v_2'\\v_3'\\v_4'\end{pmatrix}.
    $$

    **Repeat for every remaining column.** Applying steps 1–2 to column
    2, then column 3, then column 4 leaves an upper-triangular system
    with unit diagonal,
    $$
    \begin{pmatrix}1&\tilde a_{12}&\tilde a_{13}&\tilde a_{14}\\
    0&1&\tilde a_{23}&\tilde a_{24}\\0&0&1&\tilde a_{34}\\
    0&0&0&1\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}\tilde v_1\\\tilde v_2\\\tilde v_3\\\tilde v_4\end{pmatrix},
    $$
    ready for **back-substitution** (Section 2) — but only if every pivot
    $a_{kk}$ met along the way was actually nonzero, which is not
    guaranteed even for a well-posed system. That caveat, and its fix,
    is Section 3: **pivoting**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try it below: drag the entries of $\mathbf{A}$ and $\mathbf{v}$ to
    set up your own $4\times4$ system (click an entry to type an exact
    value), then step through its elimination one row operation at a
    time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    A_matrix_ui = mo.ui.matrix(
        [
            [2.0, 1.0, 4.0, 1.0],
            [3.0, 4.0, -1.0, -1.0],
            [1.0, -4.0, 1.0, 5.0],
            [2.0, -2.0, 1.0, 3.0],
        ],
        min_value=-12.0,
        max_value=12.0,
        step=0.5,
        precision=2,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["x1", "x2", "x3", "x4"],
        label="**Coefficient matrix A**",
    )
    v_matrix_ui = mo.ui.matrix(
        [-4.0, 3.0, 9.0, 7.0],
        min_value=-20.0,
        max_value=20.0,
        step=0.5,
        precision=2,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["v"],
        label="**Right-hand side v**",
    )
    mo.hstack([A_matrix_ui, v_matrix_ui], justify="center", align="center", gap=3)
    return A_matrix_ui, v_matrix_ui


@app.cell
def _(A_matrix_ui, np, v_matrix_ui):
    A0_ge = np.asarray(A_matrix_ui.value, dtype=float)
    v0_ge = np.asarray(v_matrix_ui.value, dtype=float)
    N_ge = A0_ge.shape[0]
    return A0_ge, N_ge, v0_ge


@app.function
def gaussian_trace(A0, v0):
    A = A0.astype(float).copy()
    v = v0.astype(float).copy()
    N = len(v)

    trace = [dict(A=A.copy(), v=v.copy(), step="initial", k=None, i=None, mult=None, div=None)]
    for k in range(N):
        div = A[k, k]
        A[k, :] /= div
        v[k] /= div
        trace.append(dict(A=A.copy(), v=v.copy(), step="normalize", k=k, i=None, mult=None, div=div))
        for i in range(k + 1, N):
            mult = A[i, k]
            A[i, :] -= mult * A[k, :]
            A[i, k] = 0.0
            v[i] -= mult * v[k]
            trace.append(dict(A=A.copy(), v=v.copy(), step="eliminate", k=k, i=i, mult=mult, div=None))
    return trace


@app.cell
def _(A0_ge, v0_ge):
    trace_ge = gaussian_trace(A0_ge, v0_ge)
    return (trace_ge,)


@app.cell(hide_code=True)
def _(mo, trace_ge):
    ge_step_slider = mo.ui.slider(
        0, len(trace_ge) - 1, value=0, step=1, label="Elimination step"
    )
    return (ge_step_slider,)


@app.cell(hide_code=True)
def _(N_ge, Patch, plt):
    def draw_gauss_step(step_idx, trace):
        state = trace[step_idx]
        A, v, step_type, k, i_row = state["A"], state["v"], state["step"], state["k"], state["i"]
        cell = 1.0

        fig, ax = plt.subplots(figsize=(6.5, 6.0))

        for r in range(N_ge):
            for c in range(N_ge):
                x0, y0 = c * cell, (N_ge - 1 - r) * cell
                facecolor, edgecolor, lw, textcolor = "white", "0.6", 1.0, "black"

                if step_type == "normalize":
                    if r == k and c == k:
                        facecolor, edgecolor, lw = "#f6c453", "black", 2.2
                    elif r == k:
                        facecolor = "#fdf0d5"
                elif step_type == "eliminate":
                    if r == k:
                        facecolor = "#fdf0d5"
                    if r == i_row and c == k:
                        facecolor, edgecolor, lw = "#cfe3fb", "#1d4ed8", 2.0
                    elif r == i_row:
                        facecolor = "#fdecea"

                if k is not None and c < k and r > c:
                    textcolor = "0.6"

                ax.add_patch(plt.Rectangle((x0, y0), cell, cell, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
                ax.text(x0 + cell / 2, y0 + cell / 2, f"{A[r, c]:.3f}", ha="center", va="center", fontsize=13, color=textcolor, zorder=3)
            ax.text(-0.5, (N_ge - 1 - r) * cell + cell / 2, f"$R_{{{r+1}}}$", ha="center", va="center", fontsize=13)

        for c in range(N_ge):
            ax.text(c * cell + cell / 2, N_ge * cell + 0.15, f"$x_{{{c+1}}}$", ha="center", va="bottom", fontsize=13)

        xg = N_ge * cell + 0.35
        ax.plot([N_ge * cell + 0.15, N_ge * cell + 0.15], [0, N_ge * cell], color="black", lw=1.3)
        ax.text(xg + cell / 2, N_ge * cell + 0.15, "$v$", ha="center", va="bottom", fontsize=13)
        for r in range(N_ge):
            y0 = (N_ge - 1 - r) * cell
            facecolor = "white"
            if step_type == "normalize" and r == k:
                facecolor = "#fdf0d5"
            elif step_type == "eliminate" and r == k:
                facecolor = "#fdf0d5"
            elif step_type == "eliminate" and r == i_row:
                facecolor = "#fdecea"
            ax.add_patch(plt.Rectangle((xg, y0), cell, cell, facecolor=facecolor, edgecolor="0.6", lw=1.0, zorder=2))
            ax.text(xg + cell / 2, y0 + cell / 2, f"{v[r]:.3f}", ha="center", va="center", fontsize=13, zorder=3)

        legend_handles = [
            Patch(facecolor="#f6c453", edgecolor="black", label="Pivot element"),
            Patch(facecolor="#fdf0d5", edgecolor="0.6", label="Pivot row"),
            Patch(facecolor="#fdecea", edgecolor="0.6", label="Row being updated"),
            Patch(facecolor="#cfe3fb", edgecolor="#1d4ed8", label="Entry being eliminated"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=1, frameon=False, fontsize=10)

        ax.set_xlim(-1.0, xg + cell + 0.2)
        ax.set_ylim(-0.2, N_ge * cell + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return (draw_gauss_step,)


@app.cell(hide_code=True)
def _(draw_gauss_step, ge_step_slider, mo, trace_ge):
    _state = trace_ge[ge_step_slider.value]
    _step, _k, _i, _mult, _div = _state["step"], _state["k"], _state["i"], _state["mult"], _state["div"]

    if _step == "initial":
        ge_formula_md = mo.md(
            r"""
            **Initial system.** The augmented matrix $[\mathbf{A}\,|\,\mathbf{v}]$,
            before any elimination has taken place.
            """
        )
        ge_pivot_stat, ge_row_stat, ge_value_label, ge_value_stat = "—", "—", "Value", "—"
    elif _step == "normalize":
        ge_formula_md = mo.md(
            rf"""
            **Step {ge_step_slider.value} of {len(trace_ge) - 1}** — normalize
            pivot row $R_{{{_k+1}}}$ so its diagonal entry becomes 1:
            $$
            R_{{{_k+1}}} \leftarrow \frac{{R_{{{_k+1}}}}}{{a_{{{_k+1}{_k+1}}}}},
            \qquad a_{{{_k+1}{_k+1}}} = {_div:.3f}
            $$
            """
        )
        ge_pivot_stat, ge_row_stat, ge_value_label, ge_value_stat = f"R{_k+1}", "—", "Divisor a_kk", f"{_div:.3f}"
    else:
        ge_formula_md = mo.md(
            rf"""
            **Step {ge_step_slider.value} of {len(trace_ge) - 1}** — eliminate
            $x_{{{_k+1}}}$ from row $R_{{{_i+1}}}$ using (normalized) pivot row $R_{{{_k+1}}}$:
            $$
            R_{{{_i+1}}} \leftarrow R_{{{_i+1}}} - a_{{{_i+1}{_k+1}}}\, R_{{{_k+1}}},
            \qquad a_{{{_i+1}{_k+1}}} = {_mult:.3f}
            $$
            """
        )
        ge_pivot_stat, ge_row_stat, ge_value_label, ge_value_stat = f"R{_k+1}", f"R{_i+1}", "Multiplier a_ik", f"{_mult:.3f}"

    if ge_step_slider.value == len(trace_ge) - 1:
        ge_note_md = mo.md(
            r"""
            **Elimination complete.** The augmented matrix is now upper
            triangular with unit diagonal: every equation below the
            diagonal has had its leading unknowns removed. This triangular
            system is solved directly by working from the last row upward
            — back-substitution (Section 2), next. (If dragging a diagonal
            entry ever sends it through zero, the method breaks down — the
            motivation for *pivoting*, Section 3.)
            """
        )
    else:
        ge_note_md = mo.md("")

    mo.vstack(
        [
            ge_step_slider,
            mo.hstack(
                [
                    draw_gauss_step(ge_step_slider.value, trace_ge),
                    mo.vstack(
                        [
                            ge_formula_md,
                            mo.hstack(
                                [
                                    mo.stat(label="Pivot row", value=ge_pivot_stat),
                                    mo.stat(label="Row updated", value=ge_row_stat),
                                    mo.stat(label=ge_value_label, value=ge_value_stat),
                                ]
                            ),
                            ge_note_md,
                        ]
                    ),
                ],
                widths=[0.55, 0.45],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Back-substitution
    ---

    Elimination — with a row swap along the way if needed (Section 3
    covers when and why) — leaves an upper-triangular system with unit
    diagonal,
    $$
    \begin{pmatrix}1&\tilde a_{12}&\tilde a_{13}&\tilde a_{14}\\
    0&1&\tilde a_{23}&\tilde a_{24}\\0&0&1&\tilde a_{34}\\
    0&0&0&1\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}\tilde v_1\\\tilde v_2\\\tilde v_3\\\tilde v_4\end{pmatrix},
    $$
    i.e. the explicit system of equations
    $$
    \begin{align}
    x_1+\tilde a_{12}x_2+\tilde a_{13}x_3+\tilde a_{14}x_4&=\tilde v_1,\\
    x_2+\tilde a_{23}x_3+\tilde a_{24}x_4&=\tilde v_2,\\
    x_3+\tilde a_{34}x_4&=\tilde v_3,\\
    x_4&=\tilde v_4.
    \end{align}
    $$
    The last row already **is** the answer for $x_4$; substituting it
    into the row above gives $x_3$, substituting both into the next
    row up gives $x_2$, and so on — working from the bottom row
    upward, hence *back*-substitution:
    $$
    x_4=\tilde v_4,\quad
    x_3=\tilde v_3-\tilde a_{34}x_4,\quad
    x_2=\tilde v_2-\tilde a_{23}x_3-\tilde a_{24}x_4,\quad
    x_1=\tilde v_1-\tilde a_{12}x_2-\tilde a_{13}x_3-\tilde a_{14}x_4,
    $$
    or, in general,
    $$
    x_i = \tilde v_i - \sum_{j=i+1}^{N}\tilde a_{ij}\,x_j,
    \qquad i = N, N-1, \ldots, 1.
    $$
    Unlike elimination itself, which is $\mathcal{O}(N^3)$,
    back-substitution only touches the upper triangle and is
    $\mathcal{O}(N^2)$ — cheap enough to repeat for many different
    $\mathbf{v}$ once $\mathbf{A}$ has been eliminated once (the
    motivation for *LU decomposition*, later).

    This picks up exactly where the Section 1 demo left off: drag its
    matrix above, or leave it as is, and step through solving for each
    $x_i$ below.
    """)
    return


@app.function
def backsub_trace(A_final, v_final):
    N = len(v_final)
    x = v_final.astype(float).copy()

    trace = [dict(k=None, coeffs=None, known=None, value=None, x=x.copy())]
    for k in range(N - 1, -1, -1):
        coeffs = A_final[k, k + 1 :]
        known = x[k + 1 :]
        value = v_final[k] - coeffs.dot(known)
        x[k] = value
        trace.append(dict(k=k, coeffs=coeffs.copy(), known=known.copy(), value=value, x=x.copy()))
    return trace


@app.cell
def _(trace_ge):
    bs_trace = backsub_trace(trace_ge[-1]["A"], trace_ge[-1]["v"])
    bs_v_tilde = trace_ge[-1]["v"]
    return bs_trace, bs_v_tilde


@app.cell(hide_code=True)
def _(bs_trace, mo):
    bs_step_slider = mo.ui.slider(
        0, len(bs_trace) - 1, value=0, step=1, label="Back-substitution step"
    )
    return (bs_step_slider,)


@app.cell(hide_code=True)
def _(N_ge, plt):
    def draw_backsub_vector(step_idx, trace):
        state = trace[step_idx]
        k, x = state["k"], state["x"]
        cell = 1.0

        fig, ax = plt.subplots(figsize=(2.4, 5.0))
        for r in range(N_ge):
            y0 = (N_ge - 1 - r) * cell
            solved = (k is not None) and (r >= k)
            if k is not None and r == k:
                facecolor, edgecolor, lw = "#f6c453", "black", 2.2
            elif solved:
                facecolor, edgecolor, lw = "#d9f2df", "0.6", 1.0
            else:
                facecolor, edgecolor, lw = "white", "0.6", 1.0

            ax.add_patch(plt.Rectangle((0, y0), cell, cell, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
            label = f"{x[r]:.3f}" if solved else "?"
            ax.text(cell / 2, y0 + cell / 2, label, ha="center", va="center", fontsize=13, zorder=3)
            ax.text(-0.4, y0 + cell / 2, f"$x_{{{r+1}}}$", ha="center", va="center", fontsize=13)

        ax.set_xlim(-1.0, cell + 0.2)
        ax.set_ylim(-0.2, N_ge * cell + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return (draw_backsub_vector,)


@app.cell(hide_code=True)
def _(
    A0_ge,
    bs_step_slider,
    bs_trace,
    bs_v_tilde,
    draw_backsub_vector,
    mo,
    np,
    v0_ge,
):
    _state = bs_trace[bs_step_slider.value]
    _k, _coeffs, _known, _value = _state["k"], _state["coeffs"], _state["known"], _state["value"]

    if _k is None:
        bs_formula_md = mo.md(
            r"""
            **Nothing solved yet.** The last equation of the triangular
            system, $x_N=\tilde v_N$, already gives an unknown directly —
            it is solved first.
            """
        )
    elif len(_coeffs) == 0:
        bs_formula_md = mo.md(
            rf"""
            **Step {bs_step_slider.value} of {len(bs_trace) - 1}** — solve
            for $x_{{{_k+1}}}$:
            $$
            x_{{{_k+1}}} = \tilde v_{{{_k+1}}} = {_value:.3f}
            $$
            """
        )
    else:
        _terms = " + ".join(rf"({c:.3f})({xj:.3f})" for c, xj in zip(_coeffs, _known))
        bs_formula_md = mo.md(
            rf"""
            **Step {bs_step_slider.value} of {len(bs_trace) - 1}** — solve
            for $x_{{{_k+1}}}$ using the already-known
            $x_{{{_k+2}}},\ldots,x_N$:
            $$
            x_{{{_k+1}}} = \tilde v_{{{_k+1}}} - \sum_{{j={_k+2}}}^{{N}}\tilde a_{{{_k+1},j}}x_j
            = {bs_v_tilde[_k]:.3f} - \left[{_terms}\right] = {_value:.3f}
            $$
            """
        )

    if bs_step_slider.value == len(bs_trace) - 1:
        _x_final = bs_trace[-1]["x"]
        _residual = np.max(np.abs(A0_ge.dot(_x_final) - v0_ge))
        bs_note = mo.md(
            f"""
            **All unknowns recovered.** Checking against the *original*
            (pre-elimination) system: $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}|
            = {_residual:.2e}$.
            """
        ).callout(kind="success")
    else:
        bs_note = mo.md("")

    mo.vstack(
        [
            bs_step_slider,
            mo.hstack(
                [
                    draw_backsub_vector(bs_step_slider.value, bs_trace),
                    mo.vstack([bs_formula_md, bs_note]),
                ],
                widths=[0.25, 0.75],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pivoting
    ---

    Simple Gaussian elimination relies on the diagonal element of the
    current row being nonzero. That is not always the case: the
    diagonal entry can be zero even in a perfectly well-posed,
    non-singular system, and simple elimination then fails outright.
    Take our example matrix from Section 1 and zero its first entry:
    $$
    \begin{pmatrix} 0&1&4&1\\3&4&-1&-1\\1&-4&1&5\\2&-2&1&3\end{pmatrix}
    \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}
    =\begin{pmatrix}-4\\3\\9\\7\end{pmatrix}.
    $$
    """)
    return


@app.cell
def _(np):
    A_piv0 = np.array(
        [
            [0.0, 1.0, 4.0, 1.0],
            [3.0, 4.0, -1.0, -1.0],
            [1.0, -4.0, 1.0, 5.0],
            [2.0, -2.0, 1.0, 3.0],
        ]
    )
    v_piv0 = np.array([-4.0, 3.0, 9.0, 7.0])
    return A_piv0, v_piv0


@app.function
def linsolve_gaussian(A0, v0):
    A = A0.astype(float).copy()
    v = v0.astype(float).copy()
    N = len(v)

    for k in range(N):
        div = A[k, k]
        if div == 0.0:
            raise ZeroDivisionError(f"pivot a_{k+1}{k+1} is zero")
        A[k, :] /= div
        v[k] /= div
        for i in range(k + 1, N):
            mult = A[i, k]
            A[i, :] -= mult * A[k, :]
            v[i] -= mult * v[k]

    x = v.copy()
    for k in range(N - 1, -1, -1):
        x[k] = v[k] - A[k, k + 1 :] @ x[k + 1 :]
    return x


@app.cell(hide_code=True)
def _(A_piv0, mo, v_piv0):
    try:
        linsolve_gaussian(A_piv0, v_piv0)
    except ZeroDivisionError as _err:
        ge_fail_callout = mo.md(
            f"""
            **Simple Gaussian elimination fails.** The first pivot
            $a_{{11}}$ is zero, so the very first row-normalization
            step divides by zero. This matrix is perfectly non-singular —
            the elimination *procedure*, not the system, is what breaks.
            """
        ).callout(kind="danger")

    ge_fail_callout
    return


@app.function
def linsolve_gaussian_partial_pivot(A0, v0):
    A = A0.astype(float).copy()
    v = v0.astype(float).copy()
    N = len(v)

    for k in range(N):
        p = k + int(abs(A[k:, k]).argmax())
        A[[k, p]] = A[[p, k]]
        v[[k, p]] = v[[p, k]]

        div = A[k, k]
        A[k, :] /= div
        v[k] /= div
        for i in range(k + 1, N):
            mult = A[i, k]
            A[i, :] -= mult * A[k, :]
            v[i] -= mult * v[k]

    x = v.copy()
    for k in range(N - 1, -1, -1):
        x[k] = v[k] - A[k, k + 1 :] @ x[k + 1 :]
    return x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Partial pivoting** fixes this by swapping the current row with
    whichever remaining row has the *largest* entry (in magnitude) in
    the pivot column, before eliminating. Column 1 here holds
    $0,3,1,2$; the largest is $3$, in row 2, so we swap
    $R_1\leftrightarrow R_2$ and continue exactly as before:
    $$
    \begin{pmatrix} 0&1&4&1\\3&4&-1&-1\\1&-4&1&5\\2&-2&1&3\end{pmatrix}
    \;\xrightarrow{R_1\leftrightarrow R_2}\;
    \begin{pmatrix} 3&4&-1&-1\\0&1&4&1\\1&-4&1&5\\2&-2&1&3\end{pmatrix}.
    $$
    Repeating this row selection before every pivot column is **partial
    pivoting**: the system is unchanged (row swaps are one of our two
    legal operations), but the diagonal entry used for division is now
    guaranteed nonzero whenever the matrix is non-singular. The resulting
    triangular system is solved exactly as in Section 2.
    """)
    return


@app.cell(hide_code=True)
def _(A_piv0, mo, np, v_piv0):
    x_pivoted = linsolve_gaussian_partial_pivot(A_piv0, v_piv0)
    residual = np.max(np.abs(A_piv0.dot(x_pivoted) - v_piv0))

    mo.md(
        f"""
        **Partial pivoting solves it.**
        $\\mathbf{{x}} = ({x_pivoted[0]:.4f},\\ {x_pivoted[1]:.4f},\\
        {x_pivoted[2]:.4f},\\ {x_pivoted[3]:.4f})$, with residual
        $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {residual:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More generally, even a pivot that is merely *small* — not exactly
    zero — is dangerous: it forces a huge multiplier $\ell=a_{ik}/a_{kk}$
    that amplifies floating-point rounding error when subtracted from
    the rows below (Section 1's derivation still holds — only now
    $\ell$ is enormous). For this reason, production linear-algebra
    routines (including `numpy.linalg.solve`) **always pivot**, using
    the largest available entry in each column regardless of whether
    the naive pivot happens to be exactly zero. This costs nothing
    asymptotically — pivoting is still $\mathcal{O}(N^3)$ — and is the
    default, not an exception.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. LU decomposition
    ---

    Back-substitution is cheap ($\mathcal{O}(N^2)$), but the
    elimination that produces the triangular system is not
    ($\mathcal{O}(N^3)$). Whenever the coefficient matrix $\mathbf{A}$
    is reused with many different right-hand sides $\mathbf{v}$ — as
    happens, for instance, when computing a matrix inverse column by
    column, or re-solving a circuit under different loads — repeating
    the full elimination for every $\mathbf{v}$ wastes nearly all of
    that $\mathcal{O}(N^3)$ work. **LU decomposition** factors the
    expensive, $\mathbf{v}$-independent part out once and for all:
    $$
    \mathbf{A} = \mathbf{L}\mathbf{U},
    $$
    a product of a lower triangular matrix $\mathbf{L}$ and an upper
    triangular matrix $\mathbf{U}$, both depending only on
    $\mathbf{A}$, never on $\mathbf{v}$.

    Gaussian elimination already builds $\mathbf{U}$: it is exactly
    the triangular matrix obtained in Section 1, before the
    right-hand side is even considered. Where does $\mathbf{L}$ come
    from? Every row operation performed during elimination —
    normalizing row $k$, then subtracting multiples of it from the
    rows below — is a linear transformation of $\mathbf{A}$, and so
    can itself be written as left multiplication by some matrix
    $\mathbf{L}_k^{-1}$, e.g. for the first pivot column,
    $$
    \mathbf{L}_0^{-1} =
    \frac{1}{a_{11}}
    \begin{pmatrix}
    1 & 0 & 0 & 0\\
    -a_{21} & a_{11} & 0 & 0\\
    -a_{31} & 0 & a_{11} & 0\\
    -a_{41} & 0 & 0 & a_{11}
    \end{pmatrix},
    \qquad\text{so that}\qquad
    \mathbf{U} = \mathbf{L}_{N-1}^{-1}\cdots\mathbf{L}_1^{-1}\mathbf{L}_0^{-1}\,\mathbf{A}.
    $$
    Each $\mathbf{L}_k^{-1}$ is lower triangular — it only ever
    combines row $k$ with rows below it — and so is its inverse
    $\mathbf{L}_k$:
    $$
    \mathbf{L}_0 =
    \begin{pmatrix}
    a_{11} & 0 & 0 & 0\\
    a_{21} & 1 & 0 & 0\\
    a_{31} & 0 & 1 & 0\\
    a_{41} & 0 & 0 & 1
    \end{pmatrix}.
    $$
    Multiplying both sides of the elimination by these lower
    triangular matrices, in reverse order, isolates $\mathbf{A}$:
    $$
    \mathbf{A} = \mathbf{L}_0\mathbf{L}_1\cdots\mathbf{L}_{N-1}\,\mathbf{U}
    = \mathbf{L}\,\mathbf{U}.
    $$
    The product of lower triangular matrices is itself lower
    triangular, so $\mathbf{L}=\mathbf{L}_0\mathbf{L}_1\cdots\mathbf{L}_{N-1}$
    is exactly the factor we want — and it takes an unexpectedly
    simple form: **column $k$ of $\mathbf{L}$ is column $k$ of
    $\mathbf{A}$ exactly as it stood right before pivot $k$ was
    normalized**, i.e. the pivot itself on the diagonal, and the
    eliminated multipliers below it,
    $$
    \mathbf{L} =
    \begin{pmatrix}
    a_{11} & 0 & 0 & 0\\
    a_{21} & a_{22}' & 0 & 0\\
    a_{31} & a_{32}' & a_{33}'' & 0\\
    a_{41} & a_{42}' & a_{43}'' & a_{44}'''
    \end{pmatrix}.
    $$
    In other words, Gaussian elimination computes $\mathbf{L}$ for
    free: it is just the column of entries eliminated from
    $\mathbf{A}$ at each step, *saved* instead of discarded, before
    that column is normalized to zeros-and-a-one. Note $\mathbf{L}$'s
    diagonal holds the pivots actually used, not 1s — this is the
    same information as the elimination of Section 1, just
    repackaged as a matrix product instead of a row-reduced augmented
    matrix.

    The demo below reuses the coefficient matrix $\mathbf{A}$ from
    Section 1 — drag it there to try a different system — and steps
    through building $\mathbf{L}$ and $\mathbf{U}$ side by side.
    """)
    return


@app.cell
def _(np):
    def lu_trace(A0):
        A = A0.astype(float).copy()
        N = len(A)
        U = A.copy()
        L = np.zeros((N, N), dtype=float)

        trace = [dict(L=L.copy(), U=U.copy(), step="initial", k=None, i=None, mult=None, div=None)]
        for k in range(N):
            for r2 in range(k, N):
                L[r2, k] = U[r2, k]
            trace.append(dict(L=L.copy(), U=U.copy(), step="record", k=k, i=None, mult=None, div=None))

            div = U[k, k]
            U[k, :] /= div
            trace.append(dict(L=L.copy(), U=U.copy(), step="normalize", k=k, i=None, mult=None, div=div))

            for i in range(k + 1, N):
                mult = U[i, k]
                U[i, :] -= mult * U[k, :]
                U[i, k] = 0.0
                trace.append(dict(L=L.copy(), U=U.copy(), step="eliminate", k=k, i=i, mult=mult, div=None))
        return trace

    return (lu_trace,)


@app.cell
def _(A0_ge, lu_trace):
    trace_lu = lu_trace(A0_ge)
    return (trace_lu,)


@app.cell(hide_code=True)
def _(mo, trace_lu):
    lu_step_slider = mo.ui.slider(
        0, len(trace_lu) - 1, value=0, step=1, label="Decomposition step"
    )
    return (lu_step_slider,)


@app.cell(hide_code=True)
def _(N_ge, Patch, plt):
    def draw_lu_step(step_idx, trace):
        state = trace[step_idx]
        L, U, step_type, k, i_row = state["L"], state["U"], state["step"], state["k"], state["i"]
        cell = 1.0

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))

        for ax, M, name in zip(axes, (L, U), ("L", "U")):
            for r in range(N_ge):
                for c in range(N_ge):
                    x0, y0 = c * cell, (N_ge - 1 - r) * cell
                    facecolor, edgecolor, lw, textcolor = "white", "0.6", 1.0, "black"

                    if name == "L":
                        done_col = (k is not None) and (c < k or (c == k and step_type != "record"))
                        if step_type == "record" and c == k and r >= k:
                            facecolor, edgecolor, lw = "#cfe3fb", "#1d4ed8", 2.0
                        elif done_col and r >= c:
                            facecolor = "#eef3fb"
                    else:
                        if step_type == "normalize":
                            if r == k and c == k:
                                facecolor, edgecolor, lw = "#f6c453", "black", 2.2
                            elif r == k:
                                facecolor = "#fdf0d5"
                        elif step_type == "eliminate":
                            if r == k:
                                facecolor = "#fdf0d5"
                            if r == i_row and c == k:
                                facecolor, edgecolor, lw = "#cfe3fb", "#1d4ed8", 2.0
                            elif r == i_row:
                                facecolor = "#fdecea"
                        if k is not None and c < k and r > c:
                            textcolor = "0.6"

                    ax.add_patch(plt.Rectangle((x0, y0), cell, cell, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
                    ax.text(x0 + cell / 2, y0 + cell / 2, f"{M[r, c]:.3f}", ha="center", va="center", fontsize=11, color=textcolor, zorder=3)

            ax.set_xlim(-0.3, N_ge * cell + 0.2)
            ax.set_ylim(-0.2, N_ge * cell + 0.4)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"${name}$", fontsize=15)

        legend_handles = [
            Patch(facecolor="#f6c453", edgecolor="black", label="Pivot element"),
            Patch(facecolor="#fdf0d5", edgecolor="0.6", label="Pivot row (U)"),
            Patch(facecolor="#fdecea", edgecolor="0.6", label="Row being updated (U)"),
            Patch(facecolor="#cfe3fb", edgecolor="#1d4ed8", label="Entry recorded into L / eliminated in U"),
            Patch(facecolor="#eef3fb", edgecolor="0.6", label="Finished column of L"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(bottom=0.22, wspace=0.35)
        return fig

    return (draw_lu_step,)


@app.cell(hide_code=True)
def _(draw_lu_step, lu_step_slider, mo, trace_lu):
    _state = trace_lu[lu_step_slider.value]
    _step, _k, _i, _mult, _div = _state["step"], _state["k"], _state["i"], _state["mult"], _state["div"]

    if _step == "initial":
        lu_formula_md = mo.md(
            r"""
            **Initial matrices.** $\mathbf{U}$ starts as a copy of
            $\mathbf{A}$; $\mathbf{L}$ starts at zero.
            """
        )
        lu_pivot_stat, lu_row_stat, lu_value_label, lu_value_stat = "—", "—", "Value", "—"
    elif _step == "record":
        lu_formula_md = mo.md(
            rf"""
            **Step {lu_step_slider.value} of {len(trace_lu) - 1}** —
            before normalizing row $R_{{{_k+1}}}$, copy column
            $k={_k+1}$ of $\mathbf{{U}}$ (pivot and multipliers) into
            the same column of $\mathbf{{L}}$:
            $$
            L_{{i,{_k+1}}} \leftarrow U_{{i,{_k+1}}}, \qquad i \geq {_k+1}.
            $$
            """
        )
        lu_pivot_stat, lu_row_stat, lu_value_label, lu_value_stat = f"R{_k+1}", "—", "Column copied", f"k={_k+1}"
    elif _step == "normalize":
        lu_formula_md = mo.md(
            rf"""
            **Step {lu_step_slider.value} of {len(trace_lu) - 1}** —
            normalize pivot row $R_{{{_k+1}}}$ of $\mathbf{{U}}$,
            exactly as in Section 1:
            $$
            R_{{{_k+1}}} \leftarrow \frac{{R_{{{_k+1}}}}}{{a_{{{_k+1}{_k+1}}}}},
            \qquad a_{{{_k+1}{_k+1}}} = {_div:.3f}.
            $$
            """
        )
        lu_pivot_stat, lu_row_stat, lu_value_label, lu_value_stat = f"R{_k+1}", "—", "Divisor a_kk", f"{_div:.3f}"
    else:
        lu_formula_md = mo.md(
            rf"""
            **Step {lu_step_slider.value} of {len(trace_lu) - 1}** —
            eliminate $x_{{{_k+1}}}$ from row $R_{{{_i+1}}}$ of
            $\mathbf{{U}}$ using the (normalized) pivot row:
            $$
            R_{{{_i+1}}} \leftarrow R_{{{_i+1}}} - a_{{{_i+1}{_k+1}}}\, R_{{{_k+1}}},
            \qquad a_{{{_i+1}{_k+1}}} = {_mult:.3f}.
            $$
            """
        )
        lu_pivot_stat, lu_row_stat, lu_value_label, lu_value_stat = f"R{_k+1}", f"R{_i+1}", "Multiplier a_ik", f"{_mult:.3f}"

    if lu_step_slider.value == len(trace_lu) - 1:
        lu_note_md = mo.md(
            r"""
            **Decomposition complete.** $\mathbf{L}$ and $\mathbf{U}$
            are fully built, and $\mathbf{A}=\mathbf{L}\mathbf{U}$ —
            the check below confirms it.
            """
        )
    else:
        lu_note_md = mo.md("")

    mo.vstack(
        [
            lu_step_slider,
            mo.hstack(
                [
                    draw_lu_step(lu_step_slider.value, trace_lu),
                    mo.vstack(
                        [
                            lu_formula_md,
                            mo.hstack(
                                [
                                    mo.stat(label="Pivot row", value=lu_pivot_stat),
                                    mo.stat(label="Row updated", value=lu_row_stat),
                                    mo.stat(label=lu_value_label, value=lu_value_stat),
                                ]
                            ),
                            lu_note_md,
                        ]
                    ),
                ],
                widths=[0.6, 0.4],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(A0_ge, mo, np, trace_lu):
    _L_final, _U_final = trace_lu[-1]["L"], trace_lu[-1]["U"]
    _residual = np.max(np.abs(_L_final.dot(_U_final) - A0_ge))

    mo.md(
        f"""
        **Check:** $\\max|\\mathbf{{L}}\\mathbf{{U}}-\\mathbf{{A}}|
        = {_residual:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Solving with LU decomposition
    ---

    Once $\mathbf{A}=\mathbf{L}\mathbf{U}$ is known, solving
    $\mathbf{A}\mathbf{x}=\mathbf{v}$ for a *new* $\mathbf{v}$ no
    longer requires touching $\mathbf{A}$ at all. Substituting the
    factorization,
    $$
    \mathbf{L}\mathbf{U}\mathbf{x} = \mathbf{v},
    $$
    and introducing the intermediate vector
    $\mathbf{y}=\mathbf{U}\mathbf{x}$ splits the problem into two
    triangular solves:
    $$
    \mathbf{L}\mathbf{y} = \mathbf{v}, \qquad \mathbf{U}\mathbf{x} = \mathbf{y}.
    $$
    The second is exactly the back-substitution of Section 2. The
    first is its mirror image — **forward substitution** — solving
    from the top row down instead of the bottom row up, since
    $\mathbf{L}$ is lower triangular:
    $$
    y_i = \frac{1}{L_{ii}}\left(v_i - \sum_{j=1}^{i-1}L_{ij}\,y_j\right),
    \qquad i = 1,\ldots,N.
    $$
    Both solves are $\mathcal{O}(N^2)$. The expensive
    $\mathcal{O}(N^3)$ work — computing $\mathbf{L}$ and $\mathbf{U}$
    — is done once, regardless of how many right-hand sides follow;
    every subsequent $\mathbf{v}$ costs only two cheap triangular
    solves, instead of a full re-elimination. The demo below reuses
    the $\mathbf{L}$ and $\mathbf{U}$ built in Section 4 to solve for
    both Section 1's $\mathbf{v}$ and a second right-hand side.
    """)
    return


@app.cell
def _(np):
    def lu_solve(L, U, v):
        N = len(v)
        y = np.empty(N, dtype=float)
        for i in range(N):
            y[i] = (v[i] - L[i, :i] @ y[:i]) / L[i, i]

        x = np.empty(N, dtype=float)
        for i in range(N - 1, -1, -1):
            x[i] = y[i] - U[i, i + 1 :] @ x[i + 1 :]
        return x

    return (lu_solve,)


@app.cell(hide_code=True)
def _(mo):
    v2_matrix_ui = mo.ui.matrix(
        [5.0, -2.0, 0.0, 6.0],
        min_value=-20.0,
        max_value=20.0,
        step=0.5,
        precision=2,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["v'"],
        label="**A second right-hand side v'**",
    )
    v2_matrix_ui
    return (v2_matrix_ui,)


@app.cell(hide_code=True)
def _(A0_ge, lu_solve, mo, np, trace_lu, v0_ge, v2_matrix_ui):
    _L_final, _U_final = trace_lu[-1]["L"], trace_lu[-1]["U"]
    _v2 = np.asarray(v2_matrix_ui.value, dtype=float)

    _x1 = lu_solve(_L_final, _U_final, v0_ge)
    _x2 = lu_solve(_L_final, _U_final, _v2)

    _res1 = np.max(np.abs(A0_ge.dot(_x1) - v0_ge))
    _res2 = np.max(np.abs(A0_ge.dot(_x2) - _v2))

    mo.md(
        f"""
        The **same** $\\mathbf{{L}}$ and $\\mathbf{{U}}$ from Section
        4 solve both right-hand sides — no elimination is repeated.

        - $\\mathbf{{v}}$ (Section 1's): $\\mathbf{{x}} =
          ({_x1[0]:.4f},\\ {_x1[1]:.4f},\\ {_x1[2]:.4f},\\ {_x1[3]:.4f})$,
          residual $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {_res1:.2e}$.
        - $\\mathbf{{v}}'$ (above): $\\mathbf{{x}} =
          ({_x2[0]:.4f},\\ {_x2[1]:.4f},\\ {_x2[2]:.4f},\\ {_x2[3]:.4f})$,
          residual $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}'| = {_res2:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Wall-clock demonstration.** The saving above is easy to state but
    easy to underestimate — timing it directly makes it concrete. For a
    fixed $\mathbf{A}$ and $M$ different right-hand sides $\mathbf{v}$,
    compare two routes to the same $M$ solutions:

    - **Repeated Gaussian elimination** — re-run the full
      $\mathcal{O}(N^3)$ elimination of Section 1 from scratch for
      *every* $\mathbf{v}$.
    - **LU decomposition** — factor $\mathbf{A}=\mathbf{L}\mathbf{U}$
      once ($\mathcal{O}(N^3)$, paid a single time), then reduce every
      $\mathbf{v}$ to the two $\mathcal{O}(N^2)$ triangular solves of
      this section.

    Both routes use partial pivoting (Sections 3 and 6) so the
    comparison is fair and robust for a random $\mathbf{A}$. Drag the
    sliders to change the system size $N$ and the number of
    right-hand sides $M$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lu_bench_N_slider = mo.ui.slider(20, 250, value=120, step=10, label="Matrix size $N$")
    lu_bench_M_slider = mo.ui.slider(5, 200, value=60, step=5, label="Number of right-hand sides $M$")
    return lu_bench_M_slider, lu_bench_N_slider


@app.function
def solve_many_gaussian(A, vs):
    return [linsolve_gaussian_partial_pivot(A, v) for v in vs]


@app.cell
def _(lu_decomp_partial_pivot, lu_solve_partial_pivot):
    def solve_many_lu(A, vs):
        L, U, row_order = lu_decomp_partial_pivot(A)
        return [lu_solve_partial_pivot(L, U, row_order, v) for v in vs]

    return (solve_many_lu,)


@app.cell
def _(bench, lu_bench_M_slider, lu_bench_N_slider, np, solve_many_lu):
    _rng = np.random.default_rng(0)
    lu_bench_N = lu_bench_N_slider.value
    lu_bench_M = lu_bench_M_slider.value

    lu_bench_A = _rng.uniform(-1.0, 1.0, size=(lu_bench_N, lu_bench_N))
    lu_bench_A[np.arange(lu_bench_N), np.arange(lu_bench_N)] += lu_bench_N
    lu_bench_vs = [_rng.uniform(-1.0, 1.0, size=lu_bench_N) for _ in range(lu_bench_M)]

    lu_bench_time_gaussian = bench(solve_many_gaussian, lu_bench_A, lu_bench_vs, repeats=1)
    lu_bench_time_lu = bench(solve_many_lu, lu_bench_A, lu_bench_vs, repeats=1)
    return lu_bench_M, lu_bench_N, lu_bench_time_gaussian, lu_bench_time_lu


@app.cell(hide_code=True)
def _(plt):
    def draw_lu_benchmark_bar(time_gaussian, time_lu, N, M):
        _GAUSS_COLOR = "#eb6834"
        _LU_COLOR = "#2a78d6"

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        labels = [
            f"Repeated\n Gaus.-Elem\n ({M}$\\times$)",
            f"LU decomp. once\n+ {M} triangular\n solves",
        ]
        times = [time_gaussian, time_lu]
        bars = ax.bar(
            labels, times, color=[_GAUSS_COLOR, _LU_COLOR], width=0.55,
            edgecolor="black", lw=0.8,
        )

        for bar, t in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.3f} s", ha="center", va="bottom", fontsize=11,
            )

        ax.set_ylabel("Total wall time (s)")
        ax.set_title(
            f"Solving $\\mathbf{{A}}\\mathbf{{x}}=\\mathbf{{v}}$ for "
            f"$M={M}$ right-hand sides, $N={N}$"
        )
        # ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(times) * 1.2)
        fig.tight_layout()
        return fig

    return (draw_lu_benchmark_bar,)


@app.cell(hide_code=True)
def _(
    draw_lu_benchmark_bar,
    lu_bench_M,
    lu_bench_M_slider,
    lu_bench_N,
    lu_bench_N_slider,
    lu_bench_time_gaussian,
    lu_bench_time_lu,
    mo,
):
    _speedup = lu_bench_time_gaussian / lu_bench_time_lu

    mo.vstack(
        [mo.hstack([lu_bench_N_slider, lu_bench_M_slider], justify="center", gap=3),
            draw_lu_benchmark_bar(
                lu_bench_time_gaussian, lu_bench_time_lu, lu_bench_N, lu_bench_M
            ),
            mo.md(
                f"""
                For $N={lu_bench_N}$ and $M={lu_bench_M}$ right-hand sides:
                repeated Gaussian elimination takes
                **{lu_bench_time_gaussian:.3f} s** in total, versus
                **{lu_bench_time_lu:.3f} s** for LU decomposition once
                plus $M$ triangular solves — a **{_speedup:.1f}$\\times$**
                speed-up. Drag the sliders above: the gap widens as $N$
                grows Both routes still pay $\\mathcal{{O}}(N^3)$ once,
                but Gaussian elimination pays it $M$ times whereas LU decomposition pays it only once.
                """
            ).callout(kind="success"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. LU decomposition with pivoting
    ---

    Just like plain Gaussian elimination (Section 3), plain LU
    decomposition breaks down whenever a pivot is exactly zero, and
    is numerically dangerous whenever a pivot is merely small. The
    fix is the same: swap in the largest available pivot from the
    rows below before eliminating. Tracking those swaps turns the
    factorization into
    $$
    \mathbf{P}\mathbf{A} = \mathbf{L}\mathbf{U},
    $$
    where $\mathbf{P}$ is a **permutation matrix** recording the row
    order actually used — $\mathbf{L}$ and $\mathbf{U}$ are
    triangular exactly as before; only the reference row order has
    changed. Solving $\mathbf{A}\mathbf{x}=\mathbf{v}$ now
    additionally requires permuting $\mathbf{v}$'s rows the same way
    before the forward/back substitution of Section 5. This is what
    production routines (`scipy.linalg.lu`, `numpy.linalg.solve`)
    actually compute internally — LU decomposition with partial
    pivoting is the standard general-purpose solver.

    Below, Section 3's zero-pivot matrix — on which *plain* LU
    decomposition would immediately fail, just like plain Gaussian
    elimination did — is factored and solved successfully.
    """)
    return


@app.cell
def _(np):
    def lu_decomp_partial_pivot(A0):
        A = A0.astype(float).copy()
        N = len(A)
        U = A.copy()
        L = np.zeros((N, N), dtype=float)
        row_order = list(range(N))

        for k in range(N):
            p = k + int(abs(U[k:, k]).argmax())
            row_order[k], row_order[p] = row_order[p], row_order[k]
            U[[k, p]] = U[[p, k]]
            L[[k, p]] = L[[p, k]]

            for r2 in range(k, N):
                L[r2, k] = U[r2, k]

            div = U[k, k]
            U[k, :] /= div
            for i in range(k + 1, N):
                mult = U[i, k]
                U[i, :] -= mult * U[k, :]
                U[i, k] = 0.0

        return L, U, row_order

    return (lu_decomp_partial_pivot,)


@app.cell
def _(lu_solve):
    def lu_solve_partial_pivot(L, U, row_order, v):
        return lu_solve(L, U, v[row_order])

    return (lu_solve_partial_pivot,)


@app.cell(hide_code=True)
def _(A_piv0, lu_decomp_partial_pivot, lu_solve_partial_pivot, mo, np, v_piv0):
    _L, _U, _row_order = lu_decomp_partial_pivot(A_piv0)
    _x = lu_solve_partial_pivot(_L, _U, _row_order, v_piv0)
    _residual = np.max(np.abs(A_piv0.dot(_x) - v_piv0))

    mo.md(
        f"""
        **Row order used:** {[r + 1 for r in _row_order]} (row 2 of
        $\\mathbf{{A}}$ became the new pivot row 1, since column 1
        held $0,3,1,2$ and $3$ is the largest).

        $\\mathbf{{x}} = ({_x[0]:.4f},\\ {_x[1]:.4f},\\ {_x[2]:.4f},\\
        {_x[3]:.4f})$, residual
        $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {_residual:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. The determinant, from LU
    ---

    Once $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ is known
    (Section 6), the determinant of $\mathbf{A}$ is almost free.
    Determinants are multiplicative,
    $$
    \det(\mathbf{P})\det(\mathbf{A}) = \det(\mathbf{L})\det(\mathbf{U}),
    $$
    and the determinant of a triangular matrix — $\mathbf{L}$ and
    $\mathbf{U}$ both are — is just the product of its diagonal
    entries, so
    $$
    \det(\mathbf{A}) = \frac{1}{\det(\mathbf{P})}
    \prod_{i=1}^N L_{ii} \prod_{i=1}^N U_{ii}.
    $$
    $\mathbf{U}$'s diagonal is always 1, by the normalization used
    throughout this notebook (Section 1, Step 1), so this reduces to
    the pivots already sitting on $\mathbf{L}$'s diagonal, corrected
    for sign by the row swaps performed while pivoting (Section 3):
    $\det(\mathbf{P})=(-1)^s$, where $s$ is the number of swaps. Once
    $\mathbf{L}$ is available — computed once, regardless of how it
    is later reused (Section 5) — reading off $\det(\mathbf{A})$
    costs only $\mathcal{O}(N)$, instead of the $\mathcal{O}(N^3)$ of
    a cofactor expansion or a fresh elimination.
    """)
    return


@app.cell
def _(lu_decomp_partial_pivot, np):
    def permutation_sign(row_order):
        order = list(row_order)
        sign = 1
        for i in range(len(order)):
            while order[i] != i:
                j = order[i]
                order[i], order[j] = order[j], order[i]
                sign = -sign
        return sign

    def determinant_from_lu(A0):
        L, U, row_order = lu_decomp_partial_pivot(A0)
        sign = permutation_sign(row_order)
        return sign * np.prod(np.diag(L)) * np.prod(np.diag(U))

    return (determinant_from_lu,)


@app.cell(hide_code=True)
def _(A0_ge, determinant_from_lu, mo, np):
    _det_lu = determinant_from_lu(A0_ge)
    _det_np = np.linalg.det(A0_ge)

    mo.md(
        f"""
        Applied to Section 1's matrix $\\mathbf{{A}}$ (drag its
        entries to try another): $\\det(\\mathbf{{A}}) = {_det_lu:.4f}$
        via $\\mathbf{{L}}$ and $\\mathbf{{U}}$, compared to NumPy's
        own $\\det(\\mathbf{{A}}) = {_det_np:.4f}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Matrix inversion
    ---

    The inverse $\mathbf{A}^{-1}$ of a non-singular matrix
    $\mathbf{A}$ satisfies
    $$
    \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}.
    $$
    Reading this equation one column at a time, column $k$ of
    $\mathbf{A}^{-1}$ — call it $\mathbf{x}_k$ — must satisfy
    $$
    \mathbf{A}\mathbf{x}_k = \mathbf{e}_k, \qquad k = 1,\ldots,N,
    $$
    where $\mathbf{e}_k$ is the $k$th standard basis vector (all
    zeros, except a 1 in entry $k$). Computing $\mathbf{A}^{-1}$
    therefore reduces to solving $N$ linear systems that all share
    the *same* coefficient matrix $\mathbf{A}$ and differ only in
    the right-hand side — precisely the scenario Section 5 was built
    for. LU decomposition, with partial pivoting so the method never
    fails the way plain elimination can (Section 6), factors
    $\mathbf{A}$ once, in $\mathcal{O}(N^3)$; each of the $N$
    columns of $\mathbf{A}^{-1}$ then costs only the two
    $\mathcal{O}(N^2)$ triangular solves of Section 5. The total
    cost, $\mathcal{O}(N^3)$, is the same order as a single
    elimination — not $N$ times that — and this is exactly how
    `numpy.linalg.inv` computes a matrix inverse internally.

    The demo below gives $\mathbf{A}$ its own draggable matrix,
    defaulting to Section 1's example; $\mathbf{A}^{-1}$ is
    recomputed and redrawn, read-only, every time you drag an entry
    of $\mathbf{A}$. Partial pivoting keeps this robust even if you
    drag $\mathbf{A}$ into a configuration where the naive pivot
    would be zero, as in Section 3.
    """)
    return


@app.cell
def _(lu_decomp_partial_pivot, lu_solve_partial_pivot, np):
    def matrix_inverse(A0):
        L, U, row_order = lu_decomp_partial_pivot(A0)
        N = len(row_order)

        Ainv = np.empty((N, N), dtype=float)
        for c in range(N):
            e_c = np.zeros(N, dtype=float)
            e_c[c] = 1.0
            Ainv[:, c] = lu_solve_partial_pivot(L, U, row_order, e_c)
        return Ainv

    return (matrix_inverse,)


@app.cell(hide_code=True)
def _(mo):
    A_inv_matrix_ui = mo.ui.matrix(
        [
            [2.0, 1.0, 4.0, 1.0],
            [3.0, 4.0, -1.0, -1.0],
            [1.0, -4.0, 1.0, 5.0],
            [2.0, -2.0, 1.0, 3.0],
        ],
        min_value=-12.0,
        max_value=12.0,
        step=0.5,
        precision=2,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["x1", "x2", "x3", "x4"],
        label="**Matrix A**",
    )
    return (A_inv_matrix_ui,)


@app.cell(hide_code=True)
def _(A_inv_matrix_ui, matrix_inverse, mo, np):
    A_ge_inv = np.asarray(A_inv_matrix_ui.value, dtype=float)
    Ainv_ge = matrix_inverse(A_ge_inv)

    Ainv_display_ui = mo.ui.matrix(
        Ainv_ge.tolist(),
        precision=4,
        disabled=True,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["C1", "C2", "C3", "C4"],
        label="**Computed inverse $A^{-1}$**",
    )

    _residual = np.max(np.abs(A_ge_inv.dot(Ainv_ge) - np.eye(len(A_ge_inv))))

    mo.vstack(
        [
            mo.hstack(
                [A_inv_matrix_ui, Ainv_display_ui],
                justify="center", align="center", gap=3,
            ),
            mo.md(
                f"""
                Checking against the definition:
                $\\max|\\mathbf{{A}}\\mathbf{{A}}^{{-1}}-\\mathbf{{I}}|
                = {_residual:.2e}$.
                """
            ).callout(kind="success"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Tridiagonal systems
    ---

    A particularly common special case restricts $\mathbf{A}$ to
    nonzero entries only on the main diagonal and its two
    neighbors,
    $$
    \mathbf{A} =
    \begin{pmatrix}
    d_1 & u_1 & & & \\
    l_2 & d_2 & u_2 & & \\
     & l_3 & d_3 & \ddots & \\
     & & \ddots & \ddots & u_{N-1}\\
     & & & l_N & d_N
    \end{pmatrix}.
    $$
    Such systems appear throughout physics: nearest-neighbor
    coupling (a linear chain of springs) and finite-difference
    discretizations of 1-D differential equations (e.g. the heat
    equation) both produce exactly this structure.

    General Gaussian elimination (Section 1) is
    $\mathcal{O}(N^3)$ because eliminating column $k$ touches every
    remaining row and every remaining column. Here it doesn't need
    to: row $k+1$ has only *one* nonzero entry, $l_{k+1}$, in column
    $k$, so eliminating it is a single row operation touching just 2
    entries — not $N$ rows against $N$ columns. Repeating this for
    every column keeps the whole elimination $\mathcal{O}(N)$:
    $$
    b_1 = \frac{u_1}{d_1}, \qquad
    b_k = \frac{u_k}{d_k - l_k b_{k-1}}, \quad k = 2,\ldots,N-1,
    $$
    $$
    \tilde v_1 = \frac{v_1}{d_1}, \qquad
    \tilde v_k = \frac{v_k - l_k \tilde v_{k-1}}{d_k - l_k b_{k-1}},
    \quad k = 2,\ldots,N.
    $$
    Back-substitution (Section 2) collapses the same way, since row
    $k$ now has only one entry, $b_k$, above the diagonal:
    $$
    x_N = \tilde v_N, \qquad
    x_k = \tilde v_k - b_k x_{k+1}, \qquad k = N-1,\ldots,1.
    $$
    This is the **Thomas algorithm** — Gaussian elimination and
    back-substitution, specialized to a band of width 1.
    """)
    return


@app.cell
def _(np):
    def tridiagonal_solve(d, l, u, v0):
        N = len(v0)
        a = d.astype(float).copy()
        b = u.astype(float).copy()
        v = v0.astype(float).copy()

        for r in range(N):
            b[r] /= a[r]
            v[r] /= a[r]
            if r < N - 1:
                a[r + 1] -= l[r + 1] * b[r]
                v[r + 1] -= l[r + 1] * v[r]

        x = np.empty(N, dtype=float)
        x[N - 1] = v[N - 1]
        for r in range(N - 2, -1, -1):
            x[r] = v[r] - b[r] * x[r + 1]
        return x

    return (tridiagonal_solve,)


@app.cell
def _(np):
    def dense_to_tridiagonal(A):
        N = len(A)
        d = np.diag(A).copy()
        l = np.zeros(N, dtype=float)
        u = np.zeros(N, dtype=float)
        l[1:] = np.diag(A, -1)
        u[:-1] = np.diag(A, 1)
        return d, l, u

    return (dense_to_tridiagonal,)


@app.cell(hide_code=True)
def _(
    dense_to_tridiagonal,
    lu_decomp_partial_pivot,
    lu_solve_partial_pivot,
    mo,
    np,
    tridiagonal_solve,
):
    _A_tri = np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 4.0, -5.0, 0.0],
            [0.0, -4.0, 3.0, 5.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    _v_tri = np.array([-4.0, 3.0, 9.0, 7.0])

    _x_tri = tridiagonal_solve(*dense_to_tridiagonal(_A_tri), _v_tri)
    _L, _U, _row_order = lu_decomp_partial_pivot(_A_tri)
    _x_lu = lu_solve_partial_pivot(_L, _U, _row_order, _v_tri)

    _residual = np.max(np.abs(_A_tri.dot(_x_tri) - _v_tri))
    _agreement = np.max(np.abs(_x_tri - _x_lu))

    mo.md(
        f"""
        $\\mathbf{{x}} = ({_x_tri[0]:.4f},\\ {_x_tri[1]:.4f},\\
        {_x_tri[2]:.4f},\\ {_x_tri[3]:.4f})$, residual
        $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {_residual:.2e}$
        — agreeing with Section 6's general LU-with-pivoting solver
        to within
        $\\max|\\mathbf{{x}}_{{\\rm tri}}-\\mathbf{{x}}_{{\\rm LU}}|
        = {_agreement:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Wall-clock scaling.** Both solvers reach the same answer — one
    in $\mathcal{O}(N)$, the other in $\mathcal{O}(N^3)$. Generating
    random, diagonally dominant tridiagonal systems of increasing
    size $N$ and timing both directly shows the gap between them.
    """)
    return


@app.cell
def _(time):
    def bench(fn, *args, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            best = min(best, time.perf_counter() - t0)
        return best

    return (bench,)


@app.cell
def _(lu_decomp_partial_pivot, lu_solve_partial_pivot):
    def general_lu_solve(A, v):
        L, U, row_order = lu_decomp_partial_pivot(A)
        return lu_solve_partial_pivot(L, U, row_order, v)

    return (general_lu_solve,)


@app.cell
def _(plt):
    _SPECIAL_COLOR = "#2a78d6"
    _GENERAL_COLOR = "#eb6834"

    def draw_timing_plot(Ns, times_special, times_general, label_special, label_general):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(Ns, times_special, "o-", color=_SPECIAL_COLOR, lw=2, ms=7, label=label_special)
        ax.plot(Ns, times_general, "o-", color=_GENERAL_COLOR, lw=2, ms=7, label=label_general)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$ (matrix size)")
        ax.set_ylabel("Wall time (s)")
        ax.grid(True, which="both", color="0.85", lw=0.7)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        return fig

    return (draw_timing_plot,)


@app.cell
def _(bench, dense_to_tridiagonal, general_lu_solve, np, tridiagonal_solve):
    def random_tridiagonal_dense(N, rng):
        A = np.zeros((N, N), dtype=float)
        idx = np.arange(N)
        A[idx, idx] = rng.uniform(4.0, 6.0, size=N)
        off = rng.uniform(-1.0, 1.0, size=N - 1)
        A[idx[1:], idx[:-1]] = off
        A[idx[:-1], idx[1:]] = off
        return A

    _rng = np.random.default_rng(0)
    tri_Ns = [20, 40, 80, 120, 160, 200, 260, 320, 400, 500]

    tri_times_special = []
    tri_times_general = []
    for _N in tri_Ns:
        _A = random_tridiagonal_dense(_N, _rng)
        _v = _rng.uniform(-1.0, 1.0, size=_N)
        _d, _l, _u = dense_to_tridiagonal(_A)
        tri_times_special.append(bench(tridiagonal_solve, _d, _l, _u, _v))
        tri_times_general.append(bench(general_lu_solve, _A, _v))
    return tri_Ns, tri_times_general, tri_times_special


@app.cell(hide_code=True)
def _(draw_timing_plot, mo, tri_Ns, tri_times_general, tri_times_special):
    mo.vstack(
        [
            draw_timing_plot(
                tri_Ns,
                tri_times_special,
                tri_times_general,
                "Tridiagonal solver — $\\mathcal{O}(N)$",
                "General LU + pivoting — $\\mathcal{O}(N^3)$",
            ),
            mo.md(
                r"""
                On these log–log axes, a power law $t\propto N^p$ is
                a straight line of slope $p$: the general solver's
                curve steepens toward slope 3 as $N$ grows, while the
                tridiagonal solver's stays close to slope 1 —
                exactly the $\mathcal{O}(N^3)$ vs. $\mathcal{O}(N)$
                gap the derivation above predicts. (At small $N$
                both curves flatten out, dominated by fixed
                Python-level overhead rather than either
                complexity.)
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Banded systems
    ---

    Tridiagonal systems generalize to **banded** systems, where each
    row may have up to $m_{\rm lower}$ nonzero entries to the left
    of the diagonal and up to $m_{\rm upper}$ to the right of it
    (tridiagonal is the special case $m_{\rm lower}=m_{\rm
    upper}=1$):
    $$
    \mathbf{A} =
    \begin{pmatrix}
    d_1 & u_{1,1} & u_{2,1} & & \\
    l_{1,2} & d_2 & u_{1,2} & u_{2,2} & \\
    l_{2,3} & l_{1,3} & d_3 & \ddots & \ddots\\
     & \ddots & \ddots & \ddots & u_{1,N-1}\\
     & & l_{2,N} & l_{1,N} & d_N
    \end{pmatrix}
    \qquad (m_{\rm lower}=m_{\rm upper}=2\text{ shown}).
    $$
    The same argument as Section 9 applies, just over a wider band:
    eliminating column $k$ only ever touches the $m_{\rm lower}$
    rows below it, and within each of those rows only the $m_{\rm
    upper}+1$ nonzero entries need updating — no longer a single
    entry, but still a fixed number, independent of $N$. The
    elimination is therefore $\mathcal{O}(N\,m_{\rm lower}\,m_{\rm
    upper})$: linear in $N$, same order as Section 9, whenever the
    bandwidth $m_{\rm lower},m_{\rm upper}\ll N$ is held fixed as
    $N$ grows.
    """)
    return


@app.cell
def _(np):
    def banded_solve(d, l, u, v0):
        N = len(v0)
        mlower = l.shape[0]
        mupper = u.shape[0]

        a = d.astype(float).copy()
        dlow = l.astype(float).copy()
        dup = u.astype(float).copy()
        v = v0.astype(float).copy()

        for r in range(N):
            div = a[r]
            for c in range(r + 1, min(r + mupper + 1, N)):
                dup[c - r - 1, r] /= div
            v[r] /= div

            max_row = min(r + mlower, N - 1)
            for r2 in range(r + 1, max_row + 1):
                v[r2] -= dlow[r2 - r - 1, r2] * v[r]
            for c in range(r + 1, min(r + mupper + 1, N)):
                for r2 in range(r + 1, max_row + 1):
                    if c == r2:
                        a[r2] -= dlow[r2 - r - 1, r2] * dup[c - r - 1, r]
                    elif c < r2 and r2 - c - 1 < mlower:
                        dlow[r2 - c - 1, r2] -= dlow[r2 - r - 1, r2] * dup[c - r - 1, r]
                    elif c > r2 and c - r2 - 1 < mupper:
                        dup[c - r2 - 1, r2] -= dlow[r2 - r - 1, r2] * dup[c - r - 1, r]

        x = np.empty(N, dtype=float)
        for r in range(N - 1, -1, -1):
            x[r] = v[r]
            for c in range(r + 1, min(r + mupper + 1, N)):
                x[r] -= dup[c - r - 1, r] * x[c]
        return x

    return (banded_solve,)


@app.cell
def _(np):
    def dense_to_banded(A, mlower, mupper):
        N = len(A)
        d = np.diag(A).copy()
        l = np.zeros((mlower, N), dtype=float)
        u = np.zeros((mupper, N), dtype=float)
        for r in range(N):
            for c in range(max(0, r - mlower), r):
                l[r - c - 1, r] = A[r, c]
            for c in range(r + 1, min(r + mupper + 1, N)):
                u[c - r - 1, r] = A[r, c]
        return d, l, u

    return (dense_to_banded,)


@app.cell(hide_code=True)
def _(
    banded_solve,
    dense_to_banded,
    lu_decomp_partial_pivot,
    lu_solve_partial_pivot,
    mo,
    np,
):
    _A_band = np.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 4.0, -5.0, 0.0],
            [0.0, -4.0, 3.0, 5.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    _v_band = np.array([-4.0, 3.0, 9.0, 7.0])

    _x_band = banded_solve(*dense_to_banded(_A_band, 1, 1), _v_band)
    _L, _U, _row_order = lu_decomp_partial_pivot(_A_band)
    _x_lu = lu_solve_partial_pivot(_L, _U, _row_order, _v_band)

    _residual = np.max(np.abs(_A_band.dot(_x_band) - _v_band))
    _agreement = np.max(np.abs(_x_band - _x_lu))

    mo.md(
        f"""
        With $m_{{\\rm lower}}=m_{{\\rm upper}}=1$ this is exactly
        Section 9's tridiagonal example. $\\mathbf{{x}} =
        ({_x_band[0]:.4f},\\ {_x_band[1]:.4f},\\ {_x_band[2]:.4f},\\
        {_x_band[3]:.4f})$, residual
        $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {_residual:.2e}$
        — agreeing with the general LU-with-pivoting solver to
        within
        $\\max|\\mathbf{{x}}_{{\\rm band}}-\\mathbf{{x}}_{{\\rm LU}}|
        = {_agreement:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell
def _(banded_solve, bench, dense_to_banded, general_lu_solve, np):
    def random_banded_dense(N, mlower, mupper, rng):
        bandwidth = mlower + mupper
        A = np.zeros((N, N), dtype=float)
        for r in range(N):
            A[r, r] = rng.uniform(2 * bandwidth + 2.0, 2 * bandwidth + 4.0)
            for c in range(max(0, r - mlower), r):
                A[r, c] = rng.uniform(-1.0, 1.0)
            for c in range(r + 1, min(r + mupper + 1, N)):
                A[r, c] = rng.uniform(-1.0, 1.0)
        return A

    band_mlower, band_mupper = 3, 3
    band_Ns = [20, 40, 80, 120, 160, 200, 260, 320, 400, 500]
    _rng = np.random.default_rng(1)

    band_times_special = []
    band_times_general = []
    for _N in band_Ns:
        _A = random_banded_dense(_N, band_mlower, band_mupper, _rng)
        _v = _rng.uniform(-1.0, 1.0, size=_N)
        _d, _l, _u = dense_to_banded(_A, band_mlower, band_mupper)
        band_times_special.append(bench(banded_solve, _d, _l, _u, _v))
        band_times_general.append(bench(general_lu_solve, _A, _v))
    return band_Ns, band_mlower, band_times_general, band_times_special


@app.cell(hide_code=True)
def _(
    band_Ns,
    band_mlower,
    band_times_general,
    band_times_special,
    draw_timing_plot,
    mo,
):
    mo.vstack(
        [
            draw_timing_plot(
                band_Ns,
                band_times_special,
                band_times_general,
                f"Banded solver ($m_l=m_u={band_mlower}$) — $\\mathcal{{O}}(N)$",
                "General LU + pivoting — $\\mathcal{O}(N^3)$",
            ),
            mo.md(
                r"""
                Same picture as Section 9, with a larger, fixed
                bandwidth: the banded solver's cost still grows
                linearly in $N$ — just with a bigger constant,
                $m_{\rm lower}m_{\rm upper}$ times that of the
                tridiagonal case — while the general solver's
                $\mathcal{O}(N^3)$ cost doesn't care how sparse
                $\mathbf{A}$ is. The gap only widens as $N$ grows.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. QR decomposition
    ---

    A different, equally general factorization of a square matrix
    $\mathbf{A}$ is
    $$
    \mathbf{A} = \mathbf{Q}\mathbf{R},
    $$
    where $\mathbf{Q}$ is **orthogonal**
    ($\mathbf{Q}^T\mathbf{Q}=\mathbf{Q}\mathbf{Q}^T=\mathbf{I}$, so
    $\mathbf{Q}^{-1}=\mathbf{Q}^T$ — no separate inversion needed)
    and $\mathbf{R}$ is upper triangular, exactly like the
    $\mathbf{U}$ of Section 4. Unlike $\mathbf{L}\mathbf{U}$, this
    factorization exists for *every* matrix, singular or not, and
    never needs pivoting: $\mathbf{Q}$ is always perfectly
    well-conditioned by construction
    ($\|\mathbf{Q}\mathbf{x}\|=\|\mathbf{x}\|$ for any
    $\mathbf{x}$).

    Several algorithms compute it — Gram–Schmidt
    orthogonalization, Householder reflections, Givens rotations —
    trading off simplicity against numerical stability; we won't
    rederive them here and instead use `numpy.linalg.qr` (Householder
    reflections under the hood). The demo below applies it to
    Section 1's draggable matrix $\mathbf{A}$.
    """)
    return


@app.cell(hide_code=True)
def _(A0_ge, mo, np):
    _Q, _R = np.linalg.qr(A0_ge)

    _Q_ui = mo.ui.matrix(
        _Q.tolist(),
        precision=4,
        disabled=True,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["C1", "C2", "C3", "C4"],
        label="**Q** (orthogonal)",
    )
    _R_ui = mo.ui.matrix(
        _R.tolist(),
        precision=4,
        disabled=True,
        row_labels=["R1", "R2", "R3", "R4"],
        column_labels=["C1", "C2", "C3", "C4"],
        label="**R** (upper triangular)",
    )

    _residual_A = np.max(np.abs(A0_ge - _Q.dot(_R)))
    _residual_orth = np.max(np.abs(_Q.T.dot(_Q) - np.eye(len(A0_ge))))

    mo.vstack(
        [
            mo.hstack([_Q_ui, _R_ui], justify="center", align="center", gap=3),
            mo.md(
                f"""
                Checking against Section 1's $\\mathbf{{A}}$ (drag
                its entries to try another):
                $\\max|\\mathbf{{A}}-\\mathbf{{Q}}\\mathbf{{R}}|
                = {_residual_A:.2e}$,
                $\\max|\\mathbf{{Q}}^T\\mathbf{{Q}}-\\mathbf{{I}}|
                = {_residual_orth:.2e}$.
                """
            ).callout(kind="success"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Solving with QR decomposition
    ---

    Once $\mathbf{A}=\mathbf{Q}\mathbf{R}$ is known,
    $\mathbf{A}\mathbf{x}=\mathbf{v}$ becomes
    $$
    \mathbf{Q}\mathbf{R}\mathbf{x} = \mathbf{v}.
    $$
    Left-multiplying by $\mathbf{Q}^T$ and using
    $\mathbf{Q}^T\mathbf{Q}=\mathbf{I}$ isolates
    $$
    \mathbf{R}\mathbf{x} = \mathbf{Q}^T\mathbf{v}
    $$
    — no matrix inversion, just a matrix–vector product for the new
    right-hand side. $\mathbf{R}$ is upper triangular, so the rest
    is exactly Section 2's back-substitution. Unlike the
    $\mathbf{L}\mathbf{U}$ route (Section 5), this needs no
    pivoting step to stay numerically sound — at the cost of QR
    decomposition itself being more expensive to compute than LU.

    The demo below applies it to Section 3's zero-pivot matrix,
    where QR needs no special handling at all.
    """)
    return


@app.cell
def _(np):
    def qr_solve(Q, R, v):
        N = len(v)
        y = Q.T @ v

        x = np.empty(N, dtype=float)
        for r in range(N - 1, -1, -1):
            x[r] = y[r] - R[r, r + 1 :] @ x[r + 1 :]
            x[r] /= R[r, r]
        return x

    return (qr_solve,)


@app.cell(hide_code=True)
def _(A_piv0, general_lu_solve, mo, np, qr_solve, v_piv0):
    _Q, _R = np.linalg.qr(A_piv0)
    _x_qr = qr_solve(_Q, _R, v_piv0)
    _x_lu = general_lu_solve(A_piv0, v_piv0)

    _residual = np.max(np.abs(A_piv0.dot(_x_qr) - v_piv0))
    _agreement = np.max(np.abs(_x_qr - _x_lu))

    mo.md(
        f"""
        No pivoting needed: $\\mathbf{{x}} = ({_x_qr[0]:.4f},\\
        {_x_qr[1]:.4f},\\ {_x_qr[2]:.4f},\\ {_x_qr[3]:.4f})$,
        residual
        $\\max|\\mathbf{{A}}\\mathbf{{x}}-\\mathbf{{v}}| = {_residual:.2e}$
        — agreeing with the general LU-with-pivoting solver to
        within
        $\\max|\\mathbf{{x}}_{{\\rm QR}}-\\mathbf{{x}}_{{\\rm LU}}|
        = {_agreement:.2e}$.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Eigenvalues and eigenvectors — the QR algorithm
    ---

    A different matrix problem altogether: given $\mathbf{A}$, find
    the vectors left unrotated by it,
    $$
    \mathbf{A}\mathbf{v} = \lambda\mathbf{v}.
    $$
    For a real symmetric $\mathbf{A}$ (the case we focus on here),
    all $N$ eigenvalues $\lambda_i$ are real and the eigenvectors
    orthogonal, so the whole problem collapses into one matrix
    equation,
    $$
    \mathbf{A}\mathbf{V} = \mathbf{V}\mathbf{D}, \qquad
    \mathbf{V}^T\mathbf{V} = \mathbf{I},
    $$
    where column $i$ of $\mathbf{V}$ is $\mathbf{v}_i$ and
    $\mathbf{D}$ is diagonal with entries $\lambda_i$.

    Remarkably, the QR decomposition of Section 11 — used only to
    *solve* linear systems so far — also finds $\mathbf{V}$ and
    $\mathbf{D}$, through a short iteration with no elimination or
    pivoting in sight:

    1. Set $\mathbf{A}_1=\mathbf{A}$ and factor it,
       $\mathbf{A}_1=\mathbf{Q}_1\mathbf{R}_1$.
    2. Multiply the factors back together in the *opposite* order,
       $\mathbf{A}_2 = \mathbf{R}_1\mathbf{Q}_1$.
    3. Repeat: $\mathbf{A}_{k+1} = \mathbf{R}_k\mathbf{Q}_k$, where
       $\mathbf{A}_k=\mathbf{Q}_k\mathbf{R}_k$.

    Reversing the product looks arbitrary, but it isn't:
    substituting $\mathbf{R}_k=\mathbf{Q}_k^T\mathbf{A}_k$ shows
    $$
    \mathbf{A}_{k+1} = \mathbf{Q}_k^T\mathbf{A}_k\mathbf{Q}_k,
    $$
    a **similarity transform** — $\mathbf{A}_{k+1}$ has exactly the
    same eigenvalues as $\mathbf{A}_k$, and hence as $\mathbf{A}$,
    for every $k$. Iterated, $\mathbf{A}_k$ converges toward
    diagonal form as $k\to\infty$, its diagonal settling onto the
    eigenvalues $\lambda_i$; accumulating the rotations,
    $\mathbf{Q}=\mathbf{Q}_1\mathbf{Q}_2\cdots$, gives the
    eigenvectors $\mathbf{V}=\mathbf{Q}$ directly. In practice the
    iteration stops once the off-diagonal entries of $\mathbf{A}_k$
    fall below some tolerance.

    The demo below runs this on a fixed random symmetric
    $4\times4$ matrix — drag the slider to watch $\mathbf{A}_k$
    settle onto a diagonal.
    """)
    return


@app.cell
def _(np):
    def _offdiag_max(M):
        return np.max(np.abs(M - np.diag(np.diag(M))))

    def eigen_qr_trace(A, iterations):
        N = len(A)
        Ak = A.astype(float).copy()
        Qcum = np.eye(N)

        trace = [dict(k=0, A=Ak.copy(), Q=Qcum.copy(), offdiag=_offdiag_max(Ak))]
        for k in range(1, iterations + 1):
            Q, R = np.linalg.qr(Ak)
            Ak = R.dot(Q)
            Qcum = Qcum.dot(Q)
            trace.append(dict(k=k, A=Ak.copy(), Q=Qcum.copy(), offdiag=_offdiag_max(Ak)))
        return trace

    return (eigen_qr_trace,)


@app.cell
def _(np):
    _rng = np.random.default_rng(42)
    _M = _rng.uniform(-2.0, 2.0, size=(4, 4))
    A_eig = (_M + _M.T) / 2
    return (A_eig,)


@app.cell
def _(A_eig, eigen_qr_trace):
    eig_trace = eigen_qr_trace(A_eig, 40)
    return (eig_trace,)


@app.cell(hide_code=True)
def _(eig_trace, mo):
    eig_step_slider = mo.ui.slider(
        0, len(eig_trace) - 1, value=0, step=1, label="QR algorithm iteration"
    )
    return (eig_step_slider,)


@app.cell(hide_code=True)
def _(Patch, plt):
    def draw_eigen_step(step_idx, trace):
        A_k = trace[step_idx]["A"]
        N = len(A_k)
        cell = 1.0

        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        for r in range(N):
            for c in range(N):
                x0, y0 = c * cell, (N - 1 - r) * cell
                facecolor = "#fdf0d5" if r == c else "white"
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0), cell, cell,
                        facecolor=facecolor, edgecolor="0.6", lw=1.0, zorder=2,
                    )
                )
                ax.text(
                    x0 + cell / 2, y0 + cell / 2, f"{A_k[r, c]:.3f}",
                    ha="center", va="center", fontsize=10, zorder=3,
                )

        legend_handles = [
            Patch(facecolor="#fdf0d5", edgecolor="0.6", label="Diagonal — eigenvalue estimate"),
        ]
        ax.legend(
            handles=legend_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9,
        )

        ax.set_xlim(-0.2, N * cell + 0.2)
        ax.set_ylim(-0.2, N * cell + 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return (draw_eigen_step,)


@app.cell(hide_code=True)
def _(draw_eigen_step, eig_step_slider, eig_trace, mo):
    _state = eig_trace[eig_step_slider.value]

    mo.vstack(
        [
            eig_step_slider,
            mo.hstack(
                [
                    draw_eigen_step(eig_step_slider.value, eig_trace),
                    mo.vstack(
                        [
                            mo.md(
                                rf"""
                                **Iteration {_state['k']} of
                                {len(eig_trace) - 1}.** Largest
                                off-diagonal entry:
                                $\max_{{i\neq j}}|(\mathbf{{A}}_k)_{{ij}}|
                                = {_state['offdiag']:.2e}$.
                                """
                            ),
                            mo.hstack(
                                [
                                    mo.stat(label="Iteration", value=str(_state["k"])),
                                    mo.stat(label="Max off-diagonal", value=f"{_state['offdiag']:.2e}"),
                                ]
                            ),
                        ]
                    ),
                ],
                widths=[0.55, 0.45],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(A_eig, eig_trace, mo, np):
    _D = eig_trace[-1]["A"]
    _V = eig_trace[-1]["Q"]

    _our_eigs = np.sort(np.diag(_D))
    _np_eigs = np.sort(np.linalg.eigvalsh(A_eig))

    _orth_residual = np.max(np.abs(_V.T.dot(_V) - np.eye(len(A_eig))))
    _eig_residual = np.max(np.abs(_our_eigs - _np_eigs))

    mo.md(
        f"""
        After {len(eig_trace) - 1} iterations, the eigenvalues read
        off $\\mathbf{{A}}_k$'s diagonal are
        $({_our_eigs[0]:.4f},\\ {_our_eigs[1]:.4f},\\
        {_our_eigs[2]:.4f},\\ {_our_eigs[3]:.4f})$, matching
        `numpy.linalg.eigvalsh` to within
        $\\max|\\lambda_{{\\rm ours}}-\\lambda_{{\\rm NumPy}}|
        = {_eig_residual:.2e}$. The accumulated
        $\\mathbf{{Q}}=\\mathbf{{Q}}_1\\mathbf{{Q}}_2\\cdots$ is
        orthogonal to within
        $\\max|\\mathbf{{Q}}^T\\mathbf{{Q}}-\\mathbf{{I}}|
        = {_orth_residual:.2e}$.
        """
    ).callout(kind="success")
    return


if __name__ == "__main__":
    app.run()
