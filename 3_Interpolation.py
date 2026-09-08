import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/3_Interpolation.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline, RegularGridInterpolator
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.interpolate import BarycentricInterpolator
    import math
    # from wigglystuff.formula_animation import FormulaAnimation
    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return (
        BarycentricInterpolator,
        CubicSpline,
        RegularGridInterpolator,
        go,
        make_subplots,
        math,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interpolation

    Interpolation estimates a function between a finite set of known values
    $(x_i, f_i)$. In this notebook we will develop the principal
    one-dimensional methods from their definitions, examine their errors, and
    then extend the ideas to two-dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    md_sample_title = mo.md(
    """
    ## 1. Sampling a function
    ---

    """
    )
    md_sample_count = mo.md(
    """
    Let $f(x)=\sin(x)$ on $[0,2\pi]$. Imagine that the black curve is unknown:
    only the red sample locations are available to an algorithm. Every method
    below is constrained to use those samples alone.
    """
    )
    return md_sample_count, md_sample_title


@app.cell
def _(mo):
    sample_count = mo.ui.slider(4, 25, value=7, step=1, label="Number of data points")
    return (sample_count,)


@app.cell
def _(md_sample_count, md_sample_title, mo, np, plt, sample_count):
    x_data = np.linspace(0, 6, sample_count.value)
    y_data = np.sin(x_data)
    x_reference = np.linspace(0, 6, 1000)
    y_reference = np.sin(x_reference)

    _fig, _ax = plt.subplots()
    _ax.plot(x_reference, y_reference, "k--", lw=2, label="True function")
    _ax.scatter(x_data, y_data, color="crimson", s=48, zorder=3, label="Known data")
    _ax.set(xlabel="$x$", ylabel="$f(x)$",)
    _ax.legend()
    mo.vstack([md_sample_title,md_sample_count, sample_count, _fig])
    return x_data, x_reference, y_data, y_reference


@app.cell
def _(mo):
    md_nearest_title = mo.md(
    r"""
    ## Nearest-neighbour interpolation
    ---
    """
    )
    md_nearest = mo.md(
    r"""
    The simplest interpolant assigns the value of the closest datum:
    $$I_{\rm nn}(x)=f_i,\qquad i=\arg\min_j|x-x_j|.$$

    It is piecewise constant, discontinuous at midpoints between nodes, and
    therefore unsuitable when derivatives are needed. Its value is that it is
    simple, local, and extends naturally to scattered multidimensional data.
    """
    )
    return md_nearest, md_nearest_title


@app.cell
def _(np):
    def nearest_neighbour(x, nodes, values):
        """Return the value belonging to the closest node."""
        index = np.abs(nodes - x).argmin()
        return values[index]

    return (nearest_neighbour,)


@app.cell
def _(mo, np):
    nearest_query = mo.ui.slider(0.0, float(2 * np.pi), value=1.2, step=0.01, label="Evaluation point x")
    return (nearest_query,)


@app.cell
def _(
    md_nearest,
    md_nearest_title,
    mo,
    nearest_neighbour,
    nearest_query,
    np,
    plt,
    x_data,
    x_reference,
    y_data,
    y_reference,
):
    nearest_curve = np.array([nearest_neighbour(x, x_data, y_data) for x in x_reference])
    nearest_value = nearest_neighbour(nearest_query.value, x_data, y_data)
    exact_value = np.sin(nearest_query.value)

    _fig, _ax = plt.subplots()
    _ax.plot(x_reference, y_reference, "k--", label="True function")
    _ax.step(x_reference, nearest_curve, where="mid", lw=2, label="Nearest neighbour")
    _ax.scatter(x_data, y_data, color="crimson", zorder=3, label="Data")
    _ax.axvline(nearest_query.value, color="0.4", ls=":")
    _ax.scatter(nearest_query.value, nearest_value, color="tab:blue", s=60, zorder=4, label="Interpolation")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.3), loc="upper center")
    mo.vstack([md_nearest_title, md_nearest,mo.hstack([nearest_query, mo.stat(label="Absolute error", value=f"{abs(nearest_value - exact_value):.3e}")],justify="center"), _fig],justify='center')
    return


@app.cell
def _(np, plt):
    def f(x):
        return 4 * (1 - np.exp(-0.45 * x))

    _a = 2
    _b = 8
    _x0 = 6

    # Function values
    fa = f(_a)
    fb = f(_b)
    fx = f(_x0)

    # Straight line through (a,f(a)) and (b,f(b))
    m = (fb - fa) / (_b - _a)
    line = lambda x: fa + m * (x - _a)

    lx = line(_x0)

    fig_lin, _ax = plt.subplots()
    fig_lin.patch.set_facecolor("#ffffff")
    _ax.set_facecolor("#ffffff")

    # Curve
    _xx = np.linspace(1.4, 9.5, 500)
    _ax.plot(_xx, f(_xx), color="black", lw=1.3)

    # Straight line
    xx_line = np.linspace(0.8, 10, 2)
    _ax.plot(xx_line, line(xx_line), color="black", lw=1.2)

    # Key points
    _ax.plot([_a, _x0, _b], [fa, lx, fb], "ko", ms=6)

    # Dashed construction lines
    _ax.plot([_a, _a], [0, fa], "--", color="0.35", lw=1.4)
    _ax.plot([_b, _b], [0, fb], "--", color="0.35", lw=1.4)
    _ax.plot([_a, _b], [fa, fa], "--", color="0.35", lw=1.4)

    # Vertical distance arrows
    _ax.annotate(
        "",
        xy=(_x0, lx),
        xytext=(_x0, fa),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="black")
    )

    _ax.annotate(
        "",
        xy=(_x0, fa),
        xytext=(_x0, 0),
        arrowprops=dict(arrowstyle="<->", lw=1.0, color="black")
    )

    # Labels y and z
    _ax.text(_x0 + 0.08, (lx + fa) / 2, r"$y$", fontsize=24)
    _ax.text(_x0 + 0.08, fa / 2, r"$z$", fontsize=24)

    # Point labels
    _ax.text(_a - 0.9, fa + 0.15, r"$f(a)$", fontsize=24)
    _ax.text(_b - 0.8, fb + 0.15, r"$f(b)$", fontsize=24)

    # Axis labels
    _ax.text(_a, -0.4, r"$a$", ha="center", fontsize=24)
    _ax.text(_x0, -0.4, r"$x$", ha="center", fontsize=24)
    _ax.text(_b, -0.4, r"$b$", ha="center", fontsize=24)

    # Annotation: curve
    _ax.annotate(
        r"Curve of $f(x)$",
        xy=(5.5, f(5.5)),
        xytext=(3.0, 4.6),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=22,
        ha="center"
    )

    # Annotation: straight line
    _ax.annotate(
        "Straight line",
        xy=(5.2, line(5.2)),
        xytext=(1.6, 4.0),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=22,
        ha="center"
    )

    # Axes styling
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _ax.set_xlim(-0.5, 10.4)
    _ax.set_ylim(0, 5)

    _ax.set_xticks([])
    _ax.set_yticks([])
    return (fig_lin,)


@app.cell
def _(fig_lin, mo):
    _md_linear = mo.md(
    r"""
    ## 3. Linear interpolation
    ---

    Nearest-neighbour interpolation is simple but not smooth and leads to significant errors, especially when the underlying function varies rapidly.
    Linear interpolation improves upon this by assuming that the function varies linearly between known data points.

    Suppose we know the values of a function at two points, $a$ and $b$ with $a < b$ and where the function values $f(a)$ and $f(b)$ are known.
    If we want to estimate the value of the function at some point $x$ such that $a \le x \le b$, we can use linear interpolation.

    If we do a little geometry, we can see that the linear interpolation formula is essentially finding the point on the straight line connecting \((a, f(a))\) and \((b, f(b))\) that corresponds to the horizontal position \(x\).

    The slope ($m$) of this line is given by
    \[
    m = \frac{f(b) - f(a)}{b - a}.
    \]
    Looking at the figure we can see that:
    \[
    f(x) \approx y + z = \frac{f(b) - f(a)}{b - a} (x - a) + f(a)
    \]
    \[
    = \frac{(b-x)f(a) - (a-x)f(b)}{b-a}
    \]
    """
    )
    mo.hstack([ _md_linear, fig_lin],widths=[0.6, 0.4],align="center")
    return


@app.cell
def _(fig_lin, mo):
    _md_linear = mo.md(
    r"""
    ### Accuracy of Linear Interpolation
    Let us express the $f(a)$ and $f(b)$ using their Taylor series expansions around some point $x$:
    \[
    f(a) = f(x) + f'(x)(a-x) + \frac{f''(x)}{2}(a-x)^2+ ...,
    \]
    \[
    f(b) = f(x) + f'(x)(b-x) + \frac{f''(x)}{2}(b-x)^2+ ...,
    \]
    if we multiply f(a) by $(b-x)$ and f(b) by $(a-x)$ and subtract the two expressions, we get:
    \[
    (b-x)f(a) - (a-x)f(b) = 
    \]
    \[
    (b-x)f(x) \cancel{+ (b-x)f'(x)(a-x)} + \frac{f''(x)}{2}(b-x)(a-x)^2 -
    \]
    \[
    (a-x)f(x) \cancel{- (a-x)f'(x)(b-x)} - \frac{f''(x)}{2}(a-x)(b-x)^2 - ...
    \]
    which can be rearranged to highlight the leading term and the error term in the linear interpolation formula.
    \[
    f(x) = \overbrace{\frac{(b-x)f(a) - (a-x)f(b)}{b-a}}^{\text{linear interpolation}} + \underbrace{(a-x)(b-x) \frac{f''(x)}{2}}_{\text{error term}} + ...,
    \]
    """
    )

    mo.hstack([ _md_linear, fig_lin],widths=[0.6, 0.4],align="center")
    return


@app.cell
def _(fig_lin, mo):
    _md_linear = mo.md(
    r"""
    \[
    f(x) = \overbrace{\frac{(b-x)f(a) - (a-x)f(b)}{b-a}}^{\text{linear interpolation}} + \underbrace{(a-x)(b-x) \frac{f''(x)}{2}}_{\text{error term}} + ...,
    \]
    The error will be maximum at the midpoint between $a$ and $b$. (if $f''(x)$ does not vary significantly over the interval)  
    If we define $h = b-a$, then: 

    \[
    \begin{aligned}
    x-a &= \frac{h}{2} \quad \text{at the midpoint between } a \text{ and } b.\\
    b-x &= \frac{h}{2}.
    \end{aligned}
    \]
    we can rearrange the error term as:
    \[
    \text{error} = -(a-x)(b-x) \frac{f''(x)}{2} \approx -\frac{h^2}{8} f''\left(x\right).
    \]
    So the error is bounded by the maximum value of the second derivative over the interval.
    \[
    \text{error} \leq \frac{h^2}{8} \max_{x \in [a,b]} \left| f''(x) \right|.
    \]
    """
    )

    mo.hstack([ _md_linear, fig_lin],widths=[0.6, 0.4],align="center")
    return


@app.cell
def _(np):
    def linear_interpolant(x, nodes, values):
        """Evaluate the piecewise-linear interpolant, including endpoints."""
        return np.interp(x, nodes, values)

    return (linear_interpolant,)


@app.cell
def _(mo, np):
    linear_query = mo.ui.slider(0.0, float(2 * np.pi), value=1.7, step=0.01, label="Evaluation point x")
    return (linear_query,)


@app.cell
def _(
    linear_interpolant,
    linear_query,
    mo,
    np,
    plt,
    sample_count,
    x_data,
    x_reference,
    y_data,
    y_reference,
):
    linear_curve = linear_interpolant(x_reference, x_data, y_data)
    linear_value = linear_interpolant(linear_query.value, x_data, y_data)
    point_error = abs(linear_value - np.sin(linear_query.value))
    h = x_data[1] - x_data[0]
    bound = h**2 / 8

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    _axes[0].plot(x_reference, y_reference, "k--", label="Reference function")
    _axes[0].plot(x_reference, linear_curve, lw=2, label="Linear interpolant")
    _axes[0].scatter(x_data, y_data, color="crimson", zorder=3, label="Data")
    _axes[0].scatter(linear_query.value, linear_value, color="tab:blue", s=60, zorder=4)
    _axes[0].set(xlabel="$x$", ylabel="$f(x)$", title="Piecewise linear interpolation")
    _axes[0].legend()
    _axes[1].plot(x_reference, abs(y_reference - linear_curve), label="Measured error")
    _axes[1].axhline(bound, color="crimson", ls="--", label="$h^2/8$ bound")
    _axes[1].set(xlabel="$x$", ylabel="Absolute error", title="Error and theoretical bound")
    _axes[1].set_ylim(0, max(abs(y_reference - linear_curve).max(), bound) * 1.5)
    # put axes 2 y axis label and ticks on the right
    _axes[1].yaxis.set_label_position("right")
    _axes[1].yaxis.tick_right()
    _axes[1].legend()
    mo.vstack([mo.hstack([sample_count,linear_query,mo.stat(label="Point error", value=f"{point_error:.3e}"), mo.stat(label="Bound for sin(x)", value=f"{bound:.3e}")]), _fig])
    return


@app.cell
def _(mo):
    finest_nodes = mo.ui.slider(10, 80, value=40, step=2, label="Largest node count in study")
    return (finest_nodes,)


@app.cell
def _(finest_nodes, mo, np, plt):
    _md_convergence = mo.md(r"""### Convergence Study

    If one wants to study the convergence of linear interpolation, one can do so by measuring the maximum error as the grid is refined.
    We notice that the error decreases as the grid $h$ is refined, and the slope of the log-log plot indicates the observed convergence order $\mathcal{O}(h^2)$. As expected from our previous analysis on the error of linear interpolation for smooth functions.
    """)


    node_counts = np.arange(4, finest_nodes.value + 1, 2)
    test_grid = np.linspace(0, 2 * np.pi, 5001)
    spacings = 2 * np.pi / (node_counts - 1)
    max_errors = np.array([
        np.max(abs(np.interp(test_grid, np.linspace(0, 2 * np.pi, count), np.sin(np.linspace(0, 2 * np.pi, count))) - np.sin(test_grid)))
        for count in node_counts
    ])
    order = np.polyfit(np.log(spacings), np.log(max_errors), 1)[0]
    _fig, _ax = plt.subplots()
    _ax.loglog(spacings, max_errors, "o-", label=r"Measured $L_\infty$ error")
    _ax.loglog(spacings, 2*max_errors[-1] * (spacings / spacings[-1]) ** 2, "--", label="$\mathcal{O}(h^2)$")
    _ax.invert_xaxis()
    _ax.set_xlim(4, 0.09)
    _ax.grid(True, which="both", )
    _ax.set(xlabel="Grid spacing $h$", ylabel="Maximum error", title="Convergence of linear interpolation")
    _ax.legend()
    mo.vstack([_md_convergence, mo.hstack([finest_nodes, mo.stat(label="Observed convergence order", value=f"{order:.3f}")]), _fig])
    return


@app.cell
def _(mo):
    basis_index = mo.ui.slider(0, 6, value=0, step=1, label="Lagrange basis index j")
    return (basis_index,)


@app.cell
def _(basis_index, mo, np, plt):
    _md_title = mo.md(r"""## 4. Polynomial interpolation: the Lagrange form""")

    _md_lagrange_basis = mo.md(r"""
    Given $n+1$ distinct nodes, there is a unique polynomial of degree at most
    $n$ passing through them. Its Lagrange representation is:  

    \[
    P_n(x)=\sum_{j=0}^{n} f_j L_j(x),\qquad L_j(x)=\prod_{k\ne j}\frac{x-x_k}{x_j-x_k}
    \]

    Each basis function has $L_j(x_j)=1$ and $L_j(x_k)=0$ for $k\ne j$.  
    """)
    basis_nodes = np.linspace(-1, 1, 7)
    _grid = np.linspace(-1, 1, 1000)
    basis = np.ones_like(_grid)
    for other_index in range(len(basis_nodes)):
        if other_index != basis_index.value:
            basis *= (_grid - basis_nodes[other_index]) / (basis_nodes[basis_index.value] - basis_nodes[other_index])
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_grid, basis, lw=2, label=fr"$L_{{{basis_index.value}}}(x)$")
    _ax.scatter(basis_nodes, np.zeros_like(basis_nodes), color="black", label="Nodes")
    _ax.axhline(0, color="0.3", lw=0.8)
    _ax.set(xlabel="$x$", ylabel="$L_j(x)$", title="A Lagrange basis function")
    _ax.legend()
    mo.vstack([_md_title, mo.hstack([_md_lagrange_basis, mo.vstack([basis_index, _fig])],widths=[0.5, 0.5],align='center')])
    return


@app.cell
def _(mo):
    md_lagrange_basis = mo.md(r"""
    The error of the Lagrange interpolation can be expressed as
    \[
    R(x) = f(x) - P_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{k=0}^{n} (x - x_k)
    \]
    for some \(\xi\) in the interval containing the nodes.
    Hence, the error is bounded by
    \[
    |R(x)| \le \frac{\max |f^{(n+1)}(\xi)|}{(n+1)!} \max_{x \in [x_0, x_n]} \prod_{k=0}^{n} |x - x_k|
    \]
    """)
    return (md_lagrange_basis,)


@app.cell
def _(math, md_lagrange_basis, mo, np, plt, sample_count, x_data, y_data):
    def Lagrange_basis(x, j, xdata):
        """Lagrange basis function.

        Parameters
        ----------
        x : float
            The point at which to evaluate the basis function.
        j : int
            The index of the basis function.
        xdata : array-like
            The data points used in the interpolation.

        Returns
        -------
        float
            The value of the j-th Lagrange basis function at x.
        """
        ret = 1.
        ret *= np.prod([(x - xdata[k]) / (xdata[j] - xdata[k]) for k in range(len(xdata)) if k != j])
        return ret

    def lagrange_polynomial(x, xdata, fdata):
        """Build the Lagrange polynomial at a given point x.

        Parameters
        ----------
        x : float or array-like
            The point(s) at which to evaluate the polynomial.
        xdata : array-like
            The data points used in the interpolation.
        fdata : array-like
            The function values at the data points.

        Returns
        -------
        float or array-like
            The value of the Lagrange polynomial at x.
        """  
        x = np.asarray(x)
        ret = np.zeros_like(x)
        n = len(xdata) 
        for j in range(0, n):
            basis = np.ones_like(x)
            for other_index in range(len(xdata)):
                if other_index != j:
                    basis *= (x - xdata[other_index]) / (xdata[j] - xdata[other_index])
            ret += fdata[j] * basis
        return ret

    # Calculate the Lagrange polynomial at a specific point
    x_poly = np.linspace(min(x_data), max(x_data), 100)  # Example: 100 points between min and max of xdata
    y_poly = np.array([lagrange_polynomial(x, x_data, y_data) for x in x_poly])

    # Plot the Lagrange polynomial and the original data points and the error on an additional subplot
    _fig = plt.figure(figsize=(16, 5))
    ax1 = _fig.add_subplot(1, 2, 1)
    ax2 = _fig.add_subplot(1, 2, 2)
    ax1.plot(x_poly, y_poly, label='Lagrange Poly.')
    ax1.plot(x_poly, np.sin(x_poly), label= 'True Function', linestyle='--', color='black')
    ax1.scatter(x_data, y_data, color='red', label='Data Pts',zorder = 3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend(loc='best')

    # Plot the error
    error = np.abs(y_poly - np.sin(x_poly))
    ax2.plot(x_poly, error, label='Error', color='purple')
    _n = len(x_data)-1 # polynomial degree
    #theoritical max error
    # calc the max of product \prod_{k=0}^{n} (x - x_k)
    _maxi = np.max([np.prod([(x - x_data[k]) for k in range(len(x_data))]) for x in x_poly])
    ax2.axhline(y=_maxi/math.factorial(_n+1), color='r', linestyle='--', label='Max of Product')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|f(x) - P(x)|')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.legend(loc='best')
    # make custum smaller pad between subplots
    _fig.tight_layout(pad=0.5)

    mo.vstack([md_lagrange_basis,sample_count, _fig])
    return (lagrange_polynomial,)


@app.cell
def _(np):
    def divided_differences(nodes, values):
        """Compute the divided differences for Newton interpolation.


        Parameters
        ----------
        nodes : array-like
            The interpolation nodes (x-values).
        values : array-like
            The function values at the interpolation nodes (y-values).

        Returns
        -------
        array-like
            The divided differences coefficients for the Newton interpolation polynomial.
        """    
        coefficients = np.asarray(values, dtype=float).copy()
        for order in range(1, len(nodes)):
            coefficients[order:] = (coefficients[order:] - coefficients[order - 1:-1]) / (nodes[order:] - nodes[:-order])
        return coefficients

    def newton_polynomial(x, nodes, coefficients):
        """Build the Newton interpolation polynomial at a given point x.

        Parameters
        ----------
        x : float or array-like
            The point(s) at which to evaluate the Newton polynomial.
        nodes : array-like
            The interpolation nodes (x-values).
        coefficients : array-like
            The divided differences coefficients for the Newton interpolation polynomial.

        Returns
        -------
        float
            The value of the Newton interpolation polynomial at x.
        """    
        result = coefficients[-1]
        for index in range(len(coefficients) - 2, -1, -1):
            result = coefficients[index] + (x - nodes[index]) * result
        return result

    return divided_differences, newton_polynomial


@app.cell
def _(
    divided_differences,
    lagrange_polynomial,
    mo,
    newton_polynomial,
    np,
    plt,
    x_data,
    x_reference,
    y_data,
    y_reference,
):
    _net_md = mo.md(
    r"""
    ### Newton form: adding a point efficiently

    \[
    N_{j}(x) = \prod_{k=0}^{j-1} (x - x_k).
    \]

    The divided differences method incrementally calculates coefficients for the Newton polynomial, allowing for efficient interpolation. The coefficients represent the slopes of the divided intervals and are recursively computed as:
    \[
    f[x_i] = f(x_i), \qquad f[x_i, x_{i+1}, \ldots, x_{i+j}] = \frac{f[x_{i+1}, \ldots, x_{i+j}] - f[x_i, \ldots, x_{i+j-1}]}{x_{i+j} - x_i}.
    \]

    Therefore, the interpolating polynomial can be constructed as:
    \[
    p(x) = \sum_{j=0}^n f[x_0, x_1, \ldots, x_j] N_{j}(x).
    \]

    The two forms represent the same polynomial; only their computational
    organisation differs.

    With 4 interpolation nodes, this can be written as:
    $$
    \begin{aligned}
    p_3(x)
    ={}&f(x_0)
    +\frac{f(x_1)-f(x_0)}{x_1-x_0}(x-x_0)
    \\[0.5em]
    &+
    \left[
    \frac{f(x_2)-f(x_1)}
         {(x_2-x_1)(x_2-x_0)}
    -
    \frac{f(x_1)-f(x_0)}
         {(x_1-x_0)(x_2-x_0)}
    \right]
    (x-x_0)(x-x_1)
    \\[0.5em]
    &+
    \Bigg[
    \frac{f(x_3)}
         {(x_3-x_2)(x_3-x_1)(x_3-x_0)}
    -
    \frac{f(x_2)}
         {(x_3-x_2)(x_2-x_1)(x_2-x_0)}
    \\
    &\qquad\qquad
    +
    \frac{f(x_1)}
         {(x_3-x_1)(x_2-x_1)(x_1-x_0)}
    -
    \frac{f(x_0)}
         {(x_3-x_0)(x_2-x_0)(x_1-x_0)}
    \Bigg]
    \\
    &\qquad\qquad\qquad\qquad\cdot
    (x-x_0)(x-x_1)(x-x_2).
    \end{aligned}
    $$

    """)
    coefficients = divided_differences(x_data, y_data)
    lagrange_values = lagrange_polynomial(x_reference, x_data, y_data)
    newton_values = newton_polynomial(x_reference, x_data, coefficients)
    agreement = np.max(abs(lagrange_values - newton_values))
    _fig, _ax = plt.subplots()
    _ax.plot(x_reference, y_reference, "k--", label="True function")
    _ax.plot(x_reference, newton_values, lw=2, label="Newton polynomial")
    _ax.scatter(x_data, y_data, color="crimson", zorder=3, label="Data")
    _ax.set(xlabel="$x$", ylabel="$f(x)$", title="Equivalent polynomial representations")
    _ax.legend()

    mo.hstack([_net_md, _fig], align="center", widths=[0.6, 0.4])
    return


@app.cell
def _(mo):
    runge_degree = mo.ui.slider(3, 30, value=5, step=1, label="Polynomial degree")
    runge_nodes = mo.ui.radio(options=["Equally spaced", "Chebyshev"], value="Equally spaced", label="Node distribution")
    return runge_degree, runge_nodes


@app.cell
def _(lagrange_polynomial, mo, np, plt, runge_degree, runge_nodes):
    _cheby_md = mo.md(r"""
    ## 5. Runge's phenomenon and node placement

    High degree does not automatically mean better interpolation. For
    $r(x)=\frac{1}{1+25x^2},$
    equally spaced nodes can produce increasingly severe endpoint oscillations.
    The interpolation remainder contains $\prod_i(x-x_i)$, so changing node
    geometry can radically change the error. Chebyshev nodes cluster near the
    interval ends where the problem is most sensitive.
    """)

    def chebyshev_nodes(n, a=-1, b=1):
        return 0.5 * (a + b) + 0.5 * (b - a) * np.sort(np.cos((2 * np.arange(n + 1) + 1) * np.pi / (2 * (n + 1))))

    degree = runge_degree.value
    if runge_nodes.value == "Chebyshev":
        _nodes = chebyshev_nodes(degree, -1, 1)
    else:
        _nodes = np.linspace(-1, 1, degree + 1)
    runge = lambda x: 1 / (1 + 25 * x**2)
    _grid = np.linspace(-1, 1, 2000)
    approximation = lagrange_polynomial(_grid, _nodes, runge(_nodes))
    _error = abs(approximation - runge(_grid))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    _axes[0].plot(_grid, runge(_grid), "k", lw=2, label="Runge function")
    _axes[0].plot(_grid, approximation, lw=1.8, label="Interpolating polynomial")
    _axes[0].scatter(_nodes, runge(_nodes), color="crimson", s=24, zorder=3, label="Nodes")
    _axes[0].set(xlabel="$x$", ylabel="$r(x)$")
    _axes[0].legend(fontsize=9)
    _axes[1].semilogy(_grid, _error, color="crimson")
    _axes[1].set(xlabel="$x$", ylabel="Absolute error")
    # set axis 2 y label position to the right
    _axes[1].yaxis.set_label_position("right")
    _axes[1].yaxis.tick_right()
    mo.vstack([_cheby_md,mo.hstack([runge_degree, runge_nodes,mo.stat(label="Maximum error", value=f"{_error.max():.2e}")]), _fig])
    return (chebyshev_nodes,)


@app.cell
def _(mo):
    product_degree = mo.ui.slider(
        3,
        30,
        value=12,
        step=1,
        label="Polynomial degree",
    )
    return (product_degree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Why do Chebyshev nodes help mitigate Runge's phenomenon?
    """)
    return


@app.cell
def _(mo):
    error_md1 = mo.md(r"""
    The interpolation remainder is

    \[
    f(x)-P_n(x)
    =
    \frac{f^{(n+1)}(\xi_x)}{(n+1)!}
    \underbrace{\prod_{k=0}^{n}(x-x_k)}_{\omega_{n+1}(x)}.
    \]

    For a fixed function, the node placement controls the product term
    $\omega_{n+1}(x)$.  
    Equally spaced nodes produce large values near the
    endpoints, while Chebyshev nodes distribute the extrema more evenly.
    """)

    error_md2 = mo.md(r"""
    *Chebyshev nodes*:
    \[
    x_k = \cos\left(\frac{2k+1}{2n+2} \pi \right), \qquad k = 0,\ldots,n.
    \]
    General formula on an interval [a, b]:
    \[
    x_k = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{2k+1}{2n+2} \pi \right), \qquad k = 0,\ldots,n.
    \]
    Thus, Chebyshev nodes reduce the part of the interpolation error caused
    by the geometry of the nodes.
    """)
    return error_md1, error_md2


@app.cell
def _(chebyshev_nodes, error_md1, error_md2, mo, np, plt, product_degree):
    _product_degree = product_degree.value
    _product_grid = np.linspace(-1, 1, 2000)

    _product_node_sets = {
        "Equally spaced": np.linspace(-1, 1, _product_degree + 1),
        "Chebyshev": chebyshev_nodes(_product_degree, -1, 1),
    }

    _product_values = {
        _label: np.prod(
            _product_grid[:, None] - _node_set[None, :],
            axis=1,
        )
        for _label, _node_set in _product_node_sets.items()
    }

    _product_fig, _product_axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        layout="constrained",
    )

    # Product term
    for _label, _values in _product_values.items():
        _product_axes[0].plot(
            _product_grid,
            np.abs(_values),
            lw=2,
            label=_label,
        )

    _product_axes[0].set(
        xlabel="$x$",
        ylabel=r"$|\omega_{n+1}(x)|$",
        title=fr"Product term for degree $n={_product_degree}$",
    )
    # _product_axes[0].legend()
    _product_axes[0].grid(alpha=0.25)

    # Node locations
    _product_axes[1].scatter(
        _product_node_sets["Equally spaced"],
        np.zeros(_product_degree + 1),
        s=35,
        # label="Equally spaced",
    )
    _product_axes[1].scatter(
        _product_node_sets["Chebyshev"],
        np.ones(_product_degree + 1),
        s=35,
        # label="Chebyshev",
    )

    _product_axes[1].set(
        xlabel="$x$",
        yticks=[0, 1],
        yticklabels=["Equally spaced", "Chebyshev"],
        title="Node locations",
        xlim=(-1.05, 1.05),
        ylim=(-0.35, 1.35),
    )
    # put some vertical grey line at the location of the line nodes
    for _node in _product_node_sets["Equally spaced"]:
        _product_axes[1].axvline(_node, color="grey", lw=0.5, alpha=0.5)

    _product_axes[1].yaxis.set_label_position("right")
    _product_axes[1].yaxis.set_ticks_position("right")


    mo.vstack(
        [
            mo.md("""### Why do Chebyshev nodes reduce Runge's phenomenon?"""),
            mo.hstack([error_md1, error_md2],align="start", justify="center",widths= [0.5, 0.5]),
            mo.hstack(
                [
                    product_degree,
                    mo.stat(
                        label="Equally spaced maximum",
                        value=(
                            f"{np.max(np.abs(_product_values['Equally spaced'])):.3e}"
                        ),
                    ),
                    mo.stat(
                        label="Chebyshev maximum",
                        value=(
                            f"{np.max(np.abs(_product_values['Chebyshev'])):.3e}"
                        ),
                    ),
                ]
            ),
            _product_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Polynomial interpolation pros and cons

    Polynomial interpolation has several advantages and disadvantages:

    **Pros:**
    - Generally more accurate than piecewise linear interpolation for smooth functions.
    - Derivatives of the interpolating polynomial are also continuous, which can be advantageous for certain applications.
    - Can be used for numerical differentiation and integration due to the smoothness of the polynomial.

    **Cons:**
    - Implementation can become complex.
    - Susceptible to Runge's phenomenon (Oscillations).
    - Poor numerical stability for high-degree polynomials.
    - Difficult in multivariate settings.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Cubic Spline Interpolation

    Global high-degree polynomial interpolation can suffer from oscillations, especially when many nodes are used. A cubic spline avoids this problem by replacing a single polynomial with a collection of **piecewise cubic polynomials**, one on each interval between neighboring data points.

    Suppose we have data points

    \[
    (x_0,y_0),\,(x_1,y_1),\,\ldots,\,(x_{n-1},y_{n-1}).
    \]

    On each interval \([x_{k-1},x_k]\), we define a cubic polynomial

    \[
    s_{k-1,k}(x).
    \]

    The spline is constructed so that:

    1. **Interpolation:** each cubic passes through its interval endpoints,

    \[
    s_{k-1,k}(x_{k-1}) = y_{k-1},
    \qquad
    s_{k-1,k}(x_k)=y_k.
    \]

    2. **Smooth first derivative:** neighboring cubics have the same slope at every interior node,

    \[
    s'_{k-1,k}(x_k)=s'_{k,k+1}(x_k).
    \]

    3. **Smooth second derivative:** neighboring cubics also have matching curvature,

    \[
    s''_{k-1,k}(x_k)=s''_{k,k+1}(x_k).
    \]

    For a **natural cubic spline**, the curvature is additionally set to zero at the two endpoints:

    \[
    s''(x_0)=0,
    \qquad
    s''(x_{n-1})=0.
    \]

    The result is an interpolant that is continuous together with its first and second derivatives, producing a smooth curve while avoiding the large oscillations that may occur with a single high-order interpolation polynomial.
    The math behind the construction of the cubic spline involves solving a tridiagonal linear system for the second derivatives at the interior nodes, which will be explored later in the course. (or you can read more about it section 6.3 of [Numerical Methods in Physics with Python](https://numphyspy.org/))
    """)
    return


@app.cell
def _(CubicSpline, mo, np, plt, sample_count, x_data, y_data):
    spline = CubicSpline(x_data, y_data)#, bc_type="clamped")
    _grid = np.linspace(0, 2 * np.pi, 1200)
    value_error = abs(spline(_grid) - np.sin(_grid))
    derivative_error = abs(spline.derivative()(_grid) - np.cos(_grid))
    _fig, _axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    _axes[0].plot(_grid, np.sin(_grid), "k--", label="True function")
    _axes[0].plot(_grid, spline(_grid), lw=2, label="Cubic spline")
    _axes[0].scatter(x_data, y_data, color="crimson", zorder=3, label="Knots")
    _axes[0].set(ylabel="$f(x)$", )
    _axes[0].legend()
    _axes[1].plot(_grid, np.cos(_grid), "k--", label="Exact derivative")
    _axes[1].plot(_grid, spline.derivative()(_grid), lw=2, label="Spline derivative")
    _axes[1].set(xlabel="$x$", ylabel="$f'(x)$")
    _axes[1].legend()
    mo.vstack([mo.md(r"""##Cubic Spline"""), mo.hstack([sample_count, mo.stat(label="Max value error", value=f"{value_error.max():.2e}"), mo.stat(label="Max derivative error", value=f"{derivative_error.max():.2e}")]), _fig])
    return


@app.cell
def _(mo):
    method_selector = mo.ui.multiselect(
        options=["Linear", "Polynomial", "Cubic spline", "Polynomial (Chebyshev)"],
        value=["Linear", "Polynomial", "Cubic spline", "Polynomial (Chebyshev)"],
        label="Methods to display",
    )
    return (method_selector,)


@app.cell
def _(
    BarycentricInterpolator,
    CubicSpline,
    chebyshev_nodes,
    go,
    linear_interpolant,
    make_subplots,
    method_selector,
    mo,
    np,
    x_data,
    y_data,
):


    _cmp_grid = np.linspace(x_data.min(), x_data.max(), 1500)
    _true_f = np.sin(_cmp_grid)
    _true_df = np.cos(_cmp_grid)

    # --- Linear: np.interp values + piecewise-constant derivative ---
    _lin_f = linear_interpolant(_cmp_grid, x_data, y_data)
    _slopes = np.diff(y_data) / np.diff(x_data)
    _interval = np.clip(np.searchsorted(x_data, _cmp_grid, side="right") - 1, 0, len(_slopes) - 1)
    _lin_df = _slopes[_interval]

    # --- Polynomial: same unique interpolant as the Lagrange/Newton forms ---
    _poly = BarycentricInterpolator(x_data, y_data)
    _poly_f = _poly(_cmp_grid)
    _poly_df = _poly.derivative(_cmp_grid)

    # --- Polynomial on Chebyshev nodes (same node count, remapped to the data range) ---
    _cheb_nodes = chebyshev_nodes(len(x_data) - 1, x_data.min(), x_data.max())
    _poly_cheb = BarycentricInterpolator(_cheb_nodes, np.sin(_cheb_nodes))
    _poly_cheb_f = _poly_cheb(_cmp_grid)
    _poly_cheb_df = _poly_cheb.derivative(_cmp_grid)

    # --- Natural cubic spline ---
    _spline = CubicSpline(x_data, y_data)
    _spline_f = _spline(_cmp_grid)
    _spline_df = _spline(_cmp_grid, 1)

    _curves = {
        "Linear": (_lin_f, _lin_df),
        "Polynomial": (_poly_f, _poly_df),
        "Cubic spline": (_spline_f, _spline_df),
        "Polynomial (Chebyshev)": (_poly_cheb_f, _poly_cheb_df),
    }
    _colors = {
        "Linear": "#1f77b4",
        "Polynomial": "#ff7f0e",
        "Cubic spline": "#2ca02c",
        "Polynomial (Chebyshev)": "#d62728",
    }

    _eps = 1e-16  # floor for the log-scale error plots (error is exactly 0 at the nodes)

    _fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.1,
    )

    # Data knots drawn up front
    _fig.add_trace(
        go.Scatter(x=x_data, y=y_data, mode="markers", name="Data",
                   marker=dict(color="crimson", size=7), legendgroup="data"),
        row=1, col=1,
    )

    for _name in method_selector.value:
        _f, _df = _curves[_name]
        _c = _colors[_name]
        _fig.add_trace(
            go.Scatter(x=_cmp_grid, y=_f, mode="lines", name=_name,
                       line=dict(color=_c, width=2), legendgroup=_name),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_cmp_grid, y=_df, mode="lines", name=_name,
                       line=dict(color=_c, width=2), legendgroup=_name, showlegend=False),
            row=2, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_cmp_grid, y=np.abs(_f - _true_f) + _eps, mode="lines", name=_name,
                       line=dict(color=_c, width=2), legendgroup=_name, showlegend=False),
            row=1, col=2,
        )
        _fig.add_trace(
            go.Scatter(x=_cmp_grid, y=np.abs(_df - _true_df) + _eps, mode="lines", name=_name,
                       line=dict(color=_c, width=2), legendgroup=_name, showlegend=False),
            row=2, col=2,
        )

    # True function and derivative added last so they are drawn on top
    _fig.add_trace(
        go.Scatter(x=_cmp_grid, y=_true_f, mode="lines", name="True function",
                   line=dict(color="black", dash="dash", width=2.5), legendgroup="true"),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(x=_cmp_grid, y=_true_df, mode="lines", name="True derivative",
                   line=dict(color="black", dash="dash", width=2.5), legendgroup="true",
                   showlegend=False),
        row=2, col=1,
    )

    # Axis labels; log scale on the error column
    _fig.update_yaxes(title_text="$f(x)$", row=1, col=1)
    _fig.update_yaxes(title_text="$f'(x)$", row=2, col=1)
    _fig.update_yaxes(title_text="$|f(x) - I(x)|$", type="log", row=1, col=2)
    _fig.update_yaxes(title_text="$|f'(x) - I'(x)|$", type="log", row=2, col=2)
    _fig.update_xaxes(title_text="$x$", row=2, col=1)
    _fig.update_xaxes(title_text="$x$", row=2, col=2)

    # Move the right-hand column's labels/ticks to the right side
    _fig.update_yaxes(side="right", col=2)

    _fig.update_layout(
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
        template="plotly_white",
    )

    mo.vstack([method_selector, mo.ui.plotly(_fig)])
    return


@app.cell
def _(
    BarycentricInterpolator,
    CubicSpline,
    chebyshev_nodes,
    method_selector,
    mo,
    np,
    plt,
):
    _study_counts = np.geomspace(4, 100, 20)
    _study_grid = np.linspace(0, 2 * np.pi, 4001)
    _study_true_f = np.sin(_study_grid)
    _study_true_df = np.cos(_study_grid)

    _conv_colors = {
        "Linear": "tab:blue",
        "Polynomial": "tab:orange",
        "Cubic spline": "tab:green",
        "Polynomial (Chebyshev)": "tab:red",
    }
    _conv_errors = {}

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    for _name in method_selector.value:
        _err_f, _err_df = [], []
        for _count in _study_counts:
            _nodes = np.linspace(0, 2 * np.pi, int(_count))
            _vals = np.sin(_nodes)

            if _name == "Linear":
                _est_f = np.interp(_study_grid, _nodes, _vals)
                _slopes = np.diff(_vals) / np.diff(_nodes)
                _ivl = np.clip(np.searchsorted(_nodes, _study_grid, side="right") - 1, 0, len(_slopes) - 1)
                _est_df = _slopes[_ivl]
            elif _name in ("Polynomial", "Polynomial (Chebyshev)"):
                if _name == "Polynomial (Chebyshev)":
                    _nodes = chebyshev_nodes(_count - 1, 0, 2 * np.pi)
                    _vals = np.sin(_nodes)
                _poly = BarycentricInterpolator(_nodes, _vals)
                _est_f = _poly(_study_grid)
                _est_df = _poly.derivative(_study_grid)
            else:  # Cubic spline
                _spl = CubicSpline(_nodes, _vals)
                _est_f = _spl(_study_grid)
                _est_df = _spl(_study_grid, 1)

            _err_f.append(np.max(np.abs(_est_f - _study_true_f)))
            _err_df.append(np.max(np.abs(_est_df - _study_true_df)))

        _conv_errors[_name] = (np.array(_err_f), np.array(_err_df))
        _c = _conv_colors[_name]
        _axes[0].loglog(_study_counts, _err_f, "o-", color=_c, ms=5, lw=1.8, label=_name)
        _axes[1].loglog(_study_counts, _err_df, "o-", color=_c, ms=5, lw=1.8)

    # Theoretical reference slopes anchored to the coarsest grid of the relevant method
    # Theoretical reference slopes anchored to the finest grid of the relevant method
    if "Linear" in _conv_errors:
        _ref = _conv_errors["Linear"]
        _axes[0].loglog(_study_counts, _ref[0][-1] * (_study_counts / _study_counts[-1]) ** -2,
                        "k--", lw=1.2, label=r"$\mathcal{O}(h^2)$")

    if "Cubic spline" in _conv_errors:
        _spl_ref = _conv_errors["Cubic spline"]
        _axes[1].loglog(_study_counts, _spl_ref[1][-1] * (_study_counts / _study_counts[-1]) ** -3,
                        color="0.4", ls="-.", lw=1.2, label=r"$\mathcal{O}(h^3)$")
        _axes[0].loglog(_study_counts, _spl_ref[0][-1] * (_study_counts / _study_counts[-1]) ** -4,
                        color="0.4", ls=":", lw=1.2, label=r"$\mathcal{O}(h^4)$")

    # Combined legend placed outside, to the right of the second panel
    _handles, _labels = [], []
    for _ax in _axes:
        _h, _l = _ax.get_legend_handles_labels()
        _handles += _h
        _labels += _l
    _fig.legend(_handles, _labels, loc="center", bbox_to_anchor=(1.25, 0.5),frameon=False,ncols=1) #bbox_to_anchor=(0.5, 1.15),

    mo.vstack([method_selector,_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Accuracy of Common Interpolation Methods

    Let

    \[
    h=\max_i (x_{i+1}-x_i)
    \]

    denote the largest spacing between neighboring nodes.

    | Method | Spacing | Local approximation | Leading error term | Global error |
    |----------|----------|----------|----------|----------|
    | Nearest neighbour | Any | Constant | \(\frac{f'(\xi)}{1!}h\) | \(\mathcal{O}(h)\) |
    | Linear interpolation | Any | Degree 1 | \(\frac{f''(\xi)}{2!}(x-x_i)(x-x_{i+1})\) | \(\mathcal{O}(h^2)\) |
    | Quadratic interpolation | Any | Degree 2 | \(\frac{f^{(3)}(\xi)}{3!}(x-x_0)(x-x_1)(x-x_2)\) | \(\mathcal{O}(h^3)\) |
    | Cubic interpolation | Any | Degree 3 | \(\frac{f^{(4)}(\xi)}{4!}\prod_{k=0}^{3}(x-x_k)\) | \(\mathcal{O}(h^4)\) |
    | Degree \(n\) polynomial | Equally spaced | Degree \(n\) (global) | \(\frac{f^{(n+1)}(\xi)}{(n+1)!}\omega_{n+1}(x)\) | \(\mathcal{O}(h^{n+1})\)* |
    | Degree \(n\) polynomial | Chebyshev | Degree \(n\) (global) | \(\frac{f^{(n+1)}(\xi)}{(n+1)!} \omega_{n+1}(x)\) with \(\max \lvert\omega_{n+1}\rvert = \frac{(b-a)^{n+1}}{2^{2n+1}}\) | \(\mathcal{O}(\rho^{-n})\) |
    | Cubic spline | Any | Piecewise cubic | Depends on \(f^{(4)}\) | \(\mathcal{O}(h^4)\) |

    \* but liable to Runge oscillations
    """)
    return


@app.cell
def _(mo):
    grid_points = mo.ui.slider(4, 20, value=7, step=1, label="Data points in each direction")
    return (grid_points,)


@app.cell
def _(RegularGridInterpolator, grid_points, mo, np, plt):
    _md_2D = mo.md(r"""
    ## 7. Two-dimensional interpolation

    Now sample $g(x,y)=\sin(x+y)$ on a rectangular grid. Bilinear
    interpolation applies linear interpolation in one coordinate and then in
    the other. The left panel shows the reconstructed field and its black dots
    show the available data; the right panel makes the approximation error
    visible.
    """)
    _nodes = np.linspace(0, 2 * np.pi, grid_points.value)
    node_x, node_y = np.meshgrid(_nodes, _nodes, indexing="ij")
    data = np.sin(node_x + node_y)
    interpolator = RegularGridInterpolator((_nodes, _nodes), data, method="linear")
    _grid = np.linspace(0, 2 * np.pi, 180)
    xx, yy = np.meshgrid(_grid, _grid, indexing="ij")
    estimate = interpolator(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    _error = abs(estimate - np.sin(xx + yy))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    _image = _axes[0].imshow(estimate.T, origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi), cmap="coolwarm", vmin=-1, vmax=1)
    _axes[0].scatter(node_x, node_y, color="black", s=10)
    _axes[0].set(xlabel="$x$", ylabel="$y$", title="Bilinear interpolation")
    _fig.colorbar(_image, ax=_axes[0])
    _error_image = _axes[1].imshow(_error.T, origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi), cmap="magma")
    _axes[1].set(xlabel="$x$", ylabel="$y$", title="Absolute error")
    _fig.colorbar(_error_image, ax=_axes[1])
    mo.vstack([_md_2D, mo.hstack([grid_points, mo.stat(label="Maximum 2D error", value=f"{_error.max():.2e}")]), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpolation versus fitting

    Interpolation passes exactly through all data points.

    \[
    I(x_i)=f_i.
    \]
    When data contains noise this is often undesirable, since the interpolant reproduces the noise as well.

    Curve fitting instead seeks a function that approximately matches the data while reducing the influence of noise.

    Interpolation is therefore most appropriate when the data are known accurately, for example when they come from a deterministic simulation.
    """)
    return


if __name__ == "__main__":
    app.run()
