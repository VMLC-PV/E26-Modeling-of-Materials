import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit message on Python 3.14

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import solve_triangular
    from wigglystuff import PlaySlider

    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.exceptions import InputDataWarning
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
    from gpytorch.means import ZeroMean
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.constraints import GreaterThan
    from gpytorch.mlls import ExactMarginalLogLikelihood

    warnings.filterwarnings("ignore", category=InputDataWarning)

    return (
        ExactMarginalLogLikelihood,
        GaussianLikelihood,
        GreaterThan,
        MaternKernel,
        PlaySlider,
        RBFKernel,
        ScaleKernel,
        SingleTaskGP,
        Standardize,
        ZeroMean,
        fit_gpytorch_mll,
        mo,
        np,
        plt,
        solve_triangular,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian processes

    In the lecture on Bayesian inference, the unknowns were a few parameters of a known model, the Arrhenius law. Often we do not have such a model: we measure a property $y$ as a function of some control variable $x$ (a composition, a temperature, a processing time...) and want to know the whole function $f(x)$, including at values of $x$ we have not measured, and *how uncertain* our knowledge is there.

    A **Gaussian process** (GP) is Bayesian inference in which the unknown is the function itself:

    | Bayesian inference (lecture 11) | Gaussian process |
    |---|---|
    | parameters $\theta$ | function values $f(x)$ at all $x$ |
    | prior $p(\theta)$ | prior over functions, set by a **kernel** |
    | likelihood: model + Gaussian noise | observations $y_i = f(x_i) + $ Gaussian noise |
    | posterior $p(\theta\,\vert\,d)$ | posterior over functions: mean and error band |
    | evidence $p(d)$ (ignored) | marginal likelihood, used to choose the **hyperparameters** |

    Everything is Gaussian, so all the integrals of lecture 11 can be done exactly with linear algebra: no grid and no MCMC are needed. The GP is the backbone of **Bayesian optimization**, the topic of the next lecture: finding the minimum of an expensive function with as few measurements as possible.

    | Section | Topic | Key idea |
    |---|---|---|
    | 1 | From parameters to functions | conditioning a multivariate Gaussian |
    | 2 | Kernels: priors over functions | the covariance between $f(x)$ and $f(x')$ |
    | 3 | GP regression, step by step | kernel matrices, Cholesky, posterior mean and variance |
    | 4 | Hyperparameters | maximizing the marginal likelihood |
    | 5 | Choosing the kernel | Matérn kernels; when a GP struggles |

    We write the GP ourselves in `numpy`, formula by formula, and then use [BoTorch](https://botorch.org) (built on [GPyTorch](https://gpytorch.ai)), a modern library which we also use for Bayesian optimization.
    """)
    return


@app.cell(hide_code=True)
def _():
    colors = {
        "Prior": "0.45",
        "Posterior": "#1d4ed8",
        "BoTorch": "seagreen",
        "Default": "darkorange",
        "True": "crimson",
    }
    return (colors,)


@app.cell
def _(np):
    def f_custom(x):
        return 0.8 * np.sin(12 * x) + 0.6 * np.cos(4 * x)

    def f_forrester(x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    test_functions = {
        "f(x) = 0.8 sin(12x) + 0.6 cos(4x)": f_custom,
        "Forrester: (6x - 2)² sin(12x - 4)": f_forrester,
    }

    # fixed sequences of measurement positions and noise draws (seeded: everyone sees the same data)
    x_sequences = {
        "Random in [0, 1]": np.random.default_rng(3).uniform(0, 1, 20),
        "Clustered in [0, 0.5]": np.random.default_rng(4).uniform(0, 0.5, 20),
    }
    z_sequence = np.random.default_rng(5).normal(size=20)
    x_test = np.linspace(0, 1, 300)
    return test_functions, x_sequences, x_test, z_sequence


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. From parameters to functions
    ---

    Consider two Gaussian random variables $f_1$ and $f_2$, each with mean 0 and variance 1, and correlation $\rho$. Their joint distribution is the bivariate Gaussian
    $$
    \begin{pmatrix} f_1 \\ f_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0\end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1\end{pmatrix}\right).
    $$
    Suppose we **observe** $f_1$. What do we now know about $f_2$? Bayes' theorem gives the conditional distribution, which is again Gaussian:
    $$
    f_2 \,|\, f_1 \sim \mathcal{N}\left(\rho\, f_1,\; 1-\rho^2\right).
    $$
    The stronger the correlation, the more an observation of $f_1$ tells us about $f_2$.

    On the right, the same random samples are drawn differently: each sample $(f_1, f_2)$ is a line from position 1 to position 2. With a strong correlation, the samples look like smooth "functions" of the position.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rho_slider = mo.ui.slider(-0.95, 0.95, 0.05, value=0.8, label=r"Correlation $\rho$", show_value=True)
    f1_slider = mo.ui.slider(-2.5, 2.5, 0.1, value=1.0, label=r"Observed $f_1$", show_value=True)
    return f1_slider, rho_slider


@app.cell(hide_code=True)
def _(colors, f1_slider, mo, np, plt, rho_slider):
    _rho, _f1 = rho_slider.value, f1_slider.value
    _g = np.linspace(-3, 3, 200)
    _A, _B = np.meshgrid(_g, _g)
    _joint = np.exp(-(_A**2 - 2 * _rho * _A * _B + _B**2) / (2 * (1 - _rho**2)))
    _samples = np.random.default_rng(0).multivariate_normal([0, 0], [[1, _rho], [_rho, 1]], size=25)
    _cond_mean, _cond_std = _rho * _f1, np.sqrt(1 - _rho**2)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))
    _ax1.contour(_A, _B, _joint, levels=5, colors=colors["Prior"])
    _ax1.plot(_samples[:, 0], _samples[:, 1], ".", color=colors["Prior"])
    _ax1.axvline(_f1, color=colors["Posterior"], lw=2)
    _ax1.set(xlabel="$f_1$", ylabel="$f_2$", aspect="equal", title="Joint distribution")

    _ax2.plot(_g, np.exp(-(_g**2) / 2) / np.sqrt(2 * np.pi), color=colors["Prior"], ls="--", label="$p(f_2)$, before")
    _ax2.plot(_g, np.exp(-((_g - _cond_mean) ** 2) / (2 * _cond_std**2)) / (_cond_std * np.sqrt(2 * np.pi)), color=colors["Posterior"], lw=2, label=r"$p(f_2\,|\,f_1)$, after")
    _ax2.set(xlabel="$f_2$", title="Observing $f_1$ updates $f_2$")
    _ax2.legend(fontsize=8)

    for _s in _samples:
        _ax3.plot([1, 2], _s, "-o", color=colors["Prior"], lw=0.8, ms=3)
    _cond = _cond_mean + _cond_std * np.random.default_rng(1).normal(size=10)
    for _c in _cond:
        _ax3.plot([1, 2], [_f1, _c], "-o", color=colors["Posterior"], lw=1, ms=3)
    _ax3.set(xticks=[1, 2], xlabel="Position", ylabel="Value", ylim=(-3, 3), title="Samples as 'functions' (blue: given $f_1$)")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.hstack([rho_slider, f1_slider], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Many variables

    The same works for any number of jointly Gaussian variables. Split them into observed ones $\mathbf{f}_a$ and unknown ones $\mathbf{f}_b$, with covariance blocks $\Sigma_{aa}, \Sigma_{ab}, \Sigma_{bb}$ (zero means). Then $\mathbf{f}_b\,|\,\mathbf{f}_a$ is Gaussian with
    $$
    \boldsymbol{\mu}_{b|a} = \Sigma_{ba}\Sigma_{aa}^{-1}\mathbf{f}_a, \qquad \Sigma_{b|a} = \Sigma_{bb} - \Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}.
    $$
    For $\Sigma_{aa} = 1$, $\Sigma_{ab} = \rho$ this is exactly the 2D result above.

    Now the key step: take the values of a function $f$ at many positions $x_1, x_2, \ldots, x_m$ and treat them as jointly Gaussian random variables. Observing $f$ at some positions updates our knowledge of $f$ at all the others, through the formula above. All we need is a rule giving the covariance between $f(x)$ and $f(x')$ for any two positions. That rule is the **kernel**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Kernels: priors over functions
    ---

    A **Gaussian process** is a collection of random variables $f(x)$, one for every $x$, any finite number of which are jointly Gaussian. It is defined by a mean function (here 0) and a **kernel** (covariance function)
    $$
    k(x, x') = \mathrm{cov}\left(f(x), f(x')\right).
    $$
    The most common choice is the **squared-exponential** or RBF (radial basis function) kernel
    $$
    k(x, x') = \sigma_f^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right),
    $$
    with two **hyperparameters**: the amplitude $\sigma_f$ (typical size of $f$) and the **length scale** $\ell$ (the distance over which $f(x)$ and $f(x')$ remain correlated). Nearby points are strongly correlated, distant points are independent: this is the assumption that $f$ is smooth.

    To draw random functions from the prior, we evaluate $f$ on a grid of $m$ points. The values form a vector $\mathbf{f}\sim\mathcal{N}(0, K)$ with $K_{ij} = k(x_i, x_j)$. As for the Box–Muller method (lecture 10), we transform standard Gaussian numbers, using the **Cholesky decomposition**.

    *Cholesky decomposition.* A symmetric positive-definite matrix (all eigenvalues $> 0$), such as a covariance matrix, can be factored as $K = LL^\top$ with $L$ lower triangular. It is the LU decomposition of lecture 6 specialised to this case: since $K$ is symmetric, the two triangular factors can be chosen as transposes of each other, $U = L^\top$, so only one factor is computed, about twice as fast, and no pivoting is needed. `np.linalg.cholesky(K)` returns $L$ and fails if $K$ is not positive definite.

    With $\mathbf{z}$ a vector of independent standard Gaussians, $\mathbf{f} = L\mathbf{z}$ has covariance $L\,\langle \mathbf{z}\mathbf{z}^\top\rangle L^\top = K$.
    """)
    return


@app.cell
def _(np):
    def rbf_kernel(xa, xb, ell, sigma_f):
        """Squared-exponential kernel matrix K_ij = k(xa_i, xb_j)."""
        return sigma_f**2 * np.exp(-0.5 * (xa[:, None] - xb[None, :]) ** 2 / ell**2)

    jitter = 1e-8  # small number added to the diagonal so that the Cholesky decomposition works numerically
    return jitter, rbf_kernel


@app.cell(hide_code=True)
def _(mo):
    ell2_slider = mo.ui.slider(steps=[0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0], value=0.1, label=r"Length scale $\ell$", show_value=True)
    sf2_slider = mo.ui.slider(0.2, 2.0, 0.1, value=1.0, label=r"Amplitude $\sigma_f$", show_value=True)
    return ell2_slider, sf2_slider


@app.cell
def _(ell2_slider, jitter, np, rbf_kernel, sf2_slider):
    x_grid = np.linspace(0, 1, 150)
    K_prior = rbf_kernel(x_grid, x_grid, ell2_slider.value, sf2_slider.value)
    L_prior = np.linalg.cholesky(K_prior + jitter * np.eye(x_grid.size))
    prior_samples = L_prior @ np.random.default_rng(2).normal(size=(x_grid.size, 5))  # 5 random functions
    return K_prior, prior_samples, x_grid


@app.cell(hide_code=True)
def _(
    K_prior,
    colors,
    ell2_slider,
    mo,
    np,
    plt,
    prior_samples,
    sf2_slider,
    x_grid,
):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    _im = _ax1.imshow(K_prior, extent=(0, 1, 1, 0), cmap="viridis")
    _fig.colorbar(_im, ax=_ax1, shrink=0.8)
    _ax1.set(xlabel="$x'$", ylabel="$x$", title=r"Kernel matrix $k(x, x')$")
    _sf = sf2_slider.value
    _ax2.fill_between(x_grid, -2 * _sf, 2 * _sf, color=colors["Prior"], alpha=0.15, label=r"$\pm 2\sigma_f$")
    _ax2.plot(x_grid, prior_samples, lw=1.5)
    _ax2.set(xlabel="$x$", ylabel="$f(x)$", ylim=(-4.5, 4.5), title=rf"5 samples from the prior, $\ell = {ell2_slider.value}$")
    _ax2.legend(loc="upper right")
    _fig.tight_layout()
    plt.close(_fig)
    _eig_min = np.linalg.eigvalsh(K_prior).min()
    mo.vstack(
        [
            mo.hstack([ell2_slider, sf2_slider], justify="start"),
            _fig,
            mo.md(rf"""
    The length scale sets how fast the functions wiggle; the amplitude sets their vertical size. Before any measurement, the GP says: "$f$ is some smooth function of size $\sim\sigma_f$, varying on a scale $\sim\ell$."

    **Why the jitter?** In exact arithmetic $K$ is positive definite, but rows of nearby points are almost identical. The smallest eigenvalue of this $K$, computed in floating point, is ${_eig_min:.1e}$: zero to machine precision, possibly even negative, and the Cholesky decomposition would fail. Adding $10^{{-8}}$ to the diagonal fixes it, at a negligible cost in accuracy.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. GP regression, step by step
    ---

    We measure $n$ noisy values $y_i = f(x_i) + \epsilon_i$, with Gaussian noise $\epsilon_i$ of standard deviation $\sigma_n$, and want $f$ at $m$ test points $x_{*,j}$ (a fine grid). The training values $\mathbf{y}$ and the test values $\mathbf{f}_*$ are jointly Gaussian:
    $$
    \begin{pmatrix}\mathbf{y}\\ \mathbf{f}_*\end{pmatrix} \sim \mathcal{N}\left(0, \begin{pmatrix} K + \sigma_n^2 I & K_* \\ K_*^\top & K_{**}\end{pmatrix}\right),
    $$
    with $K_{ij} = k(x_i, x_j)$ ($n\times n$), $(K_*)_{ij} = k(x_i, x_{*,j})$ ($n\times m$), $(K_{**})_{ij} = k(x_{*,i}, x_{*,j})$ ($m\times m$). The noise only adds to the diagonal of the training block, since it is independent for each measurement. Conditioning (Section 1) gives the **posterior**:
    $$
    \boldsymbol{\mu}_* = K_*^\top (K + \sigma_n^2 I)^{-1}\mathbf{y}, \qquad \Sigma_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_* .
    $$
    Instead of computing the inverse (lecture 6), we solve linear systems, which is cheaper and more accurate: we factor $K + \sigma_n^2 I = LL^\top$ and solve with the triangular factors, by forward and back substitution as in the LU method of lecture 6. The steps are below, with the data and hyperparameters set by the controls. In this section the hyperparameters, including the noise level, are assumed known, as in Sections 2–3 of lecture 11.
    """)
    return


@app.cell(hide_code=True)
def _(PlaySlider, mo, test_functions, x_sequences):
    func_select = mo.ui.dropdown(options=list(test_functions), value=list(test_functions)[0], label="Test function")
    positions_select = mo.ui.radio(options=list(x_sequences), value=list(x_sequences)[0], label="Measurement positions", inline=True)
    noise3_slider = mo.ui.slider(0.0, 0.5, 0.01, value=0.1, label=r"Noise $\sigma_n$", show_value=True)
    ell3_slider = mo.ui.slider(steps=[0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0], value=0.1, label=r"Length scale $\ell$", show_value=True)
    sf3_slider = mo.ui.slider(steps=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0], value=1.0, label=r"Amplitude $\sigma_f$", show_value=True)
    n3_play = mo.ui.anywidget(PlaySlider(value=6, min_value=0, max_value=20, step=1, interval_ms=500))
    return (
        ell3_slider,
        func_select,
        n3_play,
        noise3_slider,
        positions_select,
        sf3_slider,
    )


@app.cell
def _(
    ell3_slider,
    func_select,
    n3_play,
    noise3_slider,
    positions_select,
    sf3_slider,
    test_functions,
    x_sequences,
    z_sequence,
):
    # data and hyperparameters from the controls
    f_true = test_functions[func_select.value]
    n_obs = int(n3_play.value["value"])
    sigma_n = noise3_slider.value
    ell, sigma_f = ell3_slider.value, sf3_slider.value
    x_train = x_sequences[positions_select.value][:n_obs]
    y_train = f_true(x_train) + sigma_n * z_sequence[:n_obs]
    return ell, f_true, n_obs, sigma_f, sigma_n, x_train, y_train


@app.cell
def _(ell, jitter, n_obs, np, rbf_kernel, sigma_f, sigma_n, x_test, x_train):
    # Step 1: kernel matrices
    K = rbf_kernel(x_train, x_train, ell, sigma_f) + (sigma_n**2 + jitter) * np.eye(n_obs)  # (n, n), noise on the diagonal
    K_s = rbf_kernel(x_train, x_test, ell, sigma_f)  # (n, m)
    K_ss = rbf_kernel(x_test, x_test, ell, sigma_f)  # (m, m)
    return K, K_s, K_ss


@app.cell
def _(K, np, solve_triangular, y_train):
    # Step 2: Cholesky decomposition K + sigma_n^2 I = L L^T
    L = np.linalg.cholesky(K)
    # Step 3: alpha = (K + sigma_n^2 I)^{-1} y, with two triangular solves
    alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True), lower=False)
    return L, alpha


@app.cell
def _(K_s, K_ss, L, alpha, np, solve_triangular):
    # Step 4: posterior mean and covariance at the test points
    mu_post = K_s.T @ alpha  # K_*^T (K + s^2 I)^{-1} y
    V = solve_triangular(L, K_s, lower=True)  # L^{-1} K_*, so that V^T V = K_*^T (K + s^2 I)^{-1} K_*
    cov_post = K_ss - V.T @ V
    std_post = np.sqrt(np.clip(np.diag(cov_post), 0, None))
    return cov_post, mu_post, std_post


@app.cell(hide_code=True)
def _(
    colors,
    cov_post,
    ell3_slider,
    f_true,
    func_select,
    mo,
    mu_post,
    n3_play,
    n_obs,
    noise3_slider,
    np,
    plt,
    positions_select,
    sf3_slider,
    sigma_n,
    std_post,
    x_test,
    x_train,
    y_train,
):
    _samples = np.linalg.cholesky(cov_post + 1e-6 * np.eye(x_test.size)) @ np.random.default_rng(8).normal(size=(x_test.size, 3))
    _fig, _ax = plt.subplots(figsize=(10, 4.6))
    _std_new = np.sqrt(std_post**2 + sigma_n**2)
    _ax.fill_between(x_test, mu_post - 2 * _std_new, mu_post + 2 * _std_new, color=colors["Posterior"], alpha=0.1, label="New measurement, ±2 std")
    _ax.fill_between(x_test, mu_post - 2 * std_post, mu_post + 2 * std_post, color=colors["Posterior"], alpha=0.25, label="$f(x)$, ±2 std")
    _ax.plot(x_test, mu_post[:, None] + _samples, color=colors["Posterior"], lw=0.6, alpha=0.6)
    _ax.plot(x_test, mu_post, color=colors["Posterior"], lw=2, label="Posterior mean")
    _ax.plot(x_test, f_true(x_test), color=colors["True"], ls="--", label="True function")
    _ax.plot(x_train, y_train, "ko", label=f"{n_obs} measurements")
    _ax.set(xlabel="$x$", ylabel="$f(x)$", title="GP posterior (thin lines: 3 posterior samples)")
    _ax.legend(loc="upper left", fontsize=8, ncol=2)
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack(
        [
            mo.hstack([func_select, positions_select], justify="start"),
            mo.hstack([noise3_slider, ell3_slider, sf3_slider], justify="start"),
            mo.hstack([mo.md("Number of measurements:"), n3_play], justify="start"),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - **No noise** ($\sigma_n = 0$): the posterior mean passes exactly through the measurements and the band collapses there. The GP is then an **interpolation** method (lecture 3), which also tells us how far to trust the interpolant between the points.
    - **Far from the data** (clustered positions, look at $x > 0.5$): the mean returns to the prior mean 0 and the band to $\pm 2\sigma_f$. A GP does not extrapolate a trend; it honestly says "I don't know".
    - **Length scale too small** ($\ell = 0.02$): between the measurements the mean drops back to 0 and the band opens up. **Too large** ($\ell = 0.5$): the GP cannot follow the function and treats the misfit as noise. With noise, the band for a *new measurement* never shrinks below $\pm 2\sigma_n$, while the band for $f$ itself shrinks as data accumulate, like the two bands of the posterior predictive in lecture 11.
    - **Forrester function**: its values reach 15, far beyond $\pm 2\sigma_f$ for $\sigma_f = 1$. Increase $\sigma_f$.

    Choosing $\ell$, $\sigma_f$ and $\sigma_n$ by hand is unsatisfactory. Section 4 lets the data choose them. First, we collect the four steps into a function and compare it with BoTorch.
    """)
    return


@app.cell
def _(jitter, np, rbf_kernel, solve_triangular):
    def gp_posterior(x_train, y_train, x_test, ell, sigma_f, sigma_n):
        """Posterior mean and covariance of a zero-mean GP with RBF kernel (Steps 1-4)."""
        K = rbf_kernel(x_train, x_train, ell, sigma_f) + (sigma_n**2 + jitter) * np.eye(x_train.size)
        K_s = rbf_kernel(x_train, x_test, ell, sigma_f)
        L = np.linalg.cholesky(K)
        alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True), lower=False)
        V = solve_triangular(L, K_s, lower=True)
        return K_s.T @ alpha, rbf_kernel(x_test, x_test, ell, sigma_f) - V.T @ V

    return (gp_posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same GP in BoTorch

    BoTorch's `SingleTaskGP` is a GP regression model built from GPyTorch components. We build it with exactly our ingredients: zero mean (`ZeroMean`), the RBF kernel with an amplitude (`ScaleKernel(RBFKernel())`, the scale being $\sigma_f^2$), Gaussian noise (`GaussianLikelihood`, its `noise` being $\sigma_n^2$), and we set the hyperparameters to the slider values. BoTorch works with `torch` tensors of shape $(n, d)$ for $n$ points in $d$ dimensions; here $d = 1$.
    """)
    return


@app.cell
def _(
    GaussianLikelihood,
    GreaterThan,
    RBFKernel,
    ScaleKernel,
    SingleTaskGP,
    ZeroMean,
    ell,
    gp_posterior,
    jitter,
    mo,
    n_obs,
    np,
    sigma_f,
    sigma_n,
    torch,
    x_test,
    x_train,
    y_train,
):
    mo.stop(n_obs == 0, mo.md("*Add at least one measurement above to compare with BoTorch.*"))

    X_t = torch.tensor(x_train)[:, None]  # (n, 1) tensors, double precision
    Y_t = torch.tensor(y_train)[:, None]
    gp_check = SingleTaskGP(
        X_t,
        Y_t,
        mean_module=ZeroMean(),
        covar_module=ScaleKernel(RBFKernel()),
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-10)),
        outcome_transform=None,
    )
    gp_check.covar_module.base_kernel.lengthscale = ell
    gp_check.covar_module.outputscale = sigma_f**2
    gp_check.likelihood.noise = sigma_n**2 + jitter
    gp_check.eval()
    with torch.no_grad():
        posterior_check = gp_check.posterior(torch.tensor(x_test)[:, None])
        mean_botorch = posterior_check.mean.numpy().ravel()
        var_botorch = posterior_check.variance.numpy().ravel()

    # GPyTorch stores the hyperparameters through a softplus transform: read back the exact stored values
    _ell = gp_check.covar_module.base_kernel.lengthscale.item()
    _sf = np.sqrt(gp_check.covar_module.outputscale.item())
    _sn = np.sqrt(gp_check.likelihood.noise.item() - jitter)
    _mu, _cov = gp_posterior(x_train, y_train, x_test, _ell, _sf, _sn)
    mo.md(rf"""
    Largest difference between our GP and BoTorch over the {x_test.size} test points:
    mean ${np.abs(mean_botorch - _mu).max():.1e}$, variance ${np.abs(var_botorch - np.diag(_cov)).max():.1e}$.
    The two agree to rounding error: `SingleTaskGP` does Steps 1–4 for us.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hyperparameters: the marginal likelihood
    ---

    The hyperparameters $\theta = (\ell, \sigma_f, \sigma_n)$ are inferred like the parameters in lecture 11. Their likelihood is the probability of the measurements given $\theta$, with the unknown function integrated out. Since $f$ and the noise are Gaussian, $\mathbf{y} \sim \mathcal{N}(0, K + \sigma_n^2 I)$, so
    $$
    \ln p(\mathbf{y}\,|\,\theta) = \underbrace{-\tfrac12 \mathbf{y}^\top (K + \sigma_n^2 I)^{-1}\mathbf{y}}_{\text{data fit}} \;\underbrace{-\tfrac12 \ln\det(K + \sigma_n^2 I)}_{\text{complexity penalty}} - \tfrac{n}{2}\ln 2\pi .
    $$
    This is the **marginal likelihood**: the *evidence* $p(d)$ of lecture 11, which we could ignore there, now becomes the quantity to maximize. With the Cholesky factor, $\mathbf{y}^\top (K + \sigma_n^2 I)^{-1}\mathbf{y} = \mathbf{y}^\top\boldsymbol{\alpha}$ and $\ln\det(K + \sigma_n^2 I) = 2\sum_i \ln L_{ii}$.

    The two terms pull in opposite directions. A very short length scale fits every data point (good data fit) but corresponds to an extremely flexible prior that could have produced almost any data (large penalty). A very long length scale is simple, but fits badly. The marginal likelihood picks the simplest model that explains the data: a built-in **Occam's razor**, and the Bayesian answer to von Neumann's elephant (lecture 1).

    In practice the measurements are first **standardized** (zero mean, unit variance), so that $\sigma_f \approx 1$ is a sensible scale for any problem; BoTorch's `Standardize` transform does this and converts predictions back. The data below use the test function selected in Section 3, a true noise level of 0.1 (unknown to the model), and the first $n$ measurement positions.
    """)
    return


@app.cell
def _(jitter, np, rbf_kernel, solve_triangular):
    def log_marginal_likelihood(x, y, ell, sigma_f, sigma_n):
        """ln p(y | ell, sigma_f, sigma_n), returned with its data-fit and complexity terms."""
        K = rbf_kernel(x, x, ell, sigma_f) + (sigma_n**2 + jitter) * np.eye(x.size)
        L = np.linalg.cholesky(K)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)
        data_fit = -0.5 * y @ alpha
        complexity = -np.sum(np.log(np.diag(L)))
        return data_fit + complexity - 0.5 * x.size * np.log(2 * np.pi), data_fit, complexity

    return (log_marginal_likelihood,)


@app.cell(hide_code=True)
def _(mo):
    n4_slider = mo.ui.slider(3, 20, 1, value=8, label="Number of measurements $n$", show_value=True)
    model4_select = mo.ui.radio(
        options=["Explicit: ScaleKernel(RBFKernel()), no priors", "BoTorch default SingleTaskGP"],
        value="Explicit: ScaleKernel(RBFKernel()), no priors",
        label="Model shown on the right",
    )
    return model4_select, n4_slider


@app.cell
def _(
    ExactMarginalLogLikelihood,
    GaussianLikelihood,
    RBFKernel,
    ScaleKernel,
    SingleTaskGP,
    Standardize,
    ZeroMean,
    f_true,
    fit_gpytorch_mll,
    n4_slider,
    positions_select,
    torch,
    x_sequences,
    z_sequence,
):
    x4 = x_sequences[positions_select.value][: n4_slider.value]
    y4 = f_true(x4) + 0.1 * z_sequence[: n4_slider.value]
    X4, Y4 = torch.tensor(x4)[:, None], torch.tensor(y4)[:, None]

    # explicit model: our three hyperparameters, no priors -> maximum of the marginal likelihood
    gp_explicit = SingleTaskGP(
        X4,
        Y4,
        mean_module=ZeroMean(),
        covar_module=ScaleKernel(RBFKernel()),
        likelihood=GaussianLikelihood(),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_explicit.likelihood, gp_explicit))

    # BoTorch defaults: RBF kernel with priors on the hyperparameters, constant mean, Standardize
    gp_default = SingleTaskGP(X4, Y4)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_default.likelihood, gp_default))
    return gp_default, gp_explicit, x4, y4


@app.cell(hide_code=True)
def _(
    colors,
    f_true,
    gp_default,
    gp_explicit,
    log_marginal_likelihood,
    mo,
    model4_select,
    n4_slider,
    np,
    plt,
    torch,
    x4,
    x_test,
    y4,
):
    _ys = (y4 - y4.mean()) / y4.std(ddof=1)  # standardized data, as inside BoTorch
    _ell = gp_explicit.covar_module.base_kernel.lengthscale.item()
    _sf = np.sqrt(gp_explicit.covar_module.outputscale.item())
    _sn = np.sqrt(gp_explicit.likelihood.noise.item())

    # (a) the two terms along the length scale, at the fitted sigma_f and sigma_n
    _ells = np.geomspace(0.01, 2, 200)
    _terms = np.array([log_marginal_likelihood(x4, _ys, _e, _sf, _sn) for _e in _ells])
    # (b) landscape over (ell, sigma_n), at the fitted sigma_f
    _g_ell, _g_sn = np.geomspace(0.01, 2, 60), np.geomspace(1e-3, 2, 60)
    _land = np.array([[log_marginal_likelihood(x4, _ys, _e, _sf, _s)[0] for _e in _g_ell] for _s in _g_sn])

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    _ax1.plot(_ells, _terms[:, 1], label="Data fit")
    _ax1.plot(_ells, _terms[:, 2], label="Complexity penalty")
    _ax1.plot(_ells, _terms[:, 0], "k", lw=2, label="Log marginal likelihood")
    _ax1.axvline(_ell, color=colors["BoTorch"], ls="--", label="BoTorch fit")
    _ax1.set(xscale="log", xlabel=r"$\ell$", ylim=(_terms[:, 0].max() - 25, max(_terms[:, 1].max(), _terms[:, 2].max()) + 3), title=r"Terms of $\ln p(\mathbf{y}|\theta)$")
    _ax1.legend(fontsize=8)

    _cs = _ax2.contourf(_g_ell, _g_sn, _land, levels=np.linspace(_land.max() - 20, _land.max(), 21), cmap="viridis", extend="min")
    _fig.colorbar(_cs, ax=_ax2, shrink=0.8)
    _ax2.plot(_ell, _sn, "*", color="white", ms=14, mec="k", label="BoTorch fit")
    _ax2.set(xscale="log", yscale="log", xlabel=r"$\ell$", ylabel=r"$\sigma_n$ (standardized)", title="Log marginal likelihood")
    _ax2.legend(loc="lower left", fontsize=8)

    _gp = gp_explicit if model4_select.value.startswith("Explicit") else gp_default
    with torch.no_grad():
        _post = _gp.posterior(torch.tensor(x_test)[:, None])
        _m, _s = _post.mean.numpy().ravel(), np.sqrt(_post.variance.numpy().ravel())
    _c = colors["BoTorch"] if _gp is gp_explicit else colors["Default"]
    _ax3.fill_between(x_test, _m - 2 * _s, _m + 2 * _s, color=_c, alpha=0.25, label="$f(x)$, ±2 std")
    _ax3.plot(x_test, _m, color=_c, lw=2, label="Posterior mean")
    _ax3.plot(x_test, f_true(x_test), color=colors["True"], ls="--", label="True function")
    _ax3.plot(x4, y4, "ko")
    _ax3.set(xlabel="$x$", title=model4_select.value.split(":")[0] + " model")
    _ax3.legend(fontsize=8, loc="upper left")
    _fig.tight_layout()
    plt.close(_fig)

    _std_y = y4.std(ddof=1)
    _ell_d = gp_default.covar_module.lengthscale.item()
    _sn_d = np.sqrt(gp_default.likelihood.noise.item())
    mo.vstack(
        [
            mo.hstack([n4_slider, model4_select], justify="start"),
            _fig,
            mo.md(rf"""
    | Model | $\ell$ | $\sigma_f$ (data units) | $\sigma_n$ (data units, true: 0.1) |
    |---|---|---|---|
    | Explicit, maximum marginal likelihood | {_ell:.3f} | {_sf * _std_y:.2f} | {_sn * _std_y:.3f} |
    | BoTorch default (with priors) | {_ell_d:.3f} | {_std_y:.2f} (fixed: 1 after standardization) | {_sn_d * _std_y:.3f} |
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(gp_default, mo, np):
    _lp = gp_default.covar_module.lengthscale_prior
    _np = gp_default.likelihood.noise_covar.noise_prior
    mo.md(rf"""
    `fit_gpytorch_mll` maximizes the marginal likelihood with a gradient-based optimizer (L-BFGS, a quasi-Newton method), using derivatives computed automatically by `torch`. The white star lands on the maximum of the landscape: BoTorch and our formula agree.

    - With **few measurements**, the landscape is broad and can have more than one maximum, typically a short-$\ell$/small-noise one ("everything is signal") and a long-$\ell$/large-noise one ("everything is noise"). Which one the optimizer finds depends on its starting point.
    - With **more measurements**, the maximum sharpens and the fitted noise approaches the true value.

    **BoTorch's default model** differs from our explicit one in three ways: a learned constant mean instead of zero; no separate amplitude ($\sigma_f = 1$ after standardization); and **priors on the hyperparameters**, a log-normal prior on $\ell$ (median $e^{{{_lp.loc.item():.2f}}} \approx {np.exp(_lp.loc.item()):.1f}$ in units of $x$) and a log-normal prior on the noise variance (median $e^{{{_np.loc.item():.0f}}} \approx {np.exp(_np.loc.item()):.3f}$ in standardized units). The fit then maximizes marginal likelihood × prior: a **maximum a posteriori** estimate, the effect of the prior seen in lecture 11, now acting on hyperparameters. The priors favour smooth functions and little noise, which keeps the fit sensible with very few data points, the typical situation in Bayesian optimization. Compare both models at $n = 3$–$5$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Choosing the kernel
    ---

    The kernel is the modelling assumption of a GP: it fixes what kind of functions we expect. The RBF kernel produces infinitely differentiable functions, often smoother than reality. The **Matérn** family adds a smoothness parameter $\nu$; with $r = |x - x'|$:

    | Kernel | $k(r)/\sigma_f^2$ | Functions are |
    |---|---|---|
    | Matérn $\nu = 1/2$ | $e^{-r/\ell}$ | continuous but rough (like Brownian motion, lecture 10) |
    | Matérn $\nu = 3/2$ | $\left(1 + \frac{\sqrt 3 r}{\ell}\right)e^{-\sqrt3 r/\ell}$ | once differentiable |
    | Matérn $\nu = 5/2$ | $\left(1 + \frac{\sqrt 5 r}{\ell} + \frac{5r^2}{3\ell^2}\right)e^{-\sqrt5 r/\ell}$ | twice differentiable |
    | RBF ($\nu\to\infty$) | $e^{-r^2/2\ell^2}$ | infinitely differentiable |

    Matérn 5/2 is the usual choice in Bayesian optimization. Below, prior samples for each kernel are drawn with the same random numbers, using the GPyTorch kernel objects to build the kernel matrices.
    """)
    return


@app.cell
def _(MaternKernel, RBFKernel, jitter, np, torch, x_grid):
    kernels = {
        "Matérn 1/2": lambda: MaternKernel(nu=0.5),
        "Matérn 3/2": lambda: MaternKernel(nu=1.5),
        "Matérn 5/2": lambda: MaternKernel(nu=2.5),
        "RBF": lambda: RBFKernel(),
    }
    _z = np.random.default_rng(9).normal(size=(x_grid.size, 3))
    kernel_samples = {}
    for _name, _make in kernels.items():
        _k = _make()
        _k.lengthscale = 0.15
        with torch.no_grad():
            _K = _k(torch.tensor(x_grid)[:, None]).to_dense().numpy()
        kernel_samples[_name] = np.linalg.cholesky(_K + jitter * np.eye(x_grid.size)) @ _z
    return kernel_samples, kernels


@app.cell(hide_code=True)
def _(kernel_samples, plt, x_grid):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for _ax, (_name, _s) in zip(_axes, kernel_samples.items()):
        _ax.plot(x_grid, _s, lw=1.2)
        _ax.set(title=_name, xlabel="$x$")
    _axes[0].set_ylabel("$f(x)$")
    _fig.suptitle(r"Prior samples, $\ell = 0.15$", y=1.02)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    n5_slider = mo.ui.slider(5, 20, 1, value=12, label="Number of measurements $n$", show_value=True)
    return (n5_slider,)


@app.cell
def _(
    ExactMarginalLogLikelihood,
    GaussianLikelihood,
    ScaleKernel,
    SingleTaskGP,
    Standardize,
    ZeroMean,
    f_true,
    fit_gpytorch_mll,
    kernels,
    n5_slider,
    np,
    torch,
    x_sequences,
    x_test,
    z_sequence,
):
    x5 = x_sequences["Random in [0, 1]"][: n5_slider.value]
    y5 = f_true(x5) + 0.1 * z_sequence[: n5_slider.value]
    _X, _Y = torch.tensor(x5)[:, None], torch.tensor(y5)[:, None]
    kernel_fits = {}
    for _name, _make in kernels.items():
        _gp = SingleTaskGP(_X, _Y, mean_module=ZeroMean(), covar_module=ScaleKernel(_make()), likelihood=GaussianLikelihood(), outcome_transform=Standardize(m=1))
        _mll = ExactMarginalLogLikelihood(_gp.likelihood, _gp)
        fit_gpytorch_mll(_mll)
        _gp.train()
        with torch.no_grad():
            _lml = _mll(_gp(*_gp.train_inputs), _gp.train_targets).item() * x5.size  # GPyTorch returns ln p / n
            _gp.eval()
            _post = _gp.posterior(torch.tensor(x_test)[:, None])
        _mean = _post.mean.numpy().ravel()
        kernel_fits[_name] = {
            "mean": _mean,
            "std": np.sqrt(_post.variance.numpy().ravel()),
            "lml": _lml,
            "ell": _gp.covar_module.base_kernel.lengthscale.item(),
            "rmse": np.sqrt(np.mean((_mean - f_true(x_test)) ** 2)),
        }
    return kernel_fits, x5, y5


@app.cell(hide_code=True)
def _(
    colors,
    f_true,
    func_select,
    kernel_fits,
    mo,
    n5_slider,
    plt,
    x5,
    x_test,
    y5,
):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for _ax, (_name, _r) in zip(_axes, kernel_fits.items()):
        _ax.fill_between(x_test, _r["mean"] - 2 * _r["std"], _r["mean"] + 2 * _r["std"], color=colors["BoTorch"], alpha=0.25)
        _ax.plot(x_test, _r["mean"], color=colors["BoTorch"], lw=2)
        _ax.plot(x_test, f_true(x_test), color=colors["True"], ls="--")
        _ax.plot(x5, y5, "ko", ms=4)
        _ax.set(title=_name, xlabel="$x$")
    _fig.tight_layout()
    plt.close(_fig)
    _rows = "\n".join(f"| {_n} | {_r['ell']:.3f} | {_r['lml']:.2f} | {_r['rmse']:.3f} |" for _n, _r in kernel_fits.items())
    mo.vstack(
        [
            mo.hstack([func_select, n5_slider], justify="start"),
            _fig,
            mo.md(f"""
    | Kernel | fitted $\\ell$ | log marginal likelihood | RMS error vs true function |
    |---|---|---|---|
    {_rows}
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The marginal likelihood compares kernels on the data alone, without knowing the true function: a higher value means the data are more probable under that prior. For the smooth custom function, the smoother kernels win, and the RMS error (which needs the true function, unavailable in practice) agrees.

    **When a GP struggles: the Forrester function.** Switch the test function. The Forrester function is nearly flat for $x < 0.5$ and varies steeply with a large amplitude near $x = 1$. Our kernels are **stationary**: they depend only on $x - x'$, so a single length scale and amplitude must describe the whole range. The fit compromises: the error band is too wide in the flat part and the mean overshoots or undershoots in the steep part. Remedies exist (transforming $x$ or $y$, input warping, non-stationary kernels), but the lesson is general: **the error bars of a GP are only as good as its kernel assumptions.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: Gaussian processes

    **Ingredients**

    | Object | Formula | Remarks |
    |---|---|---|
    | Kernel (RBF) | $k(x,x') = \sigma_f^2 e^{-(x-x')^2/2\ell^2}$ | covariance of $f(x)$ and $f(x')$: the prior over functions |
    | Prior samples | $\mathbf{f} = L\mathbf{z}$, $K + \text{jitter}\cdot I = LL^\top$ | Cholesky: LU for symmetric positive-definite matrices |
    | Posterior mean | $\boldsymbol{\mu}_* = K_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y} = K_*^\top\boldsymbol{\alpha}$ | interpolates exactly when $\sigma_n = 0$ |
    | Posterior covariance | $\Sigma_* = K_{**} - V^\top V$, $V = L^{-1}K_*$ | reverts to the prior far from the data |
    | New measurement | variance $\Sigma_{*,jj} + \sigma_n^2$ | never below the noise |
    | Log marginal likelihood | $-\frac12\mathbf{y}^\top\boldsymbol{\alpha} - \sum_i \ln L_{ii} - \frac n2 \ln 2\pi$ | data fit + complexity penalty; maximized to choose $\ell, \sigma_f, \sigma_n$ |

    **In practice**

    | Topic | Recommendation |
    |---|---|
    | Numerics | Cholesky + triangular solves, never an explicit inverse; jitter on the diagonal; cost $\mathcal{O}(n^3)$, fine up to a few thousand points |
    | Data | standardize $y$; scale $x$ to $[0, 1]$ |
    | Hyperparameters | maximize the marginal likelihood (`fit_gpytorch_mll`); priors (BoTorch defaults) stabilize fits with few points |
    | Kernel | Matérn 5/2 is a robust default; compare kernels with the marginal likelihood |
    | Limits | stationary kernels assume the same smoothness everywhere; error bars are only as good as the kernel |
    | Library | BoTorch `SingleTaskGP` (GPyTorch): the same formulas, plus automatic derivatives, priors and the tools of Bayesian optimization |
    """)
    return


if __name__ == "__main__":
    app.run()
