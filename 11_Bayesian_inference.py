import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import norm
    from wigglystuff import PlaySlider


    return PlaySlider, curve_fit, mo, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian inference for parameter estimation

    A model of a material always contains parameters that must be taken from experiments: an activation energy, an elastic constant, a reaction rate... The usual recipe is a least-squares fit, which returns **one** best set of parameters, with error bars as an afterthought.

    Bayesian inference answers a richer question: *given the data, how probable is each value of the parameters?* The answer is a probability distribution over parameter space, the **posterior**. From it we get best estimates, honest uncertainties, the correlations between parameters, and the uncertainty of any prediction made with the model.

    Our running example is diffusion in a solid, which follows the **Arrhenius law**
    $$
    D(T) = D_0\, e^{-E_a/k_B T},
    $$
    with two parameters: the prefactor $D_0$ and the activation energy $E_a$. Diffusion coefficients are measured at high temperature (where diffusion is fast enough to measure) but are often needed at low temperature, which makes the uncertainty of the extrapolation a real engineering question.

    | Section | Topic | Key idea |
    |---|---|---|
    | 1 | Bayes' theorem | prior × likelihood ∝ posterior |
    | 2 | One parameter on a grid | the posterior is updated as data arrives |
    | 3 | Two parameters on a grid | correlations, marginals, credible intervals |
    | 4 | Markov chain Monte Carlo | the Metropolis algorithm samples the posterior |
    | 5 | Comparison with least squares | `curve_fit` is a Gaussian approximation |
    | 6 | Posterior predictive | uncertainty of an extrapolation |
    | 7 | The effect of the prior | informative priors, live |

    The Markov chain Monte Carlo (MCMC) part builds directly on the Metropolis algorithm of the lecture *Random processes and Monte Carlo methods* (Section 13).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cache_mode = mo.ui.radio(
        options=["Load saved results", "Recompute"],
        value="Load saved results",
        label="**Expensive results** (long MCMC runs of Section 4):",
    )
    return (cache_mode,)


@app.cell(hide_code=True)
def _(cache_mode, mo, np):
    data_dir = mo.notebook_dir() / "data" / "11_Bayesian_inference"

    def cached(name, compute, key):
        """Load results/<name>.npz if allowed and computed with the same data `key`; otherwise compute and save."""
        path = data_dir / f"{name}.npz"
        if cache_mode.value == "Load saved results" and path.exists():
            with np.load(path) as f:
                saved = dict(f)
            if np.array_equal(saved.pop("key"), key):
                return saved
        result = compute()
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, key=key, **result)
        return result

    _files = sorted(p.name for p in data_dir.glob("*.npz")) if data_dir.exists() else []
    mo.vstack(
        [
            cache_mode,
            mo.md(
                f"Saved results in `data/11_Bayesian_inference/`: {', '.join(f'`{f}`' for f in _files) or '*none yet: they are computed and saved on the first run*'}."
            ),
        ]
    )
    return (cached,)


@app.cell(hide_code=True)
def _():
    colors = {
        "Prior": "0.45",
        "Likelihood": "darkorange",
        "Posterior": "#1d4ed8",
        "curve_fit": "seagreen",
        "True": "crimson",
    }
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Bayes' theorem
    ---

    Let $\theta$ denote the parameters of a model and $d$ the measured data. The rules of probability give
    $$
    p(\theta\,|\,d) = \frac{p(d\,|\,\theta)\; p(\theta)}{p(d)},
    $$
    which reads

    | | name | meaning |
    |---|---|---|
    | $p(\theta)$ | **prior** | what we know about $\theta$ before the measurement |
    | $p(d\,\vert\,\theta)$ | **likelihood** | probability to measure $d$ if the parameters were $\theta$ (the model + the noise) |
    | $p(\theta\,\vert\,d)$ | **posterior** | what we know about $\theta$ after the measurement |
    | $p(d)$ | **evidence** | a normalization constant |

    The evidence $p(d) = \int p(d\,|\,\theta)\,p(\theta)\,d\theta$ plays the same role as the partition function $Z$ in statistical mechanics: it is hard to compute, and for parameter estimation we never need it, since
    $$
    p(\theta\,|\,d) \propto p(d\,|\,\theta)\; p(\theta).
    $$

    ### The Arrhenius model

    Taking the logarithm of the Arrhenius law makes it linear. With $y = \ln D$ and $x = 1/k_B T$ (in eV$^{-1}$),
    $$
    y = \ln D_0 - E_a\, x .
    $$
    Diffusion coefficients typically scatter by a *factor*, so we assume Gaussian noise of standard deviation $\sigma$ on $\ln D$: each measurement is $y_i = \ln D_0 - E_a x_i + \sigma\,\epsilon_i$ with $\epsilon_i$ a standard Gaussian random number. For $N$ independent measurements the likelihood is
    $$
    p(d\,|\,\theta) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left[-\frac{r_i^2}{2\sigma^2}\right], \qquad r_i = y_i - (\ln D_0 - E_a x_i),
    $$
    and its logarithm is
    $$
    \ln p(d\,|\,\theta) = -N \ln\sigma - \frac{1}{2\sigma^2}\sum_{i} r_i^2 + \text{const} = -N\ln\sigma - \frac{\chi^2}{2} + \text{const}.
    $$
    With a flat prior and a known $\sigma$, the maximum of the posterior is the minimum of $\chi^2$: **the least-squares fit is the most probable point of the posterior**. Bayesian inference keeps the whole distribution instead of only its peak.

    We use synthetic data generated with known "true" parameters, so we can check whether the posterior contains the truth. The true values, $E_a = 2.05$ eV and $D_0 = 2\times 10^{-5}$ m²/s, are fixed as a reference point of the order of self-diffusion in copper (see Section 7). The **reference dataset** below (8 measurements between 800 and 1300 K, below the melting point of copper, 1358 K) is used from Section 4 on.
    """)
    return


@app.cell
def _(np):
    kB = 8.617333262e-5  # Boltzmann constant in eV/K
    Ea_true = 2.05  # eV
    lnD0_true = np.log(2e-5)  # D0 in m^2/s
    sigma_true = 0.3  # noise on ln D: a factor exp(0.3) = 1.35 on D

    # reference dataset (fixed seed)
    T_ref = np.linspace(800, 1300, 8)
    x_ref = 1 / (kB * T_ref)
    y_ref = lnD0_true - Ea_true * x_ref + sigma_true * np.random.default_rng(2026).normal(size=T_ref.size)
    return Ea_true, T_ref, kB, lnD0_true, sigma_true, x_ref, y_ref


@app.cell(hide_code=True)
def _(Ea_true, T_ref, colors, lnD0_true, np, plt, y_ref):
    _T = np.linspace(750, 1300, 100)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.plot(1000 / _T, (lnD0_true - Ea_true / (8.617333262e-5 * _T)) / np.log(10), color=colors["True"], ls="--", label="True law")
    _ax.plot(1000 / T_ref, y_ref / np.log(10), "ko", label="Reference dataset")
    _ax.set(xlabel=r"$1000/T$ (K$^{-1}$)", ylabel=r"$\log_{10} D$ (m$^2$/s)", title="Arrhenius plot")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One parameter on a grid
    ---

    We start with a single unknown parameter, $E_a$, pretending that $D_0$ and $\sigma$ are known. The posterior is then a function of one variable, which we simply evaluate on a grid of $E_a$ values and normalize numerically.

    As prior we take a Gaussian around a reference value that we fix, $E_a = 2.2 \pm 0.05$ eV, deliberately different from the true value (you can change it). The measurements are added one at a time; after each measurement, the posterior becomes the prior for the next one.

    The data of Sections 2 and 3 is an **exploration dataset** of 30 measurements at random temperatures between 750 and 1300 K, with a noise level $\sigma$ you can choose.
    """)
    return


@app.cell(hide_code=True)
def _(PlaySlider, mo):
    noise_slider = mo.ui.slider(0.1, 1.5, 0.05, value=0.8, label=r"Noise level $\sigma$ on $\ln D$", show_value=True)
    prior_mu = mo.ui.slider(1.8, 2.4, 0.01, value=2.2, label=r"Prior centre (eV)", show_value=True)
    prior_width = mo.ui.slider(0.01, 0.3, 0.01, value=0.05, label=r"Prior width (eV)", show_value=True)
    n_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=30, step=1, interval_ms=400))
    return n_play, noise_slider, prior_mu, prior_width


@app.cell
def _(Ea_true, kB, lnD0_true, noise_slider, np):
    # exploration dataset: fixed random draws, rescaled by the noise slider
    rng_exp = np.random.default_rng(7)
    T_exp = rng_exp.uniform(750, 1300, 30)
    x_exp = 1 / (kB * T_exp)
    y_exp = lnD0_true - Ea_true * x_exp + noise_slider.value * rng_exp.normal(size=30)
    return T_exp, x_exp, y_exp


@app.cell
def _(lnD0_true, noise_slider, norm, np, prior_mu, prior_width, x_exp, y_exp):
    def posterior_Ea(n):
        """Prior, likelihood and posterior of Ea on a grid, using the first n measurements."""
        Ea = np.linspace(1.7, 2.5, 2001)
        log_prior = norm.logpdf(Ea, prior_mu.value, prior_width.value)
        r = y_exp[:n] - (lnD0_true - Ea[:, None] * x_exp[:n])
        log_like = -0.5 * np.sum(r**2, axis=1) / noise_slider.value**2
        log_post = log_prior + log_like

        def normalize(logp):
            p = np.exp(logp - logp.max())  # subtract the maximum to avoid underflow
            return p / np.trapezoid(p, Ea)

        return Ea, normalize(log_prior), normalize(log_like), normalize(log_post)

    return (posterior_Ea,)


@app.cell(hide_code=True)
def _(
    Ea_true,
    T_exp,
    colors,
    lnD0_true,
    mo,
    n_play,
    noise_slider,
    np,
    plt,
    posterior_Ea,
    prior_mu,
    prior_width,
    y_exp,
):
    _n = int(n_play.value["value"])
    _Ea, _prior, _like, _post = posterior_Ea(_n)
    _mean = np.trapezoid(_Ea * _post, _Ea)
    _std = np.sqrt(np.trapezoid((_Ea - _mean) ** 2 * _post, _Ea))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _T = np.linspace(750, 1300, 100)
    _ax1.plot(1000 / _T, (lnD0_true - Ea_true / (8.617333262e-5 * _T)) / np.log(10), color=colors["True"], ls="--", label="True law")
    _ax1.plot(1000 / T_exp[_n:], y_exp[_n:] / np.log(10), "o", mfc="none", color="0.75", label="Not yet measured")
    _ax1.plot(1000 / T_exp[:_n], y_exp[:_n] / np.log(10), "ko", label="Measured")
    _ax1.set(xlabel=r"$1000/T$ (K$^{-1}$)", ylabel=r"$\log_{10} D$ (m$^2$/s)", title=f"{_n} measurements")
    _ax1.legend(loc="lower left")

    _ax2.plot(_Ea, _prior, color=colors["Prior"], ls="--", label="Prior")
    if _n > 0:
        _ax2.plot(_Ea, _like, color=colors["Likelihood"], ls=":", lw=2, label="Likelihood (normalized)")
    _ax2.fill_between(_Ea, _post, color=colors["Posterior"], alpha=0.25)
    _ax2.plot(_Ea, _post, color=colors["Posterior"], lw=2, label="Posterior")
    _ax2.axvline(Ea_true, color=colors["True"], ls="--", label="True value")
    _ax2.set(xlabel=r"$E_a$ (eV)", ylabel="Probability density (1/eV)", xlim=(1.85, 2.4), title=rf"$E_a = {_mean:.3f} \pm {_std:.3f}$ eV")
    _ax2.legend(loc="upper left", fontsize=9)
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([mo.hstack([noise_slider, prior_mu, prior_width]), mo.hstack([mo.md("Number of measurements:"), n_play], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - With 0 measurements the posterior is the prior. With a few measurements it lies **between** prior and likelihood; with many, the data dominate and the prior no longer matters.
    - A narrow (overconfident) prior needs many more measurements to be corrected. A wide prior lets the data speak immediately.
    - With more noise, each measurement carries less information and the prior matters longer.

    Fixing $D_0$ is a strong assumption: it is the reason why a single measurement already determines $E_a$ quite well. In Section 3 we release it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Two parameters on a grid
    ---

    Now both $\ln D_0$ and $E_a$ are unknown ($\sigma$ is still known), with a flat prior over a wide rectangle. The posterior is a function of two variables, which we evaluate on a $300\times 300$ grid.

    From the 2D posterior $p(\ln D_0, E_a\,|\,d)$ we get the posterior of a single parameter by integrating out the other one: the **marginal** distribution
    $$
    p(E_a\,|\,d) = \int p(\ln D_0, E_a\,|\,d)\; d\ln D_0 .
    $$
    A **credible interval** is an interval containing a given fraction (e.g. 95%) of the posterior probability. Similarly, a **credible region** in 2D is the smallest region containing 68% or 95% of the probability, obtained by keeping the grid cells with the highest posterior until their total reaches that fraction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n2_slider = mo.ui.slider(3, 30, 1, value=10, label="Number of measurements", show_value=True)
    return (n2_slider,)


@app.cell
def _(n2_slider, noise_slider, np, x_exp, y_exp):
    lnD0_grid = np.linspace(-20, -2, 300)
    Ea_grid = np.linspace(1.4, 2.7, 300)
    _x, _y = x_exp[: n2_slider.value], y_exp[: n2_slider.value]
    _r = _y - (lnD0_grid[:, None, None] - Ea_grid[None, :, None] * _x)  # shape (300, 300, n)
    _log_post = -0.5 * np.sum(_r**2, axis=2) / noise_slider.value**2  # flat prior
    post2d = np.exp(_log_post - _log_post.max())
    post2d /= post2d.sum()  # normalized over the grid cells
    return Ea_grid, lnD0_grid, post2d


@app.cell
def _(np):
    def credible_levels(p, fractions=(0.95, 0.68)):
        """Posterior values above which the highest cells contain the given probability fractions."""
        p_sorted = np.sort(p.ravel())[::-1]
        cumulative = np.cumsum(p_sorted) / p_sorted.sum()
        return [p_sorted[np.searchsorted(cumulative, f)] for f in fractions]

    def credible_interval(x, p, fraction=0.95):
        """Central interval of a 1D distribution p(x) on a grid."""
        cdf = np.cumsum(p) / np.sum(p)
        return np.interp([(1 - fraction) / 2, (1 + fraction) / 2], cdf, x)

    return credible_interval, credible_levels


@app.cell(hide_code=True)
def _(
    Ea_grid,
    Ea_true,
    colors,
    credible_interval,
    credible_levels,
    lnD0_grid,
    lnD0_true,
    mo,
    n2_slider,
    noise_slider,
    np,
    plt,
    post2d,
):
    _pL = post2d.sum(axis=1)  # marginal of ln D0
    _pE = post2d.sum(axis=0)  # marginal of Ea
    _dL, _dE = lnD0_grid[1] - lnD0_grid[0], Ea_grid[1] - Ea_grid[0]
    _mL, _mE = np.sum(lnD0_grid * _pL), np.sum(Ea_grid * _pE)
    _cov = np.sum(post2d * np.outer(lnD0_grid - _mL, Ea_grid - _mE))
    _rho = _cov / np.sqrt(np.sum(_pL * (lnD0_grid - _mL) ** 2) * np.sum(_pE * (Ea_grid - _mE) ** 2))
    _ciE = credible_interval(Ea_grid, _pE)
    _ciL = credible_interval(lnD0_grid, _pL)

    _fig = plt.figure(figsize=(7.5, 6.5))
    _gs = _fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
    _ax = _fig.add_subplot(_gs[1, 0])
    _axL = _fig.add_subplot(_gs[0, 0], sharex=_ax)
    _axE = _fig.add_subplot(_gs[1, 1], sharey=_ax)
    _ax.contourf(lnD0_grid, Ea_grid, post2d.T, levels=credible_levels(post2d) + [post2d.max()], colors=[colors["Posterior"]], alpha=0.25)
    _ax.contour(lnD0_grid, Ea_grid, post2d.T, levels=credible_levels(post2d), colors=[colors["Posterior"]], linestyles=["--", "-"])
    _ax.plot(lnD0_true, Ea_true, "*", color=colors["True"], ms=14, label="True values")
    _ax.set(xlabel=r"$\ln D_0$", ylabel=r"$E_a$ (eV)")
    _ax.legend(loc="lower right", title="contours: 68% and 95%")
    _axL.plot(lnD0_grid, _pL / _dL, color=colors["Posterior"])
    _axL.axvline(lnD0_true, color=colors["True"], ls="--")
    _axL.axvspan(*_ciL, color=colors["Posterior"], alpha=0.15)
    _axL.set_yticks([])
    _axL.tick_params(labelbottom=False)
    _axE.plot(_pE / _dE, Ea_grid, color=colors["Posterior"])
    _axE.axhline(Ea_true, color=colors["True"], ls="--")
    _axE.axhspan(*_ciE, color=colors["Posterior"], alpha=0.15)
    _axE.set_xticks([])
    _axE.tick_params(labelleft=False)
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([n2_slider, noise_slider]),
            _fig,
            mo.md(
                        rf"""
    **95% credible intervals** (shaded in the marginals)  
    $E_a \in [{_ciE[0]:.3f}, {_ciE[1]:.3f}]$ eV   
    $\ln D_0 \in [{_ciL[0]:.2f}, {_ciL[1]:.2f}]$  
    i.e. $D_0$ known within a factor {np.exp((_ciL[1] - _ciL[0]) / 2):.1f}  
    **Correlation coefficient**
    $\rho = {_rho:.3f}$
    """
                    ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior is a long, thin **ridge**: $\ln D_0$ and $E_a$ are strongly correlated ($\rho$ close to 1). The reason is visible on the Arrhenius plot: $\ln D_0$ is the intercept at $x = 1/k_BT = 0$, i.e. at infinite temperature, far away from the data. A slightly steeper line (larger $E_a$) through the same data has a much higher intercept.

    The marginal intervals are wide, but most combinations inside the rectangle they span are excluded: **error bars on each parameter separately do not tell the whole story.** We will see in Section 6 what happens if we forget the correlation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Markov chain Monte Carlo
    ---

    ### The curse of dimensionality, again

    Now we also treat the noise level $\sigma$ as unknown, a third parameter $\theta = (\ln D_0, E_a, \sigma)$, and switch to the reference dataset. Since $\sigma$ is a scale (we do not know even its order of magnitude), we use a prior that is uniform in $\ln \sigma$, i.e. $p(\sigma) \propto 1/\sigma$. The log-posterior is
    $$
    \ln p(\theta\,|\,d) = -N\ln\sigma - \frac{1}{2\sigma^2}\sum_i r_i^2 + \text{const}
    $$
    inside the prior bounds, and $-\infty$ outside.

    With 100 points per parameter, a grid needs $100^d$ posterior evaluations: $10^6$ for $d=3$ (still feasible, we will use it as a check), $10^{20}$ for $d=10$ (impossible). Moreover, almost all of these evaluations are wasted: the posterior is concentrated in a small part of the grid.
    """)
    return


@app.cell
def _(lnD0_true, np, x_ref, y_ref):
    # 3D grid in (ln D0, Ea, ln sigma); a flat prior in ln sigma is p(sigma) ~ 1/sigma
    g_lnD0 = np.linspace(lnD0_true - 5, lnD0_true + 5, 100)
    g_Ea = np.linspace(1.65, 2.45, 100)
    g_lns = np.linspace(np.log(0.05), np.log(3), 100)
    _r = y_ref - (g_lnD0[:, None, None] - g_Ea[None, :, None] * x_ref)  # (100, 100, N)
    _chi2 = np.sum(_r**2, axis=2)[:, :, None]
    _log_post = -x_ref.size * g_lns - 0.5 * _chi2 / np.exp(2 * g_lns)
    post3d = np.exp(_log_post - _log_post.max())
    post3d /= post3d.sum()
    return g_Ea, g_lnD0, g_lns, post3d


@app.cell(hide_code=True)
def _(mo, np, post3d):
    _useful = np.mean(post3d > 1e-3 * post3d.max())
    mo.md(rf"""
    On the 3D grid, only **{100 * _useful:.2f}%** of the $10^6$ points have a posterior larger than $10^{{-3}}$ times its maximum. The rest of the work is wasted.

    This is the situation of statistical mechanics (Sections 12–13 of the Monte Carlo lecture): we need averages over a probability distribution that is concentrated in a tiny part of a huge space, and known only up to a normalization constant. The solution is the same: **importance sampling with a Markov chain**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Metropolis algorithm for a posterior

    The Metropolis algorithm samples a distribution $P(\theta)$ using only ratios $P(\theta')/P(\theta)$. For the Boltzmann distribution $P \propto e^{-\beta E}$; for a posterior, $P \propto p(d\,|\,\theta)\,p(\theta)$. The translation is

    | Statistical mechanics | Bayesian inference |
    |---|---|
    | state $i$ | parameter vector $\theta$ |
    | $\beta E_i$ | $-\ln p(\theta\,\vert\,d)$ |
    | partition function $Z$ | evidence $p(d)$ (never needed) |
    | move (flip a spin) | random step $\theta' = \theta + \boldsymbol{s}\odot\boldsymbol{\epsilon}$, $\epsilon_k$ standard Gaussian |

    The step is symmetric (going from $\theta$ to $\theta'$ is as probable as the reverse), so detailed balance holds with the acceptance probability
    $$
    P_a = \min\left(1,\; \frac{p(\theta'\,|\,d)}{p(\theta\,|\,d)}\right) = \min\left(1,\; e^{\ln p(\theta'|d) - \ln p(\theta|d)}\right).
    $$
    After an equilibration period (in statistics: **burn-in**), the visited points $\theta_k$ are samples from the posterior. Any posterior average is a sample average, e.g. $\langle E_a\rangle \approx \frac{1}{M}\sum_k E_{a,k}$, and a histogram of the samples of one parameter *is* its marginal distribution: the integration over the other parameters is automatic.
    """)
    return


@app.cell
def _(np, x_ref, y_ref):
    # prior bounds (flat prior in ln D0, Ea, ln sigma)
    bounds = {"lnD0": (-30.0, 5.0), "Ea": (0.0, 4.0), "lns": (np.log(0.01), np.log(5.0))}

    def log_posterior(theta):
        """Unnormalized log-posterior; theta = (ln D0, Ea, ln sigma)."""
        lnD0, Ea, lns = theta
        if not (bounds["lnD0"][0] < lnD0 < bounds["lnD0"][1] and bounds["Ea"][0] < Ea < bounds["Ea"][1] and bounds["lns"][0] < lns < bounds["lns"][1]):
            return -np.inf
        r = y_ref - (lnD0 - Ea * x_ref)
        return -x_ref.size * lns - 0.5 * np.sum(r**2) * np.exp(-2 * lns)

    def metropolis(log_post, theta0, step, n_steps, rng):
        """Random-walk Metropolis. Returns the chain, shape (n_steps + 1, d), and the acceptance rate."""
        theta = np.array(theta0, dtype=float)
        lp = log_post(theta)
        chain = np.empty((n_steps + 1, theta.size))
        chain[0] = theta
        eta = rng.random(n_steps)
        eps = rng.normal(size=(n_steps, theta.size))
        n_accepted = 0
        for k in range(n_steps):
            candidate = theta + step * eps[k]
            lp_candidate = log_post(candidate)
            if np.log(eta[k]) < lp_candidate - lp:  # accept with probability min(1, p'/p)
                theta, lp = candidate, lp_candidate
                n_accepted += 1
            chain[k + 1] = theta
        return chain, n_accepted / n_steps

    return log_posterior, metropolis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The parameters are sampled as $(\ln D_0, E_a, \ln\sigma)$: a flat prior in $\ln\sigma$ is the $1/\sigma$ prior, and a Gaussian step in $\ln\sigma$ never produces a negative $\sigma$.

    The step sizes $\boldsymbol{s}$ are a tuning parameter. Below, $\boldsymbol{s} = f\times(0.6,\ 0.05,\ 0.25)$, roughly the posterior widths of the three parameters, times a factor $f$. The chain starts far from the posterior and runs 5000 steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    step_factor = mo.ui.slider(steps=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0, label="Step factor $f$", show_value=True)
    return (step_factor,)


@app.cell(hide_code=True)
def _(Ea_true, colors, log_posterior, metropolis, mo, np, plt, step_factor):
    _chain, _acc = metropolis(log_posterior, (-14.0, 1.8, np.log(1.0)), step_factor.value * np.array([0.6, 0.05, 0.25]), 5000, np.random.default_rng(1))
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    _ax1.plot(_chain[:, 1], color=colors["Posterior"], lw=0.8)
    _ax1.axhline(Ea_true, color=colors["True"], ls="--")
    _ax1.set(xlabel="Step $k$", ylabel=r"$E_a$ (eV)", ylim=(1.6, 2.6), title=f"Trace plot, acceptance rate {100 * _acc:.0f}%")
    _ax2.plot(_chain[:, 0], _chain[:, 1], color=colors["Posterior"], lw=0.5, alpha=0.6)
    _ax2.plot(*_chain[0, :2], "ks", label="Start")
    _ax2.set(xlabel=r"$\ln D_0$", ylabel=r"$E_a$ (eV)", xlim=(-18, -4), ylim=(1.6, 2.6), title="Path in the $(\\ln D_0, E_a)$ plane")
    _ax2.legend()
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([step_factor, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Steps too small** ($f \le 0.03$): almost every step is accepted, but the chain crawls. It has not even reached the posterior after 5000 steps (random walk: the distance grows only as $\sqrt{k}$).
    - **Steps too large** ($f \ge 3$): almost every candidate falls outside the ridge and is rejected; the chain stays stuck for hundreds of steps.
    - In between, the chain moves quickly through the posterior. A rule of thumb for random-walk Metropolis is an acceptance rate of about 25–50%.

    Even at the best $f$, the chain moves slowly *along* the ridge: the steps must be small enough to stay inside the thin ridge, while the ridge itself is long. We fix this below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Burn-in and multiple chains

    How do we know that a chain has converged? A practical test is to start **several chains from very different points** and check that they end up exploring the same region with the same statistics. The first part of each chain, before the chains merge, is discarded as burn-in.

    Below: 4 chains of 50 000 steps each, with $f = 1$ (these are the cached results).
    """)
    return


@app.cell
def _(cached, log_posterior, metropolis, np, y_ref):
    starts = np.array([[-6.0, 2.4, np.log(1.0)], [-18.0, 1.7, np.log(0.1)], [-12.0, 2.25, np.log(2.0)], [-8.0, 1.9, np.log(0.05)]])
    n_steps = 50_000
    burn_in = 2_000

    def _run_chains():
        rng = np.random.default_rng(42)
        runs = [metropolis(log_posterior, s, np.array([0.6, 0.05, 0.25]), n_steps, rng) for s in starts]
        return {"chains": np.array([c for c, _ in runs]), "acceptance": np.array([a for _, a in runs])}

    chains_direct = cached("chains_direct", _run_chains, key=y_ref)
    return burn_in, chains_direct, n_steps, starts


@app.cell(hide_code=True)
def _(Ea_true, burn_in, chains_direct, lnD0_true, np, plt, sigma_true):
    _fig, _axes = plt.subplots(3, 2, figsize=(11, 6.5), sharey="row", gridspec_kw={"width_ratios": [1, 2.5]})
    _labels = [r"$\ln D_0$", r"$E_a$ (eV)", r"$\sigma$"]
    _truth = [lnD0_true, Ea_true, sigma_true]
    for _i in range(3):
        for _c, _chain in enumerate(chains_direct["chains"]):
            _v = np.exp(_chain[:, 2]) if _i == 2 else _chain[:, _i]
            _axes[_i, 0].plot(_v[: 2 * burn_in], lw=0.8, color=plt.cm.tab10(_c))
            _axes[_i, 1].plot(np.arange(burn_in, _v.size, 10), _v[burn_in::10], lw=0.4, color=plt.cm.tab10(_c), alpha=0.7)
        for _ax in _axes[_i]:
            _ax.axhline(_truth[_i], color="k", ls="--", lw=1)
        _axes[_i, 0].axvline(burn_in, color="k", lw=1)
        _axes[_i, 0].set_ylabel(_labels[_i])
    _axes[2, 0].set_yscale("log")
    _axes[0, 0].set_title("First steps (burn-in left of the line)")
    _axes[0, 1].set_title("After burn-in (every 10th step)")
    _axes[2, 0].set_xlabel("Step $k$")
    _axes[2, 1].set_xlabel("Step $k$")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(burn_in, chains_direct, mo):
    _Ea = chains_direct["chains"][:, burn_in:, 1]
    _rows = "\n".join(
        f"| {_c + 1} | {100 * chains_direct['acceptance'][_c]:.0f}% | {_Ea[_c].mean():.4f} | {_Ea[_c].std():.4f} |" for _c in range(_Ea.shape[0])
    )
    mo.md(rf"""
    After a few hundred steps the four chains forget their starting points and explore the same region. Their statistics after burn-in:

    | Chain | Acceptance | mean $E_a$ (eV) | std $E_a$ (eV) |
    |---|---|---|---|
    {_rows}

    The chains agree only to about the second digit: along the ridge the chain is a slow random walk, so successive samples are strongly correlated and 48 000 steps are worth far fewer independent samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo, x_ref):
    mo.md(rf"""
    ### Reparameterisation: removing the ridge

    The ridge comes from measuring the intercept at $x = 0$, far from the data. If we instead use the value of $\ln D$ **at the centre of the data**, $\bar{{x}} = \langle 1/k_BT_i\rangle = {x_ref.mean():.2f}$ eV$^{{-1}}$,
    $$
    a = \ln D_0 - E_a\, \bar x, \qquad y = a - E_a (x - \bar x),
    $$
    the parameters $a$ and $E_a$ are (almost) uncorrelated: rotating the line about the centre of the data changes the slope but not $a$. The posterior is the same distribution, only described in other coordinates; since the transformation is linear with unit Jacobian, a flat prior in $(\ln D_0, E_a)$ is a flat prior in $(a, E_a)$.

    This trick, centering the independent variable, is worth remembering for every fit of a straight line (or of any model whose parameters are strongly correlated).
    """)
    return


@app.cell
def _(
    burn_in,
    cached,
    log_posterior,
    metropolis,
    n_steps,
    np,
    starts,
    x_ref,
    y_ref,
):
    x_bar = x_ref.mean()

    def log_posterior_centered(theta):
        """Same posterior in the coordinates (a, Ea, ln sigma) with a = ln D0 - Ea * x_bar."""
        a, Ea, lns = theta
        return log_posterior((a + Ea * x_bar, Ea, lns))

    def _run_chains():
        rng = np.random.default_rng(43)
        starts_centered = starts - np.outer(starts[:, 1] * x_bar, [1, 0, 0])
        runs = [metropolis(log_posterior_centered, s, np.array([0.15, 0.05, 0.25]), n_steps, rng) for s in starts_centered]
        return {"chains": np.array([c for c, _ in runs]), "acceptance": np.array([a for _, a in runs])}

    chains_centered = cached("chains_centered", _run_chains, key=y_ref)

    # posterior samples in the original parameters (ln D0, Ea, sigma), all chains, after burn-in
    _c = chains_centered["chains"][:, burn_in:].reshape(-1, 3)
    samples = np.column_stack([_c[:, 0] + _c[:, 1] * x_bar, _c[:, 1], np.exp(_c[:, 2])])
    return chains_centered, log_posterior_centered, samples, x_bar


@app.cell(hide_code=True)
def _(burn_in, chains_centered, chains_direct, colors, mo, np, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [2, 1]})
    for _row, (_name, _res) in enumerate([("$(\\ln D_0, E_a)$", chains_direct), ("$(a, E_a)$, centered", chains_centered)]):
        _ch = _res["chains"][0]
        _axes[_row, 0].plot(np.arange(burn_in, burn_in + 3000), _ch[burn_in : burn_in + 3000, 1], color=colors["Posterior"], lw=0.7)
        _axes[_row, 0].set(ylabel=r"$E_a$ (eV)", ylim=(1.75, 2.35), title=f"Parameters {_name}: acceptance {100 * _res['acceptance'][0]:.0f}%")
        _axes[_row, 1].plot(_ch[burn_in::20, 0], _ch[burn_in::20, 1], ".", ms=1.5, color=colors["Posterior"])
        _axes[_row, 1].set(xlabel=r"$a = \ln D_0 - E_a\bar x$" if _row == 1 else r"$\ln D_0$", ylabel=r"$E_a$ (eV)")
    _axes[1, 0].set_xlabel("Step $k$")
    _fig.tight_layout()
    plt.close(_fig)

    _std = lambda res: np.std([c[burn_in:, 1].mean() for c in res["chains"]])
    mo.vstack(
        [
            _fig,
            mo.md(rf"""
    Same posterior, same number of steps, very different efficiency: in centered coordinates the posterior is a round blob, the steps can be as large as its width in every direction, and the trace plot looks like noise (successive samples are nearly independent). The spread of the mean $E_a$ between the 4 chains drops from {_std(chains_direct):.4f} eV to {_std(chains_centered):.4f} eV.

    From now on we use the samples of the centered chains, transformed back to $(\ln D_0, E_a, \sigma)$.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The full posterior: a corner plot

    A standard way to display samples of a multidimensional posterior is a **corner plot**: the marginal of each parameter on the diagonal, and the 2D marginal of each pair below it. As a check, the lines show the marginals computed on the 3D grid.
    """)
    return


@app.cell(hide_code=True)
def _(
    Ea_true,
    colors,
    g_Ea,
    g_lnD0,
    g_lns,
    lnD0_true,
    np,
    plt,
    post3d,
    samples,
    sigma_true,
):
    _labels = [r"$\ln D_0$", r"$E_a$ (eV)", r"$\sigma$"]
    _truth = [lnD0_true, Ea_true, sigma_true]
    _grid_x = [g_lnD0, g_Ea, np.exp(g_lns)]
    _marg = [post3d.sum(axis=(1, 2)), post3d.sum(axis=(0, 2)), post3d.sum(axis=(0, 1))]
    _ranges = [(lnD0_true - 3.5, lnD0_true + 3.5), (1.75, 2.35), (0.05, 1.2)]
    _fig, _axes = plt.subplots(3, 3, figsize=(8, 7.5))
    for _i in range(3):
        for _j in range(3):
            _ax = _axes[_i, _j]
            if _j > _i:
                _ax.axis("off")
                continue
            if _i == _j:
                _ax.hist(samples[:, _i], bins=80, range=_ranges[_i], density=True, color=colors["Posterior"], alpha=0.4, label="MCMC")
                # grid marginal as a density in the plotted variable
                _dens = _marg[_i] / np.trapezoid(_marg[_i], np.log(_grid_x[_i]) if _i == 2 else _grid_x[_i])
                _ax.plot(_grid_x[_i], _dens / _grid_x[_i] if _i == 2 else _dens, color="k", lw=1.2, label="3D grid")
                _ax.axvline(_truth[_i], color=colors["True"], ls="--")
                _ax.set_yticks([])
            else:
                _ax.plot(samples[::40, _j], samples[::40, _i], ".", ms=1, color=colors["Posterior"])
                _ax.plot(_truth[_j], _truth[_i], "*", color=colors["True"], ms=10)
                _ax.set_ylim(_ranges[_i])
            _ax.set_xlim(_ranges[_j])
            if _i == 2:
                _ax.set_xlabel(_labels[_j])
            if _j == 0 and _i > 0:
                _ax.set_ylabel(_labels[_i])
    _axes[0, 0].legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo, samples):
    mo.md(rf"""
    The MCMC histograms agree with the grid. With only 8 measurements, the marginal of $\sigma$ is clearly **skewed**: small values are excluded by the scatter of the data, while a large noise level cannot be ruled out as firmly. The $\sigma$ uncertainty also feeds into the other parameters: a larger $\sigma$ allows a wider range of lines.

    We now have {samples.shape[0]:,} samples of $(\ln D_0, E_a, \sigma)$. Any question about the parameters becomes a question about these samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Comparison with least squares
    ---

    `scipy.optimize.curve_fit` minimizes $\chi^2$ and returns the best parameters and a covariance matrix. Without known error bars, it estimates the noise from the residuals, $s^2 = \chi^2_{\min}/(N-2)$, and scales the covariance accordingly. Implicitly, it approximates the posterior by a **Gaussian** centred on the best fit, with the noise level fixed at $s$.
    """)
    return


@app.cell
def _(curve_fit, np, x_ref, y_ref):
    def arrhenius(x, lnD0, Ea):
        return lnD0 - Ea * x

    p_fit, cov_fit = curve_fit(arrhenius, x_ref, y_ref, p0=(-10.0, 2.0))
    s_fit = np.sqrt(np.sum((y_ref - arrhenius(x_ref, *p_fit)) ** 2) / (x_ref.size - 2))
    return cov_fit, p_fit, s_fit


@app.cell(hide_code=True)
def _(
    Ea_true,
    colors,
    cov_fit,
    lnD0_true,
    mo,
    norm,
    np,
    p_fit,
    plt,
    s_fit,
    samples,
    sigma_true,
):
    # fraction of MCMC samples inside the 95% ellipse of curve_fit (chi^2 with 2 degrees of freedom: 5.991)
    _d = samples[:, :2] - p_fit
    _inside = np.mean(np.einsum("ij,jk,ik->i", _d, np.linalg.inv(cov_fit), _d) < 5.991)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))
    _ax1.plot(samples[::20, 0], samples[::20, 1], ".", ms=1.5, color=colors["Posterior"], label="MCMC samples")
    _w, _v = np.linalg.eigh(cov_fit)
    _t = np.linspace(0, 2 * np.pi, 200)
    _ell = p_fit[:, None] + _v @ (np.sqrt(5.991 * _w)[:, None] * np.array([np.cos(_t), np.sin(_t)]))
    _ax1.plot(*_ell, color=colors["curve_fit"], lw=2, label="curve_fit 95% ellipse")
    _ax1.plot(lnD0_true, Ea_true, "*", color=colors["True"], ms=12)
    _ax1.set(xlabel=r"$\ln D_0$", ylabel=r"$E_a$ (eV)", xlim=(lnD0_true - 4, lnD0_true + 4), ylim=(1.7, 2.4))
    _ax1.legend(loc="upper left", fontsize=8)

    _E = np.linspace(1.7, 2.4, 300)
    _ax2.hist(samples[:, 1], bins=100, range=(1.7, 2.4), density=True, color=colors["Posterior"], alpha=0.4, label="MCMC")
    _ax2.plot(_E, norm.pdf(_E, p_fit[1], np.sqrt(cov_fit[1, 1])), color=colors["curve_fit"], lw=2, label="curve_fit (Gaussian)")
    _ax2.axvline(Ea_true, color=colors["True"], ls="--")
    _ax2.set(xlabel=r"$E_a$ (eV)", yscale="log", ylim=(1e-2, None), title="Marginal of $E_a$ (log scale)")
    _ax2.legend(fontsize=8)

    _ax3.hist(samples[:, 2], bins=100, range=(0.05, 1.2), density=True, color=colors["Posterior"], alpha=0.4, label="MCMC")
    _ax3.axvline(s_fit, color=colors["curve_fit"], lw=2, label="curve_fit estimate $s$")
    _ax3.axvline(sigma_true, color=colors["True"], ls="--", label="True")
    _ax3.set(xlabel=r"$\sigma$", title=r"Marginal of $\sigma$")
    _ax3.legend(fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)

    _ci_mc = np.percentile(samples[:, 1], [2.5, 97.5])
    _ci_cf = p_fit[1] + np.array([-1.96, 1.96]) * np.sqrt(cov_fit[1, 1])
    mo.vstack(
        [
            _fig,
            mo.md(rf"""
    | | best / mean $E_a$ | 95% interval for $E_a$ | $\sigma$ |
    |---|---|---|---|
    | `curve_fit` | {p_fit[1]:.3f} eV | [{_ci_cf[0]:.3f}, {_ci_cf[1]:.3f}] eV | {s_fit:.2f} (no uncertainty) |
    | MCMC | {samples[:, 1].mean():.3f} eV | [{_ci_mc[0]:.3f}, {_ci_mc[1]:.3f}] eV | 95%: [{np.percentile(samples[:, 2], 2.5):.2f}, {np.percentile(samples[:, 2], 97.5):.2f}] |

    Only **{100 * _inside:.1f}%** of the posterior samples lie inside the 95% ellipse of `curve_fit`.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this model the two approaches agree on the centre and on the shape of the ridge. This is not a coincidence: the model is linear in $(\ln D_0, E_a)$ and the prior is flat, so the exact posterior at fixed $\sigma$ *is* a Gaussian. The differences come from $\sigma$:

    - `curve_fit` treats the estimated $s$ as exact, while with $N = 8$ points the noise level is poorly known (right panel).
    - Averaging over all plausible $\sigma$ gives the parameters **heavier tails** than a Gaussian (middle panel, log scale; it is in fact a Student-$t$ distribution with $N-2$ degrees of freedom), so the `curve_fit` intervals are too narrow.

    The differences shrink with more data and grow with fewer data, with nonlinear models, and with informative priors, which `curve_fit` cannot use at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Posterior predictive: extrapolating to low temperature
    ---

    The real payoff of the samples is in **predictions**. Each sample $\theta_k$ is a plausible Arrhenius law; evaluating the model for every sample gives the distribution of the prediction, the **posterior predictive**. Two different questions:

    - *Where is the true law?* Spread of $\ln D_0 - E_a x$ over the samples (model uncertainty).
    - *Where will a new measurement fall?* Add the noise: $\ln D_0 - E_a x + \sigma\epsilon$, with each sample's own $\sigma$.

    Suppose we need $D$ at 500 K, where diffusion is far too slow to be measured directly. We compare with a common shortcut: report $E_a \pm \delta E_a$ and $\ln D_0 \pm \delta\ln D_0$ separately, then propagate the errors **as if the parameters were independent**,
    $$
    \delta \ln D(x)^2 = \delta\ln D_0^2 + x^2\, \delta E_a^2 ,
    $$
    forgetting the covariance term $-2x\,\mathrm{cov}(\ln D_0, E_a)$.
    """)
    return


@app.cell
def _(np, samples):
    T_pred = np.linspace(480, 1350, 200)
    x_pred = 1 / (8.617333262e-5 * T_pred)
    _thin = samples[::25]  # a few thousand samples are enough for percentiles
    lnD_model = _thin[:, :1] - _thin[:, 1:2] * x_pred  # (samples, temperatures)
    lnD_new = lnD_model + _thin[:, 2:3] * np.random.default_rng(5).normal(size=lnD_model.shape)
    return T_pred, lnD_model, lnD_new, x_pred


@app.cell(hide_code=True)
def _(mo):
    threshold = mo.ui.slider(-26.5, -23.0, 0.1, value=-24.5, label=r"Threshold $\log_{10} D^*$ at 500 K", show_value=True)
    return (threshold,)


@app.cell(hide_code=True)
def _(
    Ea_true,
    T_pred,
    T_ref,
    colors,
    cov_fit,
    kB,
    lnD0_true,
    lnD_model,
    lnD_new,
    mo,
    norm,
    np,
    p_fit,
    plt,
    samples,
    threshold,
    x_pred,
    y_ref,
):
    _ln10 = np.log(10)
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _X = 1000 / T_pred
    _lo, _hi = np.percentile(lnD_new, [2.5, 97.5], axis=0)
    _ax.fill_between(_X, _lo / _ln10, _hi / _ln10, color=colors["Posterior"], alpha=0.12, label="New measurement (95%)")
    _lo, _hi = np.percentile(lnD_model, [2.5, 97.5], axis=0)
    _ax.fill_between(_X, _lo / _ln10, _hi / _ln10, color=colors["Posterior"], alpha=0.35, label="True law (95%), MCMC")
    _naive = 1.96 * np.sqrt(cov_fit[0, 0] + x_pred**2 * cov_fit[1, 1])
    _best = p_fit[0] - p_fit[1] * x_pred
    _ax.plot(_X, (_best - _naive) / _ln10, color=colors["Likelihood"], ls=":", lw=2, label="Independent errors (95%)")
    _ax.plot(_X, (_best + _naive) / _ln10, color=colors["Likelihood"], ls=":", lw=2)
    _ax.plot(_X, (lnD0_true - Ea_true * x_pred) / _ln10, color=colors["True"], ls="--", lw=1, label="True law")
    _ax.plot(1000 / T_ref, y_ref / _ln10, "ko", ms=5, label="Reference dataset")
    _ax.axvline(1000 / 500, color="k", lw=0.8)
    _ax.axhline(threshold.value, color="k", lw=0.8, ls="--")
    _ax.set(xlabel=r"$1000/T$ (K$^{-1}$)", ylabel=r"$\log_{10} D$ (m$^2$/s)", title="Posterior predictive")
    _ax.legend(loc="lower left", fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)

    # predictions at 500 K
    _x_target = 1 / (kB * 500)
    _lnD_target = samples[:, 0] - samples[:, 1] * _x_target
    _lnDstar = threshold.value * _ln10
    _p_mc = np.mean(_lnD_target > _lnDstar)
    _J = np.array([1, -_x_target])
    _p_cf = norm.sf(_lnDstar, p_fit[0] - p_fit[1] * _x_target, np.sqrt(_J @ cov_fit @ _J))
    _p_naive = norm.sf(_lnDstar, p_fit[0] - p_fit[1] * _x_target, np.sqrt(cov_fit[0, 0] + _x_target**2 * cov_fit[1, 1]))
    _q = np.percentile(_lnD_target, [2.5, 97.5]) / _ln10

    mo.vstack(
        [
            _fig,
            mo.md(rf"""
    At 500 K the 95% credible interval of the true $D$ is $[10^{{{_q[0]:.1f}}}, 10^{{{_q[1]:.1f}}}]$ m²/s: **a factor {10 ** (_q[1] - _q[0]):.0f}** between its ends, although each measurement scatters by only a factor $e^{{0.3}}\approx 1.35$. Extrapolation over a long range in $1/T$ amplifies the uncertainty of $E_a$.

    Ignoring the correlation is not a small error: the independent-error band is too wide everywhere, and worst at the measured temperatures, where the data pin $D$ down well.

    **A decision question.** A coating is acceptable only if $D(500\text{{ K}}) < D^*$. What is the probability that it fails? With samples, the answer is a count, $P(D > D^*) \approx \frac{{1}}{{M}}\#\{{k: D_k > D^*\}}$:
    """),
            threshold,
            mo.md(rf"""
    | Method | $P\left(D(500\text{{ K}}) > 10^{{{threshold.value:.1f}}}\text{{ m}}^2/\text{{s}}\right)$ |
    |---|---|
    | MCMC samples | **{_p_mc:.2g}** |
    | `curve_fit`, Gaussian with full covariance | {_p_cf:.2g} |
    | Independent errors | {_p_naive:.2g} |
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Move the threshold into the tail (e.g. $10^{-24.5}$ or higher): the Gaussian approximation of `curve_fit` underestimates the probability of failure by orders of magnitude, because it misses the heavy tails coming from the uncertain $\sigma$, while the independent-error estimate overestimates it. Rare events are precisely what an engineer cares about. Note that the probability itself is a statement that no single best-fit value can make.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. The effect of the prior
    ---

    So far, Sections 3–6 used flat priors, so the posterior was just the normalized likelihood. An informative prior encodes what we assume before the measurement. We consider two:

    - **A reference value for $E_a$**: we fix $E_a = 2.20 \pm 0.05$ eV as a reference point (the prior of Section 2), deliberately different from the true value.
    - **From physics.** For vacancy-mediated self-diffusion in an fcc metal, $D_0 = f\, a^2 \nu\, e^{\Delta S/k_B}$, with $a$ the lattice parameter, $f \approx 0.78$ the correlation factor, $\nu \approx 10^{13}$ Hz the attempt (Debye) frequency and $\Delta S \approx 1$–$4\,k_B$ the vacancy formation + migration entropy (H. Mehrer, *Diffusion in Solids*, Springer, 2007). For copper ($a = 3.61$ Å) this gives $D_0 \approx 3\times 10^{-6}$–$6\times 10^{-5}$ m²/s, and tabulated diffusion data for metals (e.g. Callister & Rethwisch, *Materials Science and Engineering: An Introduction*, chapter on diffusion) lie between about $10^{-6}$ and $10^{-4}$ m²/s. Hence the prior $\log_{10} D_0 = -5 \pm 1$.

    Below, Gaussian priors on $E_a$ and on $\log_{10} D_0$ are multiplied into the posterior of the reference dataset (8 points, $\sigma$ unknown with the $1/\sigma$ prior), and the Metropolis chain is rerun live (40 000 steps in centered coordinates). The orange histograms are the flat-prior result of Section 4, i.e. the likelihood alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    use_prior_Ea = mo.ui.checkbox(value=True, label=r"Gaussian prior on $E_a$")
    prior7_Ea_mu = mo.ui.slider(1.8, 2.4, 0.01, value=2.2, debounce=True, label="centre (eV)", show_value=True)
    prior7_Ea_w = mo.ui.slider(0.01, 0.3, 0.01, value=0.05, debounce=True, label="width (eV)", show_value=True)
    use_prior_D0 = mo.ui.checkbox(value=False, label=r"Gaussian prior on $\log_{10} D_0$")
    prior7_D0_mu = mo.ui.slider(-7.0, -3.0, 0.1, value=-5.0, debounce=True, label="centre (m²/s)", show_value=True)
    prior7_D0_w = mo.ui.slider(0.1, 2.0, 0.1, value=0.5, debounce=True, label="width (decades)", show_value=True)
    return (
        prior7_D0_mu,
        prior7_D0_w,
        prior7_Ea_mu,
        prior7_Ea_w,
        use_prior_D0,
        use_prior_Ea,
    )


@app.cell
def _(
    log_posterior_centered,
    metropolis,
    np,
    p_fit,
    prior7_D0_mu,
    prior7_D0_w,
    prior7_Ea_mu,
    prior7_Ea_w,
    s_fit,
    use_prior_D0,
    use_prior_Ea,
    x_bar,
):
    def log_prior_extra(lnD0, Ea):
        """Informative Gaussian priors, multiplied with the flat priors of Section 4."""
        lp = 0.0
        if use_prior_Ea.value:
            lp -= 0.5 * ((Ea - prior7_Ea_mu.value) / prior7_Ea_w.value) ** 2
        if use_prior_D0.value:
            lp -= 0.5 * ((lnD0 / np.log(10) - prior7_D0_mu.value) / prior7_D0_w.value) ** 2
        return lp

    def log_posterior_prior(theta):
        a, Ea, lns = theta
        return log_posterior_centered(theta) + log_prior_extra(a + Ea * x_bar, Ea)

    _start = (p_fit[0] - p_fit[1] * x_bar, p_fit[1], np.log(s_fit))  # start at the least-squares fit: no long burn-in
    _chain, acceptance_prior = metropolis(log_posterior_prior, _start, np.array([0.15, 0.05, 0.25]), 40_000, np.random.default_rng(11))
    _c = _chain[2_000:]
    samples_prior = np.column_stack([_c[:, 0] + _c[:, 1] * x_bar, _c[:, 1], np.exp(_c[:, 2])])
    return acceptance_prior, samples_prior


@app.cell(hide_code=True)
def _(
    Ea_true,
    acceptance_prior,
    colors,
    kB,
    lnD0_true,
    mo,
    norm,
    np,
    plt,
    prior7_D0_mu,
    prior7_D0_w,
    prior7_Ea_mu,
    prior7_Ea_w,
    samples,
    samples_prior,
    use_prior_D0,
    use_prior_Ea,
):
    _ln10 = np.log(10)
    _x_target = 1 / (kB * 500)
    _flat, _new = samples[::4], samples_prior  # similar numbers of samples
    _log10D_target = lambda s: (s[:, 0] - s[:, 1] * _x_target) / _ln10

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))
    if use_prior_Ea.value:
        _ax1.axhspan(prior7_Ea_mu.value - 2 * prior7_Ea_w.value, prior7_Ea_mu.value + 2 * prior7_Ea_w.value, color=colors["Prior"], alpha=0.15, label=r"Prior $\pm 2$ std")
    if use_prior_D0.value:
        _ax1.axvspan((prior7_D0_mu.value - 2 * prior7_D0_w.value) * _ln10, (prior7_D0_mu.value + 2 * prior7_D0_w.value) * _ln10, color=colors["Prior"], alpha=0.15, label=None if use_prior_Ea.value else r"Prior $\pm 2$ std")
    _ax1.plot(_flat[::10, 0], _flat[::10, 1], ".", ms=1.5, color=colors["Likelihood"], label="Flat prior")
    _ax1.plot(_new[::10, 0], _new[::10, 1], ".", ms=1.5, color=colors["Posterior"], label="With prior")
    _ax1.plot(lnD0_true, Ea_true, "*", color=colors["True"], ms=12)
    _ax1.set(xlabel=r"$\ln D_0$", ylabel=r"$E_a$ (eV)", xlim=(lnD0_true - 4, lnD0_true + 4), ylim=(1.7, 2.4), title=f"Acceptance {100 * acceptance_prior:.0f}%")
    _ax1.legend(loc="upper left", fontsize=8, markerscale=5)

    _E = np.linspace(1.7, 2.45, 300)
    _ax2.hist(_flat[:, 1], bins=80, range=(1.7, 2.45), density=True, histtype="step", color=colors["Likelihood"], lw=1.5, label="Flat prior")
    _ax2.hist(_new[:, 1], bins=80, range=(1.7, 2.45), density=True, color=colors["Posterior"], alpha=0.4, label="With prior")
    if use_prior_Ea.value:
        _ax2.plot(_E, norm.pdf(_E, prior7_Ea_mu.value, prior7_Ea_w.value), color=colors["Prior"], ls="--", label="Prior")
    _ax2.axvline(Ea_true, color=colors["True"], ls="--")
    _ax2.set(xlabel=r"$E_a$ (eV)", yticks=[], title=r"Marginal of $E_a$")
    _ax2.legend(fontsize=8)

    _ax3.hist(_log10D_target(_flat), bins=80, range=(-27.5, -23.5), density=True, histtype="step", color=colors["Likelihood"], lw=1.5, label="Flat prior")
    _ax3.hist(_log10D_target(_new), bins=80, range=(-27.5, -23.5), density=True, color=colors["Posterior"], alpha=0.4, label="With prior")
    _ax3.axvline((lnD0_true - Ea_true * _x_target) / _ln10, color=colors["True"], ls="--", label="True")
    _ax3.set(xlabel=r"$\log_{10} D(500\,\mathrm{K})$ (m$^2$/s)", yticks=[], title="Prediction at 500 K")
    _ax3.legend(fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)

    def _row(name, s):
        ciE = np.percentile(s[:, 1], [2.5, 97.5])
        ciD = np.percentile(_log10D_target(s), [2.5, 97.5])
        return f"| {name} | {s[:, 1].mean():.3f} | [{ciE[0]:.3f}, {ciE[1]:.3f}] | [{ciD[0]:.2f}, {ciD[1]:.2f}] |"

    mo.vstack(
        [
            mo.hstack([use_prior_Ea, prior7_Ea_mu, prior7_Ea_w], justify="start"),
            mo.hstack([use_prior_D0, prior7_D0_mu, prior7_D0_w], justify="start"),
            _fig,
            mo.md(
                f"""
    | Prior | mean $E_a$ (eV) | 95% interval $E_a$ (eV) | 95% interval $\\log_{{10}} D(500\\text{{ K}})$ |
    |---|---|---|---|
    {_row("Flat", _flat)}
    {_row("Chosen above", _new)}
    | True value | {Ea_true:.3f} | | {(lnD0_true - Ea_true * _x_target) / _ln10:.2f} |
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - **The reference prior** $E_a = 2.20 \pm 0.05$ eV (default): with only 8 measurements it pulls $E_a$ upwards, away from the true value, and since the parameters lie on a ridge it drags $D_0$ along. The prediction at 500 K moves accordingly. A wrong but confident prior gives a wrong but confident answer.
    - **Widen the prior** (width ≥ 0.2 eV): it becomes nearly flat over the region allowed by the data, and the result returns to the flat-prior one. A weak prior is harmless.
    - **The physics prior on $D_0$ only** (uncheck $E_a$, check $D_0$, narrow it to about 0.3 decades): a prior on $D_0$ alone also narrows $E_a$ and the prediction at 500 K, because the two parameters are correlated. Information on one parameter propagates to the others through the posterior.
    - **Both priors, in conflict** with each other or with the data: the posterior is a compromise lying between them. When prior and data disagree strongly, this is a warning sign to investigate, not a result to report.

    The prior must reflect what was known **before** the measurement. Choosing it after seeing the data, to obtain a desired answer, defeats the whole purpose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: Bayesian inference for parameter estimation

    **Ingredients**

    | Object | Formula | Remarks |
    |---|---|---|
    | Posterior | $p(\theta\,\vert\,d) \propto p(d\,\vert\,\theta)\,p(\theta)$ | the evidence $p(d)$ is not needed |
    | Gaussian likelihood | $\ln p(d\,\vert\,\theta) = -N\ln\sigma - \chi^2/2\sigma^2$ | max. posterior with flat prior = least squares |
    | Prior on a scale $\sigma$ | $p(\sigma)\propto 1/\sigma$ | flat in $\ln\sigma$ |
    | Informative prior | e.g. Gaussian around a reference value | matters when data are few; acts on correlated parameters too |
    | Marginal | $p(\theta_1\,\vert\,d) = \int p(\theta\,\vert\,d)\,d\theta_2\ldots d\theta_n$ | with samples: histogram of $\theta_1$ |
    | Posterior predictive | model evaluated for every sample | + noise for new measurements |

    **Methods**

    | Method | When | Remarks |
    |---|---|---|
    | Grid | 1–3 parameters | exact up to the grid resolution; cost $n^d$ |
    | Metropolis MCMC | any number of parameters | step $\approx$ posterior width, acceptance 25–50%, discard burn-in, check with several chains |
    | Reparameterisation | correlated parameters | center the variables (here $1/T$ around its mean) |
    | `curve_fit` covariance | many data, near-linear model | Gaussian approximation, noise level treated as exact |

    **Going further**

    - Successive MCMC samples are correlated. The **autocorrelation time** tells how many steps separate independent samples, and the **effective sample size** how many independent samples a chain is worth. Convergence across chains is quantified by the $\hat R$ statistic.
    - In practice, use a library: [`emcee`](https://emcee.readthedocs.io) (an ensemble sampler insensitive to correlations), [`PyMC`](https://www.pymc.io) (gradient-based samplers for many parameters), and [`ArviZ`](https://python.arviz.org) for diagnostics and corner plots.
    - **Model comparison:** the evidence $p(d)$ that we ignored measures how well a whole model (not one parameter set) explains the data, and automatically penalizes unnecessary parameters: the Bayesian answer to von Neumann's elephant (lecture 1). Approximations such as the Bayesian information criterion (BIC) are common.
    """)
    return


if __name__ == "__main__":
    app.run()
