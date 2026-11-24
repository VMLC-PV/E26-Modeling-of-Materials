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
    from scipy.stats import norm
    from wigglystuff import PlaySlider

    import torch

    torch.set_num_threads(1)  # the matrices are tiny: one thread is faster than many
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import (
        LogExpectedImprovement,
        PosteriorMean,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )
    from botorch.acquisition.analytic import PosteriorStandardDeviation
    from botorch.optim import optimize_acqf
    from botorch.exceptions import InputDataWarning, OptimizationWarning
    from gpytorch.mlls import ExactMarginalLogLikelihood

    warnings.filterwarnings("ignore", category=InputDataWarning)
    warnings.filterwarnings("ignore", category=OptimizationWarning)  # occasional failed hyperparameter fit; BoTorch retries

    return (
        ExactMarginalLogLikelihood,
        LogExpectedImprovement,
        PlaySlider,
        PosteriorMean,
        PosteriorStandardDeviation,
        ProbabilityOfImprovement,
        SingleTaskGP,
        UpperConfidenceBound,
        fit_gpytorch_mll,
        mo,
        norm,
        np,
        optimize_acqf,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian optimization

    Many design problems in materials science are optimization problems with **expensive** function evaluations: the composition with the highest strength, the annealing schedule with the lowest resistivity, the simulation parameters that best reproduce an experiment. Each evaluation is a synthesis and a measurement, or hours of computer time. We cannot afford thousands of evaluations, we have no gradients, and the function may have several local minima.

    **Bayesian optimization** (BO) minimizes such a function with as few evaluations as possible. It combines the two tools of the previous lectures:

    - a **surrogate model**: a Gaussian process (lecture 12) fitted to the evaluations so far, which predicts $f(x)$ *and its uncertainty* everywhere;
    - an **acquisition function**: a cheap function of the GP mean and standard deviation, whose maximum is the most promising point to evaluate next.

    | Section | Topic | Key idea |
    |---|---|---|
    | 1 | The Bayesian optimization loop | fit, choose, evaluate, repeat |
    | 2 | Acquisition functions | probability of improvement, expected improvement, confidence bound |
    | 3 | Bayesian optimization in one dimension | the loop at work, iteration by iteration |
    | 4 | Exploration vs exploitation | comparing strategies over many runs |
    | 5 | Two dimensions | BO against random search |

    We write the acquisition functions ourselves in `numpy`, check them against [BoTorch](https://botorch.org), and then let BoTorch run the loop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cache_mode = mo.ui.radio(
        options=["Load saved results", "Recompute"],
        value="Load saved results",
        label="**Expensive results** (comparisons over many runs, Sections 4 and 5):",
    )
    return (cache_mode,)


@app.cell(hide_code=True)
def _(cache_mode, mo, np):
    data_dir = mo.notebook_dir() / "data" / "13_Bayesian_optimization"

    def cached(name, compute, key):
        """Load <name>.npz if allowed and computed with the same settings `key`; otherwise compute and save."""
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
                f"Saved results in `data/13_Bayesian_optimization/`: {', '.join(f'`{f}`' for f in _files) or '*none yet: they are computed and saved on the first run*'}."
            ),
        ]
    )
    return (cached,)


@app.cell(hide_code=True)
def _():
    colors = {
        "GP": "darkorange",
        "Acquisition": "#1d4ed8",
        "True": "crimson",
        "Next": "gold",
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
    x_test = np.linspace(0, 1, 300)

    # true minima, for reference only: the optimizer never sees them
    _fine = np.linspace(0, 1, 200_001)
    f_min = {name: f(_fine).min() for name, f in test_functions.items()}
    x_min = {name: _fine[np.argmin(f(_fine))] for name, f in test_functions.items()}
    return f_custom, f_min, test_functions, x_min, x_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Bayesian optimization loop
    ---

    We want the minimum of $f(x)$ on a box (here $x \in [0, 1]$) with a budget of $N$ evaluations. The alternatives we know are poorly suited:

    - a **grid** wastes evaluations and suffers from the curse of dimensionality;
    - **gradient-based** methods need derivatives and find only a local minimum;
    - **simulated annealing** (lecture 10) finds global minima, but needs thousands of evaluations.

    Bayesian optimization uses *all* previous evaluations to decide where to evaluate next:

    1. **Initial design**: evaluate $f$ at a few points $x_1, \ldots, x_{n_0}$ (random or space-filling).
    2. **Repeat** until the budget is spent:
        1. fit a GP to all evaluations $(x_i, y_i)$, hyperparameters included (lecture 12, Section 4);
        2. compute the acquisition function $\alpha(x)$ from the GP mean $\mu(x)$ and standard deviation $\sigma(x)$;
        3. evaluate $f$ at $x_{\rm next} = \arg\max_x \alpha(x)$ and add the result to the data.
    3. **Recommend** the best point evaluated.

    Maximizing $\alpha$ is itself an optimization problem, but a cheap one: $\alpha$ costs microseconds to evaluate (it only needs the GP), and its gradient is available. All the cost is in evaluating $f$; the method is worth it as soon as one evaluation of $f$ costs more than a few seconds.

    Throughout, the GP is BoTorch's default `SingleTaskGP` (lecture 12, Section 4): its priors on the hyperparameters keep the fit sensible with the handful of points available at the start. The evaluations here are noise-free.
    """)
    return


@app.cell
def _(ExactMarginalLogLikelihood, SingleTaskGP, fit_gpytorch_mll, np, torch):
    def fit_gp(X, Y):
        """Fit BoTorch's default GP to inputs X (n, d) and values Y (n,)."""
        model = SingleTaskGP(torch.tensor(X), torch.tensor(Y)[:, None])
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model

    def gp_predict(model, X):
        """Posterior mean and standard deviation of f at the points X (m, d), as numpy arrays."""
        with torch.no_grad():
            post = model.posterior(torch.tensor(X))
        return post.mean.numpy().ravel(), np.sqrt(post.variance.numpy().ravel())

    return fit_gp, gp_predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Acquisition functions
    ---

    Let $f^* = \min_i y_i$ be the best value found so far. The GP says that at a point $x$, $f(x)$ is Gaussian with mean $\mu(x)$ and standard deviation $\sigma(x)$. With the standardized improvement
    $$
    z(x) = \frac{f^* - \mu(x) - \xi}{\sigma(x)},
    $$
    where $\xi \ge 0$ is an optional margin, three classic acquisition functions are:

    | Acquisition $\alpha(x)$ (maximized) | Formula | Meaning |
    |---|---|---|
    | Probability of improvement (PI) | $\Phi(z)$ | probability that $f(x) < f^* - \xi$ |
    | Expected improvement (EI) | $(f^* - \mu - \xi)\,\Phi(z) + \sigma\,\phi(z)$ | mean of the improvement $\max(f^* - f(x), 0)$ |
    | Lower confidence bound (LCB) | $-\left(\mu(x) - \kappa\,\sigma(x)\right)$ | optimistic estimate of $f(x)$ |

    Here $\Phi$ and $\phi$ are the cumulative distribution and density of the standard Gaussian. The EI formula follows from integrating the improvement over the Gaussian distribution of $f(x)$.

    Every acquisition function balances two terms: a low mean $\mu$ (**exploitation**: look where $f$ is known to be good) and a large $\sigma$ (**exploration**: look where we know little). In EI, the first term rewards a low mean and the second term a large uncertainty. In LCB, $\kappa$ sets the balance explicitly.
    """)
    return


@app.cell
def _(norm):
    def acq_pi(mu, sd, f_best, xi=0.0):
        return norm.cdf((f_best - mu - xi) / sd)

    def acq_ei(mu, sd, f_best, xi=0.0):
        z = (f_best - mu - xi) / sd
        return (f_best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)

    def acq_lcb(mu, sd, kappa=2.0):
        return -(mu - kappa * sd)  # maximizing -LCB = minimizing the lower confidence bound

    return acq_ei, acq_lcb, acq_pi


@app.cell(hide_code=True)
def _(mo):
    n2_slider = mo.ui.slider(2, 8, 1, value=4, label="Number of evaluations", show_value=True)
    xi_slider = mo.ui.slider(0.0, 0.5, 0.01, value=0.0, label=r"Margin $\xi$ (PI, EI)", show_value=True)
    kappa_slider = mo.ui.slider(0.0, 5.0, 0.1, value=2.0, label=r"$\kappa$ (LCB)", show_value=True)
    return kappa_slider, n2_slider, xi_slider


@app.cell
def _(f_custom, fit_gp, gp_predict, n2_slider, np, x_test):
    # a few evaluations of the custom function at fixed random positions
    x2 = np.random.default_rng(3).uniform(0, 1, 8)[: n2_slider.value]
    y2 = f_custom(x2)
    model2 = fit_gp(x2[:, None], y2)
    mu2, sd2 = gp_predict(model2, x_test[:, None])
    return model2, mu2, sd2, x2, y2


@app.cell(hide_code=True)
def _(
    acq_ei,
    acq_lcb,
    acq_pi,
    colors,
    f_custom,
    kappa_slider,
    mo,
    mu2,
    n2_slider,
    plt,
    sd2,
    x2,
    x_test,
    xi_slider,
    y2,
):
    _best = y2.min()
    _acqs = {
        "Probability of improvement": acq_pi(mu2, sd2, _best, xi_slider.value),
        "Expected improvement": acq_ei(mu2, sd2, _best, xi_slider.value),
        "Lower confidence bound (negated)": acq_lcb(mu2, sd2, kappa_slider.value),
    }
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    _axes[0].fill_between(x_test, mu2 - 2 * sd2, mu2 + 2 * sd2, color=colors["GP"], alpha=0.25, label=r"$\mu \pm 2\sigma$")
    _axes[0].plot(x_test, mu2, color=colors["GP"], lw=2, label=r"GP mean $\mu$")
    _axes[0].plot(x_test, f_custom(x_test), color=colors["True"], ls="--", label="True function (unknown)")
    _axes[0].plot(x2, y2, "ko", label="Evaluations")
    _axes[0].axhline(_best, color="k", lw=0.8, ls=":", label=r"best so far $f^*$")
    _axes[0].set(ylabel="$f(x)$", ylim=(-2.5, 2.5))
    _axes[0].legend(fontsize=8, ncol=3, loc="upper center")
    for _ax, (_name, _a) in zip(_axes[1:], _acqs.items()):
        _ax.plot(x_test, _a, color=colors["Acquisition"], lw=2)
        _ax.fill_between(x_test, _a.min(), _a, color=colors["Acquisition"], alpha=0.15)
        _ax.axvline(x_test[_a.argmax()], color="k", lw=1)
        _ax.plot(x_test[_a.argmax()], _a.max(), "*", color=colors["Next"], mec="k", ms=14)
        _ax.set(ylabel=r"$\alpha(x)$", title=_name)
        _axes[0].axvline(x_test[_a.argmax()], color="0.6", lw=0.6)
    _axes[-1].set_xlabel("$x$")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.hstack([n2_slider, xi_slider, kappa_slider], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try:

    - **PI with $\xi = 0$** is greedy: it prefers points that are *almost certainly* a tiny bit better than $f^*$, right next to the best evaluation. Increase $\xi$ to make it explore.
    - **EI** also weighs *how much* better a point could be, so it naturally values uncertain regions: it is the most widely used acquisition function.
    - **LCB with $\kappa = 0$** is pure exploitation (minimize the mean); a large $\kappa$ is nearly pure exploration (maximize $\sigma$).
    - With more evaluations, the acquisition functions become peaked around a few candidate regions.

    ### Check against BoTorch

    BoTorch provides these acquisition functions as `torch` modules, which it maximizes with gradient-based optimizers. Its recommended version of EI is `LogExpectedImprovement`, the logarithm of EI computed in a numerically stable way: far from the best point EI underflows to exactly 0, which leaves the optimizer without a gradient (S. Ament et al., *Unexpected improvements to expected improvement for Bayesian optimization*, NeurIPS 2023). BoTorch maximizes by convention, so for a minimization we pass `maximize=False`.
    """)
    return


@app.cell
def _(
    LogExpectedImprovement,
    acq_ei,
    mo,
    model2,
    mu2,
    np,
    sd2,
    torch,
    x_test,
    y2,
):
    log_ei = LogExpectedImprovement(model2, best_f=y2.min(), maximize=False)
    with torch.no_grad():
        ei_botorch = log_ei(torch.tensor(x_test)[:, None, None]).exp().numpy()  # shape (m, q=1, d=1) -> (m,)
    _ei_ours = acq_ei(mu2, sd2, y2.min())
    mo.md(f"Largest difference between our EI and BoTorch's `exp(LogExpectedImprovement)`: {np.abs(ei_botorch - _ei_ours).max():.1e} (largest EI value: {_ei_ours.max():.2e}).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Bayesian optimization in one dimension
    ---

    Now the full loop. We start from 3 random evaluations and run 15 iterations; at each iteration BoTorch refits the GP, builds the acquisition function and maximizes it with `optimize_acqf` (random starting points followed by gradient-based optimization). Use the play button to follow the iterations.
    """)
    return


@app.cell
def _(
    LogExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    fit_gp,
    np,
    optimize_acqf,
    torch,
):
    # acquisition functions in BoTorch, all set up for minimization (each is maximized)
    strategies = {
        "Expected improvement (LogEI)": lambda model, f_best: LogExpectedImprovement(model, best_f=f_best, maximize=False),
        "Lower confidence bound, κ = 2": lambda model, f_best: UpperConfidenceBound(model, beta=2.0**2, maximize=False),
        "Probability of improvement": lambda model, f_best: ProbabilityOfImprovement(model, best_f=f_best, maximize=False),
        "Pure exploitation (minimize the mean)": lambda model, f_best: PosteriorMean(model, maximize=False),
        "Pure exploration (maximize σ)": lambda model, f_best: PosteriorStandardDeviation(model),
        "Random search": None,
    }

    def run_bo(f, dim, n_init, n_iter, strategy, seed, record=None):
        """Minimize f on [0,1]^dim. f takes an array (n, dim) and returns (n,).

        Returns all evaluated points X, values Y, and, if `record` (an array of points) is given,
        the GP mean and standard deviation at those points before each iteration."""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float64)
        X = rng.uniform(size=(n_init, dim))
        Y = f(X)
        means, stds = [], []
        for _ in range(n_iter):
            if strategies[strategy] is None:
                x_new = rng.uniform(size=(1, dim))
            else:
                model = fit_gp(X, Y)
                acq = strategies[strategy](model, Y.min())
                candidate, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=5, raw_samples=128)
                x_new = candidate.numpy()
                if record is not None:
                    with torch.no_grad():
                        post = model.posterior(torch.tensor(record))
                    means.append(post.mean.numpy().ravel())
                    stds.append(np.sqrt(post.variance.numpy().ravel()))
            X = np.vstack([X, x_new])
            Y = np.append(Y, f(x_new))
        return X, Y, np.array(means), np.array(stds)

    return run_bo, strategies


@app.cell(hide_code=True)
def _(PlaySlider, mo, strategies, test_functions):
    func3_select = mo.ui.dropdown(options=list(test_functions), value=list(test_functions)[0], label="Test function")
    strategy3_select = mo.ui.dropdown(options=list(strategies)[:3], value=list(strategies)[0], label="Acquisition function")
    seed3_slider = mo.ui.slider(0, 9, 1, value=0, label="Random initial design (seed)", show_value=True)
    iter3_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=15, step=1, interval_ms=700))
    return func3_select, iter3_play, seed3_slider, strategy3_select


@app.cell
def _(
    func3_select,
    run_bo,
    seed3_slider,
    strategy3_select,
    test_functions,
    x_test,
):
    f3 = test_functions[func3_select.value]
    n_init3, n_iter3 = 3, 15
    X3, Y3, mu3, sd3 = run_bo(lambda X: f3(X[:, 0]), 1, n_init3, n_iter3, strategy3_select.value, seed3_slider.value, record=x_test[:, None])
    return X3, Y3, f3, mu3, n_init3, n_iter3, sd3


@app.cell(hide_code=True)
def _(
    X3,
    Y3,
    acq_ei,
    acq_lcb,
    acq_pi,
    colors,
    f3,
    f_min,
    func3_select,
    iter3_play,
    mo,
    mu3,
    n_init3,
    n_iter3,
    np,
    plt,
    sd3,
    seed3_slider,
    strategy3_select,
    x_min,
    x_test,
):
    _k = min(int(iter3_play.value["value"]), n_iter3 - 1)  # iteration shown
    _n = n_init3 + _k  # evaluations available at this iteration
    _x, _y = X3[:_n, 0], Y3[:_n]
    _mu, _sd = mu3[_k], sd3[_k]
    _name = strategy3_select.value
    if _name.startswith("Expected"):
        _a = acq_ei(_mu, _sd, _y.min())
    elif _name.startswith("Lower"):
        _a = acq_lcb(_mu, _sd, 2.0)
    else:
        _a = acq_pi(_mu, _sd, _y.min())
    _x_next = X3[_n, 0]
    _fmin = f_min[func3_select.value]

    _fig = plt.figure(figsize=(13, 6))
    _gs = _fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[2, 1])
    _ax1, _ax2, _ax3 = _fig.add_subplot(_gs[0, 0]), _fig.add_subplot(_gs[1, 0]), _fig.add_subplot(_gs[:, 1])
    _ax1.fill_between(x_test, _mu - 2 * _sd, _mu + 2 * _sd, color=colors["GP"], alpha=0.25)
    _ax1.plot(x_test, _mu, color=colors["GP"], lw=2, label="GP mean ± 2σ")
    _ax1.plot(x_test, f3(x_test), color=colors["True"], ls="--", label="True function")
    _ax1.plot(_x[:n_init3], _y[:n_init3], "ks", label="Initial design")
    _ax1.plot(_x[n_init3:], _y[n_init3:], "ko", label="BO evaluations")
    _ax1.plot(_x_next, f3(_x_next), "*", color=colors["Next"], mec="k", ms=16, label="Next evaluation")
    _ax1.axvline(x_min[func3_select.value], color=colors["True"], lw=0.6)
    _lo = min(f3(x_test).min(), (_mu - 2 * _sd).min())
    _hi = max(f3(x_test).max(), (_mu + 2 * _sd).max())
    _ax1.set(ylabel="$f(x)$", ylim=(_lo - 0.05 * (_hi - _lo), _hi + 0.35 * (_hi - _lo)), title=f"Iteration {_k + 1}: {_n} evaluations so far")
    _ax1.legend(fontsize=8, loc="upper center", ncol=5)  # headroom above the curves keeps the legend clear of the data
    _ax2.plot(x_test, _a, color=colors["Acquisition"], lw=2)
    _ax2.axvline(_x_next, color="k", lw=1)
    _ax2.set(xlabel="$x$", ylabel=r"$\alpha(x)$", title=_name)

    _best = np.minimum.accumulate(Y3) - _fmin
    _ax3.semilogy(np.arange(1, Y3.size + 1), np.maximum(_best, 1e-8), "o-", color="0.6", ms=3)
    _ax3.semilogy(np.arange(1, _n + 2), np.maximum(_best[: _n + 1], 1e-8), "o-", color=colors["Acquisition"], ms=4)
    _ax3.axvline(n_init3 + 0.5, color="k", lw=0.8, ls=":")
    _ax3.set(xlabel="Number of evaluations", ylabel=r"best $f$ so far $- f_{\min}$", title="Convergence", ylim=(1e-8, 10))
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([func3_select, strategy3_select, seed3_slider], justify="start"),
            mo.hstack([mo.md("Iteration:"), iter3_play], justify="start"),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The thin red line marks the true minimum, which the algorithm does not know. Things to observe:

    - **Expected improvement** alternates between refining the best region and probing uncertain regions. Within about 10 evaluations it typically finds the global minimum near $x \approx 0.91$, even when the initial design points to the local minimum near $x \approx 0.41$. Once the GP is confident, the evaluations cluster around the minimum and the error drops by orders of magnitude.
    - **Probability of improvement** ($\xi = 0$) is greedy and can stay in the first basin it finds; change the seed to see it get stuck.
    - **Forrester function**: the global minimum at $x \approx 0.76$ lies next to a deep descent; the GP, which assumes the same smoothness everywhere (lecture 12, Section 5), needs a few evaluations before it models it well.

    A single run can be lucky or unlucky. To compare strategies fairly, we repeat each from many random initial designs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Exploration vs exploitation
    ---

    We run each strategy from the same 10 random initial designs (3 points), with 15 further evaluations, and record the error of the best point found, $f_{\rm best} - f_{\min}$. Two extreme strategies show why the balance matters:

    - **pure exploitation**: evaluate at the minimum of the GP mean;
    - **pure exploration**: evaluate where the GP standard deviation is largest (this reduces the uncertainty everywhere, like a smart grid);

    and **random search** is the baseline every method must beat. The curves show the median over the 10 runs, with a band from the 25th to the 75th percentile. The comparison is made for the test function selected in Section 3 (cached results).
    """)
    return


@app.cell
def _(cached, f3, func3_select, np, run_bo, strategies, test_functions):
    _seeds, _n_init, _n_iter = np.arange(10), 3, 15

    def _compare():
        out = {}
        for i, name in enumerate(strategies):
            out[f"Y{i}"] = np.array([run_bo(lambda X: f3(X[:, 0]), 1, _n_init, _n_iter, name, s)[1] for s in _seeds])
        return out

    _fid = list(test_functions).index(func3_select.value)
    comparison = cached(f"comparison_1d_function{_fid}", _compare, key=np.array([_fid, _n_init, _n_iter, *_seeds]))
    return (comparison,)


@app.cell(hide_code=True)
def _(comparison, f_min, func3_select, mo, np, plt, strategies):
    _fmin = f_min[func3_select.value]
    _styles = ["#1d4ed8", "seagreen", "darkorange", "0.35", "purple", "0.6"]
    _fig, _ax = plt.subplots(figsize=(9, 4.8))
    _final = []
    for _i, (_name, _c) in enumerate(zip(strategies, _styles)):
        _err = np.maximum(np.minimum.accumulate(comparison[f"Y{_i}"], axis=1) - _fmin, 1e-8)
        _n = np.arange(1, _err.shape[1] + 1)
        _q25, _q50, _q75 = np.percentile(_err, [25, 50, 75], axis=0)
        _ax.fill_between(_n, _q25, _q75, color=_c, alpha=0.12)
        _ax.semilogy(_n, _q50, marker="o", ms=3, color=_c, lw=2 if _i == 0 else 1.3, ls="--" if _name == "Random search" else "-", label=_name)
        _final.append((_name, _q50[-1], np.mean(_err[:, -1] < 1e-2)))
    _ax.axvline(3.5, color="k", lw=0.8, ls=":")
    _ax.set(xlabel="Number of evaluations", ylabel=r"$f_{\rm best} - f_{\min}$ (median of 10 runs)", ylim=(1e-8, 20), title=func3_select.value)
    _ax.legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    plt.close(_fig)
    _rows = "\n".join(f"| {n} | {m:.1e} | {int(round(10 * p))}/10 |" for n, m, p in _final)
    mo.vstack(
        [
            _fig,
            mo.md(f"""
    After 18 evaluations:

    | Strategy | median error | runs with error < 0.01 |
    |---|---|---|
    {_rows}
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Pure exploitation** gets stuck: it keeps evaluating near the minimum of the current mean, which may be a local minimum, and never learns that a better region exists.
    - **Pure exploration** never gets stuck, but wastes evaluations in uninteresting regions: it improves only as fast as it fills the space.
    - **EI and LCB** combine both and reach the global minimum to high accuracy in most runs. Their advantage over random search grows with the required accuracy: random search improves only by luck, while BO concentrates its evaluations where they matter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Two dimensions
    ---

    The same code works in any dimension: the GP takes inputs of shape $(n, d)$ and `optimize_acqf` maximizes over the box $[0, 1]^d$. As a 2D test function we combine two copies of our 1D function with a coupling term,
    $$
    f(x_1, x_2) = g(x_1) + g(x_2) + 0.3\,\sin\left(6(x_1 + x_2)\right), \qquad g(x) = 0.8\sin(12x) + 0.6\cos(4x),
    $$
    which has several local minima and its global minimum near $(0.91, 0.91)$. We start from 5 random points and run 25 iterations of BO with expected improvement.
    """)
    return


@app.cell
def _(f_custom, np):
    def f_2d(X):
        return f_custom(X[:, 0]) + f_custom(X[:, 1]) + 0.3 * np.sin(6 * (X[:, 0] + X[:, 1]))

    g1 = np.linspace(0, 1, 60)
    grid_2d = np.array(np.meshgrid(g1, g1)).reshape(2, -1).T  # (3600, 2)
    _fine = np.linspace(0, 1, 801)
    _F = np.array(np.meshgrid(_fine, _fine)).reshape(2, -1).T
    f_min_2d = f_2d(_F).min()
    x_min_2d = _F[np.argmin(f_2d(_F))]
    return f_2d, f_min_2d, g1, grid_2d, x_min_2d


@app.cell
def _(cached, f_2d, grid_2d, np, run_bo):
    _n_init, _n_iter = 5, 25

    def _one_run():
        X, Y, means, stds = run_bo(f_2d, 2, _n_init, _n_iter, "Expected improvement (LogEI)", 0, record=grid_2d)
        return {"X": X, "Y": Y, "means": means, "stds": stds}

    def _compare():
        seeds = np.arange(10)
        return {
            "bo": np.array([run_bo(f_2d, 2, _n_init, _n_iter, "Expected improvement (LogEI)", s)[1] for s in seeds]),
            "random": np.array([run_bo(f_2d, 2, _n_init, 4 * _n_iter, "Random search", s)[1] for s in seeds]),
        }

    run_2d = cached("run_2d", _one_run, key=np.array([_n_init, _n_iter, grid_2d.shape[0]]))
    comparison_2d = cached("comparison_2d", _compare, key=np.array([_n_init, _n_iter, 10]))
    n_init_2d, n_iter_2d = _n_init, _n_iter
    return comparison_2d, n_init_2d, n_iter_2d, run_2d


@app.cell(hide_code=True)
def _(PlaySlider, mo, n_iter_2d):
    iter5_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=n_iter_2d - 1, step=1, interval_ms=600))
    return (iter5_play,)


@app.cell(hide_code=True)
def _(
    acq_ei,
    colors,
    f_2d,
    g1,
    grid_2d,
    iter5_play,
    mo,
    n_init_2d,
    plt,
    run_2d,
    x_min_2d,
):
    _k = int(iter5_play.value["value"])
    _n = n_init_2d + _k
    _X, _Y = run_2d["X"], run_2d["Y"]
    _mu, _sd = run_2d["means"][_k], run_2d["stds"][_k]
    _ei = acq_ei(_mu, _sd, _Y[:_n].min())
    _shape = (g1.size, g1.size)
    _panels = [
        (f_2d(grid_2d), "True function (unknown)", "viridis"),
        (_mu, "GP mean", "viridis"),
        (_sd, "GP standard deviation", "magma"),
        (_ei, "Expected improvement", "Blues"),
    ]
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for _ax, (_v, _title, _cmap) in zip(_axes, _panels):
        _vmin, _vmax = (f_2d(grid_2d).min(), f_2d(grid_2d).max()) if _cmap == "viridis" else (None, None)
        _ax.contourf(g1, g1, _v.reshape(_shape), levels=20, cmap=_cmap, vmin=_vmin, vmax=_vmax)
        _ax.plot(_X[:n_init_2d, 0], _X[:n_init_2d, 1], "ws", mec="k", ms=6)
        _ax.plot(_X[n_init_2d:_n, 0], _X[n_init_2d:_n, 1], "wo", mec="k", ms=5)
        _ax.plot(*_X[_n], "*", color=colors["Next"], mec="k", ms=16)
        _ax.plot(*x_min_2d, "x", color=colors["True"], ms=10, mew=2)
        _ax.set(title=_title, xlabel="$x_1$", aspect="equal")
    _axes[0].set_ylabel("$x_2$")
    _fig.tight_layout()
    plt.close(_fig)
    _best = _Y[: _n + 1].min()
    mo.vstack(
        [
            mo.hstack([mo.md("Iteration:"), iter5_play], justify="start"),
            _fig,
            mo.md(f"Iteration {_k + 1}: {_n} evaluations so far (squares: initial design; star: next evaluation; red cross: true minimum). Best value after the next evaluation: {_best:.4f}."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(comparison_2d, f_min_2d, mo, n_init_2d, n_iter_2d, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    for _key, _label, _c, _ls in [("bo", "BO, expected improvement", "#1d4ed8", "-"), ("random", "Random search", "0.5", "--")]:
        _err = np.maximum(np.minimum.accumulate(comparison_2d[_key], axis=1) - f_min_2d, 1e-8)
        _n = np.arange(1, _err.shape[1] + 1)
        _q25, _q50, _q75 = np.percentile(_err, [25, 50, 75], axis=0)
        _ax.fill_between(_n, _q25, _q75, color=_c, alpha=0.15)
        _ax.semilogy(_n, _q50, _ls, color=_c, lw=2, label=_label)
    _ax.axvline(n_init_2d + n_iter_2d, color="k", lw=0.8, ls=":")
    _ax.set(xlabel="Number of evaluations", ylabel=r"$f_{\rm best} - f_{\min}$ (median of 10 runs)", ylim=(1e-8, 10), title="2D test function")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _bo = np.median(np.minimum.accumulate(comparison_2d["bo"], axis=1)[:, -1] - f_min_2d)
    _rnd = np.median(np.minimum.accumulate(comparison_2d["random"], axis=1)[:, -1] - f_min_2d)
    mo.vstack(
        [
            _fig,
            mo.md(rf"""
    With {n_init_2d + n_iter_2d} evaluations, BO reaches a median error of {_bo:.1e}; random search, given **four times** as many evaluations ({n_init_2d + 4 * n_iter_2d}), reaches {_rnd:.1e}. On the maps above, the first BO evaluations are spread over the domain (exploration, where $\sigma$ is large), and the later ones concentrate around the global minimum (exploitation).
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: Bayesian optimization

    **The method**

    | Step | Tool | Remarks |
    |---|---|---|
    | Surrogate model | GP, BoTorch `SingleTaskGP` | hyperparameters refitted at every iteration; priors stabilize the first fits |
    | Acquisition function | EI (`LogExpectedImprovement`), LCB (`UpperConfidenceBound`), PI | balance exploitation (low $\mu$) and exploration (large $\sigma$) |
    | Next point | `optimize_acqf` | maximizes the cheap acquisition function with random starts + gradients |
    | Recommendation | best evaluated point | |

    **Acquisition functions** (minimization, $z = (f^* - \mu - \xi)/\sigma$)

    | Name | $\alpha(x)$ | Behaviour |
    |---|---|---|
    | Probability of improvement | $\Phi(z)$ | greedy unless $\xi > 0$ |
    | Expected improvement | $(f^* - \mu - \xi)\Phi(z) + \sigma\phi(z)$ | the standard choice; use its logarithm numerically |
    | Lower confidence bound | $-(\mu - \kappa\sigma)$ | $\kappa$ sets the balance explicitly |

    **When to use it**: evaluations are expensive (minutes to days), there are few input variables (up to about 10–20), no gradients are available and the budget is tens to hundreds of evaluations. For cheap functions, classical optimizers are faster; for many variables, the GP needs too many points to be informative.

    **Going further**, all available in BoTorch:

    - **Noisy measurements**: the best *observed* value is then unreliable; use the GP mean instead, e.g. `qLogNoisyExpectedImprovement`.
    - **Batches**: choose $q$ points at once to run experiments in parallel (`q > 1` in `optimize_acqf`).
    - **Constraints** (e.g. compositions summing to 100%) and **several objectives** (strength vs ductility: a Pareto front).
    """)
    return


if __name__ == "__main__":
    app.run()
