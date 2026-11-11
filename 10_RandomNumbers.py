import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from wigglystuff import PlaySlider

    # try:
    #     from utils import plot_settings_screen
    # except ImportError:
    #     pass
    return go, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random processes and Monte Carlo methods

    Random numbers play an important role, both in modeling physical processes (some of which are regarded as truly random, such as radioactive decay) and as a tool to tackle otherwise intractable problems. Examples:

    - numerical integration, especially in many dimensions;
    - sampling microstates in statistical mechanics;
    - simulating quantum processes;
    - Monte Carlo event generators (e.g. for particle collisions).

    Numbers generated on a computer are not truly random: they come out of a deterministic algorithm. A good generator, however, produces numbers that reflect the desired properties of a random variable, hence they are called **pseudo-random**.

    The most basic routine typically produces a random integer $x$ between 0 and some maximum value $m$. Dividing by $m$ gives a real pseudo-random number $\eta = x/m$ uniformly distributed in the interval $\eta \in [0,1)$. By applying various transformations and techniques to the sequence of $\eta$ one can obtain other distributions.
    """)
    return


@app.cell(hide_code=True)
def _():
    method_colors = {
        "Hit-or-miss": "darkorange",
        "Mean value": "seagreen",
        "Importance": "#1d4ed8",
        "Rectangle": "0.4",
        "Exact": "crimson",
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Random numbers: the linear congruential generator
    ---

    Historically, one of the simplest generators is the **linear congruential generator** (LCG).
    Although it should not be used in practice, it illustrates some of the key concepts and issues.

    It generates a sequence of pseudo-random integers with the iterative procedure
    $$
    x_{n+1} = (a x_n + c) ~ \text{mod} ~ m,
    $$
    for some integer parameters $a, c, m$. Dividing by $m$ gives $\eta_n = x_n/m \in [0,1)$.

    The next number only depends on the present one, so the sequence is **periodic**, with a period of at most $m$.
    """)
    return


@app.function
def lcg(n, a=1664525, c=1013904223, m=2**32, seed=1):
    """Return n pseudo-random numbers in [0,1) from the linear congruential generator.

    Default parameters from Numerical Recipes. The integer x is iterated as
    x <- (a*x + c) mod m, starting from the seed.
    """
    out = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x / m)
    return out


@app.cell(hide_code=True)
def _(mo):
    lcg_N = mo.ui.slider(100, 5000, step=100, value=1000, label="Number of samples $N$")
    lcg_params = mo.ui.dropdown(
        options={
            "Numerical Recipes: a=1664525, c=1013904223, m=2³²": (1664525, 1013904223, 2**32),
            "Small modulus: a=106, c=1283, m=6075": (106, 1283, 6075),
            "Poor choice: a=65, c=1, m=2048": (65, 1, 2048),
        },
        value="Numerical Recipes: a=1664525, c=1013904223, m=2³²",
        label="LCG parameters",
    )
    return lcg_N, lcg_params


@app.cell(hide_code=True)
def _(lcg_N, lcg_params, mo, plt):
    _a, _c, _m = lcg_params.value
    _eta = lcg(2 * lcg_N.value, _a, _c, _m)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    _ax1.plot(_eta[: lcg_N.value], "o", ms=2.5, color="steelblue")
    _ax1.set(xlabel="$n$", ylabel=r"$\eta_n$", title="The sequence")
    _ax2.plot(_eta[0::2], _eta[1::2], "o", ms=2.5, color="steelblue")
    _ax2.set(xlabel=r"$\eta_{2k}$", ylabel=r"$\eta_{2k+1}$", title="Consecutive pairs as points $(x, y)$")
    _ax2.set_aspect("equal")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([lcg_params, lcg_N], justify="center", gap=2),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Apart from a rather short period, the LCG also places points on a few **hyperplanes** (rather than filling the full space) if consecutive random numbers are taken as the components $(x,y,z,\ldots)$ of a vector. Checking for this structure is called the *spectral test*.
    For a good choice of parameters and small dimension this issue is hardly visible, but for a small modulus or a poor choice of $a$ the lattice structure is obvious already in 2D: switch the parameters above.

    The infamous generator **RANDU** ($a = 65539$, $c=0$, $m = 2^{31}$), widely used in the 1960s and 1970s, looks fine in 2D, but in 3D all triplets fall on just 15 planes. Rotate the plot below until you see them.
    """)
    return


@app.cell(hide_code=True)
def _(go):
    _eta = lcg(3 * 3000, a=65539, c=0, m=2**31)
    _fig = go.Figure(
        go.Scatter3d(x=_eta[0::3], y=_eta[1::3], z=_eta[2::3], mode="markers", marker=dict(size=2, color="steelblue"))
    )
    _fig.update_layout(
        title="RANDU: consecutive triplets",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", camera=dict(eye=dict(x=-1.6, y=1.4, z=0.3))),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this reason the LCG is not, and should not be, used in any serious calculation. (Many old library routines, such as C's `rand()`, are LCGs: avoid them.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Random number seeds
    ---

    Many better methods have been developed over the years. A long-time method of choice is the **Mersenne Twister**, implemented by default in many programming environments (Python's `random` module, the legacy `np.random.rand`). It has a period of $2^{19937} - 1$, passes most statistical randomness tests, is fast, and is suitable for most physical applications (but not for cryptography: its future output can be predicted from a few hundred observed numbers).
    NumPy's modern interface, `np.random.default_rng`, uses the PCG64 generator, which has even better statistical properties. We will not reimplement these generators but take them for granted.

    Pseudo-random number generators are iterative constructs that produce a **pre-determined sequence** of numbers from an initial state. The initial state is set by the **seed**: the same seed always gives the same sequence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""YouTube video about Mersenne Twister by [Statistics Crystallized](https://youtu.be/TF4PLUcJO5w?si=pyCnfHDm5coxvBEk) if you want to know more about how it works.
    """).callout("info")
    return


@app.cell(hide_code=True)
def _(mo):
    seed_a = mo.ui.number(start=0, stop=10**6, value=1, label="Seed of run A")
    seed_b = mo.ui.number(start=0, stop=10**6, value=1, label="Seed of run B")
    return seed_a, seed_b


@app.cell(hide_code=True)
def _(mo, np, plt, seed_a, seed_b):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _runs = {}
    for _ax, _seed, _name in zip(_axes, [seed_a.value, seed_b.value], ["A", "B"]):
        _rng = np.random.default_rng(_seed)
        _xy = _rng.random((300, 2))
        _runs[_name] = _xy
        _ax.plot(_xy[:, 0], _xy[:, 1], "o", ms=3, color="steelblue")
        _ax.set(xlabel="x", ylabel="y", title=f"Run {_name}: seed = {_seed}")
        _ax.set_aspect("equal")
    _fig.tight_layout()
    plt.close(_fig)

    _same = np.array_equal(_runs["A"], _runs["B"])
    mo.vstack(
        [
            mo.hstack([seed_a, seed_b], justify="center", gap=2),
            _fig,
            mo.callout(
                mo.md("The two runs are **identical**." if _same else "The two runs are **different**."),
                kind="success" if _same else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The deterministic nature of a seeded generator is useful for debugging: the sequence of pseudo-random numbers is guaranteed to be the same, so an error is reproduced each time the code is run.

    In production runs (e.g. many jobs on a cluster) one has to make sure that a different seed (or, better yet, an independent generator state, e.g. via `np.random.SeedSequence.spawn`) is used for each run, otherwise all runs produce the same "random" result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Random numbers and secret codes
    ---

    The fact that a seed determines the whole sequence can be turned into a (toy) encryption scheme.
    Sender and receiver agree on a secret seed, the **key**. The sender adds a pseudo-random number to each character of the message, here with a bitwise exclusive or (XOR),
    $$
    \text{code}_i = \text{char}_i \oplus r_i, \qquad r_i \in \{0, \ldots, 255\}.
    $$
    Since $(c \oplus r) \oplus r = c$, the receiver recovers the message by generating the same sequence $r_i$ from the same seed.
    Without the key, the coded message looks like random noise.

    This idea, with a cryptographically secure generator in place of ours, is the basis of *stream ciphers*.
    """)
    return


@app.cell
def _(np):
    def xor_cipher(data, key):
        """Encode (or decode) bytes by XOR with a pseudo-random stream generated from the key."""
        rng = np.random.default_rng(key)
        stream = rng.integers(0, 256, size=len(data), dtype=np.uint8)
        return bytes(np.frombuffer(data, dtype=np.uint8) ^ stream)

    return (xor_cipher,)


@app.cell(hide_code=True)
def _(mo):
    secret_message = mo.ui.text(value="Meet me at the library at noon", label="Message", full_width=True)
    secret_key = mo.ui.number(start=0, stop=10**9, value=1234, label="Key used to encode")
    guess_key = mo.ui.number(start=0, stop=10**9, value=1234, label="Key used to decode")
    return guess_key, secret_key, secret_message


@app.cell(hide_code=True)
def _(guess_key, mo, secret_key, secret_message, xor_cipher):
    _coded = xor_cipher(secret_message.value.encode(), int(secret_key.value))
    _decoded = xor_cipher(_coded, int(guess_key.value)).decode(errors="replace")
    mo.vstack(
        [
            secret_message,
            mo.hstack([secret_key, guess_key], justify="center", gap=2),
            mo.md(f"**Coded message (hex):** `{_coded.hex(' ')}`"),
            mo.md(f"**Decoded message:** `{_decoded}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Probabilities and biased coins
    ---

    Many processes require an event to happen with a given probability $p$, like the toss of a biased coin.
    With a uniform random number $\eta\in[0,1)$ this is simple: the event happens if
    $$
    \eta < p,
    $$
    which occurs with probability exactly $p$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    coin_p = mo.ui.slider(0.0, 1.0, step=0.05, value=0.2, label="Probability of heads $p$")
    coin_N = mo.ui.slider(10, 10000, step=10, value=1000, label="Number of tosses $N$")
    return coin_N, coin_p


@app.cell(hide_code=True)
def _(coin_N, coin_p, mo, np, plt):
    _rng = np.random.default_rng(2)
    _heads = _rng.random(coin_N.value) < coin_p.value
    _n = np.arange(1, coin_N.value + 1)
    _fraction = np.cumsum(_heads) / _n

    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.plot(_n, _fraction, color="steelblue", lw=1.5, label="Fraction of heads")
    _ax.axhline(coin_p.value, ls="--", color="crimson", label="$p$")
    _ax.set(xscale="log", xlabel="Number of tosses", ylabel="Fraction of heads", ylim=(-0.05, 1.05))
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([coin_p, coin_N], justify="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.stat(label="Fraction of heads", value=f"{_fraction[-1]:.4f}", caption=f"expected {coin_p.value:.2f}"),
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
    ### Example: radioactive decay

    Some physical processes are truly random (recall quantum mechanics), for instance radioactive decay.
    The number of radioactive isotopes with a half-life $\tau$ evolves as
    $$
    N(t) = N(0)\, 2^{-t/\tau},
    $$
    therefore the probability for a single atom to decay during a time interval $h$ is
    $$
    p = 1 - 2^{-h/\tau}.
    $$
    Each atom is a biased coin: at every time step we toss one coin per remaining atom.

    Let us simulate a sample of thallium-208 atoms (half-life $\tau = 3.053$ minutes) decaying into lead-208.
    """)
    return


@app.cell
def _(np):
    def simulate_decay(N0, tau, h, tmax, rng):
        """Simulate radioactive decay with time step h. Returns times, N_Tl(t), N_Pb(t)."""
        p = 1 - 2 ** (-h / tau)
        t = np.arange(0.0, tmax, h)
        NTl = np.zeros(len(t), dtype=int)
        NPb = np.zeros(len(t), dtype=int)
        n_tl, n_pb = N0, 0
        for i in range(len(t)):
            NTl[i], NPb[i] = n_tl, n_pb
            decays = np.sum(rng.random(n_tl) < p)  # one biased coin per atom
            n_tl -= decays
            n_pb += decays
        return t, NTl, NPb

    return (simulate_decay,)


@app.cell(hide_code=True)
def _(mo):
    decay_N0 = mo.ui.dropdown(options=["10", "100", "1000", "10000"], value="100", label="Initial number of Tl atoms")
    return (decay_N0,)


@app.cell(hide_code=True)
def _(decay_N0, mo, np, plt, simulate_decay):
    _N0 = int(decay_N0.value)
    _tau = 3.053 * 60
    _t, _NTl, _NPb = simulate_decay(_N0, _tau, h=1.0, tmax=1000.0, rng=np.random.default_rng(3))

    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(_t, _NTl, color="#1d4ed8", label=r"$N_{\rm Tl}$")
    _ax.plot(_t, _NPb, color="crimson", label=r"$N_{\rm Pb}$")
    _ax.plot(_t, _N0 * 2 ** (-_t / _tau), "--", color="black", lw=1, label=r"Expected $N_{\rm Tl}$")
    _ax.set(xlabel="Time [s]", ylabel="Number of atoms")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([decay_N0, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a few atoms the evolution is very noisy; increasing the number of atoms makes the relative fluctuations smaller (they scale as $1/\sqrt{N}$) and the curve approaches the exponential law.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _md1 = mo.md(r"""### Exercise: radioactive decay
    This exercise invites you to create a more advanced version of the simple radioactive decay simulation in the previous Example.

    The isotope $^{213}$Bi decays to stable $^{209}$Bi via one of two different routes, with probabilities and half-lives as shown in the figure.

    (Technically, $^{209}$Bi is not actually stable, but it has a half-life of more than $10^{19}$ years, a billion times the age of the universe, so it might as well be.)  

    Starting with a sample consisting of $10\,000$ atoms of $^{213}$Bi, simulate the decay of the atoms by dividing time into slices of length $\delta t=1\,$s each and on each step doing the following:

    - For each atom of $^{209}$Pb in turn, decide at random, with the appropriate probability, whether it decays or not.  The probability can be calculated from second equation in the previous example.  Count the total number of atoms that decay, subtract it from the current number of $^{209}$Pb atoms, and add it to the number of $^{209}$Bi atoms.
    - Now do the same for $^{209}$Tl, subtracting the number of atoms that decay from the total for $^{209}$Tl and adding it to the total for $^{209}$Pb.
    - For $^{213}$Bi the situation is more complicated.  When a $^{213}$Bi atom decays you also have to decide at random with the appropriate probability the route by which it decays.  Count the number that decay by each route and add and subtract accordingly.

    Note that you should work up the chain from the bottom like this, not down from the top, to avoid inadvertently making the same atom decay twice on a single step.

    Keep track of the number of atoms of each of the four isotopes at all times for $20\,000$ seconds and make a single graph showing the four numbers as a function of time on the same axes.""")

    _image = mo.image("images/decaychain.png", width=300,)

    mo.hstack([_md1,_image],widths=[0.6,0.4],align="start",gap=2)
   
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Nonuniform random numbers
    ---

    In many cases we deal with random numbers $\xi$ that are distributed non-uniformly.
    Common examples are:

    - Exponential distribution $\rho(x) = e^{-x}$.
    - Gaussian distribution $\rho(x) \propto e^{-\frac{x^2}{2\sigma^2}}$.
    - Power-law distribution $\rho(x) \propto x^{\alpha}$.
    - Arbitrary peaked distributions.

    There are two common methods for generating nonuniform random variates, *inverse transform sampling* and *rejection sampling*. Both make use of uniformly distributed variates.

    ### Inverse transform sampling

    If $\eta$ is a uniformly distributed random variable, some function of it, $\xi = f(\eta)$, is not. The idea is to sample $\eta$ and calculate $\xi$ via a function chosen such that $\xi$ follows the desired probability density $\rho(\xi)$. How to find $f(\eta)$?

    Without loss of generality assume that $\xi \in (-\infty, \infty)$ and that $f$ is increasing. Consider the cumulative distribution function
    $$
    G(x) = \Pr(\xi < x) = \int_{-\infty}^x \rho(\xi)\, d\xi.
    $$
    It equals the probability that $\eta < y$, where $x = f(y)$. Since $\eta$ is uniformly distributed, this probability is $y$. Therefore
    $$
    G[f(y)] = y, \qquad \text{thus} \qquad f(y) = G^{-1}(y).
    $$

    The algorithm is the following:

    1. Calculate the cumulative distribution $G(x) = \int_{-\infty}^x \rho(\xi)\, d\xi$.
    2. Find the inverse function $G^{-1}(y)$ as the solution of $G(x) = y$ with respect to $x$.
    3. Sample uniformly distributed random variables $\eta$ and calculate $\xi = G^{-1}(\eta)$.

    Sometimes evaluating $G(x)$ and/or $G^{-1}(y)$ explicitly is challenging.
    In such cases one resorts to numerical integration and/or nonlinear equation solvers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: exponential distribution

    Recall radioactive decay. The time of decay of an atom with mean lifetime $\tau$ is distributed as
    $$
    \rho(t) = \frac{1}{\tau} e^{-t/\tau}.
    $$
    The cumulative distribution function reads
    $$
    G(t) = \int_0^{t} \frac{1}{\tau} e^{-t'/\tau}\, dt' = 1 - e^{-t/\tau}.
    $$
    Solving $1 - e^{-t/\tau} = \eta$ gives
    $$
    t(\eta) = -\tau \ln(1-\eta).
    $$
    """)
    return


@app.cell
def _(np):
    def sample_tdecay(tau, rng):
        """Sample a decay time from rho(t) = exp(-t/tau)/tau by inverse transform."""
        eta = rng.random()
        return -tau * np.log(1 - eta)

    return (sample_tdecay,)


@app.cell(hide_code=True)
def _(mo):
    exp_N = mo.ui.slider(100, 20000, step=100, value=10000, label="Number of samples $N$")
    return (exp_N,)


@app.cell(hide_code=True)
def _(exp_N, mo, np, plt, sample_tdecay):
    _tau = 3.053
    _rng = np.random.default_rng(4)
    _tdecays = [sample_tdecay(_tau, _rng) for _ in range(exp_N.value)]
    _t = np.linspace(0, 20, 200)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_tdecays, bins=40, range=(0, 20), density=True, color="steelblue", alpha=0.7, label="Sample")
    _ax.plot(_t, np.exp(-_t / _tau) / _tau, color="crimson", lw=2, label=r"$\rho(t)$")
    _ax.set(xlabel="$t$ [minutes]", ylabel="Probability density")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([exp_N, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cool YouTube video explaining Inverse Transform Sampling by ritvikmath: [Link](https://youtu.be/9ixzzPQWuAY?si=Ci97p_aFY_TBiAHT)
    """).callout("info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: points inside a circle

    Let us sample points uniformly inside the unit circle.
    One way to do that is to switch to polar coordinates
    $$
    x = r \cos\phi, \qquad y = r \sin\phi,
    $$
    and sample $r \in [0,1)$ and $\phi \in [0,2\pi)$.
    Naively one could sample $r$ and $\phi$ independently from two uniform distributions. Select *naive* below to see what happens.

    The points clump in the centre! Why? Because $r$ is not uniformly distributed. Recall
    $$
    dx\, dy = r\, dr\, d\phi,
    $$
    therefore
    $$
    \rho_r(r) = 2r, \qquad \rho_\phi(\phi) = \frac{1}{2\pi}.
    $$
    The cumulative distribution function is $G_r(r) = \int_0^r 2r'\, dr' = r^2$, and solving $G_r(r) = \eta$ gives
    $$
    r = \sqrt{\eta}.
    $$
    """)
    return


@app.cell
def _(np):
    def sample_disk(N, rng, correct=True):
        """Sample N points in the unit disk; correct=False uses a (wrong) uniform r."""
        eta = rng.random(N)
        r = np.sqrt(eta) if correct else eta
        phi = 2 * np.pi * rng.random(N)
        return r * np.cos(phi), r * np.sin(phi)

    return (sample_disk,)


@app.cell(hide_code=True)
def _(mo):
    disk_method = mo.ui.radio(options=["naive: r = η", "correct: r = √η"], value="naive: r = η", label="Sampling of $r$", inline=True)
    return (disk_method,)


@app.cell(hide_code=True)
def _(disk_method, mo, np, plt, sample_disk):
    _correct = disk_method.value.startswith("correct")
    _x, _y = sample_disk(2000, np.random.default_rng(5), correct=_correct)
    _r = np.sqrt(_x**2 + _y**2)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    _ax1.plot(_x, _y, "o", ms=2.5, color="steelblue")
    _ax1.add_patch(plt.Circle((0, 0), 1, fill=False, color="crimson"))
    _ax1.set(xlabel="x", ylabel="y", title="Sampled points")
    _ax1.set_aspect("equal")
    _rr = np.linspace(0, 1, 100)
    _ax2.hist(_r, bins=30, range=(0, 1), density=True, color="steelblue", alpha=0.7, label="Sample")
    _ax2.plot(_rr, 2 * _rr, color="crimson", lw=2, label=r"$\rho_r(r) = 2r$ (uniform disk)")
    _ax2.set(xlabel="$r$", ylabel="Probability density", title="Distribution of $r$")
    _ax2.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([disk_method, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: isotropic directions (points on a unit sphere)

    A common task in Monte Carlo simulations is sampling an isotropic direction in 3D space,
    e.g. the random orientation of an axially symmetric object (a rod) or the momentum of a particle.

    This is equivalent to choosing a random point on the unit sphere, parametrized by the azimuthal and polar angles $\phi \in [0,2\pi)$ and $\theta \in [0,\pi]$:
    $$
    x = \sin\theta \cos\phi,\qquad
    y = \sin\theta \sin\phi,\qquad
    z = \cos\theta.
    $$
    Since $d\Omega = \sin\theta\, d\theta\, d\phi$, the variables $\phi$ and $\theta$ are independent.
    $\phi$ is uniform in $[0,2\pi)$, but the polar angle has the density
    $$
    \rho_{\theta} (\theta) = \frac{1}{2} \sin\theta,
    \qquad
    G_{\theta}(\theta) = \int_0^\theta \frac{1}{2} \sin\theta'\, d \theta' = \frac{1-\cos\theta}{2},
    $$
    thus $\theta = \arccos(2\eta - 1)$.
    In practice it is convenient to work directly with
    $$
    \cos\theta = 2 \eta - 1, \qquad \sin\theta = \sqrt{1 - \cos^2\theta}.
    $$
    Sampling $\theta$ uniformly instead would crowd the points near the poles.
    """)
    return


@app.cell
def _(np):
    def sample_isotropic(N, rng, correct=True):
        """Sample N unit vectors; correct=False uses a (wrong) uniform polar angle."""
        phi = 2 * np.pi * rng.random(N)
        costh = 2 * rng.random(N) - 1 if correct else np.cos(np.pi * rng.random(N))
        sinth = np.sqrt(1 - costh**2)
        return sinth * np.cos(phi), sinth * np.sin(phi), costh

    return (sample_isotropic,)


@app.cell(hide_code=True)
def _(mo):
    sphere_method = mo.ui.radio(
        options=["naive: θ uniform", "correct: cos θ uniform"], value="correct: cos θ uniform", label="Sampling of $\\theta$", inline=True
    )
    return (sphere_method,)


@app.cell(hide_code=True)
def _(go, mo, np, sample_isotropic, sphere_method):
    _x, _y, _z = sample_isotropic(1500, np.random.default_rng(6), correct=sphere_method.value.startswith("correct"))
    _fig = go.Figure(go.Scatter3d(x=_x, y=_y, z=_z, mode="markers", marker=dict(size=2.5, color=_z, colorscale="Viridis")))
    _fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    mo.vstack([sphere_method, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    _md1 = mo.md(r"""### Exercise: radioactive decay again
    Redo the calculation from the previous exercise, but this time using the
    faster method described in the preceding section.  Using the transformation
    method, generate 1000 random numbers from the nonuniform distribution decribe above to represent the times of decay of 1000 atoms of $^{208}$Tl
    (which has half-life 3.053 minutes).  Then make a plot showing the number
    of atoms that have not decayed as a function of time, i.e., a plot as a
    function of $t$ showing the number of atoms whose chosen decay times are
    greater than $t$.

    **Hint:** You may find it useful to know that the package `numpy` contains a function `sort` that will rearrange the elements of an array in increasing order. That is, `b = numpy.sort(a)` returns a new array `b` containing the same numbers as `a`, but rearranged in order from smallest to largest.
    """)

    _md1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rejection sampling

    In the rejection sampling method one samples a candidate value from an envelope distribution and accepts it with a certain probability.

    Consider again the distribution of the polar angle, $\rho_\theta(\theta) = \sin(\theta)/2$, which is bounded from above by $\rho^{\rm max} = 1/2$.
    Rejection sampling proceeds by

    1. Sampling a candidate value $\theta_{\rm cand}$ from a uniform distribution over $(0,\pi)$.
    2. Accepting $\theta_{\rm cand}$ with probability $p = \rho_\theta(\theta_{\rm cand}) / \rho^{\rm max}$. Otherwise, go back to 1.

    The second step is a biased coin (Section 4): sample $y$ uniformly over $(0,\rho^{\rm max})$ and accept $\theta_{\rm cand}$ if $y < \rho_\theta(\theta_{\rm cand})$.

    The procedure has a simple geometrical interpretation. Taking $\theta_{\rm cand}$ and $y$ as the coordinates of a point in the plane, we accept the points that lie below the curve $\rho_{\theta}$. This ensures that $\theta_{\rm cand}$ is accepted at a rate proportional to $\rho_\theta(\theta)$, as desired.
    One advantage of rejection sampling is that $\rho_\theta(\theta)$ need not be normalized.
    """)
    return


@app.function
def sample_rejection(rho, a, b, rhomax, rng):
    """Sample x in (a,b) from the (unnormalized) density rho <= rhomax by rejection.

    Returns the sample and the number of candidates that were tried.
    """
    tries = 0
    while True:
        tries += 1
        x_cand = a + (b - a) * rng.random()
        y = rhomax * rng.random()
        if y < rho(x_cand):
            return x_cand, tries


@app.cell(hide_code=True)
def _(mo):
    rej_rhomax = mo.ui.slider(0.5, 3.0, step=0.1, value=0.5, label=r"Envelope height $\rho^{\rm max}$")
    return (rej_rhomax,)


@app.cell(hide_code=True)
def _(mo, np, plt, rej_rhomax):
    def _rho_theta(theta):
        return np.sin(theta) / 2

    _rng = np.random.default_rng(7)
    _N = 20000
    _res = [sample_rejection(_rho_theta, 0.0, np.pi, rej_rhomax.value, _rng) for _ in range(_N)]
    _samples = np.array([s for s, _ in _res])
    _tries = sum(n for _, n in _res)

    # a few candidates, to illustrate accepted / rejected points
    _xc = np.pi * _rng.random(600)
    _yc = rej_rhomax.value * _rng.random(600)
    _acc = _yc < _rho_theta(_xc)

    _th = np.linspace(0, np.pi, 200)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    _ax1.plot(_xc[_acc], _yc[_acc], "o", ms=2.5, color="seagreen", label="accepted")
    _ax1.plot(_xc[~_acc], _yc[~_acc], "o", ms=2.5, color="0.6", label="rejected")
    _ax1.plot(_th, _rho_theta(_th), color="crimson", lw=2)
    _ax1.axhline(rej_rhomax.value, ls="--", color="black", lw=1)
    _ax1.set(xlabel=r"$\theta_{\rm cand}$", ylabel="$y$", title="Candidates")
    _ax1.legend(loc="upper right")
    _ax2.hist(_samples, bins=40, range=(0, np.pi), density=True, color="steelblue", alpha=0.7, label="Sample")
    _ax2.plot(_th, _rho_theta(_th), color="crimson", lw=2, label=r"$\rho_\theta$")
    _ax2.set(xlabel=r"$\theta$", ylabel="Probability density", title="Accepted samples")
    _ax2.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            rej_rhomax,
            mo.hstack(
                [_fig, mo.stat(label="Acceptance rate", value=f"{_N / _tries:.3f}", caption=f"expected 1/(π ρmax) = {1 / (np.pi * rej_rhomax.value):.3f}")],
                widths=[0.8, 0.2],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The histogram is correct for any envelope height $\rho^{\rm max}$ above the true maximum, but the acceptance rate drops as the envelope gets higher.

    Pros of rejection sampling:

    - The distribution does not need to be normalized.
    - It still works if $\rho^{\rm max}$ is larger than the true maximum of $\rho(x)$.
    - It works for generic distributions and does not require the cumulative distribution function.

    Cons:

    - It can be inefficient if the rejection rate is very high (highly peaked distributions).
    - It is not directly applicable to distributions over infinite ranges.

    Generalizations take care of some of these deficiencies: adaptive rejection sampling with several enveloping rectangles, a variable transformation mapping an infinite interval onto a finite one, or sampling from a non-uniform envelope distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cool YouTube video explaining Accept-Reject Sampling by ritvikmath: [Link](https://youtu.be/OXDqjdVVePY?si=Gk92IK1InRPuYRhq)
    """).callout("info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Gaussian random numbers
    ---

    One of the most common distributions is the normal (or Gaussian) distribution
    $$
    \rho(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
    $$
    With the change of variable $x \to \mu + \sigma x$ it suffices to sample the standard normal distribution
    $$
    \rho(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}.
    $$
    Its cumulative distribution function cannot be inverted analytically.
    Instead of one variable, consider a pair of independent normally distributed variables $x,y$:
    $$
    \rho(x,y)\, dx\, dy = \frac{1}{2\pi} e^{-\frac{x^2+y^2}{2}}\, dx\, dy.
    $$
    In polar coordinates, $x = r \cos\phi$, $y = r \sin\phi$ and $dx\, dy = r\, dr\, d\phi$, so that
    $$
    \rho(r,\phi) = \frac{1}{2\pi}\, r e^{-r^2/2}.
    $$
    Hence $r$ and $\phi$ are independent: $\phi$ is uniform in $[0,2\pi)$, while $r$ has the density and cumulative distribution
    $$
    \rho_r(r) = r e^{-r^2/2}, \qquad G_r(r) = \int_0^r r' e^{-r'^2/2} dr' = 1 - e^{-r^2/2},
    $$
    which we invert:
    $$
    r = \sqrt{-2 \ln(1-\eta)}.
    $$
    This is the **Box–Muller** method: two uniform numbers give two independent Gaussian numbers.
    """)
    return


@app.cell
def _(np):
    def box_muller(N, rng):
        """Return 2N independent standard normal numbers with the Box-Muller method."""
        phi = 2 * np.pi * rng.random(N)
        r = np.sqrt(-2 * np.log(1 - rng.random(N)))
        return np.concatenate([r * np.cos(phi), r * np.sin(phi)])

    return (box_muller,)


@app.cell(hide_code=True)
def _(mo):
    gauss_mu = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.0, label=r"Mean $\mu$")
    gauss_sigma = mo.ui.slider(0.2, 2.0, step=0.1, value=1.0, label=r"Standard deviation $\sigma$")
    return gauss_mu, gauss_sigma


@app.cell(hide_code=True)
def _(box_muller, gauss_mu, gauss_sigma, mo, np, plt):
    _mu, _sigma = gauss_mu.value, gauss_sigma.value
    _samples = _mu + _sigma * box_muller(5000, np.random.default_rng(8))
    _x = np.linspace(-6, 6, 300)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_samples, bins=60, range=(-6, 6), density=True, color="steelblue", alpha=0.7, label="Box–Muller sample")
    _ax.plot(_x, np.exp(-((_x - _mu) ** 2) / (2 * _sigma**2)) / (np.sqrt(2 * np.pi) * _sigma), color="crimson", lw=2, label=r"$\rho(x)$")
    _ax.set(xlabel="$x$", ylabel="Probability density", title="Normal distribution")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([gauss_mu, gauss_sigma], justify="center", gap=2),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Sample mean", value=f"{_samples.mean():.3f}"),
                            mo.stat(label="Sample std", value=f"{_samples.std():.3f}"),
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
    In practice, use the library routine `rng.normal(mu, sigma, size)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _md1 = mo.md(r"""
    ### Exercise: Rutherford scattering

    At the beginning of the 20th century, Ernest Rutherford and his collaborators showed that when an $\alpha$ particle, i.e., a helium nucleus of two protons and two neutrons, passes close to an atom as shown in the figure, it is scattered, primarily by electrical repulsion from the positively charged nucleus of the atom, and that the angle $\theta$ through which it is scattered obeys

    $$
    \tan \frac{1}{2}\theta = \frac{Ze^2}{2\pi\epsilon_0 E b}, 
    $$

    """)

    _image = mo.image("images/rutherford_scattering.png", width=400)

    _md2 = mo.md(r"""
    where $Z$ is the atomic number of the nucleus, $e$ is the electron charge, $\epsilon_0$ is the permittivity of free space, $E$ is the kinetic energy of the incident $\alpha$ particle, and $b$ is the impact parameter, i.e., the perpendicular distance between the particle's initial trajectory and the axis running through the nucleus.

    Consider a beam of $\alpha$ particles with energy $7.7\text{ MeV}$ that has a Gaussian profile in both its $x$ and $y$ axes with standard deviation $\sigma = a_0/100$, where $a_0$ is the Bohr radius. The beam is fired directly at a gold atom. Let us simulate the scattering process for one million $\alpha$ particles and calculate the fraction of particles that "bounce back" on scattering, i.e., that scatter through angles greater than $90^\circ$. The threshold value of the impact parameter for which this happens can be calculated from Eq. (10.19) with $\theta = 90^\circ$, which gives

    $$
    b = \frac{Ze^2}{2\pi\epsilon_0 E}.
    $$

    If $b$ is less than this value then the particle bounces back. 

    Write a program to do the calculation and report the fraction of particles that bounce back. 
    """)

    mo.vstack(
        [
            _md1, 
            _image,
            _md2,
        ],
        # align="center",
    )
    return


if __name__ == "__main__":
    app.run()
