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
    return PlaySlider, go, mo, np, plt


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
    return (method_colors,)


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
    ### Example: Rutherford scattering

    At the beginning of the 20th century, Ernest Rutherford and his collaborators showed that when an $\alpha$ particle, i.e., a helium nucleus of two protons and two neutrons, passes close to an atom as shown in the figure, it is scattered, primarily by electrical repulsion from the positively charged nucleus of the atom, and that the angle $\theta$ through which it is scattered obeys

    $$
    \tan \frac{1}{2}\theta = \frac{Ze^2}{2\pi\epsilon_0 E b}, 
    $$

    """)

    _image = mo.image("images/rutherford_scattering.png", width=400)

    _md2 = mo.md(r"""
    where $Z$ is the atomic number of the nucleus, $e$ is the electron charge, $\epsilon_0$ is the permittivity of free space, $E$ is the kinetic energy of the incident $\alpha$ particle, and $b$ is the impact parameter, i.e., the perpendicular distance between the particle's initial trajectory and the axis running through the nucleus.

    Consider a beam of $\alpha$ particles with energy $7.7\text{ MeV}$ that has a Gaussian profile in both its $x$ and $y$ axes with standard deviation $\sigma = a_0/100$, where $a_0$ is the Bohr radius. The beam is fired directly at a gold atom. Let us simulate the scattering process for one million $\alpha$ particles and calculate the fraction of particles that "bounce back" on scattering, i.e., that scatter through angles greater than $90^\circ$. The threshold value of the impact parameter for which this happens can be calculated from the previous equation with $\theta = 90^\circ$, which gives

    $$
    b = \frac{Ze^2}{2\pi\epsilon_0 E}.
    $$

    If $b$ is less than this value then the particle bounces back. 

    Write a program to do the calculation and report the fraction of particles that bounce back. 
    """)

    _md3 = mo.md(r"""YouTube videos:  
    [The Discovery of the Atomic Nucleus](https://www.youtube.com/watch?v=wzALbzTdnc8)  
    [The Alpha Scattering Experiment - A Level Physics](https://youtu.be/mWbStD9e2hQ?si=ia5pAARpXg7D5ckN)""").callout("info")
    mo.vstack(
        [
            _md1, 
            _image,
            _md2,
            _md3
        ],
        # align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Monte Carlo integration
    ---

    Recall the interpretation of a definite integral as the area under the curve.
    We can use this interpretation to approximate integrals with random numbers.
    Consider
    $$
    I = \int_a^b f(x)\, dx,
    $$
    where $0 \leq f(x) \leq y_{\rm max}$ on $[a,b]$.
    We sample $N$ points uniformly in the enveloping rectangle $[a,b]\times[0,y_{\rm max}]$ and count the number $C$ of points that fall under the curve $f(x)$. The fraction $C/N$ estimates the ratio of the area under the curve to the area of the rectangle:
    $$
    I \approx (b-a)\,y_{\rm max} \frac{C}{N}.
    $$
    This is the **hit-or-miss** method. Each point is a biased coin with probability $p = I/[(b-a)y_{\rm max}]$, so $C$ follows the binomial distribution, and the statistical error is
    $$
    \delta I = (b-a)\, y_{\rm max} \sqrt{\frac{p(1-p)}{N}}, \qquad p \approx \frac{C}{N}.
    $$
    As an example take
    $$
    I = \int_0^\pi \sin x\, dx = 2.
    $$
    """)
    return


@app.cell
def _(np):
    def hit_or_miss(f, N, a, b, ymax, rng):
        """Hit-or-miss Monte Carlo estimate of int_a^b f(x) dx, with 0 <= f <= ymax.

        Returns the estimate, its statistical error and the sampled points with a mask of the hits.
        """
        x = a + (b - a) * rng.random(N)
        y = ymax * rng.random(N)
        hits = y < f(x)
        p = hits.mean()
        area = (b - a) * ymax
        return area * p, area * np.sqrt(p * (1 - p) / N), x, y, hits

    return (hit_or_miss,)


@app.cell(hide_code=True)
def _(mo):
    hm_N = mo.ui.slider(50, 5000, step=50, value=1000, label="Number of points $N$")
    return (hm_N,)


@app.cell(hide_code=True)
def _(hit_or_miss, hm_N, mo, np, plt):
    _I, _err, _x, _y, _hits = hit_or_miss(np.sin, hm_N.value, 0, np.pi, 1.0, np.random.default_rng(9))
    _xx = np.linspace(0, np.pi, 200)

    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(_x[_hits], _y[_hits], "o", ms=2.5, color="darkorange", label="under the curve")
    _ax.plot(_x[~_hits], _y[~_hits], "o", ms=2.5, color="0.6", label="above the curve")
    _ax.plot(_xx, np.sin(_xx), color="crimson", lw=2, label=r"$\sin x$")
    _ax.set(xlim=(0, np.pi), ylim=(0, 1), xlabel="$x$", ylabel="$y$")
    _ax.legend(loc="upper right")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            hm_N,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Estimate", value=f"{_I:.4f} ± {_err:.4f}", caption="exact: 2"),
                            mo.stat(label="Actual error", value=f"{abs(_I - 2):.4f}"),
                        ]
                    ),
                ],
                widths=[0.75, 0.25],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The error decreases only as $1/\sqrt{N}$: every additional correct digit requires 100 times more points.
    In one dimension this is much worse than the quadrature rules of the numerical integration notebook (e.g. Simpson's rule has an error $\propto 1/N^4$).
    The strength of Monte Carlo methods lies elsewhere, as we will see in Section 9.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. The mean value method
    ---

    The integral
    $$
    I = \int_a^b f(x)\, dx
    $$
    is related to the mean value $\langle f \rangle$ of $f$ over $(a,b)$:
    $$
    \langle f \rangle = \frac{1}{b-a}\int_a^b f(x)\, dx = \frac{I}{b-a}, \qquad\text{so that}\qquad I = (b-a) \langle f \rangle.
    $$
    The integral can therefore be estimated by evaluating $\langle f \rangle$ as the average of $f(x_i)$ over points $x_i$ sampled uniformly in $(a,b)$:
    $$
    \langle f \rangle \approx \frac{1}{N} \sum_{i=1}^N f(x_i).
    $$
    The error estimate follows from the variance of $f$, which requires $\langle f^2 \rangle$ as well:
    $$
    \delta I = (b-a) \sqrt{\frac{\langle f^2 \rangle - \langle f \rangle^2}{N}}.
    $$
    Unlike hit-or-miss, the mean value method needs neither $f\geq 0$ nor an upper bound $y_{\rm max}$, and it is always at least as accurate.
    """)
    return


@app.cell
def _(np):
    def int_mc(f, N, a, b, rng):
        """Mean value Monte Carlo estimate of int_a^b f(x) dx and its statistical error."""
        x = a + (b - a) * rng.random(N)
        fval = f(x)
        f_av = fval.mean()
        fsq_av = (fval**2).mean()
        return (b - a) * f_av, (b - a) * np.sqrt((fsq_av - f_av**2) / N)

    return (int_mc,)


@app.cell(hide_code=True)
def _(hit_or_miss, int_mc, method_colors, mo, np, plt):
    _Ns = np.unique(np.logspace(1, 6, 30).astype(int))
    _rng = np.random.default_rng(11)
    _hm = np.array([hit_or_miss(np.sin, n, 0, np.pi, 1.0, _rng)[:2] for n in _Ns])
    _mv = np.array([int_mc(np.sin, n, 0, np.pi, _rng) for n in _Ns])

    _fig, _ax = plt.subplots(figsize=(8, 4.4))
    for _res, _name in [(_hm, "Hit-or-miss"), (_mv, "Mean value")]:
        _ax.plot(_Ns, np.abs(_res[:, 0] - 2), "o", ms=4, color=method_colors[_name], alpha=0.6, label=f"{_name}: actual error")
        _ax.plot(_Ns, _res[:, 1], "-", color=method_colors[_name], lw=2, label=f"{_name}: estimated $\\delta I$")
    _ax.plot(_Ns, 1 / np.sqrt(_Ns), "--", color="black", lw=1, label=r"$\propto 1/\sqrt{N}$")
    _ax.set(xscale="log", yscale="log", xlabel="$N$", ylabel="Error", title=r"$\int_0^\pi \sin x\, dx$")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md("Both methods converge as $1/\\sqrt{N}$, but the mean value method has a smaller prefactor:"),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Integrals in many dimensions
    ---

    Monte Carlo methods really shine for integrals in many dimensions.
    Consider the $D$-dimensional integral
    $$
    I = \int_{a_1}^{b_1} d x_1 \ldots \int_{a_D}^{b_D} d x_D\, f(x_1,\ldots,x_D).
    $$
    The rectangle rule turns it into a $D$-dimensional sum
    $$
    I \approx \sum_{k_1 = 1}^{N_1} \ldots \sum_{k_D = 1}^{N_D} f(x_{k_1},\ldots,x_{k_D}) \prod_{d=1}^D h_d,
    $$
    with $h_d = (b_d - a_d)/N_d$ and $x_{k_d} = a_d + h_d (k_d - 1/2)$.
    The total number of integrand evaluations is $N_{\rm tot} = \prod_{d=1}^D N_d$. With the same number $N$ of points in each dimension, it grows exponentially with $D$,
    $$
    N_{\rm tot} = N^D,
    $$
    which makes the method impractical in many dimensions: the **curse of dimensionality**.
    For a fixed budget of $N_{\rm tot}$ evaluations, the error of the rectangle rule scales as $h^2 \propto N_{\rm tot}^{-2/D}$.

    Monte Carlo methods are not plagued by this curse. The integral is the mean value of the integrand in the $D$-dimensional box,
    $$
    I = \langle f(x_1,\ldots,x_D) \rangle \prod_{d=1}^D (b_d-a_d),
    $$
    where $x_1,\ldots,x_D$ are independent random variables uniformly distributed in $[a_d,b_d]$.
    Adding one dimension only means sampling one more number per point. The error estimate,
    $$
    \delta I = \sqrt{\frac{\langle f^2 \rangle - \langle f \rangle^2}{N}} \, \prod_{d=1}^D (b_d-a_d),
    $$
    still scales as $1/\sqrt{N}$, **whatever the dimension**. Monte Carlo therefore wins beyond $D \approx 4$.
    """)
    return


@app.cell
def _(np):
    def rectangle_rule_multi(f, N, a, b):
        """Composite rectangle rule in D = len(a) dimensions with N slices per dimension.

        f takes an array of shape (D, ...) holding the coordinates of the points.
        """
        h = [(bd - ad) / N for ad, bd in zip(a, b)]
        axes = [ad + hd * (np.arange(N) + 0.5) for ad, hd in zip(a, h)]
        grid = np.array(np.meshgrid(*axes, indexing="ij"))
        return np.sum(f(grid)) * np.prod(h)

    def int_mc_multi(f, N, a, b, rng):
        """Mean value Monte Carlo estimate of a D-dimensional integral over the box [a,b]."""
        a, b = np.asarray(a), np.asarray(b)
        x = a[:, None] + (b - a)[:, None] * rng.random((len(a), N))
        fval = f(x)
        vol = np.prod(b - a)
        f_av, fsq_av = fval.mean(), (fval**2).mean()
        return vol * f_av, vol * np.sqrt((fsq_av - f_av**2) / N)

    return int_mc_multi, rectangle_rule_multi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: $\int \sin(x_1+\ldots+x_D)$

    Let us consider
    $$
    I_D = \int_0^{\pi/2} d x_1 \ldots \int_{0}^{\pi/2} d x_D \, \sin(x_1 + x_2 + \ldots + x_D).
    $$
    Integrating over one variable after another, one finds $I_D = 2^{D/2}\sin(D\pi/4)$:

    | $D$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
    |---|---|---|---|---|---|---|---|---|
    | $I_D$ | 1 | 2 | 2 | 0 | −4 | −8 | −8 | 0 |

    We compare the rectangle rule and Monte Carlo for the **same budget** $N_{\rm tot}$ of function evaluations: the rectangle rule uses $N = N_{\rm tot}^{1/D}$ slices per dimension (rounded down).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    multi_budget = mo.ui.dropdown(options={"10⁴": 10**4, "10⁵": 10**5, "10⁶": 10**6}, value="10⁵", label="Budget $N_{\\rm tot}$")
    return (multi_budget,)


@app.cell(hide_code=True)
def _(
    int_mc_multi,
    method_colors,
    mo,
    multi_budget,
    np,
    plt,
    rectangle_rule_multi,
):
    def _f_sinsum(x):
        return np.sin(np.sum(x, axis=0))

    _Ntot = multi_budget.value
    _rng = np.random.default_rng(13)
    _rows = []
    for _D in range(1, 9):
        _a, _b = [0.0] * _D, [np.pi / 2] * _D
        _exact = 2 ** (_D / 2) * np.sin(_D * np.pi / 4)
        _Nrect = int(np.floor(_Ntot ** (1 / _D) + 1e-9))
        _Irect = rectangle_rule_multi(_f_sinsum, _Nrect, _a, _b)
        _Imc, _err = int_mc_multi(_f_sinsum, _Ntot, _a, _b, _rng)
        _rows.append(
            {
                "D": _D,
                "exact": round(_exact, 6),
                "N per dim (rectangle)": _Nrect,
                "rectangle": round(_Irect, 6),
                "rectangle error": float(f"{abs(_Irect - _exact):.2e}"),
                "Monte Carlo": f"{_Imc:.4f} ± {_err:.4f}",
                "MC error": float(f"{abs(_Imc - _exact):.2e}"),
            }
        )

    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _Ds = [r["D"] for r in _rows]
    _ax.semilogy(_Ds, [r["rectangle error"] for r in _rows], "o-", color=method_colors["Rectangle"], label="Rectangle rule")
    _ax.semilogy(_Ds, [r["MC error"] for r in _rows], "s-", color=method_colors["Mean value"], label="Monte Carlo")
    _ax.set(xlabel="Dimension $D$", ylabel="Actual error", title=f"Same budget: $N_{{\\rm tot}}$ = {_Ntot:.0e}")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([multi_budget, mo.ui.table(_rows, selection=None, pagination=False), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In low dimension the rectangle rule is far more accurate, but its error grows quickly with $D$, as fewer and fewer slices per dimension fit in the budget. The Monte Carlo error stays of the same order in every dimension.
    """)
    return


@app.cell
def _(np):
    def ball_volume_rect(D, R, nrect):
        """Volume of a D-dimensional ball from the recursion relation and the rectangle rule."""
        if D == 0:
            return 1.0
        h = 2.0 / nrect
        t = -1.0 + h * (np.arange(nrect) + 0.5)
        return h * R**D * sum(ball_volume_rect(D - 1, np.sqrt(1 - tk**2), nrect) for tk in t)

    def ball_volume_mc(D, R, N, rng):
        """Hit-or-miss Monte Carlo volume of a D-dimensional ball. Returns volume, error, fraction inside."""
        if D == 0:
            return 1.0, 0.0, 1.0
        x = -R + 2 * R * rng.random((N, D))
        p = np.mean(np.sum(x**2, axis=1) < R**2)
        return (2 * R) ** D * p, (2 * R) ** D * np.sqrt(p * (1 - p) / N), p

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Importance sampling
    ---

    Recall the mean value method,
    $$
    I = \int_a^b f(x)\, dx = (b - a) \langle f \rangle, \qquad \langle f \rangle \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \qquad x_i \in U(a,b).
    $$
    When $f(x)$ is highly peaked, only a small part of the interval $(a,b)$ contributes to the integral, and sampling $x$ with equal probability everywhere is wasteful. Similarly, when $f(x)$ has an integrable singularity the method still works, but the statistical uncertainty converges slowly (or $\langle f^2\rangle$ may even be infinite).

    **Importance sampling** improves on this. Let $w(x) > 0$ be a normalized probability density on $(a,b)$, $\int_a^b w(x)\, dx = 1$. The mean value of a function $g(x)$ of a random variable $x$ distributed according to $w(x)$ is
    $$
    \langle g \rangle_w = \int_a^b g(x) w(x)\, dx.
    $$
    Dividing and multiplying the integrand by $w(x)$ we rewrite $I$ as
    $$
    I = \int_a^b \frac{f(x)}{w(x)}\, w(x)\, dx = \left\langle \frac{f}{w} \right\rangle_w \approx \frac{1}{N}\sum_{i=1}^N \frac{f(x_i)}{w(x_i)}, \qquad x_i \text{ sampled from } w(x).
    $$
    For $w(x) = 1/(b-a)$ we recover the mean value method.
    The error estimate reads as before,
    $$
    \delta I = \sqrt{\frac{\left\langle (f/w)^2 \right\rangle_w - \left\langle f/w \right\rangle_w^2}{N}}.
    $$
    Choosing $w(x)$ similar in shape to $f(x)$ makes $f/w$ nearly constant, and the variance small: we sample most where the integrand matters, or cancel its singularities.

    In the extreme case $w(x) = f(x)/I$ (for $f \geq 0$), the ratio $f(x)/w(x) = I$ is the same for every sample, and $\delta I = 0$: a single sample gives the exact answer! This ideal is out of reach, since normalizing $w$ requires knowing $I$, but it shows what to aim for: the closer $w$ is to being proportional to $f$, the smaller the error.
    """)
    return


@app.cell
def _(np):
    def int_mc_weighted(f, N, w, sampler, rng):
        """Importance sampling estimate of int f(x) dx and its statistical error.

        w: normalized probability density, sampler(N, rng): N samples drawn from w.
        """
        x = sampler(N, rng)
        ratio = f(x) / w(x)
        r_av, rsq_av = ratio.mean(), (ratio**2).mean()
        return r_av, np.sqrt((rsq_av - r_av**2) / N)

    return (int_mc_weighted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: an integrable singularity

    Consider
    $$
    I = \int_0^1 \frac{x^{-1/2}}{e^x + 1}\, dx \approx 0.83893,
    $$
    which has an integrable singularity at $x = 0$.
    A natural weight cancels the singularity in $f/w$:
    $$
    w(x) = \frac{1}{2\sqrt{x}}, \qquad \frac{f(x)}{w(x)} = \frac{2}{e^x + 1}, \qquad I = \left\langle \frac{2}{e^x + 1} \right\rangle_w.
    $$
    We sample from $w(x)$ by inverse transform: $G_w(x) = \int_0^x w(x')\,dx' = \sqrt{x} = \eta$, thus
    $$
    x = \eta^2.
    $$
    """)
    return


@app.cell
def _(np):
    def f_singular(x):
        return 1.0 / np.sqrt(x) / (np.exp(x) + 1)

    def uniform_w(x):
        return np.ones_like(x)

    def uniform_sampler(N, rng):
        return rng.random(N)

    def rsqrt_w(x):
        return 1.0 / (2.0 * np.sqrt(x))

    def rsqrt_sampler(N, rng):
        return rng.random(N) ** 2

    return f_singular, rsqrt_sampler, rsqrt_w, uniform_sampler, uniform_w


@app.cell(hide_code=True)
def _(f_singular, method_colors, np, plt, rsqrt_w):
    _x = np.linspace(0.002, 1, 500)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _ax1.plot(_x, f_singular(_x), color=method_colors["Mean value"], lw=2)
    _ax1.fill_between(_x, f_singular(_x), alpha=0.3, color=method_colors["Mean value"])
    _ax1.set(xlim=(0, 1), ylim=(0, 6), xlabel="$x$", ylabel="$f(x)$", title="Integrand: what uniform sampling averages")
    _ax2.plot(_x, f_singular(_x) / rsqrt_w(_x), color=method_colors["Importance"], lw=2)
    _ax2.set(xlim=(0, 1), ylim=(0, 6), xlabel="$x$", ylabel="$f(x)/w(x)$", title="What importance sampling averages")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    imp_N = mo.ui.slider(2, 6, step=0.5, value=5, label=r"$\log_{10} N$")
    return (imp_N,)


@app.cell(hide_code=True)
def _(
    f_singular,
    imp_N,
    int_mc_weighted,
    mo,
    np,
    rsqrt_sampler,
    rsqrt_w,
    uniform_sampler,
    uniform_w,
):
    _N = int(10**imp_N.value)
    _Iu, _eu = int_mc_weighted(f_singular, _N, uniform_w, uniform_sampler, np.random.default_rng(15))
    _Ii, _ei = int_mc_weighted(f_singular, _N, rsqrt_w, rsqrt_sampler, np.random.default_rng(15))
    mo.vstack(
        [
            imp_N,
            mo.hstack(
                [
                    mo.stat(label="Uniform sampling, w = 1", value=f"{_Iu:.5f} ± {_eu:.5f}", caption=f"N = {_N:,}"),
                    mo.stat(label="Importance sampling, w = 1/(2√x)", value=f"{_Ii:.5f} ± {_ei:.5f}", caption=f"N = {_N:,}"),
                    mo.stat(label="Error ratio", value=f"{_eu / _ei:.1f}", caption="uniform / importance"),
                ],
                justify="center",
                gap=2,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The statistical error of importance sampling is more than 10 times smaller for the same number of samples. Since the error scales as $1/\sqrt{N}$, uniform sampling would need more than 100 times as many samples to reach the same accuracy.
    In general, the smoother one can make $f(x)/w(x)$, the smaller the statistical error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Monte Carlo simulation
    ---

    Besides integration, random numbers are used to **simulate** physical processes that are random themselves, such as the radioactive decay of Section 4, or that involve so many degrees of freedom that a statistical description is the only practical one.

    ### Example: Brownian motion

    Brownian motion is the motion of a particle in a gas or liquid, buffeted by collisions with the much smaller particles of the medium.
    We consider a simplified 2D model on a lattice: at each step, the particle moves by one unit in one of the four directions, chosen at random. This is a **random walk**.

    After $n$ steps of length 1, the displacement $\mathbf{r}_n = \sum_{i=1}^n \mathbf{s}_i$ is a sum of independent steps with $\langle \mathbf{s}_i\rangle = 0$ and $\langle \mathbf{s}_i \cdot \mathbf{s}_j\rangle = \delta_{ij}$, so the mean squared displacement grows linearly in time,
    $$
    \langle r_n^2 \rangle = n,
    $$
    the signature of **diffusion** ($\langle r^2\rangle = 4Dt$ in 2D).
    """)
    return


@app.cell
def _(np):
    def random_walk(n_steps, rng, n_walkers=1):
        """2D lattice random walks. Returns x, y of shape (n_walkers, n_steps+1), starting at the origin."""
        dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        steps = dirs[rng.integers(4, size=(n_walkers, n_steps))]
        pos = np.concatenate([np.zeros((n_walkers, 1, 2), dtype=int), np.cumsum(steps, axis=1)], axis=1)
        return pos[..., 0], pos[..., 1]

    return (random_walk,)


@app.cell(hide_code=True)
def _(mo):
    walk_steps = mo.ui.slider(100, 100000, step=100, value=10000, label="Number of steps $n$")
    walk_seed = mo.ui.number(start=0, stop=10**6, value=0, label="Seed")
    return walk_seed, walk_steps


@app.cell(hide_code=True)
def _(mo, np, plt, random_walk, walk_seed, walk_steps):
    _n = walk_steps.value
    _x, _y = random_walk(_n, np.random.default_rng(walk_seed.value))
    _x, _y = _x[0], _y[0]

    # mean squared displacement over many walkers
    _nw = 500
    _xs, _ys = random_walk(1000, np.random.default_rng(walk_seed.value + 1), n_walkers=_nw)
    _msd = np.mean(_xs**2 + _ys**2, axis=0)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    _ax1.plot(_x, _y, color="steelblue", lw=0.6)
    _ax1.plot([0], [0], "o", color="seagreen", label="start")
    _ax1.plot([_x[-1]], [_y[-1]], "o", color="crimson", label="end")
    # square axes centred on the start, scaled with the diffusion length sqrt(n)
    _lim = 1.05 * max(2 * np.sqrt(_n), np.abs(_x).max(), np.abs(_y).max())
    _ax1.add_patch(plt.Circle((0, 0), np.sqrt(_n), fill=False, ls="--", color="black", lw=1, label=r"$r = \sqrt{n}$"))
    _ax1.set(xlim=(-_lim, _lim), ylim=(-_lim, _lim), xlabel="x", ylabel="y", title=f"One random walk, {_n} steps")
    _ax1.set_aspect("equal")
    _ax1.legend()
    _ax2.plot(_msd, color="darkorange", lw=2, label=f"average over {_nw} walkers")
    _ax2.plot([0, 1000], [0, 1000], "--", color="black", lw=1, label=r"$\langle r^2\rangle = n$")
    _ax2.set(xlabel="Step $n$", ylabel=r"$\langle r_n^2 \rangle$", title="Mean squared displacement")
    _ax2.legend()
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.hstack([walk_steps, walk_seed], justify="center", gap=2),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A single walk is erratic, but the average over many walkers follows the deterministic law $\langle r_n^2\rangle = n$: the typical distance travelled grows only as $\sqrt{n}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Monte Carlo simulation and statistical mechanics
    ---

    For a system in thermal equilibrium at temperature $T$, the probability to find the system in microstate $i$ is given by the **Boltzmann distribution**
    $$
    P_i = \frac{e^{-\beta E_i}}{Z}, \qquad Z = \sum_i e^{-\beta E_i},
    $$
    where $\beta = 1/(k_BT)$ is the inverse temperature, $E_i$ the energy of state $i$, and $Z$ the partition function.
    We are typically interested in the average of a physical observable $X$,
    $$
    \langle X \rangle = \sum_i X_i P_i,
    $$
    where $X_i$ is the value of $X$ in microstate $i$.
    The number of microstates grows exponentially with the size of the system, so neither $Z$ nor the sum can be computed exactly beyond a handful of particles: we need Monte Carlo methods.

    **Uniform sampling.** A first idea is to pick $N$ microstates uniformly at random and weight each of them with its Boltzmann factor,
    $$
    \langle X \rangle \approx \frac{\sum_{k=1}^N X_k e^{-\beta E_k}}{\sum_{k=1}^N e^{-\beta E_k}}~.
    $$
    This does not require $Z$, but it is very inefficient: almost all randomly chosen states have a high energy and a negligible Boltzmann factor, while the few low-energy states that dominate the average are hardly ever picked.

    **Importance sampling.** As in Section 10, it is much better to sample the states where the "integrand" matters, i.e. with probability $P_i \propto e^{-\beta E_i}$ itself. The average is then a simple mean over $N$ samples,
    $$
    \langle X \rangle \approx \frac{1}{N} \sum_{k=1}^N X_k.
    $$
    How to sample from $P_i$ without knowing $Z$ is the subject of Section 13.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    def draw_spins(spins, ax, arrow_length=0.7):
        """Draw a 2D spin configuration as arrows: up (+1) in dark red, down (-1) in navy.

        spins[r, c] is the spin in row r (counted from the top) and column c.
        """
        rows, cols = spins.shape
        y, x = np.mgrid[rows - 1 : -1 : -1, 0:cols]  # row 0 at the top
        for s, color in [(1, "darkred"), (-1, "navy")]:
            sel = spins == s
            ax.quiver(
                x[sel], y[sel], np.zeros(sel.sum()), s * arrow_length * np.ones(sel.sum()),
                color=color, pivot="middle", angles="xy", scale_units="xy", scale=1,
                width=0.012, headwidth=3.5, headlength=3.5, headaxislength=3,
            )
        ax.set(xlim=(-0.6, cols - 0.4), ylim=(-0.6, rows - 0.4), aspect="equal")
        ax.axis("off")

    # The configuration of the example figure: +1 = up, -1 = down, top row first
    _example = np.array(
        [
            [-1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
            [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, +1],
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, +1],
            [+1, +1, +1, -1, -1, -1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
            [+1, -1, -1, +1, +1, +1, +1, +1, +1, +1],
            [+1, -1, -1, +1, +1, +1, +1, +1, +1, -1],
            [-1, -1, +1, +1, +1, +1, +1, +1, +1, -1],
        ]
    )
    ising_example_fig, _ax = plt.subplots(figsize=(4.4, 3.5))
    draw_spins(_example, _ax)
    ising_example_fig.tight_layout()
    plt.close(ising_example_fig)
    return (ising_example_fig,)


@app.cell(hide_code=True)
def _(ising_example_fig, mo):
    mo.vstack(
        [
            mo.md(r"""
    ### The Ising model

    As our test system we take the **Ising model**: spins $s_i = \pm 1$ (magnetic dipoles pointing up or down) on a square $L\times L$ lattice, with energy
    $$
    E = -J \sum_{\langle ij\rangle} s_i s_j,
    $$
    where the sum runs over the pairs of nearest neighbours, and $J > 0$ for a ferromagnet: aligned neighbours lower the energy.
    """),
            mo.center(ising_example_fig),
            mo.md(r"""
    Each spin interacts with its four nearest neighbours (up, down, left, right).
    We use periodic boundary conditions (the right neighbour of the last column is the first column, etc.) so that every spin has four neighbours, and units with $J = k_B = 1$.
    The magnetization is
    $$
    M = \sum_i s_i.
    $$
    In 2D the model has a phase transition at the Curie temperature
    $$
    T_C = \frac{2}{\ln(1+\sqrt{2})} \approx 2.269:
    $$
    below $T_C$ the system develops a spontaneous magnetization $|M| > 0$.

    A $4\times4$ lattice has only $2^{16} = 65536$ microstates, few enough to compute $\langle E \rangle$ **exactly** by summing over all of them. This gives a reference to test the sampling methods against.
    """),
        ]
    )
    return


@app.cell
def _(np):
    T_curie = 2 / np.log(1 + np.sqrt(2))

    def ising_energy(spins, periodic=True):
        """Energy of 2D Ising configuration(s), J = 1. spins has shape (..., L, L)."""
        if periodic:
            return -np.sum(spins * np.roll(spins, 1, axis=-1), axis=(-2, -1)) - np.sum(
                spins * np.roll(spins, 1, axis=-2), axis=(-2, -1)
            )
        return -np.sum(spins[..., :, 1:] * spins[..., :, :-1], axis=(-2, -1)) - np.sum(
            spins[..., 1:, :] * spins[..., :-1, :], axis=(-2, -1)
        )

    # All 2^16 configurations of the 4x4 lattice, and their energies
    _codes = np.arange(2**16)
    ising4_states = (1 - 2 * ((_codes[:, None] >> np.arange(16)) & 1)).reshape(-1, 4, 4)
    ising4_E = ising_energy(ising4_states)

    def ising4_exact_E(T):
        """Exact <E> per spin of the periodic 4x4 Ising model, by summing over all states."""
        w = np.exp(-(ising4_E - ising4_E.min()) / T)  # shift the energies to avoid overflow
        return np.sum(ising4_E * w) / np.sum(w) / 16

    return T_curie, ising4_E, ising4_exact_E, ising_energy


@app.cell(hide_code=True)
def _(mo):
    uni_T = mo.ui.slider(0.5, 5.0, step=0.1, value=1.5, label="Temperature $T$")
    return (uni_T,)


@app.cell(hide_code=True)
def _(ising4_E, ising4_exact_E, method_colors, mo, np, plt, uni_T):
    _T = uni_T.value
    _exact = ising4_exact_E(_T)
    _rng = np.random.default_rng(16)
    _Nmax = 100000
    _E = ising4_E[_rng.integers(2**16, size=_Nmax)]  # uniformly chosen microstates
    _w = np.exp(-(_E - ising4_E.min()) / _T)
    _Ns = np.arange(1, _Nmax + 1)
    _estimate = np.cumsum(_E * _w) / np.cumsum(_w) / 16

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_Ns, _estimate, color=method_colors["Hit-or-miss"], lw=1.5, label="Uniform sampling with Boltzmann weights")
    _ax.axhline(_exact, ls="--", color=method_colors["Exact"], label="Exact (sum over all 65536 states)")
    _ax.set(xscale="log", xlabel="Number of samples $N$", ylabel=r"$\langle E\rangle / L^2$", ylim=(-2.1, 0.1), title=f"4×4 Ising model, T = {_T:.1f}")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    plt.close(_fig)

    _p_ground = 2 / np.sum(np.exp(-(ising4_E - ising4_E.min()) / _T))  # the 2 ground states have E = E_min
    mo.vstack(
        [
            uni_T,
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            mo.stat(label="Exact ⟨E⟩/L²", value=f"{_exact:.4f}"),
                            mo.stat(label=f"Estimate, N = {_Nmax:,}", value=f"{_estimate[-1]:.4f}"),
                            mo.stat(label="Weight of the 2 ground states", value=f"{_p_ground:.3f}", caption="vs 2/65536 = 3·10⁻⁵ of the samples"),
                        ]
                    ),
                ],
                widths=[0.75, 0.25],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At high temperature uniform sampling works, but lower the temperature: the two fully aligned ground states carry most of the Boltzmann weight, yet they make up only $2/65536$ of the uniformly drawn samples. The estimate jumps every time one of them happens to be drawn, and for larger lattices (a $20\times20$ lattice has $2^{400}\approx 10^{120}$ states) it would never be drawn at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. The Markov chain method
    ---

    How can we pick states from the distribution $P_i = e^{-\beta E_i} / Z$ when we cannot even compute $Z$?
    It turns out we don't have to: the distribution can be generated by a **Markov chain**, a sequence of states in which each state is obtained from the previous one by a random move.

    Suppose the current state is $i$. We move to a new state $j$ with a transition probability $T_{ij}$, where by construction
    $$
    \sum_j T_{ij} = 1.
    $$
    The Markov chain method chooses the $T_{ij}$ such that (**detailed balance**)
    $$
    \frac{T_{ij}}{T_{ji}} = \frac{P_j}{P_i} = \frac{e^{-\beta E_j} /Z}{e^{-\beta E_i} / Z} = e^{-\beta(E_j - E_i)}.
    $$
    The unknown $Z$ cancels in the ratio!

    Imagine that at the current step the probability to be in state $i$ is the Boltzmann probability $P_i$.
    The probability to be in state $j$ at the next step is then
    $$
    \sum_i P_i T_{ij} = \sum_i P_j T_{ji} = P_j \sum_i T_{ji} = P_j.
    $$
    Therefore, once the chain has reached the Boltzmann distribution, all subsequent states also follow the Boltzmann distribution. It can be proven that, starting from an arbitrary initial state, the chain eventually converges to the Boltzmann distribution: after an **equilibration** period, the states of the chain are samples from $P_i$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Metropolis algorithm

    The **Metropolis algorithm** realizes transition probabilities that satisfy detailed balance.
    Suppose we have a move set: $M$ possible moves from state $i$ to other states, such that each move can be undone by a move of the set.

    1. Pick a move to a candidate state $j$ uniformly at random (probability $1/M$ for each move).
    2. Compare the energy $E_j$ of the candidate with the energy $E_i$ of the current state:
        - if $E_j \le E_i$, accept the move;
        - if $E_j > E_i$, accept the move with probability (a biased coin, Section 4)
    $$
    P_{a} = e^{-\beta (E_j - E_i)}.
    $$

    Why does this work? Take $E_j > E_i$. The probability to select $j$ is $1/M$ and to accept it $e^{-\beta (E_j - E_i)}$, thus $T_{ij} = e^{-\beta (E_j - E_i)}/M$. From $j$, the probability to select $i$ is $1/M$, and the move is always accepted since $E_i < E_j$, thus $T_{ji} = 1/M$. The ratio is
    $$
    \frac{T_{ij}}{T_{ji}} = e^{-\beta (E_j - E_i)},
    $$
    as required. The full algorithm:

    1. Choose a starting state.
    2. Choose a move uniformly at random.
    3. Accept the move with probability $P_a = \min\left(1, e^{-\beta (E_j - E_i)}\right)$.
    4. Measure the quantity $X$ in the current state (whether or not the move was accepted) and add it to the average.
    5. Repeat from step 2.

    When a move is rejected the system stays in state $i$, and this state is counted again in the average: $i \to i$ is a perfectly valid step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: the Ising model

    For the Ising model, a move flips one randomly chosen spin, $s_i \to -s_i$. Only the bonds of spin $i$ with its four neighbours change, so the energy change is cheap to compute:
    $$
    \Delta E = 2 J s_i \sum_{j \in \text{neighbours}(i)} s_j \in \{-8, -4, 0, 4, 8\}.
    $$
    $L^2$ attempted flips, one per spin on average, are called a **sweep**.
    """)
    return


@app.cell
def _(ising_energy, np):
    import math

    def ising_metropolis(spins, T, steps, rng, snapshot_every=0):
        """Metropolis simulation of the periodic 2D Ising model (J = 1) with single spin flips.

        Returns the energy and magnetization after each step, the final spins,
        and a snapshot of the spins every snapshot_every steps (if > 0).
        """
        L = spins.shape[0]
        s = spins.tolist()  # Python lists are much faster than NumPy for element-wise access
        E, M = int(ising_energy(spins)), int(spins.sum())
        ii = rng.integers(L, size=steps).tolist()
        jj = rng.integers(L, size=steps).tolist()
        eta = rng.random(steps).tolist()
        Es, Ms = np.empty(steps + 1), np.empty(steps + 1)
        Es[0], Ms[0] = E, M
        snapshots = [np.array(s)]
        for k in range(steps):
            i, j = ii[k], jj[k]
            row = s[i]
            neighbours = s[i - 1][j] + s[(i + 1) % L][j] + row[j - 1] + row[(j + 1) % L]
            dE = 2 * row[j] * neighbours
            if dE <= 0 or eta[k] < math.exp(-dE / T):  # Metropolis acceptance
                row[j] = -row[j]
                E += dE
                M += 2 * row[j]
            Es[k + 1], Ms[k + 1] = E, M
            if snapshot_every and (k + 1) % snapshot_every == 0:
                snapshots.append(np.array(s))
        return Es, Ms, np.array(s), snapshots

    def random_spins(L, rng):
        """A random (infinite temperature) L x L spin configuration."""
        return rng.choice(np.array([-1, 1]), size=(L, L))

    return ising_metropolis, math, random_spins


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let us check the Metropolis algorithm on the $4\times4$ lattice, where we know the exact answer. We discard the first 1000 steps (equilibration) and average the energy over the rest.
    """)
    return


@app.cell(hide_code=True)
def _(
    ising4_E,
    ising4_exact_E,
    ising_metropolis,
    method_colors,
    mo,
    np,
    plt,
    random_spins,
    uni_T,
):
    _T = uni_T.value
    _rng = np.random.default_rng(17)
    _Nmax, _neq = 100000, 1000
    _Es, _, _, _ = ising_metropolis(random_spins(4, _rng), _T, _Nmax + _neq, _rng)
    _metro = np.cumsum(_Es[_neq + 1 :]) / np.arange(1, _Nmax + 1) / 16
    _E = ising4_E[_rng.integers(2**16, size=_Nmax)]
    _w = np.exp(-(_E - ising4_E.min()) / _T)
    _uniform = np.cumsum(_E * _w) / np.cumsum(_w) / 16
    _Ns = np.arange(1, _Nmax + 1)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_Ns, _uniform, color=method_colors["Hit-or-miss"], lw=1.2, label="Uniform sampling with weights")
    _ax.plot(_Ns, _metro, color=method_colors["Importance"], lw=1.5, label="Metropolis (importance sampling)")
    _ax.axhline(ising4_exact_E(_T), ls="--", color=method_colors["Exact"], label="Exact")
    _ax.set(xscale="log", xlabel="Number of samples $N$", ylabel=r"$\langle E\rangle / L^2$", ylim=(-2.1, 0.1), title=f"4×4 Ising model, T = {_T:.1f}")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md("The temperature is set by the slider of Section 12:"),
            uni_T,
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Metropolis estimate converges to the exact value at every temperature, including low temperatures where uniform sampling fails.

    Now a larger, $L = 20$ lattice ($2^{400}$ states). Starting from a random configuration, we follow the magnetization during several independent runs.
    """)
    return


@app.cell(hide_code=True)
def _(T_curie, mo):
    mag_T = mo.ui.slider(0.5, 5.0, step=0.1, value=1.5, label="Temperature $T$")
    mag_run = mo.ui.run_button(label="Run 5 simulations")
    mo.vstack([mo.hstack([mag_T, mag_run], justify="center", gap=2), mo.md(f"$T_C \\approx {T_curie:.3f}$")], align="center")
    return mag_T, mag_run


@app.cell(hide_code=True)
def _(T_curie, ising_metropolis, mag_T, mag_run, mo, np, plt, random_spins):
    mo.stop(not mag_run.value, mo.md("*Choose a temperature and press the button (about 3 seconds).*"))

    _L, _sweeps = 20, 1000
    _rng = np.random.default_rng()
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _run in range(5):
        _, _Ms, _, _ = ising_metropolis(random_spins(_L, _rng), mag_T.value, _sweeps * _L**2, _rng)
        _ax.plot(np.arange(len(_Ms)) / _L**2, _Ms / _L**2, lw=1, label=f"Run {_run + 1}")
    _ax.set(xlabel="Sweeps", ylabel="$M / L^2$", ylim=(-1.05, 1.05), title=f"20×20 Ising model, T = {mag_T.value:.1f} ({'below' if mag_T.value < T_curie else 'above'} $T_C$)")
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below $T_C$ each run ends up magnetized, $M/L^2 \to \pm 1$: the sign is chosen at random, a **spontaneous symmetry breaking**. Close to $T_C$ the magnetization fluctuates strongly and slowly, and above $T_C$ it fluctuates around zero.

    It is instructive to watch the spins themselves. Press ▶ to follow a $100 \times 100$ lattice, one frame per sweep.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    anim_T = mo.ui.slider(0.5, 4.0, step=0.1, value=1.5, label="Temperature $T$")
    anim_run = mo.ui.run_button(label="Simulate 100 sweeps")
    mo.hstack([anim_T, anim_run], justify="center", gap=2)
    return anim_T, anim_run


@app.cell(hide_code=True)
def _(anim_T, anim_run, ising_metropolis, np, random_spins):
    # nothing is simulated until the button is pressed (the display cell shows a message instead)
    anim_M, anim_snapshots, anim_T_used = None, None, None
    if anim_run.value:
        _L = 100
        _rng = np.random.default_rng()
        _, anim_M, _, anim_snapshots = ising_metropolis(random_spins(_L, _rng), anim_T.value, 4000 * _L**2, _rng, snapshot_every=_L**2)
        anim_T_used = anim_T.value
    return anim_M, anim_T_used, anim_snapshots


@app.cell(hide_code=True)
def _(PlaySlider, mo):
    anim_play = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=4000, step=1, interval_ms=150))  # 100 sweeps
    return (anim_play,)


@app.cell(hide_code=True)
def _(anim_M, anim_T_used, anim_play, anim_snapshots, mo, np, plt):
    mo.stop(anim_snapshots is None, mo.md("*Choose a temperature and press the button (about 3 seconds).*"))
    _n = min(int(anim_play.value["value"]), len(anim_snapshots) - 1)
    _L = anim_snapshots[0].shape[0]
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    _ax1.imshow(anim_snapshots[_n].T, cmap="gray_r", vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    _ax1.set(xlabel="x", ylabel="y", title=f"T = {anim_T_used:.1f}, sweep {_n}")
    _sweeps = np.arange(len(anim_M)) / _L**2
    _ax2.plot(_sweeps[: _n * _L**2 + 1], anim_M[: _n * _L**2 + 1] / _L**2, color="steelblue")
    _ax2.set(xlim=(0, len(anim_snapshots) - 1), ylim=(-1.05, 1.05), xlabel="Sweeps", ylabel="$M/L^2$", title="Magnetization")
    _fig.tight_layout()
    plt.close(_fig)
    mo.vstack([mo.hstack([mo.md("**Sweep:**"), anim_play], justify="center"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below $T_C$, domains of aligned spins (black: up, white: down) form and grow; the domain walls cost energy and slowly disappear. Above $T_C$ the spins stay disordered, with short-lived clusters; near $T_C$ the clusters exist at all length scales.

    Finally, we scan the temperature and measure the equilibrium averages $\langle |M| \rangle / L^2$ and $\langle E \rangle/L^2$, discarding the first sweeps for equilibration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    scan_run = mo.ui.run_button(label="Scan the temperature (about 5 seconds)")
    scan_run
    return (scan_run,)


@app.cell(hide_code=True)
def _(T_curie, ising_metropolis, mo, np, plt, scan_run):
    mo.stop(not scan_run.value, mo.md("*Press the button to run the temperature scan.*"))

    _L, _neq, _nmeas = 16, 200, 800
    _rng = np.random.default_rng(18)
    _Ts = np.linspace(1.0, 4.0, 16)
    _absM, _Eav = [], []
    _spins = np.ones((_L, _L), dtype=int)
    for _T in _Ts:  # start each temperature from the end state of the previous one
        _Es, _Ms, _spins, _ = ising_metropolis(_spins, _T, (_neq + _nmeas) * _L**2, _rng)
        _absM.append(np.mean(np.abs(_Ms[_neq * _L**2 :])) / _L**2)
        _Eav.append(np.mean(_Es[_neq * _L**2 :]) / _L**2)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.plot(_Ts, _absM, "o-", color="#1d4ed8")
    _ax1.set(xlabel="$T$", ylabel=r"$\langle |M| \rangle / L^2$", title="Magnetization")
    _ax2.plot(_Ts, _Eav, "o-", color="crimson")
    _ax2.set(xlabel="$T$", ylabel=r"$\langle E \rangle / L^2$", title="Energy")
    for _ax in (_ax1, _ax2):
        _ax.axvline(T_curie, ls="--", color="black", lw=1, label="$T_C$")
        _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The magnetization drops from 1 to nearly 0 around $T_C$. On a finite lattice the transition is smoothed out; it becomes sharp only in the limit $L \to \infty$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: finite volume effects in the ideal gas

    As a second example, consider an ideal gas of $N$ non-interacting particles in a cubic box of side $L$. Quantum mechanically, the energy levels of a particle are obtained by solving the Schrödinger equation of a free particle,
    $$
    -\frac{\hbar^2}{2m} \nabla^2 \psi = E \psi.
    $$

    - For a box with hard walls, $\psi = 0$ on the walls, and
    $$
    E_{n_x,n_y,n_z} =  \frac{\pi^2 \hbar^2}{2mL^2} (n_x^2 + n_y^2 + n_z^2), \qquad n_x, n_y, n_z = 1,2,\ldots
    $$
    - For periodic boundary conditions, $\psi(x) = \psi(x+L)$, and
    $$
    E_{n_x,n_y,n_z} =  \frac{2 \pi^2 \hbar^2}{mL^2} (n_x^2 + n_y^2 + n_z^2), \qquad n_x, n_y, n_z = 0, \pm1, \pm2,\ldots
    $$

    Neglecting quantum statistics, the total energy is $E = \sum_{i=1}^N E_{n_x^{(i)},n_y^{(i)},n_z^{(i)}}$, and a microstate is specified by the $3N$ quantum numbers.

    The move set: pick a particle and one of its three quantum numbers at random, and increase or decrease it by one. A move to a forbidden state ($n = 0$ for hard walls) is rejected, i.e. the particle stays in its present state. For periodic boundary conditions, $n$ and $-n$ have the same energy, and we only keep track of $|n|$: a move from $n = 0$ always goes to $|n| = 1$, which correctly accounts for the two states $n = \pm 1$.

    Classically, the equipartition theorem gives $E = \frac{3}{2} N k_B T$. With quantized levels this holds only when $k_BT$ is large compared with the level spacing. We work in units $m = \hbar = L = k_B = 1$.
    """)
    return


@app.cell
def _(math, np):
    def ideal_gas_metropolis(T, N, steps, rng, periodic=False):
        """Metropolis simulation of N particles in a cube with quantized levels.

        Returns E/(N T) after each step. All particles start in the lowest level n = (1,1,1).
        """
        factor = 2 * np.pi**2 if periodic else np.pi**2 / 2
        nmin = 0 if periodic else 1
        n = [[1, 1, 1] for _ in range(N)]
        E = factor * 3 * N
        particle = rng.integers(N, size=steps).tolist()
        component = rng.integers(3, size=steps).tolist()
        up = (rng.random(steps) < 0.5).tolist()
        eta = rng.random(steps).tolist()
        e_trace = np.empty(steps + 1)
        e_trace[0] = E / (N * T)
        for k in range(steps):
            i, j = particle[k], component[k]
            old = n[i][j]
            new = abs(old + 1) if up[k] else abs(old - 1)  # |n| for periodic BC
            if new >= nmin:
                dE = factor * (new**2 - old**2)
                if dE <= 0 or eta[k] < math.exp(-dE / T):
                    n[i][j] = new
                    E += dE
            e_trace[k + 1] = E / (N * T)
        return e_trace

    return (ideal_gas_metropolis,)


@app.cell(hide_code=True)
def _(mo):
    gas_bc = mo.ui.radio(options=["hard walls", "periodic"], value="hard walls", label="Boundary conditions", inline=True)
    gas_bc
    return (gas_bc,)


@app.cell(hide_code=True)
def _(gas_bc, ideal_gas_metropolis, mo, np, plt):
    _N, _steps = 1000, 500000
    _periodic = gas_bc.value == "periodic"
    _rng = np.random.default_rng(19)
    # exact equilibrium value: sum over the levels of one quantum number, times 3
    _eps = 2 * np.pi**2 if _periodic else np.pi**2 / 2
    _levels = np.arange(-400, 401) if _periodic else np.arange(1, 401)
    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _final = []
    for _T, _c in zip([5.0, 10.0, 100.0], ["#1d4ed8", "seagreen", "darkorange"]):
        _e = ideal_gas_metropolis(_T, _N, _steps, _rng, periodic=_periodic)
        _w = np.exp(-_eps * _levels**2 / _T)
        _exact = 3 * np.sum(_eps * _levels**2 * _w) / np.sum(_w) / _T
        _ax.plot(_e, color=_c, lw=1.5, label=f"$T = {_T:g}$")
        _final.append(mo.stat(label=f"E/(NT) at T = {_T:g}", value=f"{_e[-_steps // 3 :].mean():.3f}", caption=f"exact: {_exact:.3f}"))
    _ax.axhline(1.5, ls="--", color="black", lw=1, label="classical limit 3/2")
    _ax.set(ylim=(0, 5), xlabel="Step", ylabel="$E/(NT)$", title=f"Ideal gas of N = {_N} particles, {gas_bc.value}")
    _ax.legend()
    _fig.tight_layout()
    plt.close(_fig)
    mo.hstack([_fig, mo.vstack(_final)], widths=[0.75, 0.25], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After equilibration (a few hundred steps per particle), the energy fluctuates around its equilibrium value, which agrees with the exact result obtained by summing the Boltzmann factors over the levels (possible here only because the particles do not interact). At high temperature it approaches the classical result $E = \frac{3}{2}NT$, while at lower temperatures the quantization of the levels matters: with hard walls the zero-point energy pushes $E/(NT)$ above $3/2$, while with periodic boundary conditions the $n = 0$ level lowers it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: random numbers and Monte Carlo methods

    **Generating random numbers**

    | Task | Method | Recipe |
    |---|---|---|
    | Uniform $\eta\in[0,1)$ | library generator (Mersenne Twister, PCG64) | `rng = np.random.default_rng(seed)`, `rng.random()` |
    | Event with probability $p$ | biased coin | accept if $\eta < p$ |
    | Density $\rho(x)$ with invertible CDF $G$ | inverse transform | $x = G^{-1}(\eta)$, e.g. $t = -\tau\ln(1-\eta)$ |
    | Bounded density $\rho(x) \le \rho^{\rm max}$ on a finite range | rejection sampling | accept $x_{\rm cand}$ if $\rho^{\rm max}\eta < \rho(x_{\rm cand})$ |
    | Gaussian | Box–Muller | $r = \sqrt{-2\ln(1-\eta_1)}$, $\phi = 2\pi\eta_2$ |
    | Uniform in a disk / on a sphere | inverse transform | $r = \sqrt{\eta}$ / $\cos\theta = 2\eta - 1$ |

    **Monte Carlo integration**: the error always scales as $1/\sqrt{N}$, independently of the dimension.

    | Method | Estimate | Error | Remarks |
    |---|---|---|---|
    | Hit-or-miss | $(b-a)\,y_{\rm max}\, C/N$ | $(b-a)\,y_{\rm max}\sqrt{p(1-p)/N}$ | needs $0\le f \le y_{\rm max}$ |
    | Mean value | $(b-a)\langle f\rangle$ | $(b-a)\sqrt{(\langle f^2\rangle - \langle f\rangle^2)/N}$ | always at least as good as hit-or-miss |
    | Importance sampling | $\langle f/w\rangle_w$ | $\sqrt{(\langle (f/w)^2\rangle_w - \langle f/w\rangle_w^2)/N}$ | best when $f/w$ is nearly constant |

    In one dimension, quadrature rules are far more accurate; Monte Carlo wins for integrals in more than about four dimensions, where grid-based rules suffer from the curse of dimensionality.

    **Markov chain Monte Carlo**

    | Method | Move accepted with probability | Use |
    |---|---|---|
    | Metropolis algorithm | $\min\left(1, e^{-\beta\Delta E}\right)$ | samples the Boltzmann distribution without $Z$; averages $\langle X\rangle \approx \frac{1}{N}\sum_k X_k$ after equilibration |
    """)
    return


if __name__ == "__main__":
    app.run()
