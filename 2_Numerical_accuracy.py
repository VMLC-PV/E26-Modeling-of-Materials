import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/2_Numerical_accuracy.slides.json",
)


@app.cell
def _():

    import os
    import math
    import struct
    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    import matplotlib.patches as mpatches
    import pymatgen.core as mg
    import pymatviz as pmv
    from dataclasses import dataclass
    from decimal import Decimal
    from scipy.special import gamma

    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    return (
        Decimal,
        dataclass,
        gamma,
        go,
        math,
        mg,
        mo,
        mpatches,
        np,
        pd,
        plt,
        pmv,
        random,
        struct,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Numerical Accuracy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
        # Who is better at math?
        # You or your computer?
        """
            )
        ],
        align="center",
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
        # Let's test it!
        # What is 1.1 + 2.2 = ?
        """
            )
        ],
        align="center",
        justify="center",
    )
    return


@app.cell
def _(mo):
    x = 3.3
    a = 1.1
    b = 2.2

    mo.md(rf"""
    a = {a}  
    b = {b}  
    x = {x}  
    a + b == x --> {a + b == x}  
    a + b = {a + b}""")
    return


@app.cell
def _(mo):
    # User controls integer input
    number_input = mo.ui.number(
        start=0,
        stop=255,
        step=1,
        value=42,
        # label="Select an 8-bit Integer (0 - 255):",
    )
    num_md = mo.md(f"""### Select an 8-bit Integer (0 - 255):""")
    num_in = mo.vstack([num_md, number_input], align="center")
    return num_in, number_input


@app.cell
def _(mo, num_in, number_input):
    num = int(number_input.value)
    binary_str = f"{num:08b}"
    bits = [int(b) for b in binary_str]

    # Calculate individual bit place values
    powers = [2**i for i in range(7, -1, -1)]
    contributions = [b * p for b, p in zip(bits, powers)]

    # HTML representation for bit visualization
    bit_cards = ""
    idx = 0
    for bit, power, contrib in zip(bits, powers, contributions):
        is_active = bit == 1
        bg_color = "#3b82f6" if is_active else "#f3f4f6"
        text_color = "#ffffff" if is_active else "#9ca3af"
        border_color = "#2563eb" if is_active else "#d1d5db"
        opacity = "1.0" if is_active else "0.5"

        bit_cards += f"""
        <div style="
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            margin: 6px;
            font-family: system-ui, -apple-system, sans-serif;
        ">
            <div style="font-size: 0.8rem; color: #6b7280; font-weight: 600; margin-bottom: 4px;">
                2<sup>{7 - powers.index(power)}</sup> = {power}
            </div>
            <div style="
                width: 52px;
                height: 64px;
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.8rem;
                font-weight: 700;
                box-shadow: {"0 4px 12px rgba(59, 130, 246, 0.3)" if is_active else "none"};
                transition: all 0.25s ease-in-out;
            ">
                {bit}
            </div>
            <div style="
                font-size: 0.85rem;
                color: {"#1d4ed8" if is_active else "#9ca3af"};
                font-weight: 600;
                margin-top: 6px;
                opacity: {opacity};
            ">
                +{contrib}
            </div>
        </div>
        """

    # Build formula string
    equation_terms = [f"({b} × {p})" for b, p in zip(bits, powers) if b == 1]
    equation_str = " + ".join(equation_terms) if equation_terms else "0"

    bit_md = mo.md(
        f"""  
        ### Bitwise Breakout (8-Bit Unsigned Representation)

        <div style="
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            background: #fafafa;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-bottom: 16px;
        ">
            {bit_cards}
        </div>

        **Sum Calculation:**  
        $$\\text{{Total}} = {equation_str} = \\mathbf{{{num}}}$$
        """
    )

    mo.vstack([num_in, bit_md])
    return


@app.cell
def _(mo):
    precision_selector = mo.ui.dropdown(
        options=["float32", "float64"],
        value="float32",
        label="###Select Format:",
    )

    text_val_input = mo.ui.text(
        value="-13.6",
        label="###Enter Floating-Point Number (as String):",
    )
    return precision_selector, text_val_input


@app.cell
def _(Decimal, mo, np, precision_selector, struct, text_val_input):

    raw_str = text_val_input.value.strip()
    mode = precision_selector.value

    # Default fallback if input is invalid or empty

    dec_val = Decimal(raw_str)

    if mode == "float32":
        # Parse exact string float to 32-bit float
        f_val = np.float32(raw_str)
        packed = struct.pack(">f", f_val)
        integ = struct.unpack(">I", packed)[0]
        bit_str = f"{integ:032b}"
        sign_bits = bit_str[0]
        exp_bits = bit_str[1:9]
        mantissa_bits = bit_str[9:]
        bias = 127
    else:
        # Parse exact string float to 64-bit float
        f_val = np.float64(raw_str)
        packed = struct.pack(">d", f_val)
        integ = struct.unpack(">Q", packed)[0]
        bit_str = f"{integ:064b}"
        sign_bits = bit_str[0]
        exp_bits = bit_str[1:12]
        mantissa_bits = bit_str[12:]
        bias = 1023

    raw_exp = int(exp_bits, 2)

    # Compute exact mantissa fraction as a Decimal sum of powers of 2
    mantissa_dec = Decimal(0)
    for i, bit_ in enumerate(mantissa_bits):
        if bit_ == "1":
            mantissa_dec += Decimal(2) ** Decimal(-(i + 1))

    if raw_exp == 0:
        exp_val = 1 - bias
        significand_dec = mantissa_dec
        implicit_one = 0
    else:
        exp_val = raw_exp - bias
        significand_dec = Decimal(1) + mantissa_dec
        implicit_one = 1

    sign_factor = -1 if sign_bits == "1" else 1

    # Calculate exact mathematical value stored in memory
    exact_stored_val = (
        Decimal(sign_factor) * significand_dec * (Decimal(2) ** Decimal(exp_val))
    )

    def make_badge(bit_seq, color, label):
        return f"""
        <div style="display: inline-block; margin: 4px; padding: 8px 12px; background: {color}; color: white; border-radius: 6px; font-family: monospace;">
            <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.9;">{label}</div>
            <div style="font-size: 1.1rem; font-weight: bold; letter-spacing: 1px;">{bit_seq}</div>
        </div>
        """

    sign_html = make_badge(sign_bits, "#ef4444", "Sign (1b)")
    exp_html = make_badge(exp_bits, "#3b82f6", f"Exponent ({len(exp_bits)}b)")
    mant_html = make_badge(
        mantissa_bits, "#10b981", f"Mantissa ({len(mantissa_bits)}b)"
    )

    md_float1 = mo.md(
        r"""
        ## Floating Point Bit Visualizer (IEEE 754)
        """
    )

    html_disp = mo.Html(
        f"""
        <div style="display: flex; justify-content: center; flex-wrap: wrap;">
            {sign_html} {exp_html} {mant_html}
        </div>
        """
    )

    md_float2 = mo.md(
        f"""
        ---
        $$\\text{{Value}} = (-1)^S \\times ({implicit_one} + M) \\times 2^{{E - \\text{{Bias}}}}$$

        Step-by-Step Calculation:
        * **Sign Term ($(-1)^S$):** $(-1)^{{{sign_bits}}} = \\mathbf{{{sign_factor}}}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Exponent Term ($2^{{E - \\text{{Bias}}}}$):** $2^{{{raw_exp} - {bias}}} = 2^{{{exp_val}}}$
        * **Significand Term ($1 + M$):** ${implicit_one} + {mantissa_dec} = \\mathbf{{{significand_dec}}}$

        $$\\text{{Value}} = ({sign_factor}) \\times ({significand_dec}) \\times 2^{{{exp_val}}}$$
        $$\\mathbf{{\\text{{Exact Value Stored in Memory: }}}} \\mathbf{{{exact_stored_val}}}$$
        """
    )

    mo.vstack(
        [
            md_float1,
            mo.hstack(
                [precision_selector, text_val_input],
                align="center",
                justify="center",
                gap=2,
            ),
            html_disp,
            md_float2,
        ]
    )
    return


@app.cell
def _(Decimal, struct):
    def get_float64_bits(val):
        """Get the IEEE 754 double-precision (64-bit) representation of a floating-point number.

        Parameters
        ----------
        val : float
            The floating-point number to be represented in IEEE 754 double-precision format.

        Returns
        -------
            dict
                A dictionary containing the IEEE 754 double-precision representation, including the bit string, sign, exponent bits, exponent value, mantissa, significand, and exact value.
        """
        packed = struct.pack(">d", val)
        integ = struct.unpack(">Q", packed)[0]
        bit_str = f"{integ:064b}"
        sign = bit_str[0]
        exp = bit_str[1:12]
        mantissa = bit_str[12:]
        raw_exp = int(exp, 2)
        exp_val = raw_exp - 1023
        mantissa_dec = Decimal(0)
        for i, bit_ in enumerate(mantissa):
            if bit_ == "1":
                mantissa_dec += Decimal(2) ** Decimal(-(i + 1))
        significand = Decimal(1) + mantissa_dec
        exact_val = (
            Decimal(-1 if sign == "1" else 1)
            * significand
            * (Decimal(2) ** Decimal(exp_val))
        )
        return {
            "bits": bit_str,
            "sign": sign,
            "exp_bits": exp,
            "exp_val": exp_val,
            "mantissa": mantissa,
            "significand": significand,
            "exact": exact_val,
        }

    f1 = get_float64_bits(1.1)
    f2 = get_float64_bits(2.2)
    f_sum = get_float64_bits(1.1 + 2.2)
    f3 = get_float64_bits(3.3)
    return f1, f2, f3, f_sum


@app.cell
def _(f1, f2, mo):
    mo.md(
        r"""
            ## Why `1.1 + 2.2 == 3.3` is `False` ?
            ---
            ### Binary Storage Truncation

            In Python, floating-point numbers follow the **IEEE 754 64-bit double-precision standard**. Because non-powers-of-two fractions (like $0.1$ and $0.2$) are **infinitely repeating binary fractions**, rounding errors occur at two distinct stages.

            ---

            #### Step 1: Storage Truncation
            Neither $1.1$ nor $2.2$ can be stored exactly in binary. Their mantissas repeat infinitely (`00110011...`) and are truncated to $52$ bits:

            * **Stored `1.1`:**  
              $1.0001100110011001100110011001100110011001100110011010_2 \times 2^0$
              $\approx \mathbf{"""
        + f"{f1['exact']}"
        + r"""}$

            * **Stored `2.2`:**  
              $1.0001100110011001100110011001100110011001100110011010_2 \times 2^1$
              $\approx \mathbf{"""
        + f"{f2['exact']}"
        + r"""}$
            """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exponent Alignment (Bit Shifting)
    ---
    Before the CPU's Floating-Point Unit (FPU) can add two numbers, their exponents **must match** ($E_1 = 0$ vs. $E_2 = 1$).

    ### Step 2: Alignment
    The significand of $1.1$ is shifted right by $1$ bit to match the exponent of $2.2$ ($2^1$):

    $$
    \begin{aligned}
    1.1 \rightarrow&\,\, 0.100011001100110011001100110011001100110011001100110100_2 \times 2^1 \\
    +\, 2.2 \rightarrow&\,\, 1.000110011001100110011001100110011001100110011001101000_2 \times 2^1
    \end{aligned}
    $$

    > **Note:** Right-shifting drops precision at the tail end of the smaller number before addition even begins!
    """)
    return


@app.cell
def _(f_sum, mo):
    mo.md(
        r"""
            ## Bit Addition & Hardware Rounding
             ---
            Adding the two aligned binary significands bit-by-bit yields a temporary 53-bit unrounded sum.

            ### Step 3: Fixed-Bit Binary Addition & Rounding
            $$
            \text{Raw Unrounded Sum} = 1.10100110011001100110011001100110011001100110011001110_2 \times 2^1
            $$

            Because 64-bit float registers only hold **52 mantissa bits**, the FPU applies the **round-to-nearest (ties to even)** rule, causing the least significant bit to round **up**:

            $$\mathbf{\text{Stored Result of } (1.1 + 2.2):}$$
            $$\mathbf{"""
        + f"{f_sum['exact']}"
        + r"""}$$
            """
    )
    return


@app.cell
def _(f3, f_sum, mo):
    mo.md(
        r"""
            ## Comparing `(1.1 + 2.2)` vs `3.3`
            ---
            When $3.3$ is parsed directly from string input, it is rounded independently from the raw string value to the nearest 64-bit float without accumulating addition/alignment error.

            ### Comparison Table
            """
        + f"""
            | Value | Exact Value Stored in Hardware Memory |
            | :--- | :--- |
            | **`1.1 + 2.2`** | `{f_sum["exact"]}` |
            | **`3.3`** | `{f3["exact"]}` |
            | **Difference** | **`+0.000000000000000444089209850062616169452667236328125`** ($2^{{-52}}$) |


            Because `1.1 + 2.2` accumulates rounding noise during **exponent alignment** and **52-bit truncation**, it differs from `3.3` by a single bit at the $52^\\text{{nd}}$ mantissa position. 

            Therefore, `1.1 + 2.2 == 3.3` evaluates to **`False`**!
            """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Numerical Accuracy: `float32` vs. `float64`

    ---

    When working with real numbers in scientific computing, precision is constrained by the number of bits allocated to the stored mantissa (significand).

    | Metric | Single Precision (`float32`) | Double Precision (`float64`) |
    | :--- | :---: | :---: |
    | **Total Bits** | $32\text{ bits}$ | $64\text{ bits}$ |
    | **Mantissa Bits ($M$)** | $23\text{ bits}$ | $52\text{ bits}$ |
    | **Decimal Precision** | $\approx \mathbf{7\text{ decimal digits}}$ | $\approx \mathbf{15\text{--}17\text{ decimal digits}}$ |
    | **Machine Epsilon ($\epsilon_m = 2^{-M}$)** | $2^{-23} \approx \mathbf{1.19 \times 10^{-7}}$ | $2^{-52} \approx \mathbf{2.22 \times 10^{-16}}$ |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # What is a better way to compare floats?
    """)
    return


@app.cell
def _(mo):
    # Interactive UI controls for tolerance epsilon
    eps_exponent = mo.ui.slider(
        start=-20,
        stop=-10,
        step=1,
        value=-15,
        label="Epsilon Tolerance (10ⁿ):",
    )
    return (eps_exponent,)


@app.cell
def _(eps_exponent, mo):
    # Calculate current epsilon from slider (negative exponent for small values)
    eps_val = eps_exponent.value  # e.g., 15
    eps = 10 ** (eps_val)

    # Perform floating-point calculation and comparison
    val_a = 1.1
    val_b = 2.2
    val_x = 3.3
    diff = abs(val_a + val_b - val_x)
    is_equal = diff < eps

    # Dynamic styling badge for True/False
    status_bg = "#10b981" if is_equal else "#ef4444"
    status_text = "TRUE" if is_equal else "FALSE"

    status_badge = f"""<span style="background-color: {status_bg}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-family: monospace; font-size: 1rem;">{status_text}</span>"""

    comparison_md1 = mo.md(
        f"""
        # Comparing Floating-Point Numbers Safely

        ---

        When comparing floating-point numbers in Python, instead of exact equality (`==`), use a **tolerance threshold** ($\\epsilon$):
        """
    )

    comparison_md2 = mo.md(
        f"""
        $$\lvert ({val_a} + {val_b}) - {val_x} \\rvert < \epsilon$$

        ---

        ### Live Evaluation

        * **Calculated Absolute Difference:** $\lvert {val_a + val_b} - {val_x} \\rvert = \\mathbf{{{diff:.17e}}}$
        * **Selected Tolerance ($\\epsilon$):** $10^{{{eps_val}}}$
        * **Comparison Status:** {status_badge}

        """
    )

    mo.vstack(
        [
            comparison_md1,
            mo.hstack([eps_exponent], align="center", justify="center"),
            comparison_md2,
        ]
    )
    return


@app.cell
def _(Decimal, mo, np):
    num1_str = "1000000000000000"
    num2_str = "1000000000000001.23456789234"

    # Exact mathematical target difference (1.23456789234)
    exact_diff = Decimal(num2_str) - Decimal(num1_str)

    # 64-bit float computation
    f64_1 = np.float64(num1_str)
    f64_2 = np.float64(num2_str)
    diff_64 = f64_2 - f64_1

    # Absolute and relative error
    abs_err_64 = abs(Decimal(str(diff_64)) - exact_diff)
    rel_err_64 = (abs_err_64 / exact_diff) * 100
    frac_err_64 = np.abs(float(exact_diff) - diff_64) / float(exact_diff) * 100

    mo.md(
        f"""
        # Catastrophic Cancellation

        ---

        ### What is Catastrophic Cancellation?

        **Catastrophic cancellation** occurs when subtracting two nearly equal numbers. Because both numbers share identical leading significant digits, those digits cancel out, leaving behind only the **accumulated rounding noise** in the least significant bits as the "result."

        ---

        ### Example: Subtracting $10^{{15}}$ from $10^{{15}} + 1.23456789234$

        * **Number 1 ($A$):** `{num1_str}` ($10^{{15}}$)
        * **Number 2 ($B$):** `{num2_str}`
        * **Exact Mathematical Difference ($B - A$):** $\\mathbf{{{exact_diff}}}$
        * **Number 1 in float64 :** `{f64_1}` 
        * **Number 2 in float64 :** `{f64_2}` 
        * **Computed Difference:** `{f64_2 - f64_1}` 
        * **Fractionnal error:** `{frac_err_64}`%

    """
    )
    return


@app.cell
def _(mo):
    mo.md(rf"""
    # Exercise 2.1

    ---

    The quadratic equation

    $$ax^2 + bx + c = 0$$

    has the following two roots:

    $$x_{{1,2}} = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}$$

    Let us calculate the roots for $a = 10^{{-4}}$, $b = 10^4$, and $c = 10^{{-4}}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Do the results look accurate to you?

    The value of $x_1$ is not accurate due to subtracting two large numbers with small difference $b$ and $\sqrt{b^2-4ac}$.

    Consider another form of the solution.
    By multiplying the numerator and denominator of the above expression for $x_{1,2}$ by $(-b\mp\sqrt{b^2-4ac})$ one obtains

    $$
    x_{1,2} = \frac{2c}{-b \mp \sqrt{b^2-4ac}}
    $$

    Let us see what we get now
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Errors and Error Propagation

    ---

    ## Accuracy is not the same as precision

    - **Accuracy** describes how close a calculated value is to the true value.
    - **Precision** describes how many digits are retained in the calculation.

    A value can contain many digits and still be inaccurate.

    ## Two important numerical errors

    1. **Approximation error**
       Introduced when an exact mathematical procedure is replaced by an
       approximation, such as truncating a Taylor series.

    2. **Roundoff error**
       Introduced because floating-point numbers contain only a finite number
       of significant bits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Absolute and Relative Error

    Suppose $x$ is the exact value and $\tilde{x}$ is an approximation.

    The signed absolute error is

    $$
    \Delta x = \tilde{x}-x.
    $$

    Its magnitude is

    $$
    |\Delta x|=|\tilde{x}-x|.
    $$

    If $x\neq0$, the relative error is

    $$
    \delta_x
    =
    \frac{\Delta x}{x}
    =
    \frac{\tilde{x}-x}{x}.
    $$

    Relative error is dimensionless and accounts for the scale of
    the quantity.
    """)
    return


@app.cell
def _(mo):
    taylor_x = mo.ui.slider(
        start=-3.0,
        stop=3.0,
        step=0.1,
        value=1.0,
        label="Value of x:",
    )

    taylor_terms = mo.ui.slider(
        start=0,
        stop=40,
        step=1,
        value=3,
        label="Largest Taylor order:",
    )
    return taylor_terms, taylor_x


@app.cell
def _(math, mo, taylor_terms, taylor_x):
    x_value = taylor_x.value
    maximum_order = taylor_terms.value

    taylor_approximation = sum(
        x_value**n / math.factorial(n) for n in range(maximum_order + 1)
    )

    exponential_exact = math.exp(x_value)
    taylor_absolute_error = abs(taylor_approximation - exponential_exact)

    if exponential_exact != 0:
        taylor_relative_error = taylor_absolute_error / abs(exponential_exact)
    else:
        taylor_relative_error = float("nan")

    mo.vstack(
        [
            mo.md(
                r"""
                ## Approximation error: Taylor series

                The exponential can be written as

                $$
                e^x
                =
                \sum_{n=0}^{\infty}\frac{x^n}{n!}.
                $$

                On a computer, we retain only a finite number of terms:

                $$
                e^x
                \approx
                \sum_{n=0}^{n_{\max}}\frac{x^n}{n!}.
                $$
                """
            ),
            mo.hstack(
                [taylor_x, taylor_terms],
                justify="center",
                align="center",
            ),
            mo.md(
                f"""
                - $x={x_value}$
                - $n_{{max}}={maximum_order}$
                - Taylor approximation: `{taylor_approximation:.17g}`
                - Library value: `{exponential_exact:.17g}`
                - Absolute error: `{taylor_absolute_error:.5e}`
                - Relative error: `{taylor_relative_error:.5e}`
                Increasing $n_{{max}}$ initially reduces the approximation
                error. Eventually, finite floating-point precision limits
                further improvement.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    error_example = mo.ui.dropdown(
        options={
            "Small-scale example": "small",
            "Large-scale example": "large",
        },
        value="Small-scale example",
        label="Select an example:",
    )
    return (error_example,)


@app.cell
def _(error_example, mo):
    if error_example.value == "small":
        exact_value = 1.000
        approximate_value = 0.999
    else:
        exact_value = 1_000_000_000.0
        approximate_value = 999_999_999.0

    signed_error = approximate_value - exact_value
    absolute_error = abs(signed_error)
    relative_error = signed_error / exact_value
    relative_error_magnitude = abs(relative_error)

    mo.vstack(
        [
            error_example,
            mo.md(
                f"""
                ## Roundoff Error: Example

                - Exact value: $x={exact_value:g}$
                - Approximation: $\\tilde{{x}}={approximate_value:g}$
                - Signed error: $\\Delta x={signed_error:g}$
                - Absolute-error magnitude:
                  $|\\Delta x|={absolute_error:g}$
                - Relative error:
                  $\\delta_x={relative_error:.3e}$
                - Relative-error magnitude:
                  $|\\delta_x|={relative_error_magnitude:.3e}$
                - Percentage error:
                  ${100 * relative_error_magnitude:.3e}\\%$

                The large-scale example has the larger absolute error,
                but the smaller relative error.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Error Bounds

    The exact value is often unknown. Instead of knowing the actual
    error, we may only know an upper bound:

    $$
    |\Delta x|\leq\epsilon.
    $$

    Since $\Delta x=\tilde{x}-x$, the exact value must lie inside

    $$
    \tilde{x}-\epsilon
    \leq x \leq
    \tilde{x}+\epsilon.
    $$

    These are **maximum or worst-case bounds**.

    They are not the same as statistical standard deviations or
    standard errors, which are usually combined differently.
    """)
    return


@app.cell
def _(mo):
    bound_center = mo.ui.number(
        value=4.56,
        step=0.01,
        label="Approximate value:",
    )

    bound_size = mo.ui.number(
        value=0.14,
        start=0.0,
        step=0.01,
        label="Absolute-error bound:",
    )
    return bound_center, bound_size


@app.cell
def _(bound_center, bound_size, mo):
    approximate_bound_value = bound_center.value
    epsilon_bound = bound_size.value

    lower_bound = approximate_bound_value - epsilon_bound
    upper_bound = approximate_bound_value + epsilon_bound

    mo.vstack(
        [
            mo.hstack(
                [bound_center, bound_size],
                justify="center",
            ),
            mo.md(
                f"""
                If

                $$
                |{approximate_bound_value:g}-x|
                \leq {epsilon_bound:g},
                $$

                then

                $$
                {lower_bound:g}
                \leq x \leq
                {upper_bound:g}.
                $$
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Propagation Through Addition and Subtraction

    Consider

    $$
    z=a\pm b.
    $$

    If the approximate inputs are

    $$
    \tilde{a}=a+\Delta a,
    \qquad
    \tilde{b}=b+\Delta b,
    $$

    then the error in the result is

    $$
    \Delta z=\Delta a\pm\Delta b.
    $$

    Using the triangle inequality gives the worst-case bound

    $$
    |\Delta z|
    \leq
    |\Delta a|+|\Delta b|.
    $$

    Therefore, **absolute-error bounds add for addition and
    subtraction**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Propagation Through Multiplication and Division

    Consider a product

    $$
    z=ab.
    $$

    Write the approximate inputs using relative errors:

    $$
    \tilde{a}=a(1+\delta_a),
    \qquad
    \tilde{b}=b(1+\delta_b).
    $$

    Their product is

    $$
    \tilde{z}
    =
    ab(1+\delta_a)(1+\delta_b).
    $$

    Therefore,

    $$
    \delta_z
    =
    \delta_a+\delta_b+\delta_a\delta_b.
    $$

    If the errors are small, the product
    $\delta_a\delta_b$ can be neglected:

    $$
    |\delta_z|
    \lesssim
    |\delta_a|+|\delta_b|.
    $$

    The same first-order rule applies to division.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Rule of thumb

    - For addition and subtraction, add **absolute-error bounds**.
    - For multiplication and division, add **relative-error bounds**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## General Error Propagation

    Suppose

    $$
    y=f(x)
    $$

    and the input contains a small error $\Delta x$.

    A first-order Taylor expansion gives

    $$
    f(x+\Delta x)
    \approx
    f(x)+f'(x)\Delta x.
    $$

    Therefore,

    $$
    \Delta y
    \approx
    f'(x)\Delta x.
    $$

    The derivative determines how strongly an absolute input error
    affects the output.

    Dividing by $y=f(x)$ gives the relative-error relation

    $$
    \delta_y
    \approx
    \frac{x f'(x)}{f(x)}
    \delta_x.
    $$

    The quantity

    $$
    \left|
    \frac{x f'(x)}{f(x)}
    \right|
    $$

    is the relative condition number of the function.
    """)
    return


@app.cell
def _(mo):
    propagation_power = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=4,
        label="Power p:",
    )

    input_error_exponent = mo.ui.slider(
        start=-10,
        stop=-1,
        step=1,
        value=-3,
        label="Input relative error, 10ⁿ:",
    )
    return input_error_exponent, propagation_power


@app.cell
def _(input_error_exponent, mo, propagation_power):
    power_value = propagation_power.value
    input_relative_error = 10.0**input_error_exponent.value

    exact_input = 2.0
    approximate_input = exact_input * (1.0 + input_relative_error)

    exact_output = exact_input**power_value
    approximate_output = approximate_input**power_value

    measured_output_relative_error = (approximate_output - exact_output) / exact_output

    predicted_output_relative_error = power_value * input_relative_error

    ratio = measured_output_relative_error / predicted_output_relative_error

    mo.vstack(
        [
            mo.md(
                r"""
                ## Example: $y=x^p$

                For
                $$
                f(x)=x^p,
                $$
                the relative condition number is
                $$
                \frac{x f'(x)}{f(x)}
                =
                \frac{x p x^{p-1}}{x^p}
                =
                p.
                $$
                Therefore,
                $$
                \delta_y\approx p\delta_x.
                $$
                """
            ),
            mo.hstack(
                [propagation_power, input_error_exponent],
                justify="center",
            ),
            mo.md(
                f"""
                **Using $x={exact_input}$:**

                | Input Relative Error | First-Order Prediction | Actual Output Relative Error | Actual/Predicted Ratio |
                | :--- | :--- | :--- | :--- |
                | `{input_relative_error:.3e}` | `{predicted_output_relative_error:.3e}` | `{measured_output_relative_error:.3e}` | `{ratio:.8f}` |

                The first-order prediction becomes more accurate as the input error becomes smaller.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## General Error Propagation for Several Variables

    For

    $$
    y=f(x_0,x_1,\ldots,x_{n-1}),
    $$

    a first-order Taylor expansion gives

    $$
    \Delta y
    \approx
    \sum_{i=0}^{n-1}
    \frac{\partial f}{\partial x_i}\Delta x_i.
    $$

    A worst-case bound follows from the triangle inequality:

    $$
    |\Delta y|
    \lesssim
    \sum_{i=0}^{n-1}
    \left|
    \frac{\partial f}{\partial x_i}
    \right|
    |\Delta x_i|.
    $$

    This is a **first-order approximation**. It assumes that the input
    errors are sufficiently small for quadratic and higher-order terms
    to be neglected.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### From Error Propagation back to Catastrophic Cancellation

    For a subtraction

    $$
    z=a-b,
    $$

    the absolute-error bound is

    $$
    |\Delta z|
    \leq |\Delta a|+|\Delta b|.
    $$

    If $|\Delta a| \approx |\Delta b|$ The corresponding relative-error bound is approximately

    $$
    |\delta_z|
    \lesssim
    \frac{|a|}{|a-b|}
    \left(|\delta_a|+|\delta_b|\right),
    $$

    when $a$ and $b$ have similar magnitudes.

    If $a\approx b$, then $|a-b|$ is small and the amplification factor

    $$
    \frac{|a|}{|a-b|}
    $$

    becomes very large. This is the mechanism behind
    **catastrophic cancellation**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Speed
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Computational Cost

    Numerical accuracy is not the only concern.

    We must also consider:

    - execution time,
    - memory use,
    - how cost scales with problem size.

    Big-O ($\mathcal{O}$) notation describes the asymptotic growth of an algorithm's
    resource requirements.
    """)
    return


@app.cell
def _(mo, np, plt, time):
    def countingtime(num):
        """Measure the time taken to perform a counting loop up to a specified number.

        Parameters
        ----------
        num : int
            The number up to which the counting loop will run.

        Returns
        -------
        float
            The time taken to complete the counting loop, in seconds.
        """
        timer = time.time()
        counter = 0
        for _ in range(int(num)):
            # for __ in range(int(num)):
            pass
        end_time = time.time()
        return end_time - timer

    count_lst = np.geomspace(1e3, 1e8, num=6)
    time_taken = []
    for _ in count_lst:
        time_taken.append(countingtime(_))

    # plot in loglog
    figtime = plt.figure()
    plt.plot(count_lst, time_taken, "o-", label="Time Taken")
    # add a reference line for O(n)
    plt.plot(
        count_lst, [1e-7 * _ for _ in count_lst], "r--", label=r"$\mathcal{O}(n)$"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Count Limit")
    plt.ylabel("Time Taken (seconds)")
    plt.legend(frameon=True)
    plt.grid(which="both")

    # extrapolate time for 1e9 with linear regression
    time_for_1e9 = np.interp(np.log10(1e9), np.log10(count_lst), np.log10(time_taken))

    onot = mo.md(r"""## Counting speed test
    ---
    """)

    mo.vstack(
        [onot, figtime],
    )
    return


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class SortFrame:
        values: tuple[int, ...]
        comparing: tuple[int, ...] = ()
        moving: tuple[int, ...] = ()
        sorted_indices: tuple[int, ...] = ()
        comparisons: int = 0
        movements: int = 0
        message: str = "Ready"

    return (SortFrame,)


@app.cell
def _(SortFrame):
    def bubble_sort_frames(values):
        """Generate frames for visualizing the bubble sort algorithm.

        Parameters
        ----------
        values : tuple[int, ...]
            The initial array of integers to be sorted.

        Returns
        -------
        list[SortFrame]
            A list of SortFrame instances representing each step of the bubble sort process.
        """

        a = list(values)
        n = len(a)
        comparisons = swaps = 0
        frames = [SortFrame(tuple(a))]

        for pass_number in range(n - 1):
            swapped = False
            already_sorted = tuple(range(n - pass_number, n))

            for j in range(n - pass_number - 1):
                comparisons += 1
                frames.append(
                    SortFrame(
                        tuple(a),
                        comparing=(j, j + 1),
                        sorted_indices=already_sorted,
                        comparisons=comparisons,
                        movements=swaps,
                        message=f"Comparing {a[j]} and {a[j + 1]}",
                    )
                )

                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    swaps += 1
                    swapped = True
                    frames.append(
                        SortFrame(
                            tuple(a),
                            moving=(j, j + 1),
                            sorted_indices=already_sorted,
                            comparisons=comparisons,
                            movements=swaps,
                            message=f"Swapping positions {j} and {j + 1}",
                        )
                    )

            sorted_start = n - pass_number - 1
            frames.append(
                SortFrame(
                    tuple(a),
                    sorted_indices=tuple(range(sorted_start, n)),
                    comparisons=comparisons,
                    movements=swaps,
                    message=f"Position {sorted_start} is sorted",
                )
            )
            if not swapped:
                break

        frames.append(
            SortFrame(
                tuple(a),
                sorted_indices=tuple(range(n)),
                comparisons=comparisons,
                movements=swaps,
                message="Bubble sort complete",
            )
        )
        return frames

    return (bubble_sort_frames,)


@app.cell
def _(SortFrame):
    def merge_sort_frames(values):
        """Generate frames for visualizing the merge sort algorithm.

        Parameters
        ----------
        values : tuple[int, ...]
            The initial array of integers to be sorted.

        Returns
        -------
        list[SortFrame]
            A list of SortFrame instances representing each step of the merge sort process.
        """
        a = list(values)
        n = len(a)
        comparisons = writes = 0
        frames = [SortFrame(tuple(a))]

        def add(*, comparing=(), moving=(), message=""):
            frames.append(
                SortFrame(
                    tuple(a),
                    comparing=tuple(comparing),
                    moving=tuple(moving),
                    comparisons=comparisons,
                    movements=writes,
                    message=message,
                )
            )

        def merge(left, middle, right):
            nonlocal comparisons, writes
            left_values = a[left : middle + 1]
            right_values = a[middle + 1 : right + 1]
            i = j = 0
            destination = left
            add(
                comparing=range(left, right + 1),
                message=f"Merging positions {left} to {right}",
            )

            while i < len(left_values) and j < len(right_values):
                comparisons += 1
                add(
                    comparing=(left + i, middle + 1 + j),
                    message=f"Comparing {left_values[i]} and {right_values[j]}",
                )
                if left_values[i] <= right_values[j]:
                    a[destination] = left_values[i]
                    i += 1
                else:
                    a[destination] = right_values[j]
                    j += 1
                writes += 1
                add(
                    moving=(destination,),
                    message=f"Writing {a[destination]} at position {destination}",
                )
                destination += 1

            while i < len(left_values):
                a[destination] = left_values[i]
                i += 1
                writes += 1
                add(
                    moving=(destination,),
                    message=f"Writing remaining value at position {destination}",
                )
                destination += 1

            while j < len(right_values):
                a[destination] = right_values[j]
                j += 1
                writes += 1
                add(
                    moving=(destination,),
                    message=f"Writing remaining value at position {destination}",
                )
                destination += 1

            add(
                comparing=range(left, right + 1),
                message=f"Positions {left} to {right} are merged",
            )

        def sort(left, right):
            if left >= right:
                return
            middle = (left + right) // 2
            add(
                comparing=range(left, right + 1),
                message=f"Dividing positions {left} to {right}",
            )
            sort(left, middle)
            sort(middle + 1, right)
            merge(left, middle, right)

        if n:
            sort(0, n - 1)
        frames.append(
            SortFrame(
                tuple(a),
                sorted_indices=tuple(range(n)),
                comparisons=comparisons,
                movements=writes,
                message="Merge sort complete",
            )
        )
        return frames

    return (merge_sort_frames,)


@app.cell
def _(mo):
    get_frame, set_frame = mo.state(0)
    get_playing, set_playing = mo.state(False)
    return get_frame, get_playing, set_frame, set_playing


@app.cell
def _(mo):
    algorithm = mo.ui.radio(
        options={"Bubble sort": "bubble", "Merge sort": "merge"},
        value="Bubble sort",
        label="Algorithm",
        inline=True,
    )
    array_size = mo.ui.slider(
        10,
        100,
        step=5,
        value=40,
        label="Array size",
        show_value=True,
    )
    speed = mo.ui.slider(
        1,
        10,
        step=1,
        value=4,
        label="Speed",
        show_value=True,
    )
    shuffle = mo.ui.button(
        label="Shuffle",
        value=0,
        on_click=lambda count: count + 1,
    )
    return algorithm, array_size, shuffle, speed


@app.cell
def _(array_size, random, shuffle):
    seed = 12_345 + 10_000 * shuffle.value + 100 * array_size.value
    initial_values = random.Random(seed).sample(
        range(1, array_size.value + 1), array_size.value
    )
    return (initial_values,)


@app.cell
def _(algorithm, bubble_sort_frames, initial_values, merge_sort_frames):
    animation_frames = (
        bubble_sort_frames(initial_values)
        if algorithm.value == "bubble"
        else merge_sort_frames(initial_values)
    )
    return (animation_frames,)


@app.cell
def _(algorithm, array_size, set_frame, set_playing, shuffle):
    _inputs = (algorithm.value, array_size.value, shuffle.value)
    set_playing(False)
    set_frame(0)
    return


@app.cell
def _(mo, set_frame, set_playing):
    play_pause = mo.ui.button(
        label="Start / pause",
        value=0,
        on_click=lambda count: count + 1,
        on_change=lambda _value: set_playing(lambda playing: not playing),
        kind="success",
    )

    def step_forward(_value):
        set_playing(False)
        set_frame(lambda frame: frame + 1)

    def step_back(_value):
        set_playing(False)
        set_frame(lambda frame: max(frame - 1, 0))

    def restart(_value):
        set_playing(False)
        set_frame(0)

    step = mo.ui.button(
        label="Step", value=0, on_click=lambda count: count + 1, on_change=step_forward
    )
    previous = mo.ui.button(
        label="Previous", value=0, on_click=lambda count: count + 1, on_change=step_back
    )
    restart_button = mo.ui.button(
        label="Restart", value=0, on_click=lambda count: count + 1, on_change=restart
    )
    return play_pause, previous, restart_button, step


@app.cell
def _(mo):
    # Keep this widget visibly mounted in its own stable cell.
    animation_timer = mo.ui.refresh(
        default_interval=0.1,
        label="Animation timer",
    )
    # animation_timer
    return (animation_timer,)


@app.cell
def _(
    animation_frames,
    animation_timer,
    get_frame,
    get_playing,
    set_frame,
    set_playing,
    speed,
):
    _tick = animation_timer.value
    if get_playing():
        last = len(animation_frames) - 1
        next_frame = min(get_frame() + speed.value, last)
        set_frame(next_frame)
        if next_frame >= last:
            set_playing(False)
    return


@app.cell
def _(animation_frames, get_frame):
    frame_index = min(max(get_frame(), 0), len(animation_frames) - 1)
    current_frame = animation_frames[frame_index]
    return current_frame, frame_index


@app.cell
def _():
    information = {
        "bubble": (
            "Bubble sort",
            "Repeatedly compares adjacent elements and swaps those in the wrong order.",
            "$\mathcal{O}(n)$",
            "$\mathcal{O}(n^2)$",
            "$\mathcal{O}(n^2)$",
            "$\mathcal{O}(1)$",
        ),
        "merge": (
            "Merge sort",
            "Recursively divides the sequence and merges the parts in sorted order.",
            "$\mathcal{O}$(n log n)",
            "$\mathcal{O}$(n log n)",
            "$\mathcal{O}$(n log n)",
            "$\mathcal{O}$(n)",
        ),
    }
    return (information,)


@app.cell
def _(
    algorithm,
    animation_frames,
    animation_timer,
    array_size,
    current_frame,
    frame_index,
    get_playing,
    go,
    information,
    mo,
    play_pause,
    previous,
    restart_button,
    shuffle,
    speed,
    step,
):
    item_count = len(current_frame.values)
    comparing = set(current_frame.comparing)
    moving = set(current_frame.moving)
    sorted_indices = set(current_frame.sorted_indices)
    positions = list(range(item_count))
    colors = [
        "#7c3aed"
        if i in sorted_indices
        else "#d97706"
        if i in moving
        else "#e11d48"
        if i in comparing
        else "#b8b3ab"
        for i in positions
    ]

    figure = go.Figure(
        go.Bar(
            x=positions,
            y=list(current_frame.values),
            marker=dict(color=colors, line=dict(width=0)),
            width=0.85,
            customdata=list(current_frame.values),
            hovertemplate="Position: %{x}<br>Value: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        height=400,
        margin=dict(l=45, r=20, t=60, b=35),
        title=dict(text=current_frame.message, x=0.5, xanchor="center"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        bargap=0,
        transition_duration=0,
        uirevision="sorting-visualizer",
        xaxis=dict(
            title="Position",
            range=[-0.5, item_count - 0.5],
            autorange=False,
            fixedrange=True,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Value",
            range=[0, item_count * 1.08],
            autorange=False,
            fixedrange=True,
            showgrid=True,
            gridcolor="#edeae4",
            zeroline=False,
        ),
    )
    plot = mo.ui.plotly(figure)

    complete = frame_index >= len(animation_frames) - 1
    status = (
        "Complete"
        if complete
        else "Running"
        if get_playing()
        else "Paused"
        if frame_index > 0
        else "Ready"
    )
    name, description, best, average, worst, space = information[algorithm.value]

    mo.vstack(
        [
            mo.md("# Sorting algorithms"),
            mo.hstack(
                [algorithm, array_size, speed], justify="center", gap=2, wrap=True
            ),
            mo.hstack(
                [play_pause, step, previous, restart_button, shuffle, animation_timer],
                justify="center",
                gap=1,
                wrap=True,
            ),
            mo.hstack(
                [
                    plot,
                    mo.vstack(
                        [
                            mo.stat(
                                label="Comparisons", value=current_frame.comparisons
                            ),
                            mo.stat(
                                label="Moves / writes", value=current_frame.movements
                            ),
                            mo.stat(
                                label="Status",
                                value=status,
                                caption=f"Frame {frame_index + 1} of {len(animation_frames)}",
                            ),
                        ],
                        gap=0,
                    ),
                ],
                align="center",
                justify="center",
                gap=0,
                widths=[0.8, 0.2],
            ),
            mo.md(
                "**Colour key:** 🔴 Comparing · 🟠 Moving/writing · 🟣 Sorted · ⚪ Unsorted"
            ),
            # mo.callout(mo.md(
            #     f"### {name}\n{description}\n\n"
            #     f"**Best:** `{best}` · **Average:** `{average}` · "
            #     f"**Worst:** `{worst}` · **Space:** `{space}`"
            # ), kind="info"),
        ],
        gap=0,
    )
    return


@app.cell
def _(gamma, mo, mpatches, np, plt):

    N_max = 20
    n = np.linspace(1, N_max, 400)

    # Calculate complexity curves
    y_1 = np.ones_like(n)
    y_log = np.log2(n)
    y_n = n
    y_nlog = n * np.log2(n)
    y_n2 = n**2
    y_2n = 2**n
    y_fact = gamma(n + 1)  # Continuous approximation of n!

    # Curve color palette
    c_green = "#2ecc71"
    c_yellow = "#f1c40f"
    c_red = "#e74c3c"

    # Plot styling
    fig_com, ax_com = plt.subplots(figsize=(11, 6), dpi=120)
    plt.subplots_adjust(
        left=0.1, right=0.75, bottom=0.15
    )  # Adjust right margin for legend

    # Plot the curves
    ax_com.plot(n, y_1, color=c_green, lw=2.5)
    ax_com.plot(n, y_log, color=c_green, lw=2.5)
    ax_com.plot(n, y_n, color=c_yellow, lw=2.5)
    ax_com.plot(n, y_nlog, color=c_yellow, lw=2.5)
    ax_com.plot(n, y_n2, color=c_red, lw=2.5)
    ax_com.plot(n, y_2n, color=c_red, lw=2.5)
    ax_com.plot(n, y_fact, color=c_red, lw=2.5)

    # Limit Y-axis to keep visual scale
    _ymax = 50
    ax_com.set_ylim(0, _ymax)
    ax_com.set_xlim(2, N_max)

    # Add inline labels directly above the curves
    font_kwargs = {
        "fontsize": 13,
        "fontweight": "bold",
        "va": "bottom",
        "bbox": dict(facecolor="white", edgecolor="none", alpha=0.8),
    }
    ax_com.text(17.5, 1.2, "$\mathcal{O}(1)$", color=c_green, **font_kwargs)
    ax_com.text(16, 4.3, "$\mathcal{O}(log n)$", color=c_green, **font_kwargs)
    ax_com.text(17.5, 18, "$\mathcal{O}(n)$", color=c_yellow, **font_kwargs)
    ax_com.text(
        int(N_max / 3),
        _ymax / 2,
        "$\mathcal{O}(n log n)$",
        color=c_yellow,
        **font_kwargs,
    )
    ax_com.text(6.5, _ymax + 1, "$\mathcal{O}(n²)$", color=c_red, **font_kwargs)
    ax_com.text(4.5, _ymax + 1, "$\mathcal{O}(2ⁿ)$", color=c_red, **font_kwargs)
    ax_com.text(2.5, _ymax + 1, "$\mathcal{O}(n!)$", color=c_red, **font_kwargs)

    # Style Axes (Remove top/right spines, keep defaults for others)
    ax_com.spines["top"].set_visible(False)
    ax_com.spines["right"].set_visible(False)
    ax_com.spines["left"].set_linewidth(1.5)
    ax_com.spines["bottom"].set_linewidth(1.5)

    # Standard Axis Labels
    ax_com.set_xlabel("Input size (n)", fontsize=14, fontweight="bold", labelpad=10)
    ax_com.set_ylabel(
        "Running Time Complexity ($\mathcal{O}$)",
        fontsize=14,
        fontweight="bold",
        labelpad=10,
    )

    # Remove standard ticks
    ax_com.set_xticks([])
    ax_com.set_yticks([])

    # Add custom Good/OK/Slow Legend
    legend_elements = [
        (0.55, c_green, "Good"),
        (0.40, c_yellow, "OK"),
        (0.25, c_red, "Slow"),
    ]
    box_x = 1.05

    for y_pos, color, text in legend_elements:
        # Draw rounded rectangle patch
        rect = mpatches.FancyBboxPatch(
            (box_x, y_pos),
            0.1,
            0.08,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            fc=color,
            ec="none",
            transform=ax_com.transAxes,
            clip_on=False,
        )
        ax_com.add_patch(rect)

        # Draw standard arrow pointing from box to text
        ax_com.annotate(
            "",
            xy=(box_x + 0.18, y_pos + 0.04),
            xytext=(box_x + 0.10, y_pos + 0.04),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            xycoords="axes fraction",
            clip_on=False,
        )

        # Render text label
        ax_com.text(
            box_x + 0.20,
            y_pos + 0.04,
            text,
            color="black",
            fontsize=14,
            va="center",
            transform=ax_com.transAxes,
        )

    # Display in marimo
    mo.vstack([mo.md("#**Big-O Complexity Graph**"), mo.center(fig_com)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cool Youtube videos:

    - [Big-O Notation in 3 Minutes](https://youtu.be/x2CRZaN2xgM?si=EY35o4kCMLXxMciy) by [ByteByteGo](https://www.youtube.com/@ByteByteGo)

    - [Getting Sorted & Big O Notation - Computerphile](https://youtu.be/kgBjXUE_Nwc?si=htk0GoaDqUPDkkMp) by [Computerphile](https://www.youtube.com/@Computerphile)

    - [Visualization of 24 Sorting Algorithms In 2 Minutes](https://www.youtube.com/watch?v=BeoCbJPuvSE) by [Tushar Roy - Coding Made Simple](https://www.youtube.com/@TusharRoy-CodingMadeSimple)
    """)
    return


@app.cell
def _(mo):
    NaCl_md1 = mo.md(r"""
    ## The Madelung Constant ($\mathcal{M}$)
    *Visualizing Electrostatic Lattice Potentials in NaCl (Computational Physics Ch. 4)*
    """)
    NaCl_md2 = mo.md(r"""
    In a solid sodium chloride ($\text{NaCl}$) crystal, sodium ($\text{Na}^+$) and chlorine ($\text{Cl}^-$)
    ions form an alternating simple cubic lattice. The total electrostatic potential felt by a
    sodium ion located at the origin $(0, 0, 0)$ is given by:  

    $$V_{\text{total}} = \frac{e}{4\pi\varepsilon_0 a} \mathcal{M}$$  

    where $a$ is the lattice constant and $\mathcal{M}$ is the dimensionless **Madelung Constant**:  

    $$\mathcal{M} = \sum_{\substack{i,j,k = -L \\ (i,j,k) \neq (0,0,0)}}^{L} \frac{(-1)^{i+j+k}}{\sqrt{i^2 + j^2 + k^2}}$$
    """)
    return NaCl_md1, NaCl_md2


@app.cell
def _(NaCl_md1, NaCl_md2, mg, mo, pmv):
    # NaCl forms a face-centered cubic (FCC) rock-salt structure (Fm-3m)
    # Lattice constant a ≈ 5.64 Å
    NaCl_struct = mg.Structure.from_spacegroup(
        "Fm-3m", mg.Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    struct_widget = pmv.StructureWidget(
        structure=NaCl_struct,
        show_bonds=True,
        style="height: 500px;",
        png_dpi=100,
        enable_info_pane=True,
    )  # , version_override="0.6.0") #There is a bug in v0.4.0 that makes the slide exit fullscreen (unless the widget what interacted with before) v0.4.1 does not exit full screen but outputs the "wrong structure"

    mo.vstack(
        [
            NaCl_md1,
            mo.hstack(
                [NaCl_md2, struct_widget], align="center", gap=0.5, widths=[0.4, 0.6]
            ),
        ]
    )
    return


@app.cell
def _(mo, np):
    L = 10
    M = 0.0
    for ii in range(-L, L + 1):
        for jj in range(-L, L + 1):
            for kk in range(-L, L + 1):
                if ii == 0 and jj == 0 and kk == 0:
                    continue
                if (ii + jj + kk) % 2 == 0:
                    M += 1 / np.sqrt(ii * ii + jj * jj + kk * kk)
                else:
                    M -= 1 / np.sqrt(ii * ii + jj * jj + kk * kk)
    mo.md(rf"""
    # Madelung constant if L = {L} is {M}

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # What is the time complexity of the Madelung constant calculation here?
    """)
    return


@app.cell
def _(mo):
    l_slider = mo.ui.slider(
        start=1, stop=30, step=1, value=6, label="Lattice Horizon ($L$):"
    )

    view_mode = mo.ui.dropdown(
        options={"Show All Atoms": "all", "Show Cross-Section (k=0)": "slice"},
        value="Show All Atoms",
        label="3D View Filter:",
    )
    return l_slider, view_mode


@app.cell
def _(l_slider, np, time):
    L_ = l_slider.value
    start_time = time.perf_counter()

    # Create 3D grid of coordinates using NumPy vectorized operations
    axis = np.arange(-L_, L_ + 1)
    i_, j_, k_ = np.meshgrid(axis, axis, axis, indexing="ij")

    # Mask out the origin (0,0,0) where the potential is evaluated
    mask = ~((i_ == 0) & (j_ == 0) & (k_ == 0))

    i_valid = i_[mask]
    j_valid = j_[mask]
    k_valid = k_[mask]

    # Calculate radial distances r = sqrt(i^2 + j^2 + k^2)
    r = np.sqrt(i_valid**2 + j_valid**2 + k_valid**2)

    # Determine ionic sign: (-1)^(i+j+k) -> Even sum = +1 (Na+), Odd sum = -1 (Cl-)
    signs = (-1.0) ** (i_valid + j_valid + k_valid)

    # Calculate contribution terms
    terms = signs / r
    madelung_val = np.sum(terms)

    calc_time = (time.perf_counter() - start_time) * 1000  # in ms
    total_atoms = (2 * L_ + 1) ** 3 - 1

    # Pack everything into a unified state tuple
    madelung_data = (
        L_,
        i_valid,
        j_valid,
        k_valid,
        signs,
        madelung_val,
        calc_time,
        total_atoms,
    )
    return (madelung_data,)


@app.cell
def _(go, np):
    def make_3d_lattice_plot(madelung_data, view_mode_val):
        (L, i_v, j_v, k_v, signs, m_val, c_time, num_atoms) = madelung_data

        if view_mode_val == "slice":
            filter_mask = k_v == 0
        else:
            filter_mask = np.ones_like(i_v, dtype=bool)

        i_p = i_v[filter_mask]
        j_p = j_v[filter_mask]
        k_p = k_v[filter_mask]
        s_p = signs[filter_mask]

        na_mask = s_p == 1
        cl_mask = s_p == -1

        fig = go.Figure()

        # Target Na+ Ion at Origin
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(
                    size=14,
                    color="gold",
                    symbol="circle",
                    line=dict(color="black", width=2),
                    opacity=1,
                ),
                name="Target Na⁺ (Origin)",
            )
        )

        # Na+ Ions (Positive Charges)
        fig.add_trace(
            go.Scatter3d(
                x=i_p[na_mask],
                y=j_p[na_mask],
                z=k_p[na_mask],
                mode="markers",
                marker=dict(size=6, color="#d0c081", opacity=0.8),
                name="Na⁺ (+e)",
            )
        )

        # Cl- Ions (Negative Charges)
        fig.add_trace(
            go.Scatter3d(
                x=i_p[cl_mask],
                y=j_p[cl_mask],
                z=k_p[cl_mask],
                mode="markers",
                marker=dict(size=6, color="#8fcc6c", opacity=0.8),
                name="Cl⁻ (-e)",
            )
        )

        fig.update_layout(
            title=f"NaCl Crystal Lattice (L = {L}, Total Atoms = {num_atoms:,})",
            scene=dict(
                xaxis_title="i", yaxis_title="j", zaxis_title="k", aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=450,
        )

        return fig

    return (make_3d_lattice_plot,)


@app.cell
def _(np, pd):
    def get_convergence_table(madelung_data):
        curr_L = madelung_data[0]
        l_range = np.arange(1, min(curr_L + 1, 20))
        m_results = []

        for l_idx in l_range:
            axis = np.arange(-l_idx, l_idx + 1)
            i, j, k = np.meshgrid(axis, axis, axis, indexing="ij")
            mask = ~((i == 0) & (j == 0) & (k == 0))
            r = np.sqrt(i[mask] ** 2 + j[mask] ** 2 + k[mask] ** 2)
            s = (-1.0) ** (i[mask] + j[mask] + k[mask])
            m_results.append(np.sum(s / r))

        return pd.DataFrame(
            {
                "L": l_range,
                "Atoms": (2 * l_range + 1) ** 3 - 1,
                "Madelung M": m_results,
                "Abs Error": np.abs(np.array(m_results) - (-1.747565)),
            }
        )

    return (get_convergence_table,)


@app.cell
def _(
    get_convergence_table,
    l_slider,
    madelung_data,
    make_3d_lattice_plot,
    mo,
    view_mode,
):
    # (L, i_v, j_v, k_v, signs, madelung_val, calc_time, total_atoms) = madelung_data

    exact_m = -1.747565
    rel_err = abs(madelung_data[5] - exact_m) / abs(exact_m) * 100

    metrics = mo.hstack(
        [
            mo.stat(label="Calculated M", value=f"{madelung_data[5]:.6f}"),
            mo.stat(label="Exact Limit M", value=f"{exact_m:.6f}"),
            mo.stat(label="Relative Error", value=f"{rel_err:.3f}%"),
            mo.stat(label="Compute Time", value=f"{madelung_data[6]:.2f} ms"),
        ],
        justify="start",
    )

    figmad = make_3d_lattice_plot(madelung_data, view_mode.value)
    df_table = get_convergence_table(madelung_data)

    mo.vstack(
        [
            mo.hstack([l_slider, view_mode], align="center", justify="start"),
            metrics,
            mo.hstack(
                [figmad, mo.vstack([mo.md("### Convergence Table"), df_table])],
                align="center",
                justify="start",
                gap=2,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2.2: Factorials and Precision

    Write a program to calculate and print the factorial of a number entered by the user. If you wish, you can base your program on a user-defined function for factorial, but write your program so that it calculates the factorial using **integer** variables, not floating-point ones. Use your program to calculate the factorial of $200$.

    Now modify your program to use floating-point variables instead and again calculate the factorial of $200$. What do you find? Explain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2.3: Calculating Derivatives

    Suppose we have a function $f(x)$ and we want to calculate its derivative at a point $x$. We can do that with pencil and paper if we know the mathematical form of the function, or we can do it on the computer by making use of the definition of the derivative:

    $$\frac{df}{dx} = \lim_{h\to0} \frac{f(x+h)-f(x)}{h}$$

    On the computer we can't actually take the limit as $h$ goes to zero, but we can get a reasonable approximation just by making $h$ small.

    1. Write a program that defines a function `f(x)` returning the value $x(x-1)$, then calculates the derivative of the function at the point $x=1$ using the formula above with $h=10^{-2}$. Calculate the true value of the same derivative analytically and compare with the answer your program gives. The two will not agree perfectly. Why not?

    2. Repeat the calculation for $h=10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}, 10^{-12}$, and $10^{-14}$. You should see that the accuracy of the calculation initially gets better as $h$ gets smaller, but then gets worse again. Why is this?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2.4: Calculating Variances

    The mean and variance of a set of $N$ numbers $x_1 \dots x_N$ are given by the standard formulas:

    $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \qquad \text{var}(x) = \frac{1}{N} \sum_{i=1}^N (x_i-\bar{x})^2$$

    The latter expression is often rewritten as:

    $$\text{var}(x) = \frac{1}{N} \sum_{i=1}^N \left( x_i^2 - 2 x_i \bar{x} + \bar{x}^2 \right) = \overline{x^2} - \bar{x}^2$$

    where $\overline{x^2}$ is the mean-square value of $x$.

    1. By any means you like, work out on paper the variance of the five numbers $x-2$, $x-1$, $x$, $x+1$, and $x+2$. You should find that the result is just a constant, independent of $x$.

    2. Setting $x$ to 1 billion, write a program to calculate the variance of these five numbers in two different ways using the two expressions above. What do you observe and how do you explain it?

    3. In practice, if you were going to write a program to calculate the variance, which formula should you use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2.5: Calculating Integrals

    Suppose we want to calculate the value of the integral:

    $$I = \int_{-1}^1 \sqrt{1-x^2} \, dx$$

    The integrand looks like a semicircle of radius 1, and hence the value of the integral—the area under the curve—must be $\frac{1}{2}\pi = 1.57079632679\dots$

    Alternatively, we can evaluate the integral on the computer by dividing the domain of integration into a large number $N$ of slices of width $h=2/N$ each and then using the Riemann definition of the integral:

    $$I = \lim_{N\to\infty} \sum_{k=1}^N h y_k$$

    where:

    $$y_k = \sqrt{1 - x_k^2} \qquad \text{and} \qquad x_k = -1 + hk$$

    We cannot in practice take the limit $N\to\infty$, but we can make a reasonable approximation by just making $N$ large.

    1. Write a program to evaluate the integral above with $N=100$ and compare the result with the exact value. The two will not agree very well, because $N=100$ is not a sufficiently large number of slices.

    2. Increase the value of $N$ to get a more accurate value for the integral. If we require that the program runs in about one second or less, how accurate a value can you get?
    """)
    return


if __name__ == "__main__":
    app.run()
