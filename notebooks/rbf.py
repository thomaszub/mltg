import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    from collections.abc import Callable

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Callable, mo, np, plt


@app.cell
def _(Callable, mo, np):
    def get_tanh_rbf(eps: float) -> Callable:
        def _tanh_rbf(x: np.ndarray[tuple[int], np.dtype[np.number]]) -> np.ndarray[tuple[int], np.dtype[np.number]]:
            r = np.linalg.norm(x)
            return r * np.tanh(eps * r)

        return _tanh_rbf

    xs = np.linspace(0.0, 2.0, 50)
    epsilon = mo.ui.slider(0.0, 4.0, 0.2, label="epsilon", show_value=True, value=1.0)
    epsilon
    return epsilon, get_tanh_rbf, xs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Equation:
    $$
    f_\epsilon(\vec{x}) = \|\vec{x}\|_2 ~ \tanh{\left( \epsilon \|\vec{x}\|_2 \right)}
    $$
    """)
    return


@app.cell
def _(epsilon, get_tanh_rbf, mo, plt, xs):
    plt.plot(xs, xs, label="x")
    plt.plot(xs, list(map(get_tanh_rbf(epsilon.value), xs)), label=f"eps={epsilon.value}")
    plt.legend()
    ax = mo.ui.matplotlib(plt.gca())
    ax
    return


if __name__ == "__main__":
    app.run()
