# /// script
# dependencies = [
#     "altair==6.0.0",
#     "anthropic==0.79.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "openai==2.26.0",
#     "plotly==6.5.2",
#     "polars==1.38.1",
#     "pydantic-ai-slim==1.57.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Week 9 Session 2 - Adam
    """)
    return


@app.cell
def _():
    from abc import ABC, abstractmethod

    class Optimizer(ABC):
        @abstractmethod
        def step(self, point, grad_f):
            pass

    return (Optimizer,)


@app.function
# Gradient function for z=x^2 + 10*y^2
def grad_f(x, y):
    return (2 * x, 20 * y)


@app.cell
def _(Optimizer):
    class SGD(Optimizer):
        def __init__(self, learning_rate, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.weight_decay=weight_decay

        # Taking a step using GD for the function z=x^2 + 10*y^2
        def step(self, point, grad_f):
            (x, y) = point
            (dzdx, dzdy) = grad_f(x, y)
            new_x = (1-self.learning_rate*self.weight_decay)*x - self.learning_rate * dzdx
            new_y = (1-self.learning_rate*self.weight_decay)*y - self.learning_rate * dzdy
            return (new_x, new_y)

    return (SGD,)


@app.cell
def _(SGD):
    sgd = SGD(0.09)
    starting_point = (-3, 1)
    sgd_next_step = starting_point
    for j in range(15):
        sgd_next_step = sgd.step(sgd_next_step, grad_f)
        print(sgd_next_step)
    return


@app.cell
def _(Optimizer):
    class MomentumSGD(Optimizer):
        def __init__(self, learning_rate, beta, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.beta = beta
            self.vx_prev = 0
            self.vy_prev = 0
            self.weight_decay = weight_decay

        # Taking a step using Momentum GD for the function z=x^2 + 10*y^2
        # Currently velocity state is hardcoded, I should generalize
        # TODO: Replace point with a list length d and use tuple from grad_f, v_prev should be a list
        def step(self, point, grad_f):
            (x, y) = point
            (dzdx, dzdy) = grad_f(x, y)
            vx = self.beta * self.vx_prev + dzdx
            vy = self.beta * self.vy_prev + dzdy
            self.vx_prev = vx
            self.vy_prev = vy
            new_x = (1-self.learning_rate*self.weight_decay)*x - self.learning_rate * vx
            new_y = (1-self.learning_rate*self.weight_decay)*y - self.learning_rate * vy
            return (new_x, new_y)

    return (MomentumSGD,)


@app.cell
def _(Optimizer, np):
    class Adam(Optimizer):
        def __init__(self, learning_rate, beta1, beta2, epsilon, correction=True):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = (0.0,0.0)
            self.v = (0.0,0.0)
            self.correction=correction
            self.step_counter = 1

        def step(self, point, grad_f):
            (x, y) = point
            (dzdx, dzdy) = grad_f(x, y)
            (mx,my) = self.m
            (vx,vy) = self.v
            mx = self.beta1 * mx + (1-self.beta1)*dzdx
            my = self.beta1 * my + (1-self.beta1)*dzdy
            vx = self.beta2 * vx + (1-self.beta2)*dzdx**2
            vy = self.beta2 * vy + (1-self.beta2)*dzdy**2
            self.m=(mx,my)
            self.v=(vx,vy)
            mx_hat = mx
            my_hat = my 
            vx_hat = vx
            vy_hat = vy
            if self.correction:
                mx_hat = mx / (1 - self.beta1 ** self.step_counter)
                my_hat = my / (1 - self.beta1 ** self.step_counter)
                vx_hat = vx / (1 - self.beta2 ** self.step_counter)
                vy_hat = vy / (1 - self.beta2 ** self.step_counter)

            new_x = x - (self.learning_rate * mx_hat)/(np.sqrt(vx_hat) + self.epsilon)
            new_y = y - (self.learning_rate * my_hat)/(np.sqrt(vy_hat) + self.epsilon)
            self.step_counter += 1
            return (new_x, new_y)

    return (Adam,)


@app.cell
def _(Adam, MomentumSGD, SGD, plt):
    def plot_loss_sgd_and_momentum_decay(
        lr: float = 0.09,
        start: tuple[float, float] = (-3.0, 1.0),
        steps: int = 40,
        beta: float = 0.9,
        logy: bool = False,
        weight_decay: float = 0.5, # Using a very large decay to make the difference visible, in practice this would be 10^{-4} to 10^{-2}
    ):
        def f(_x: float, _y: float) -> float:
            return _x**2 + 1000 * _y**2

        def _run(
            optimizer, start_point: tuple[float, float], n_steps: int
        ) -> list[float]:
            p = start_point
            losses: list[float] = [f(*p)]
            for _ in range(n_steps):
                p = optimizer.step(p, grad_f)
                losses.append(f(*p))
            (x,y)=p
            print(f"total squared norm: {x**2+y**2}")
            return losses

        fig, ax = plt.subplots(figsize=(10, 6))

        # SGD (no decay)
        print("sgd no decay")
        sgd_losses = _run(SGD(lr), start, steps)
        ax.plot(
            range(len(sgd_losses)),
            sgd_losses,
            label=f"SGD (lr={lr}, wd=0)",
            color="k",
            linewidth=2,
            linestyle="-",
            alpha=0.9,
        )
        ax.plot(len(sgd_losses) - 1, sgd_losses[-1], marker="o", color="k", markersize=6)

        # SGD with decay
        print("sgd with decay")
        _wd = weight_decay
        sgd_decay_losses = _run(SGD(lr, weight_decay=_wd), start, steps)
        ax.plot(
            range(len(sgd_decay_losses)),
            sgd_decay_losses,
            label=f"SGD (lr={lr}, wd={_wd:g})",
            color="tab:red",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
        )
        ax.plot(
            len(sgd_decay_losses) - 1,
            sgd_decay_losses[-1],
            marker="s",
            color="tab:red",
            markersize=6,
        )

        # Momentum SGD without decay
        print("msgd without decay")
        msgd_losses = _run(MomentumSGD(lr, beta), start, steps)
        ax.plot(
            range(len(msgd_losses)),
            msgd_losses,
            label=f"Momentum (β={beta}, lr={lr}, wd=0)",
            color="tab:blue",
            linewidth=2,
            linestyle="-",
            alpha=0.9,
        )
        ax.plot(
            len(msgd_losses) - 1,
            msgd_losses[-1],
            marker="^",
            color="tab:blue",
            markersize=6,
        )

        # Momentum SGD with decay
        print("msgd with decay")
        msgd_decay_losses = _run(MomentumSGD(lr, beta, weight_decay=_wd), start, steps)
        ax.plot(
            range(len(msgd_decay_losses)),
            msgd_decay_losses,
            label=f"Momentum (β={beta}, lr={lr}, wd={_wd:g})",
            color="tab:green",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
        )
        ax.plot(
            len(msgd_decay_losses) - 1,
            msgd_decay_losses[-1],
            marker="D",
            color="tab:green",
            markersize=6,
        )

        # Adam
        print("adam")
        beta1=0.9
        beta2=0.999
        epsilon=1e-1
        lr=0.09
        adam_losses = _run(Adam(lr, beta1,beta2, epsilon), start, steps)
        ax.plot(
            range(len(adam_losses)),
            adam_losses,
            label=f"Adam (β1={beta1}, β2={beta2},lr={lr})",
            color="tab:purple",
            linewidth=2,
            linestyle="-",
            alpha=0.9,
        )
        ax.plot(
            len(msgd_losses) - 1,
            msgd_losses[-1],
            marker="^",
            color="tab:purple",
            markersize=6,
        )

        print("adam w/o correction")

        adam_losses = _run(Adam(lr, beta1,beta2, epsilon, correction=False), start, steps)
        ax.plot(
            range(len(adam_losses)),
            adam_losses,
            label=f"Adam w/o correction (β1={beta1}, β2={beta2},lr={lr})",
            color="tab:orange",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
        )
        ax.plot(
            len(msgd_losses) - 1,
            msgd_losses[-1],
            marker="^",
            color="tab:orange",
            markersize=6,
        )

        if logy:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss f(x, y) = x^2 + 10y^2")
        ax.set_title(f"Loss vs Steps: SGD vs Momentum (start={start}, steps={steps})")
        ax.grid(True, alpha=0.3)
        ax.legend()


        return plt.gca()

    plot_loss_sgd_and_momentum_decay()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations
    SGD still performs very well here, even with a large $\kappa$ value since the problem is simple. However, Adam is more robust w/r/t learning rate and has less osciallation than the momentum implementations which begin to oscillate heavily with large $\kappa$. Removing the bias correction shows much stronger osciallation in the beginning, until around step 20 when the two Adam graphs converge. This is due to the small values of $\sqrt{v_1}$ in the denominator early on causing large steps and overshooting the ravine.
    """)
    return


if __name__ == "__main__":
    app.run()
