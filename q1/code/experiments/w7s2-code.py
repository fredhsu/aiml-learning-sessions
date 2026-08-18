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

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(plt):
    def generic_contour_with_gradient_path(
        X, Y, Z, points_list, title="", origin_label="Global Minimum (0,0)"
    ):
        fig, ax = plt.subplots(figsize=(10, 8))
        CS = ax.contour(X, Y, Z, levels=20)
        ax.clabel(CS, fontsize=9)
        ax.set_title(title)

        # Plot the path with arrows
        for i in range(len(points_list) - 1):
            px, py = points_list[i]
            next_px, next_py = points_list[i + 1]

            # Draw arrow from current point to next
            ax.annotate(
                "",
                xy=(next_px, next_py),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
            )

            # Mark the point
            ax.plot(px, py, "ro", markersize=8)

        # Mark final point
        ax.plot(
            points_list[-1][0],
            points_list[-1][1],
            "go",
            markersize=10,
            label=f"Final: ({points_list[-1][0]:.2f}, {points_list[-1][1]:.2f})",
        )

        # Mark minimum at (0, 0)
        ax.plot(0, 0, "b*", markersize=20, label=origin_label)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return plt.gca()

    return (generic_contour_with_gradient_path,)


@app.cell
def _(np):
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    return X, Y


@app.cell
def _(X, Y, generic_contour_with_gradient_path):
    def isotropic_bowl():
        Z = X**2 + Y**2
        point_x = 3
        point_y = -1
        learning_rate = 0.3
        points_list = [(point_x, point_y)]
        dzdx = 2 * point_x
        dzdy = 2 * point_y
        newx = point_x - learning_rate * dzdx
        newy = point_y - learning_rate * dzdy
        points_list.append((newx, newy))
        for _ in range(5):
            dzdx = 2 * newx
            dzdy = 2 * newy
            newx = newx - learning_rate * dzdx
            newy = newy - learning_rate * dzdy
            points_list.append((newx, newy))
        return generic_contour_with_gradient_path(
            X, Y, Z, points_list, "Gradient Descent on bowl $Z=X^2 + Y^2$"
        )

    isotropic_bowl()
    return


@app.cell
def _(mo):
    mo.md(
        text="The isotropic bowl is pretty well behaved, as long as the learning rate is reasonable it will converge in a straight line to the global minimum. If LR is very large it will explode."
    )
    return


@app.cell
def _(generic_contour_with_gradient_path, np):
    def narrow_ravine():
        learning_rate = 0.09
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 10 * Y**2

        point_x = -3
        point_y = 1
        points_list = [(point_x, point_y)]
        dzdx = 2 * point_x
        dzdy = 20 * point_y
        newx = point_x - learning_rate * dzdx
        newy = point_y - learning_rate * dzdy
        points_list.append((newx, newy))
        for _ in range(15):
            dzdx = 2 * newx
            dzdy = 20 * newy
            newx = newx - learning_rate * dzdx
            newy = newy - learning_rate * dzdy
            points_list.append((newx, newy))
        return generic_contour_with_gradient_path(
            X, Y, Z, points_list, "Gradient Descent on narrow ravine $Z=X^2 + 10Y^2$"
        )

    narrow_ravine()
    return


@app.cell
def _(mo):
    mo.md(
        text="The descent oscillates back and forth across the ravine, and does not make it to the global min after 15 steps. Changing the LR larger can lead to non-convergence. Smaller will fall even further short of the minimum in the given number of steps. Ravines come from very unequal eigenvalues"
    )
    return


@app.cell
def _(l2_gradient_descent, np, plt):
    def momentum_gradient_descent(
        f,
        grad_f,
        x0: float,
        y0: float,
        lr: float = 0.09,
        momentum: float = 0.9,
        steps: int = 20,
    ) -> list[tuple[float, float]]:
        """Gradient descent with momentum."""
        points = [(x0, y0)]
        x, y = x0, y0
        vx, vy = 0.0, 0.0  # velocity terms

        for _ in range(steps):
            gx, gy = grad_f(x, y)

            # Update velocity with momentum
            vx = momentum * vx - lr * gx
            vy = momentum * vy - lr * gy

            # Update position
            x = x + vx
            y = y + vy
            points.append((x, y))

        return points

    def compare_momentum_on_ravine(momentum=0.9, lr=0.09):
        """Compare regular GD vs GD with momentum on the narrow ravine."""

        # Define ravine function: f(x,y) = x² + 10y²
        def f(x: float, y: float) -> float:
            return x**2 + 10 * y**2

        def grad_f(x: float, y: float) -> tuple[float, float]:
            return (2 * x, 20 * y)

        # Create mesh for contour plot
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-1.5, 1.5, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 10 * Y**2

        # Starting point
        x0, y0 = -3.0, 1.0

        # Run both optimizers
        regular_points = l2_gradient_descent(f, grad_f, x0, y0, lr, steps=20)
        momentum_points = momentum_gradient_descent(
            f, grad_f, x0, y0, lr, momentum, steps=20
        )

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        CS = ax.contour(X, Y, Z, levels=25)
        ax.clabel(CS, fontsize=9)

        # Regular GD path (oscillating)
        reg_x = [p[0] for p in regular_points]
        reg_y = [p[1] for p in regular_points]
        ax.plot(
            reg_x, reg_y, "r-", linewidth=2, alpha=0.7, label="Regular GD (zig-zag)"
        )
        ax.plot(reg_x[0], reg_y[0], "ro", markersize=10)
        ax.plot(reg_x[-1], reg_y[-1], "rs", markersize=10)

        # Momentum GD path (smooth)
        mom_x = [p[0] for p in momentum_points]
        mom_y = [p[1] for p in momentum_points]
        ax.plot(
            mom_x, mom_y, "b-", linewidth=2, alpha=0.7, label="Momentum GD (smooth)"
        )
        ax.plot(mom_x[0], mom_y[0], "bo", markersize=10)
        ax.plot(mom_x[-1], mom_y[-1], "bs", markersize=10)

        # Mark minimum
        ax.plot(0, 0, "g*", markersize=20, label="Global minimum")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("Momentum Solves Zig-Zag: $f(x,y) = x^2 + 10y^2$", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return plt.gca()

    compare_momentum_on_ravine(lr=0.09, momentum=0.3)
    return (compare_momentum_on_ravine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the above we can see that adding momentum dampens out the zig-zag. However if $\beta$ is too large it can overshoot and spiral as seen below:
    """)
    return


@app.cell
def _(compare_momentum_on_ravine):
    compare_momentum_on_ravine(lr=0.09, momentum=0.7)
    return


@app.cell
def _(generic_contour_with_gradient_path, np):
    def saddle():
        learning_rate = 0.03
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 - Y**2

        point_x = -3
        point_y = 1
        points_list = [(point_x, point_y)]
        dzdx = 2 * point_x
        dzdy = -2 * point_y
        newx = point_x - learning_rate * dzdx
        newy = point_y - learning_rate * dzdy
        points_list.append((newx, newy))
        for _ in range(15):
            dzdx = 2 * newx
            dzdy = -2 * newy
            newx = newx - learning_rate * dzdx
            newy = newy - learning_rate * dzdy
            points_list.append((newx, newy))
        return generic_contour_with_gradient_path(
            X,
            Y,
            Z,
            points_list,
            "Gradient Descent on saddle $Z=X^2 - Y^2$",
            "Saddle point (0,0)",
        )

    saddle()
    return


@app.cell
def _(mo):
    mo.md(
        text="With the saddle we veer off into negative infinity in the y direction. The only way to reach the saddle is to start with y=0, but that will still be unstable. If x and y = 0 we could land on the saddle point, but is unlikely to get stuck there in practice."
    )
    return


@app.cell
def _(mo):
    mo.md(
        text="All three behaviors follow from solving the same constrained linear problem under different local curvature geometries."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## L1 vs L2 Metrics: Different Norms, Different Paths

    The "steepest descent" direction depends on which norm we use to measure step size.
    Under Euclidean (L2) norm, the constraint $\|\Delta x\|_2 \leq \epsilon$ is a **circle**.
    Under L1 norm, the constraint $\|\Delta x\|_1 \leq \epsilon$ is a **diamond**.

    Linear optimization over a diamond hits the **corners** → coordinate-aligned steps!
    """)
    return


@app.cell
def _(np, plt):
    def plot_constraint_regions(gradient=(2, 1), epsilon=1.0):
        """Visualize L1 and L2 constraint regions with optimal step directions."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # L2 circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = epsilon * np.cos(theta)
        y_circle = epsilon * np.sin(theta)
        ax.plot(x_circle, y_circle, "b-", linewidth=2, label="L2 constraint (circle)")

        # L1 diamond
        x_diamond = np.array([epsilon, 0, -epsilon, 0, epsilon])
        y_diamond = np.array([0, epsilon, 0, -epsilon, 0])
        ax.plot(
            x_diamond, y_diamond, "r-", linewidth=2, label="L1 constraint (diamond)"
        )

        # Gradient direction (negated for descent)
        gx, gy = gradient
        grad_norm = np.sqrt(gx**2 + gy**2)

        # Plot the gradient covector as a line (level set normal)
        t = np.linspace(-1.5, 1.5, 100)
        # Line perpendicular to gradient: gx*x + gy*y = 0
        # Solve for y: y = -(gx/gy)*x
        if abs(gy) > 1e-6:
            y_line = -(gx / gy) * t
            ax.plot(
                t,
                y_line,
                "k--",
                alpha=0.3,
                linewidth=1,
                label="Level set (⊥ to gradient)",
            )

        # Arrow showing gradient direction
        ax.arrow(
            0,
            0,
            gx / grad_norm * 0.8,
            gy / grad_norm * 0.8,
            head_width=0.15,
            head_length=0.1,
            fc="black",
            ec="black",
            linewidth=2,
            label="Gradient direction",
        )

        # Optimal L2 step: -ε * ∇f / ||∇f||
        step_L2_x = -epsilon * gx / grad_norm
        step_L2_y = -epsilon * gy / grad_norm
        ax.plot([0, step_L2_x], [0, step_L2_y], "b-", linewidth=3, alpha=0.7)
        ax.plot(step_L2_x, step_L2_y, "bo", markersize=12, label="Optimal L2 step")

        # Optimal L1 step: goes to corner
        # Minimize gx*Δx + gy*Δy subject to |Δx| + |Δy| ≤ ε
        # Solution: put all budget in dimension with larger gradient magnitude
        if abs(gx) > abs(gy):
            step_L1_x = -epsilon * np.sign(gx)
            step_L1_y = 0
        else:
            step_L1_x = 0
            step_L1_y = -epsilon * np.sign(gy)

        ax.plot([0, step_L1_x], [0, step_L1_y], "r-", linewidth=3, alpha=0.7)
        ax.plot(step_L1_x, step_L1_y, "ro", markersize=12, label="Optimal L1 step")

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.set_xlabel("Δx", fontsize=12)
        ax.set_ylabel("Δy", fontsize=12)
        ax.set_title(
            f"Constraint Regions & Optimal Steps (gradient = {gradient})", fontsize=14
        )
        ax.legend(loc="upper right", fontsize=10)

        return plt.gca()

    plot_constraint_regions(gradient=(2, 1), epsilon=1.0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Key observation:** Same gradient covector, but the L1 step snaps to the corner (coordinate-aligned),
    while L2 step is smooth and diagonal. This is why L1 regularization produces **sparse** solutions!
    """)
    return


@app.cell
def _(np, plt):
    def l1_gradient_descent(f, grad_f, x0, y0, lr=0.1, steps=20, epsilon=None):
        """Gradient descent using L1 projected steps."""
        points = [(x0, y0)]
        x, y = x0, y0

        for _ in range(steps):
            gx, gy = grad_f(x, y)

            # L1 "steepest descent": coordinate with larger gradient gets all the budget
            if abs(gx) > abs(gy):
                step_x = -lr * np.sign(gx)
                step_y = 0
            else:
                step_x = 0
                step_y = -lr * np.sign(gy)

            x = x + step_x
            y = y + step_y
            points.append((x, y))

        return points

    def l2_gradient_descent(f, grad_f, x0, y0, lr=0.1, steps=20):
        """Standard gradient descent using L2 norm."""
        points = [(x0, y0)]
        x, y = x0, y0

        for _ in range(steps):
            gx, gy = grad_f(x, y)
            x = x - lr * gx
            y = y - lr * gy
            points.append((x, y))

        return points

    def compare_l1_l2_on_ravine():
        """Compare L1 vs L2 gradient descent on the ravine problem."""

        # Define ravine function: f(x,y) = x² + 10y²
        def f(x, y):
            return x**2 + 10 * y**2

        def grad_f(x, y):
            return (2 * x, 20 * y)

        # Create mesh for contour plot
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-1.5, 1.5, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 10 * Y**2

        # Starting point
        x0, y0 = -2.5, 0.8

        # Run both optimizers
        l2_points = l2_gradient_descent(f, grad_f, x0, y0, lr=0.09, steps=25)
        l1_points = l1_gradient_descent(f, grad_f, x0, y0, lr=0.15, steps=40)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        CS = ax.contour(X, Y, Z, levels=20)
        ax.clabel(CS, fontsize=9)

        # L2 path (smooth, oscillating)
        l2_x = [p[0] for p in l2_points]
        l2_y = [p[1] for p in l2_points]
        ax.plot(l2_x, l2_y, "b-", linewidth=2, alpha=0.7, label="L2 (Euclidean)")
        ax.plot(l2_x[0], l2_y[0], "bo", markersize=10)
        ax.plot(l2_x[-1], l2_y[-1], "bs", markersize=10)

        # L1 path (zigzag, coordinate-aligned)
        l1_x = [p[0] for p in l1_points]
        l1_y = [p[1] for p in l1_points]
        ax.plot(l1_x, l1_y, "r-", linewidth=2, alpha=0.7, label="L1 (Manhattan)")
        ax.plot(l1_x[0], l1_y[0], "ro", markersize=10)
        ax.plot(l1_x[-1], l1_y[-1], "rs", markersize=10)

        # Mark minimum
        ax.plot(0, 0, "g*", markersize=20, label="Global minimum")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(
            "L1 vs L2 Gradient Descent on Ravine: $f(x,y) = x^2 + 10y^2$", fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return plt.gca()

    compare_l1_l2_on_ravine()
    return (l2_gradient_descent,)


@app.cell
def _(mo):
    mo.md(r"""
    **Observations:**
    - **L2 (blue):** Smooth diagonal steps, but oscillates across the ravine due to high condition number
    - **L1 (red):** Zigzags along coordinate axes! Alternates between pure x-steps and pure y-steps
    - **Why L1 acts this way:** At each step, it puts all its "budget" into whichever coordinate has the larger gradient component
    - **Connection to sparsity:** This coordinate-aligned behavior is why L1 regularization drives weights to exactly zero

    **The profound lesson:** "Steepest descent" is not a property of the function alone—it depends on your metric!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Week 7 Session 2 - Optimizer code
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
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

        # Taking a step using GD for the function z=x^2 + 10*y^2
        def step(self, point, grad_f):
            (x, y) = point
            (dzdx, dzdy) = grad_f(x, y)
            new_x = x - self.learning_rate * dzdx
            new_y = y - self.learning_rate * dzdy
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
    return (starting_point,)


@app.cell
def _(SGD, generic_contour_with_gradient_path, np):
    def narrow_ravine_sgd():
        learning_rate = 0.09
        sgd = SGD(learning_rate)
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 10 * Y**2

        point = (-3, 1)
        points_list = [point]
        for _ in range(15):
            point = sgd.step(point, grad_f)
            points_list.append(point)
        return generic_contour_with_gradient_path(
            X, Y, Z, points_list, "Gradient Descent on narrow ravine $Z=X^2 + 10Y^2$"
        )

    narrow_ravine_sgd()
    return


@app.cell
def _(Optimizer):
    class MomentumSGD(Optimizer):
        def __init__(self, learning_rate, beta):
            self.learning_rate = learning_rate
            self.beta = beta
            self.vx_prev = 0
            self.vy_prev = 0

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
            new_x = x - self.learning_rate * vx
            new_y = y - self.learning_rate * vy
            return (new_x, new_y)

    return (MomentumSGD,)


@app.cell
def _(MomentumSGD, starting_point):
    msgd = MomentumSGD(0.09, 0.9)
    msgd_next_step = starting_point
    for i in range(15):
        msgd_next_step = msgd.step(msgd_next_step, grad_f)
        print(msgd_next_step)
    msgd_next_step
    return


@app.cell
def _(MomentumSGD, generic_contour_with_gradient_path, np):
    def narrow_ravine_msgd(learning_rate, beta):
        sgd = MomentumSGD(learning_rate, beta)
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + 10 * Y**2

        point = (-3, 1)
        points_list = [point]
        for _ in range(15):
            point = sgd.step(point, grad_f)
            points_list.append(point)
        return generic_contour_with_gradient_path(
            X, Y, Z, points_list, "Gradient Descent on narrow ravine $Z=X^2 + 10Y^2$"
        )

    return (narrow_ravine_msgd,)


@app.cell
def _(narrow_ravine_msgd):
    narrow_ravine_msgd(0.09, 0.5)
    return


@app.cell
def _(narrow_ravine_msgd):
    narrow_ravine_msgd(0.09, 0.9)
    return


@app.cell
def _(narrow_ravine_msgd):
    narrow_ravine_msgd(0.09, 0.99)

    return


@app.cell
def _(MomentumSGD, SGD, plt):
    def plot_loss_vs_steps_sgd_and_momentum(
        lr: float = 0.09,
        start: tuple[float, float] = (-3.0, 1.0),
        steps: int = 15,
        betas: tuple[float, ...] = (0.5, 0.9, 0.99),
        logy: bool = False,
    ):
        def f(_x: float, _y: float) -> float:
            return _x**2 + 10 * _y**2

        def _run(
            optimizer, start_point: tuple[float, float], n_steps: int
        ) -> list[float]:
            p = start_point
            losses: list[float] = [f(*p)]
            for _ in range(n_steps):
                p = optimizer.step(p, grad_f)
                losses.append(f(*p))
            return losses

        fig, ax = plt.subplots(figsize=(10, 6))

        # SGD
        sgd_losses = _run(SGD(lr), start, steps)
        ax.plot(
            range(len(sgd_losses)),
            sgd_losses,
            label=f"SGD (lr={lr})",
            color="k",
            linewidth=2,
        )

        # Momentum SGD with provided betas
        _colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
        for idx, beta in enumerate(betas):
            msgd_losses = _run(MomentumSGD(lr, beta), start, steps)
            color = _colors[idx % len(_colors)]
            ax.plot(
                range(len(msgd_losses)),
                msgd_losses,
                label=f"Momentum (β={beta}, lr={lr})",
                color=color,
                linewidth=2,
            )

        if logy:
            ax.set_yscale("log")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss f(x, y) = x^2 + 10y^2")
        ax.set_title(f"Loss vs Steps: SGD vs Momentum (start={start}, steps={steps})")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return plt.gca()

    plot_loss_vs_steps_sgd_and_momentum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations

    - Why $\beta=0.5$ helps but $\beta=0.9$ and $\beta=0.99$ oscillate (memory length + sign-flipping accumulation)
        - For this ravine the effective memory length can go from cancelling the sign flipping across the ravine to accumulating. When we cross that threshold, we get osciallations instead of convergence.

    - The effective learning rate interaction: $\alpha/(1-\beta)$ and why it pushes past the stability threshold

        - When adding momentum, the effective learning rate changes, and in this case for .9 and .99 it exceeds the stability threshold of $\frac{2}{\lambda_{max}}$

    - Why oscillation stays bounded: exponential decay caps velocity, $\beta=1$ would diverge

        - The values for $\beta$ are all <1, keeping velocity bounded, even with the oscillations, rather than diverging. If $\beta = 1$ we would lose the decay and get divergence.
    """)
    return


if __name__ == "__main__":
    app.run()
