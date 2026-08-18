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

__generated_with = "0.23.6"
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
    # Week 11 Session 2 -
    """)
    return


@app.cell
def _():
    from abc import ABC, abstractmethod

    class Optimizer(ABC):
        @abstractmethod
        def step(self, w, grad_f):
            pass

    return (Optimizer,)


@app.cell
def _(np):
    n=10
    p=100
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n,p)) / np.sqrt(p)
    w_true = rng.standard_normal(p)
    y = X @ w_true
    return X, rng, y


@app.cell
def _(X, y):
    # Gradient function 
    def grad_f(w):
        return (X.T @ (X @ w - y))

    return (grad_f,)


@app.cell
def _(Optimizer):
    class SGD(Optimizer):
        def __init__(self, learning_rate, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.weight_decay=weight_decay

        # Taking a step using GD
        def step(self, w, grad_f):
            dw = grad_f(w)
            new_w = (1-self.learning_rate*self.weight_decay)*w - self.learning_rate * dw
            return (new_w)

    return (SGD,)


@app.cell
def _(Optimizer):
    class MomentumSGD(Optimizer):
        def __init__(self, learning_rate, beta, weight_decay=0.0):
            self.learning_rate = learning_rate
            self.beta = beta
            self.v_prev = 0
            self.weight_decay = weight_decay

        def step(self, w, grad_f):
            dw = grad_f(w)
            v = self.beta * self.v_prev + dw
            self.v_prev = v
            new_w = (1-self.learning_rate*self.weight_decay)*w - self.learning_rate * v
            return (new_w)

    return (MomentumSGD,)


@app.cell
def _(Optimizer, np):
    class Adam(Optimizer):
        def __init__(self, learning_rate, beta1, beta2, epsilon, correction=True):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.m = 0.0
            self.v = 0.0
            self.correction=correction
            self.step_counter = 1

        def step(self, w, grad_f):
            dw = grad_f(w)
            m = self.m
            v = self.v
            m = self.beta1 * m + (1-self.beta1)*dw
            v = self.beta2 * v + (1-self.beta2)*dw**2
            self.m=m
            self.v=v
            m_hat = m
            v_hat = v
            if self.correction:
                m_hat = m / (1 - self.beta1 ** self.step_counter)
                v_hat = v / (1 - self.beta2 ** self.step_counter)

            new_w = w - (self.learning_rate * m_hat)/(np.sqrt(v_hat) + self.epsilon)

            self.step_counter += 1
            return (new_w)

    return (Adam,)


@app.cell
def _(X, np, y):
    closed_form = X.T@(np.linalg.inv(X@X.T)) @ y
    # Better to use numpy solver instead of computing inv
    w_star = X.T @ np.linalg.solve(X @ X.T, y)
    w_star_norm = np.linalg.norm(w_star)
    print(np.allclose(X @ w_star, y))

    return w_star, w_star_norm


@app.cell
def _(X, grad_f, np, w_star, w_star_norm, y):
    def run_pred1(optimizer, w0, n_steps, tol):
        w_k = w0
        result = []
        for i in range(n_steps):
            w_norm = np.linalg.norm(w_k)
            error_norm = np.linalg.norm(X@w_k - y)
            relative_error_norm = np.linalg.norm(w_k - w_star)
            if error_norm < tol:
                break
            w_k = optimizer.step(w_k, grad_f)
            print(f"step: {i}")
            print(f"||w||/||w*||: {w_norm/w_star_norm}")
            print(f"||relative_error||/||w*||: {relative_error_norm/w_star_norm}")
            result.append((i, w_norm))
        return result



    return


@app.cell
def _(X, grad_f, np, w_star, w_star_norm, y):
    def run_pred2(optimizer, w0, n_steps, tol):
        w_k = w0
        result = []
        for i in range(n_steps):
            w_norm = np.linalg.norm(w_k)
            error_norm = np.linalg.norm(X@w_k - y)
            relative_error_norm = np.linalg.norm(w_k - w_star)
            if error_norm < tol:
                break
            w_k = optimizer.step(w_k, grad_f)
            print(f"step: {i}")
            print(w_norm/w_star_norm)
            print(relative_error_norm/w_star_norm)
            result.append((i,w_norm))
        w_infty = w_k
    
        print(f"||w_infty - w^*|| = {np.linalg.norm(w_infty-w_star)}")
        P_row = X.T @ np.linalg.solve(X @ X.T, X)   # rowspace projector (p × p)
        w0_perp = w0 - P_row @ w0                    # nullspace component
        print(f"||w_0,perp|| = {np.linalg.norm(w0_perp)}")
        print(f"||w_infty|| = {np.linalg.norm(w_infty)}")
        print(f"||w*||={w_star_norm}")
        print(f"||w_infty - w^* - w_0,perp|| = {np.linalg.norm(w_infty - w_star - w0_perp)}")
        print(f"||w_infty||^2 - ||w*||^2 - ||w_0,perp||^2 = {np.linalg.norm(w_infty)**2 - w_star_norm**2 - np.linalg.norm(w0_perp)**2}")
        return result

    return (run_pred2,)


@app.cell
def _(X, grad_f, np, w_star, w_star_norm, y):
    def run_pred3(optimizer, w0, n_steps, tol):
        result = []
        w_k = w0
        for i in range(n_steps):
            w_norm = np.linalg.norm(w_k)
            error_norm = np.linalg.norm(X@w_k - y)
            relative_error_norm = np.linalg.norm(w_k - w_star)
            if error_norm < tol:
                break
            w_k = optimizer.step(w_k, grad_f)
            print(f"step: {i}")
            print(f"||w||/||w*||: {w_norm/w_star_norm}")
            print(f"||relative_error||/||w*||: {relative_error_norm/w_star_norm}")
            result.append((i, w_norm))
        w_adam = w_k
        print(f"||w_adam|| = {np.linalg.norm(w_adam)}")
        print(f"||w*|| = {w_star_norm}")
    
        P_row = X.T @ np.linalg.solve(X @ X.T, X)   # rowspace projector (p × p)
        rowspace_check = np.linalg.norm(w_adam - P_row @ w_adam)
        print(f"rowspace check: {rowspace_check}")
        return result

    return


@app.cell
def _(X, grad_f, np, w_star, y):
    def run(optimizer, w0, n_steps, tol):
        w_k = w0.copy()
        history = {"w_norm": [], "residual": [], "error": []}
        for i in range(n_steps):
            history["w_norm"].append(np.linalg.norm(w_k))
            history["residual"].append(np.linalg.norm(X @ w_k - y))
            history["error"].append(np.linalg.norm(w_k - w_star))
            if history["residual"][-1] < tol:
                print(f"{i} steps")
                break
            w_k = optimizer.step(w_k, grad_f)
        return w_k, history

    return (run,)


@app.cell
def _(X, np):
    n_steps=1000
    tol = 1e-8
    w0=np.zeros(100)
    lambda_max = np.linalg.eigvalsh(X.T @ X).max()
    print(2/lambda_max)
    learning_rate = 0.9
    return learning_rate, n_steps, tol, w0


@app.cell
def _(SGD, learning_rate, n_steps, run, tol, w0):
    #sgd = SGD(learning_rate)
    #run_pred1(sgd, w0, n_steps, tol)
    w_sgd1, hist_sgd1 = run(SGD(learning_rate), w0, n_steps, tol)
    return hist_sgd1, w_sgd1


@app.cell
def _(rng):
    def _(rng):
        #w_random_0 = np.random.normal(scale=0.1, size=(100))
        w_random_0 = rng.standard_normal(100) * 0.1
        return w_random_0
    w_random_0 = _(rng=rng)    
    return (w_random_0,)


@app.cell
def _(SGD, learning_rate, n_steps, run, run_pred2, tol, w_random_0):
    #run_pred2(SGD(learning_rate), w_random_0, n_steps, tol)
    w_sgd2, hist_sgd2=run(SGD(learning_rate), w_random_0, n_steps, tol)
    run_pred2(SGD(learning_rate), w_random_0, n_steps, tol)
    return hist_sgd2, w_sgd2


@app.cell
def _(X, np, w_random_0, w_sgd2, w_star, w_star_norm, y):
    print(f"||X w_random_0 - y|| = {np.linalg.norm(X @ w_random_0 - y)}")  # did it converge?
    print(f"||w_random_0||_2 = {np.linalg.norm(w_random_0)}")
    print(f"||w_inf||_2 = {np.linalg.norm(w_sgd2)}")
    print(f"||w*||_2 = {w_star_norm}")
    print(f"||w_random_0||_inf = {np.max(np.abs(w_random_0))}")
    print(f"||w_inf||_inf = {np.max(np.abs(np.max(w_sgd2)))}")
    print(f"||w*||_inf = {np.max(np.abs(w_star))}")
    P_row = X.T @ np.linalg.solve(X @ X.T, X)
    print(f"rowspace leak (same as w_0,perp) = {np.linalg.norm(w_random_0 - P_row @ w_random_0)}")

    print(f"top 5 |w_random_0|: {np.sort(np.abs(w_random_0))[-5:]}")
    print(f"top 5 |w*|:     {np.sort(np.abs(w_star))[-5:]}")
    print(f"||w_inf - w*|| = {np.linalg.norm(w_sgd2-w_star)}")

    return (P_row,)


@app.cell
def _(Adam, n_steps, run, tol, w0):
    #adam=Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #run_pred3(adam, w0, n_steps, tol)
    w_adam, hist_adam=run(Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8), w0, n_steps, tol)
    return hist_adam, w_adam


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For Adam it was not converging with the same learning rate, so I had to change the rate specifically for Adam to a smaller one (in this case 0.01).
    """)
    return


@app.cell
def _(MomentumSGD, learning_rate, n_steps, run, tol, w0):
    #msgd=MomentumSGD(beta=0.9,learning_rate=learning_rate)
    #run_pred1(msgd, w0, n_steps, tol)
    w_msgd, hist_msgd=run(MomentumSGD(beta=0.9,learning_rate=learning_rate), w0, n_steps, tol)
    return hist_msgd, w_msgd


@app.cell
def _(P_row, X, hist_adam, np, w_adam, w_star, w_star_norm, y):
    print(f"||X w_adam - y|| = {np.linalg.norm(X @ w_adam - y)}")  # did it converge?
    print(f"||w_adam||_2 = {np.linalg.norm(w_adam)}")
    print(f"||w*||_2 = {w_star_norm}")
    print(f"||w_adam||_inf = {np.max(np.abs(w_adam))}")
    print(f"||w*||_inf = {np.max(np.abs(w_star))}")
    w_adam_inf= np.max(np.abs(w_adam))
    #P_row = X.T @ np.linalg.solve(X @ X.T, X)
    print(f"rowspace leak = {np.linalg.norm(w_adam - P_row @ w_adam)}")

    print(f"top 5 |w_adam|: {np.sort(np.abs(w_adam))[-5:]}")
    print(f"top 5 |w*|:     {np.sort(np.abs(w_star))[-5:]}")
    print(f"||w_inf - w*|| = {hist_adam['error'][-1]}")
    return


@app.cell
def _(hist_adam, hist_msgd, hist_sgd1, hist_sgd2, plt, w_star_norm):
    histories = {
        'SGD from zero': hist_sgd1,
        'SGD from random': hist_sgd2,
        'Adam from zero': hist_adam,
        'MSGD from zero': hist_msgd,
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, h in histories.items():
        ax.plot(h["w_norm"], label=label, linewidth=1.5)
    ax.axhline(w_star_norm, linestyle='--', color='gray', label='||w*||')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Step')
    ax.set_ylabel('||w||')
    ax.set_title('Weight norm over training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt, w_adam, w_msgd, w_sgd1, w_star):
    def _():
        fig, ax = plt.subplots(figsize=(10, 6))
        solutions = {
            'w* (min-norm)':   w_star,
            'SGD from zero':   w_sgd1,
            'MSGD from zero':  w_msgd,
            'Adam from zero':  w_adam,
        }
        for label, w in solutions.items():
            sorted_mags = np.sort(np.abs(w))[::-1]
            if label == 'w* (min-norm)':
                ax.plot(sorted_mags, label=label, linewidth=2.5, color='black', alpha=0.8)
            elif label == 'Adam from zero':
                ax.plot(sorted_mags, label=label, linewidth=2, color='red')
            else:
                ax.plot(sorted_mags, label=label, linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_xlabel('Coordinate rank (sorted by |w_i|)')
        ax.set_ylabel('|w_i|')
        ax.set_title('Coordinate magnitudes, sorted descending')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
