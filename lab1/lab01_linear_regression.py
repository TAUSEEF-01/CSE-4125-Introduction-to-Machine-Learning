"""
Lab 01: Linear Regression with Gradient Descent
=================================================
Implements univariate linear regression from scratch using
vectorized NumPy operations and gradient descent optimization.

Modules:
    Part 1 - Synthetic data:  y = 3 + 5x + ε,  x ∈ {1,...,100}, ε ~ N(0,1)
    Part 2 - Real data:       data_01.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────── Configuration ──────────────────────────────────

SYNTHETIC_CONFIG = {
    "intercept": 3,
    "slope": 5,
    "n_samples": 100,
    "x_range": (1, 100),
    "learning_rate": 0.01,  # with feature scaling enabled
    "iterations": 1000,
    "feature_scaling": True,  # needed for good convergence with lr=0.01
}

REAL_DATA_CONFIG = {
    "file_path": os.path.join(os.path.dirname(__file__), "data_01.csv"),
    "learning_rate": 0.01,
    "iterations": 1000,
    "feature_scaling": True,  # real data benefits from scaling
}

# ─────────────────── Core ML Pipeline Functions ─────────────────────────────


def generate_synthetic_data(n=100, intercept=3, slope=5, seed=42):
    """Generate synthetic data: y = intercept + slope * x + N(0,1)."""
    np.random.seed(seed)
    x = np.arange(1, n + 1, dtype=np.float64)
    noise = np.random.randn(n)
    y = intercept + slope * x + noise
    return x, y


def save_data(filepath, x, y):
    """Save x and y columns to a CSV file."""
    data = np.column_stack((x, y))
    np.savetxt(filepath, data, delimiter=",", header="x,y", comments="")
    print(f"[INFO] Data saved to {filepath}")


def load_data(filepath):
    """
    Load dataset from a CSV file.

    Returns:
        x : (m,) array of feature values
        y : (m,) array of target values
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=0)
    # If the first row is NaN (header), skip it
    if np.isnan(data[0]).any():
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    print(f"[INFO] Loaded {x.shape[0]} samples from {filepath}")
    return x, y


def process_data(x, y, feature_scaling=False):
    """
    Prepare data for training.

    Steps:
        1. (Optional) Feature scaling – zero-mean, unit-variance.
        2. Add dummy feature x0 = 1 (bias / intercept term).

    Returns:
        X      : (m, 2) design matrix with bias column
        y      : (m, 1) target vector
        params : dict with scaling parameters (mean, std) for inverse transform
    """
    m = x.shape[0]
    params = {"mean": 0.0, "std": 1.0, "scaled": feature_scaling}

    x_proc = x.copy()

    if feature_scaling:
        mu = np.mean(x_proc)
        sigma = np.std(x_proc)
        x_proc = (x_proc - mu) / sigma
        params["mean"] = mu
        params["std"] = sigma
        print(f"[INFO] Feature scaling applied  (mean={mu:.4f}, std={sigma:.4f})")

    # Add dummy feature x0 = 1  →  X = [1 | x]
    X = np.column_stack((np.ones(m), x_proc))  # (m, 2)
    y = y.reshape(-1, 1)  # (m, 1)

    return X, y, params


def compute_cost(X, y, theta):
    """
    Compute the Mean Squared Error cost (J).

    J(θ) = (1 / 2m) * Σ (Xθ - y)²

    Parameters:
        X     : (m, n) design matrix
        y     : (m, 1) target vector
        theta : (n, 1) parameter vector

    Returns:
        cost : scalar
    """
    m = X.shape[0]
    predictions = X @ theta  # (m, 1)  – vectorized
    errors = predictions - y  # (m, 1)
    cost = (1.0 / (2 * m)) * np.sum(errors**2)
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Perform batch gradient descent.

    Update rule (vectorized):
        θ := θ - (α / m) * Xᵀ (Xθ - y)

    Parameters:
        X         : (m, n) design matrix
        y         : (m, 1) target vector
        theta     : (n, 1) initial parameters
        alpha     : learning rate
        num_iters : number of iterations

    Returns:
        theta        : (n, 1) learned parameters
        cost_history : list of cost at each iteration
    """
    m = X.shape[0]
    cost_history = []

    for i in range(num_iters):
        predictions = X @ theta  # (m, 1)
        errors = predictions - y  # (m, 1)
        gradient = (1.0 / m) * (X.T @ errors)  # (n, 1)
        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


def train(X, y, alpha=0.01, num_iters=1000):
    """
    Train a linear regression model using gradient descent.

    Parameters:
        X         : (m, n) design matrix (with bias column)
        y         : (m, 1) target vector
        alpha     : learning rate
        num_iters : number of gradient descent iterations

    Returns:
        theta        : (n, 1) learned parameters
        cost_history : list of cost values per iteration
    """
    n = X.shape[1]
    theta = np.zeros((n, 1))  # initialize parameters to zero

    print(f"[INFO] Training  |  lr={alpha}, iterations={num_iters}")
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)
    print(f"[INFO] Training complete  |  Final cost = {cost_history[-1]:.6f}")

    return theta, cost_history


def evaluate(X, y, theta):
    """
    Evaluate the learned model.

    Prints:
        - Final cost (MSE / 2)
        - R² score

    Returns:
        cost : final cost value
        r2   : R² coefficient of determination
    """
    cost = compute_cost(X, y, theta)
    predictions = X @ theta
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"[EVAL] Cost (J)  = {cost:.6f}")
    print(f"[EVAL] R^2 score = {r2:.6f}")
    return cost, r2


# ─────────────────────── Plotting Utilities ─────────────────────────────────


def plot_data(x, y, title="Dataset", xlabel="x", ylabel="y", save_as=None):
    """Scatter plot of data points."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=15, alpha=0.6, label="Data points")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def plot_cost_history(cost_history, title="Training Error Curve", save_as=None):
    """Plot cost vs. iterations."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def plot_regression_line(
    x_original,
    y_original,
    theta,
    scale_params,
    title="Regression Line",
    xlabel="x",
    ylabel="y",
    save_as=None,
):
    """
    Plot dataset together with the learned regression line.

    Works correctly regardless of whether feature scaling was applied.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(x_original, y_original, s=15, alpha=0.6, label="Data points")

    # Generate a smooth line across the range of original x
    x_line = np.linspace(x_original.min(), x_original.max(), 300)

    if scale_params["scaled"]: # scaling done over here
        x_line_scaled = (x_line - scale_params["mean"]) / scale_params["std"]
    else:
        x_line_scaled = x_line

    X_line = np.column_stack((np.ones(len(x_line)), x_line_scaled))
    y_line = X_line @ theta

    plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        print(f"[INFO] Plot saved to {save_as}")
    plt.show()


def print_parameters(theta, scale_params):
    """Print learned model parameters (both scaled and original-space)."""
    print("\n" + "=" * 50)
    print("LEARNED MODEL PARAMETERS")
    print("=" * 50)
    print(f"  theta_0 (intercept) = {theta[0, 0]:.6f}")
    print(f"  theta_1 (slope)     = {theta[1, 0]:.6f}")

    if scale_params["scaled"]: # scaling done over here
        # Convert back to original feature space:
        #   y = θ₀ + θ₁ * (x - μ) / σ
        #     = (θ₀ - θ₁ * μ / σ) + (θ₁ / σ) * x
        mu, sigma = scale_params["mean"], scale_params["std"]
        orig_slope = theta[1, 0] / sigma
        orig_intercept = theta[0, 0] - theta[1, 0] * mu / sigma
        print(f"\n  (Original feature space)")
        print(f"  intercept = {orig_intercept:.6f}")
        print(f"  slope     = {orig_slope:.6f}")
    print("=" * 50 + "\n")


# ─────────────────────────── Main Runners ───────────────────────────────────


def run_synthetic():
    """Part 1: Linear regression on synthetic data."""
    cfg = SYNTHETIC_CONFIG
    print("\n" + "=" * 60)
    print("  PART 1: SYNTHETIC DATA  (y = 3 + 5x + noise)")
    print("=" * 60 + "\n")

    # 1. Generate data
    x, y = generate_synthetic_data(
        n=cfg["n_samples"],
        intercept=cfg["intercept"],
        slope=cfg["slope"],
    )

    # 2. Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "lab01_data.csv")
    save_data(csv_path, x, y)

    # 3. Plot raw data
    plot_data(
        x,
        y,
        title="Synthetic Data: y = 3 + 5x + noise",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_data.png"),
    )

    # ── Experiment: try without feature scaling first ──
    print("\n--- Without Feature Scaling ---")
    X_noscale, y_vec_ns, params_noscale = process_data(x, y, feature_scaling=False)
    lr_noscale = 1e-5
    try:
        theta_ns, cost_ns = train(X_noscale, y_vec_ns, alpha=lr_noscale, num_iters=1000)
        if np.isnan(cost_ns[-1]) or np.isinf(cost_ns[-1]):
            raise ValueError("Cost diverged")
        print(f"  Cost without scaling (lr={lr_noscale}): {cost_ns[-1]:.6f}")
        print_parameters(theta_ns, params_noscale)
    except (ValueError, FloatingPointError):
        print("  [WARNING] Gradient descent diverged without feature scaling.")

    print("\n--- With Feature Scaling (recommended) ---")

    # 4. Process (add bias, with feature scaling)
    X, y_vec, scale_params = process_data(x, y, feature_scaling=cfg["feature_scaling"])

    # 5. Train
    theta, cost_history = train(
        X,
        y_vec,
        alpha=cfg["learning_rate"],
        num_iters=cfg["iterations"],
    )

    # 6. Plot training error curve
    plot_cost_history(
        cost_history,
        title="Training Error - Synthetic Data",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_cost.png"),
    )

    # 7. Print parameters
    print_parameters(theta, scale_params)

    # 8. Evaluate
    evaluate(X, y_vec, theta)

    # 9. Plot regression line over data
    plot_regression_line(
        x,
        y,
        theta,
        scale_params,
        title="Synthetic Data - Regression Line",
        save_as=os.path.join(os.path.dirname(__file__), "synthetic_regression.png"),
    )


def run_real_data():
    """Part 2: Linear regression on real data (data_01.csv)."""
    cfg = REAL_DATA_CONFIG
    print("\n" + "=" * 60)
    print("  PART 2: REAL DATA  (data_01.csv)")
    print("=" * 60 + "\n")

    # 1. Load data
    x, y = load_data(cfg["file_path"])

    # 2. Plot raw data
    plot_data(
        x,
        y,
        title="Real Data (data_01.csv)",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "real_data.png"),
    )

    # ── Experiment: try without feature scaling first ──
    print("\n--- Without Feature Scaling ---")
    X_noscale, y_vec, params_noscale = process_data(x, y, feature_scaling=False)

    # Use a very small learning rate without scaling
    lr_noscale = 1e-5
    try:
        theta_ns, cost_ns = train(X_noscale, y_vec, alpha=lr_noscale, num_iters=1000)
        if np.isnan(cost_ns[-1]) or np.isinf(cost_ns[-1]):
            raise ValueError("Cost diverged")
        print(f"  Cost without scaling (lr={lr_noscale}): {cost_ns[-1]:.6f}")
    except (ValueError, FloatingPointError):
        print("  [WARNING] Gradient descent diverged without feature scaling.")

    print("\n--- With Feature Scaling (recommended) ---")

    # 3. Process with feature scaling
    X, y_vec, scale_params = process_data(x, y, feature_scaling=cfg["feature_scaling"])

    # 4. Train
    theta, cost_history = train(
        X,
        y_vec,
        alpha=cfg["learning_rate"],
        num_iters=cfg["iterations"],
    )

    # 5. Plot training error curve
    plot_cost_history(
        cost_history,
        title="Training Error - Real Data",
        save_as=os.path.join(os.path.dirname(__file__), "real_cost.png"),
    )

    # 6. Print parameters
    print_parameters(theta, scale_params)

    # 7. Evaluate
    evaluate(X, y_vec, theta)

    # 8. Plot regression line over data
    plot_regression_line(
        x,
        y,
        theta,
        scale_params,
        title="Real Data - Regression Line",
        xlabel="x",
        ylabel="y",
        save_as=os.path.join(os.path.dirname(__file__), "real_regression.png"),
    )


# ─────────────────────────────── Entry ──────────────────────────────────────

if __name__ == "__main__":
    run_synthetic()
    run_real_data()
