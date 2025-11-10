"""
Demonstration of B-spline Fitting with Regularization.

This script illustrates how to use B-splines to approximate a function from a set of 
data points. B-splines are piecewise polynomial functions that are smoothly connected 
at points called 'knots'. They are highly flexible and widely used in computer graphics 
and data fitting.

The core idea is to represent the function as a linear combination of B-spline basis 
functions. The script demonstrates the following process:

1.  **Spline Definition**: A B-spline is defined by its order (e.g., cubic) and a sequence 
    of knots. The script sets up an 'open-uniform' knot vector, which is a common choice.
2.  **Basis Matrix Construction**: It constructs a 'B-spline basis matrix', `B`. Each column 
    of this matrix is one of the B-spline basis functions evaluated at the data points' x-coordinates.
3.  **Least Squares Fitting**: The problem of fitting the spline then becomes a linear least 
    squares problem: `B * a = y`, where `y` is the data values and `a` is the vector of 
    unknown spline coefficients (or control points) we want to find.
4.  **Regularization**: To prevent overfitting and ensure a stable solution, a small 
    regularization term is added. The script solves the regularized normal equations:
    `(B^T * B + lambda * I) * a = B^T * y`
5.  **Animation**: The script generates a sequence of plots, adding one data point at a time 
    and re-fitting the spline, to create an animation that shows how the fit improves as 
    more data becomes available.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def f_true(x, lamb=2e0, omega=2e1):
    """
    The true underlying function from which we draw samples.
    
    This is a damped cosine wave, used here as the function to be approximated by the spline.

    Args:
        x (np.ndarray): Input points.
        lamb (float, optional): Damping factor. Defaults to 2.0.
        omega (float, optional): Frequency of the cosine. Defaults to 20.0.

    Returns:
        np.ndarray: The function values at the input points.
    """
    return np.cos(omega * x) * np.exp(-lamb * x)

def B_matrix(spline_order, knots, x):
    """
    Constructs the B-spline basis matrix.

    This matrix `B_mat` is constructed such that the product `B_mat.dot(a)` gives the values 
    of the spline function f(x), where `a` is the vector of spline coefficients.

    Args:
        spline_order (int): The order of the spline (e.g., 4 for cubic).
        knots (np.ndarray): The knot vector defining the spline.
        x (np.ndarray): The x-coordinates at which to evaluate the basis functions.

    Returns:
        np.ndarray: The B-spline basis matrix of shape (len(x), num_coefficients).
    """
    Nx = x.shape[0]
    Nknots = knots.shape[0]
    # The number of control points (coefficients) is N_knots - spline_order
    Nc = Nknots - spline_order
    B_mat = np.zeros((Nx, Nc))
    
    # To get the i-th basis function, we create a coefficient vector with a 1 at the i-th position
    # and zeros elsewhere, then use scipy's BSpline to evaluate it.
    for i in range(Nc):
        c = np.zeros(Nc)
        c[i] = 1.0
        # Note: scipy's BSpline uses degree k = order - 1
        spl = BSpline(knots, c, spline_order - 1)
        B_mat[:, i] = spl(x)
    return B_mat

# --- Main Script ---

# --- 1. Setup ---
# Total number of data points to fit to in the end
Nx_total = 40
# Number of data points to start with
Nx_start = 5

# --- 2. Spline Definition ---
# Splines are piecewise polynomial functions of a given order.
# Knots are the points where the polynomial pieces are joined smoothly.
N_knots = 40
spline_order = 4  # k=4 corresponds to cubic splines

# We use an 'open-uniform' knot vector. This means the first and last knots are repeated
# `spline_order` times. This makes the spline pass through the first and last control points.
knot_vec = np.linspace(0.0, 1.0, N_knots - (2 * spline_order) + 2)
# Repeat the first knot
knot_vec = np.concatenate([np.full(spline_order - 1, knot_vec[0]), knot_vec])
# Repeat the last knot
knot_vec = np.concatenate([knot_vec, np.full(spline_order - 1, knot_vec[-1])])

# --- 3. Fitting and Animation Loop ---

# Create a high-resolution grid for plotting the final spline smoothly
x_grid = np.linspace(0.0, 1.0, 1000)
B_grid = B_matrix(spline_order, knot_vec, x_grid)

# Initialize the data with a few random points
x = np.random.rand(Nx_start)

# Regularization parameter to prevent ill-conditioning, especially with few data points
regularization = 1e-6

# Loop to add one data point at a time and re-fit the spline
for ix in range(Nx_start, Nx_total):
    # Add a new random data point
    x = np.append(x, np.random.rand())
    y = f_true(x)

    # Construct the B-spline basis matrix for the current data points
    B = B_matrix(spline_order, knot_vec, x)
    
    # Solve the regularized normal equations for the spline coefficients `a`
    # (B.T * B + lambda * I) * a = B.T * y
    BTB = np.dot(B.T, B)
    I = np.linalg.inv(BTB + regularization * np.eye(BTB.shape[0]))
    BTY = np.dot(B.T, y)
    a = np.dot(I, BTY)

    # --- Plotting for the Animation Frame ---
    fig = plt.figure(dpi=200, figsize=(4, 4))
    # Plot the current data points
    plt.scatter(x, y, label='Samples')

    # Plot the fitted spline by multiplying the grid basis matrix by the solved coefficients
    plt.plot(x_grid, np.dot(B_grid, a), label='Spline Fit')
    # Plot the true function for comparison
    plt.plot(x_grid, f_true(x_grid), 'k--', label='True Function')
    
    plt.legend(frameon=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.1, 1.1)
    plt.ylabel("f(x)")
    plt.xlabel("x")
    fig.tight_layout()
    
    # Save the figure for this frame of the animation
    fig.savefig(f'./Images/Spline_ix_{ix - Nx_start:03d}.png')
    plt.close(fig)

print(f"Generated {Nx_total - Nx_start} animation frames.")
