"""
Demonstration of Non-Linear Least Squares Fitting.

This script illustrates how to solve a non-linear least squares problem by fitting a 
non-linear model (a Gaussian function) to synthetic data. The goal is to find the 
model parameters that minimise the sum of the squared differences between the model's 
predictions and the observed data.

Two common iterative optimisation algorithms are implemented and compared:

1.  **Gauss-Newton Algorithm**: A classic and efficient method for non-linear least squares. 
    It approximates the non-linear model with a linear one at each step and solves for 
    the optimal parameter update. The update rule is:
    `d_theta = (J^T * J)^-1 * J^T * R`
    where `J` is the Jacobian of the model and `R` is the residual vector.

2.  **Gradient Descent**: A more general optimisation algorithm. It updates the parameters 
    by taking a step in the direction of the negative gradient of the cost function. 
    The update rule is:
    `d_theta = -learning_rate * grad(Cost)`
    where `grad(Cost) = -2 * J^T * R`.

The script visualises the optimisation process by showing the path of the parameter 
guesses on a contour plot of the cost function (sum of squared errors) and by plotting 
the corresponding model fit at each iteration.
"""
import numpy as np
import matplotlib.pyplot as plt

def Gaussian(x, mu, sig):
    """
    A simple Gaussian function, normalized to have a peak value of 1.

    Args:
        x (np.ndarray): Input coordinates.
        mu (float): The mean of the Gaussian.
        sig (float): The standard deviation of the Gaussian.

    Returns:
        np.ndarray: The value of the Gaussian function at x.
    """
    return np.exp(-0.5 * ((x - mu) / sig)**2)

def Jacobian(x, mu, sig):
    """
    Computes the Jacobian matrix of the Gaussian function with respect to its parameters.

    The Jacobian `J` is a matrix where `J[i, j]` is the partial derivative of the model 
    output for the i-th data point with respect to the j-th parameter. It measures how 
    sensitive the model output is to changes in the parameters.

    Args:
        x (np.ndarray): Input coordinates.
        mu (float): The current estimate of the mean.
        sig (float): The current estimate of the standard deviation.

    Returns:
        np.ndarray: The Jacobian matrix of shape (len(x), 2).
    """
    J = np.zeros((x.shape[0], 2))
    # Derivative with respect to mu
    J[:, 0] = (1.0 / sig) * ((x - mu) / sig) * Gaussian(x, mu, sig)
    # Derivative with respect to sig
    J[:, 1] = (1.0 / sig) * ((x - mu) / sig)**2 * Gaussian(x, mu, sig)
    return J

# --- 1. Setup and Data Generation ---

# True parameters for the data generation
mu_true, sig_true = 0.5, 0.25
noise = 0.01

# Generate synthetic data
x_arr = np.linspace(-2.0, 2.0, 100)
y_arr = Gaussian(x_arr, mu_true, sig_true) + noise * np.random.normal(size=x_arr.shape[0])

# --- 2. Visualise the Cost Function Landscape ---

# Create a grid of mu and sigma values to compute the cost function over
mu_arr = np.linspace(-1.0, 2.0, 100)
sig_arr = np.linspace(1e-2, 1.25, 100)
mm, ss = np.meshgrid(mu_arr, sig_arr)

# Calculate the sum of squared errors (the cost) for each point in the grid
S_arr = np.zeros((mu_arr.shape[0], sig_arr.shape[0]))
for i in range(mu_arr.shape[0]):
    for j in range(sig_arr.shape[0]):
        S_arr[i, j] = np.sum((y_arr - Gaussian(x_arr, mu_arr[i], sig_arr[j]))**2)

# --- 3. Gauss-Newton Algorithm ---

fig_gn = plt.figure(dpi=200, figsize=(10, 4))
ax1_gn = fig_gn.add_subplot(121)
ax2_gn = fig_gn.add_subplot(122)
fig_gn.suptitle('Gauss-Newton Method')

# Plot the cost function landscape
ax1_gn.pcolormesh(mm, ss, S_arr.T, shading='auto')
ax1_gn.set_xlabel(r"$\mu$")
ax1_gn.set_ylabel(r"$\sigma$")
ax2_gn.set_xlabel('x')
ax2_gn.set_ylabel('y')

# Initial guess for the parameters
mu_guess, sig_guess = 0.0, 1.0
ax1_gn.scatter(mu_guess, sig_guess, label='Start')
ax2_gn.plot(x_arr, Gaussian(x_arr, mu_guess, sig_guess), label=f'Iter 0')

# Run the Gauss-Newton iterations
N_iterations = 3
for i in range(N_iterations):
    # Compute the Jacobian at the current parameter guess
    J = Jacobian(x_arr, mu_guess, sig_guess)
    # Compute the residual (the difference between data and model)
    R = y_arr - Gaussian(x_arr, mu_guess, sig_guess)

    # Solve the normal equations: (J.T * J) * d_theta = J.T * R
    JTJ = np.matmul(J.T, J)
    JTJ_inv = np.linalg.inv(JTJ)
    JTR = np.matmul(J.T, R)
    dtheta = np.matmul(JTJ_inv, JTR)

    # Update the parameter guess
    mu_guess += dtheta[0]
    sig_guess += dtheta[1]

    # Plot the new guess
    ax1_gn.scatter(mu_guess, sig_guess)
    ax2_gn.plot(x_arr, Gaussian(x_arr, mu_guess, sig_guess), label=f'Iter {i+1}')

# Plot the true parameter values and the original data
ax1_gn.scatter(mu_true, sig_true, c='k', marker='x', label='Truth')
ax2_gn.scatter(x_arr, y_arr, c='k', marker='x', label='Data')
ax1_gn.legend()
ax2_gn.legend()

# --- 4. Gradient Descent Algorithm ---

fig_gd = plt.figure(dpi=200, figsize=(10, 4))
ax1_gd = fig_gd.add_subplot(121)
ax2_gd = fig_gd.add_subplot(122)
fig_gd.suptitle('Gradient Descent Method')

# Plot the cost function landscape
ax1_gd.pcolormesh(mm, ss, S_arr.T, shading='auto')
ax1_gd.set_xlabel(r"$\mu$")
ax1_gd.set_ylabel(r"$\sigma$")
ax2_gd.set_xlabel('x')
ax2_gd.set_ylabel('y')

# Initial guess for the parameters
mu_guess, sig_guess = 0.0, 1.0
ax1_gd.scatter(mu_guess, sig_guess, label='Start')
ax2_gd.plot(x_arr, Gaussian(x_arr, mu_guess, sig_guess), label=f'Iter 0')

# Set the learning rate (lambda)
learning_rate = 1e-2
N_iterations = 4
for i in range(N_iterations):
    # Compute the Jacobian and the residual
    J = Jacobian(x_arr, mu_guess, sig_guess)
    R = y_arr - Gaussian(x_arr, mu_guess, sig_guess)

    # The gradient of the sum-of-squares cost function is -2 * J.T * R
    gradient = -2 * np.matmul(J.T, R)
    
    # Update the parameters by taking a step in the negative gradient direction
    dtheta = -learning_rate * gradient

    mu_guess += dtheta[0]
    sig_guess += dtheta[1]

    # Plot the new guess
    ax1_gd.scatter(mu_guess, sig_guess)
    ax2_gd.plot(x_arr, Gaussian(x_arr, mu_guess, sig_guess), label=f'Iter {i+1}')

# Plot the true parameter values and the original data
ax1_gd.scatter(mu_true, sig_true, c='k', marker='x', label='Truth')
ax2_gd.scatter(x_arr, y_arr, c='k', marker='x', label='Data')
ax1_gd.legend()
ax2_gd.legend()

plt.show()