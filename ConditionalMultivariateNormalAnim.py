"""
Animation of Conditional Probability in a Bivariate Normal Distribution.

This script generates a sequence of images that can be combined to create an animation.
The animation illustrates how the conditional probability distribution p(x|y) of a 
bivariate normal distribution changes as the correlation between the variables x and y is varied.

For each frame of the animation, the script does the following:
1. Defines a bivariate normal distribution with a specific correlation coefficient.
2. Plots the joint probability distribution p(x,y) as a 2D heatmap.
3. Specifies a fixed value for y (y_data) and draws a line on the heatmap at this value.
4. Calculates the conditional probability distribution p(x|y=y_data).
5. Plots this 1D conditional distribution below the heatmap.

The script iterates over a range of correlation coefficients, generating a plot for each one, 
which allows for the creation of an animation showing the effect of correlation on the conditional distribution.
"""
import numpy as np
import matplotlib.pyplot as plt

def bivariate_normal(v, mu, covar):
    """
    Computes the probability density of a bivariate normal distribution at given points.

    Args:
        v (np.ndarray): A 3D array of shape (N, M, 2) representing the (x, y) coordinates of the points.
        mu (np.ndarray): A 1D array of shape (2,) representing the mean of the distribution.
        covar (np.ndarray): A 2D array of shape (2, 2) representing the covariance matrix.

    Returns:
        np.ndarray: A 2D array of shape (N, M) with the probability density at each point.
    """
    covar_inv = np.linalg.inv(covar)
    d = v - mu
    const = 1 / ((2 * np.pi) * np.sqrt(np.linalg.det(covar)))
    # Using einsum for efficient vectorized computation of the quadratic form
    p = const * np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', d, covar_inv, d))
    return p

def cond_univariate_normal(x, y_data, mu, covar):
    """
    Computes the conditional probability p(x|y) of a bivariate normal distribution.

    Given a bivariate normal distribution N(mu, covar), this function calculates the 
    probability distribution of x conditioned on a specific value of y.

    Args:
        x (np.ndarray): The x-values at which to evaluate the conditional probability.
        y_data (float): The value of y on which to condition.
        mu (np.ndarray): The mean of the bivariate normal distribution.
        covar (np.ndarray): The covariance matrix of the bivariate normal distribution.

    Returns:
        np.ndarray: The conditional probability p(x|y=y_data) for each x value.
    """
    # Calculate the conditional mean and variance using the standard formulas
    mu_cond = mu[0] + covar[0, 1] / covar[1, 1] * (y_data - mu[1])
    sig2_cond = covar[0, 0] - covar[0, 1] * covar[1, 0] / covar[1, 1]
    
    # Return the probability density of the resulting univariate normal distribution
    return np.exp(-0.5 * (x - mu_cond)**2 / sig2_cond) / np.sqrt(2 * np.pi * sig2_cond)

# --- Main Script ---

# Set up the grid for plotting
Nx = 200
x = np.linspace(-3.0, 3.0, Nx)
y = np.copy(x)

xx, yy = np.meshgrid(x, y)
v = np.dstack([xx, yy]) # Stack to get (x,y) pairs for each grid point

# Define the parameters of the bivariate normal distribution
mu = np.array([0.0, 0.0])
sig2 = 1.0

# --- Animation Loop ---
# Loop to generate frames for the animation
Nrot = 30
for i in range(Nrot):
    # Cycle through various correlation values, from -0.9 to 0.9
    r = -0.9 + 1.8 * (i / (Nrot - 1))

    # Set up the figure and axes for this frame
    fig = plt.figure(dpi=200, figsize=(3, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    # Define the covariance matrix for the current correlation value
    corr = np.array([[1.0, r], [r, 1.0]])
    covar = sig2 * corr

    # Calculate the joint probability distribution p(x,y)
    P = bivariate_normal(v, mu, covar)

    # Plot the joint probability as a heatmap
    ax1.pcolormesh(xx, yy, P, cmap='Greys')
    ax1.set_aspect('equal')

    # Define the value of y to condition on and plot a line on the heatmap
    y_data = 0.5
    ax1.axhline(y_data, c='r', ls='--')

    # Calculate the conditional probability p(x|y=y_data)
    p_cond = cond_univariate_normal(x, y_data, mu, covar)

    # Plot the conditional probability
    ax2.plot(x, p_cond)
    
    # --- Formatting the Plot ---
    ax1.set_xlim(x[0], x[-1])
    ax2.set_xlabel("x")
    ax2.set_ylabel(f"P(x|y={y_data})")
    ax2.set_ylim(-0.05, 0.95)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("p(x,y)")

    fig.tight_layout()

    # Save the figure for this frame
    fig.savefig(f'./Images/CondNormal_{i}.png')