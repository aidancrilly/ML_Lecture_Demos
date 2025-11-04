"""
Demonstration of Gaussian Process (GP) Regression.

This script provides a from-scratch implementation of a simple Gaussian Process 
regression model. GPs are a powerful non-parametric tool for regression tasks, providing 
not only predictions but also a measure of uncertainty about those predictions.

This script covers the fundamental components of a GP:
1.  **Kernel (Correlation Function)**: A function that defines the similarity between data points. 
    Here, we use the squared exponential (or Gaussian) kernel.
2.  **Hyperparameter Optimisation**: The script finds the optimal hyperparameters for the GP 
    (in this case, the length scale of the kernel) by maximising the marginal log-likelihood.
3.  **Prediction (Conditional Distribution)**: It uses the trained GP to make predictions at new, 
    unseen data points, calculating the conditional mean and variance.

The main part of the script demonstrates these steps by:
- Generating synthetic data from a known function.
- Fitting a GP to this data.
- Plotting the GP's mean prediction and confidence interval against the true function.

This file is also designed to be imported by other scripts that build upon this GP implementation, 
such as the Bayesian Optimisation demo.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f_true(x, lamb=2e0, omega=2e1):
    """
    The true underlying function from which we draw samples.
    
    This is a damped cosine wave, used here as an example of a non-trivial function 
    for the GP to learn.

    Args:
        x (np.ndarray): Input points.
        lamb (float, optional): Damping factor. Defaults to 2.0.
        omega (float, optional): Frequency of the cosine. Defaults to 20.0.

    Returns:
        np.ndarray: The function values at the input points.
    """
    return np.cos(omega * x) * np.exp(-lamb * x)

def correlation_function(x1, x2, theta):
    """
    The correlation function (or kernel) of the GP.

    The kernel defines the covariance between any two points. This function uses the 
    squared exponential (or Gaussian) kernel, which is a common choice. It assumes that 
    points that are closer together are more correlated.

    Args:
        x1 (np.ndarray): First set of input points.
        x2 (np.ndarray): Second set of input points.
        theta (float): The length scale hyperparameter, which controls how quickly the 
                     correlation decays with distance.

    Returns:
        np.ndarray: The correlation between x1 and x2.
    """
    return np.exp(-0.5 * ((x1 - x2) / theta)**2)

def correlation_matrix(x, theta):
    """
    Constructs the correlation matrix for a set of points.

    The correlation matrix (or Gram matrix) is a key component of a GP. Its elements 
    are the pairwise correlations of all points in the input set, as defined by the kernel.

    Args:
        x (np.ndarray): The input data points.
        theta (float): The length scale hyperparameter of the kernel.

    Returns:
        np.ndarray: The correlation matrix.
    """
    x1, x2 = np.meshgrid(x, x)
    corr = correlation_function(x1, x2, theta)
    return corr

def sigma2_MLE(corr, y):
    """
    Calculates the Maximum Likelihood Estimate (MLE) of the variance (sigma^2).

    Given a correlation matrix and the observed data, this function provides the 
    MLE for the overall variance of the GP.
    (See Santner, Williams, and Notz, "The Design and Analysis of Computer Experiments", pg. 65-66)

    Args:
        corr (np.ndarray): The correlation matrix.
        y (np.ndarray): The observed function values.

    Returns:
        float: The MLE of sigma^2.
    """
    n = y.shape[0]
    # This is the analytical solution for the variance that maximises the likelihood.
    return np.dot(y, np.matmul(np.linalg.inv(corr), y)) / n

def neg_log_likelihood(theta, x, y):
    """
    Calculates the negative marginal log-likelihood of the GP.

    This function is what we minimise to find the optimal value for the hyperparameter `theta`. 
    Maximising the likelihood (or minimising the negative log-likelihood) gives us the `theta` 
    that best explains the observed data.
    (See Santner, Williams, and Notz, pg. 65-66)

    Args:
        theta (float): The length scale hyperparameter.
        x (np.ndarray): The input data points.
        y (np.ndarray): The observed function values.

    Returns:
        float: The negative marginal log-likelihood.
    """
    n = x.shape[0]
    corr = correlation_matrix(x, theta)
    # Handle potential numerical instability in the correlation matrix
    if np.linalg.det(corr) == 0:
        return np.inf
    sigma2 = sigma2_MLE(corr, y)
    log_l = n * np.log(sigma2) + np.log(np.linalg.det(corr))
    return log_l

def conditional_distribution(x_trial, x_known, y_known, sigma, theta):
    """
    Calculates the conditional distribution p(y_trial | y_known) for the GP.

    This is the prediction step. Given the observed data and the GP's hyperparameters, 
    this function computes the mean and covariance of the GP at a new set of trial points.

    Args:
        x_trial (np.ndarray): The points where we want to make a prediction.
        x_known (np.ndarray): The input points we have already observed.
        y_known (np.ndarray): The function values at the known points.
        sigma (float): The overall standard deviation of the GP.
        theta (float): The length scale hyperparameter of the kernel.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The conditional mean at the trial points.
            - np.ndarray: The conditional covariance matrix at the trial points.
    """
    x_total = np.concatenate((x_trial, x_known), axis=0)
    N_trial = x_trial.shape[0]
    
    # Compute the full correlation matrix for both known and trial points
    corr = sigma**2 * correlation_matrix(x_total, theta)
    
    # Partition the correlation matrix into blocks
    corr_11 = corr[:N_trial, :N_trial]  # K(x_trial, x_trial)
    corr_12 = corr[:N_trial, N_trial:]  # K(x_trial, x_known)
    corr_21 = corr_12.T
    corr_22 = corr[N_trial:, N_trial:]  # K(x_known, x_known)
    
    # Compute the conditional mean and covariance using the standard GP formulas
    # (assuming a zero prior mean)
    corr_22_inv = np.linalg.inv(corr_22)
    mu_cond = np.dot(np.matmul(corr_12, corr_22_inv), y_known)
    sig_cond = corr_11 - np.matmul(corr_12, np.matmul(corr_22_inv, corr_21))
    return mu_cond, sig_cond

# This block runs only when the script is executed directly
if __name__ == '__main__':
    np.random.seed(10)
    
    # --- 1. Generate Sample Data ---
    N_known = 8
    X_known = np.random.rand(N_known)
    Y_known = f_true(X_known)

    # --- 2. Hyperparameter Optimisation ---
    # We use scipy.optimize.minimize to find the value of theta that minimizes the negative log-likelihood.
    # This is Maximum Likelihood Estimation (MLE) for the hyperparameter.
    res = minimize(neg_log_likelihood, 0.1, args=(X_known, Y_known), bounds=[(1e-2, None)])
    
    # Extract the optimal hyperparameters from the optimisation result
    theta_opt = res.x[0]
    corr_opt = correlation_matrix(X_known, theta_opt)
    sigma_opt = np.sqrt(sigma2_MLE(corr_opt, Y_known))

    # --- 3. Prediction ---
    # Create a set of trial points to make predictions on
    X_trial = np.linspace(0.0, 1.0, 100)
    
    # Compute the conditional mean and variance (our GP prediction)
    mu_cond, sig_cond = conditional_distribution(X_trial, X_known, Y_known, sigma_opt, theta_opt)

    # --- 4. Plotting ---
    plt.figure(dpi=200)
    # Plot the original data points
    plt.scatter(X_known, Y_known, marker='D', label='Data')
    
    # Plot the GP mean prediction
    plt.plot(X_trial, mu_cond, 'b', label='GP Mean')
    
    # Plot the confidence interval (mean +/- 1 standard deviation)
    # The standard deviation is the square root of the diagonal of the covariance matrix.
    plt.plot(X_trial, mu_cond + np.sqrt(np.diag(sig_cond)), 'b--', label='GP Confidence Interval')
    plt.plot(X_trial, mu_cond - np.sqrt(np.diag(sig_cond)), 'b--')

    # Plot the true function for comparison
    plt.plot(X_trial, f_true(X_trial), 'k', label='True Function')

    plt.xlim(0.0, 1.0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()
