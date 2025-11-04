"""
Demonstration of Deconvolution with and without Tikhonov Regularization.

This script illustrates the process of deconvolution, which aims to recover an original 
signal from a measured signal that has been convolved with a known response function 
(and potentially corrupted by noise).

It compares two methods:
1.  **Direct Deconvolution**: A straightforward approach using the pseudo-inverse of the 
    response matrix. This method is simple but highly sensitive to noise.
2.  **Regularized Deconvolution**: A more robust method that incorporates a Tikhonov 
    regularization term. This term penalizes solutions with large derivatives, promoting 
    smoothness and stability, which is particularly useful when dealing with noisy data.

The script performs the following steps:
- Defines a 'true' signal (a Gaussian function).
- Defines a response matrix that represents a simple 'boxcar' averaging or blurring effect.
- Generates a 'measured' signal by convolving the true signal with the response matrix 
  and adding random noise.
- Attempts to recover the true signal using both direct and regularized deconvolution.
- Plots the results to visually compare the effectiveness of the two methods.
"""
import numpy as np
import matplotlib.pyplot as plt

def R(x1, x2):
    """
    Constructs the response matrix for a boxcar averaging effect.

    This function creates a matrix where each row represents the instrument's response 
    to a delta function at a specific input point. The 'boxcar' or 'moving average' 
    response means that the measured signal at a point is an average of the true signal 
    over a small window.

    Args:
        x1 (np.ndarray): The grid of output points (measurement locations).
        x2 (np.ndarray): The grid of input points (true signal locations).

    Returns:
        np.ndarray: The response matrix `R` where `R[i, j]` is the response at `x1[i]` 
                    due to a source at `x2[j]`.
    """
    xx1, xx2 = np.meshgrid(x1, x2)
    R_matrix = np.zeros_like(xx1)
    # The response is 1 if the input and output points are within a certain distance, 0 otherwise.
    R_matrix[np.abs(xx1 - xx2) < 1.5] = 1.0
    # Normalize each row so that the total response to a constant signal is preserved.
    R_matrix = R_matrix / np.sum(R_matrix, axis=1, keepdims=True)
    return R_matrix

# --- Script Parameters ---
Nx = 500          # Number of points in our spatial grid
noise_amp = 0.01  # Amplitude of the random noise to add to the signal
lamb = 1e-4       # Regularization parameter (lambda)

# --- Signal and Response Setup ---
# Define the spatial grid
x = np.linspace(-4.0, 4.0, Nx)
dx = x[1] - x[0]

# Define the 'true' underlying signal (a Gaussian)
truth = np.exp(-x**2)

# Create the instrument response function (IRF) or response matrix
IRF = R(x, x)

# Create the measured signal by applying the response matrix and adding noise
signal = np.dot(IRF, truth) + noise_amp * np.random.normal(size=Nx)

# --- Deconvolution Methods ---

# 1. Straightforward Deconvolution using the Moore-Penrose pseudo-inverse
# This is equivalent to solving the least-squares problem argmin||IRF*f - signal||^2
# While simple, it is very sensitive to noise in the signal.
deconv_signal = np.dot(np.linalg.pinv(IRF), signal)

# 2. Deconvolution with Tikhonov Regularization
# We aim to solve argmin||IRF*f - signal||^2 + lambda^2 * ||D*f||^2
# where D is a matrix that penalizes non-smooth solutions (e.g., a derivative operator).
# The solution is f = (IRF.T*IRF + lambda^2*D.T*D)^-1 * IRF.T*signal

RTR = np.dot(IRF.T, IRF)
RTsignal = np.dot(IRF.T, signal)

# Set up a finite difference matrix for the first derivative (forward difference)
# This matrix, when applied to the signal vector, approximates its derivative.
D_matrix = (-1 * np.eye(Nx) + 1 * np.eye(Nx, k=1)) / dx
DTD = np.dot(D_matrix.T, D_matrix) # D.T*D is related to the second derivative (Laplacian)

# Calculate the inverse of the regularized matrix
regularised_inv = np.linalg.inv(RTR + lamb * DTD)

# Calculate the regularized deconvolution result
deconv_reg_signal = np.dot(regularised_inv, RTsignal)

# --- Plotting ---
fig = plt.figure(dpi=200, figsize=(3, 4))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)

# Top plot: The original true signal and the measured (convolved and noisy) signal
ax1.plot(x, truth, 'k', label='Truth')
ax1.plot(x, signal, 'r', label='Signal')

# Bottom plot: The true signal vs. the two deconvolution results
ax2.plot(x, truth, 'k')
ax2.plot(x, deconv_signal, 'b', alpha=0.5, label='LA Deconv.')
ax2.plot(x, deconv_reg_signal, 'g', label='Regularised Deconv.')

ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(-0.1, 1.6)

ax1.legend(frameon=False)
ax2.legend(frameon=False)
fig.tight_layout()

plt.show()