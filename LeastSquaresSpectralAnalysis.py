"""
Demonstration of Least Squares Spectral Analysis (LSSA).

This script illustrates the concept of Least Squares Spectral Analysis (LSSA), a method 
for determining the frequency spectrum of a signal. Unlike the Fast Fourier Transform (FFT), 
which is a standard algorithm for this task, LSSA works by fitting a set of sinusoidal 
basis functions to the data using the method of least squares.

This makes LSSA a powerful tool, especially for unevenly sampled data (though this demo 
uses an evenly sampled signal for simplicity and comparison with FFT).

The script performs the following steps:
1.  **Signal Generation**: Creates a synthetic signal composed of two cosine waves with a 
    Gaussian envelope.
2.  **FFT Calculation**: Computes the FFT of the signal to serve as a baseline for the 
    true frequency spectrum.
3.  **LSSA Implementation**: 
    a. Constructs a basis matrix where each column is a cosine function of a specific 
       frequency.
    b. Solves the linear least squares problem to find the coefficients (amplitudes) for 
       each basis function that best reconstruct the original signal.
    c. These coefficients form the least-squares spectrum.
4.  **Comparison**: Plots the original signal, the LSSA-reconstructed signal, the FFT 
    spectrum, and the LSSA spectrum, demonstrating the effectiveness of the method.
"""
import numpy as np
import matplotlib.pyplot as plt

def cos_basis_matrix(Nm, Nx, t):
    """
    Builds a basis matrix of cosine functions.

    Each column of the matrix is a cosine basis function `cos(2*pi*k*t)` evaluated at 
    the time points `t` for a specific frequency `k`.

    Args:
        Nm (int): The number of basis functions (and thus frequencies) to use.
        Nx (int): The number of time points.
        t (np.ndarray): The array of time points.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The basis matrix `A` of shape (Nx, Nm).
            - np.ndarray: The array of frequencies `k` corresponding to the basis functions.
    """
    A = np.zeros((Nx, Nm))
    # Define the discrete frequencies for the basis functions
    k = 0.5 * np.arange(Nm) / t[-1]
    for m in range(Nm):
        A[:, m] = np.cos(2 * np.pi * k[m] * t)
    return A, k

def trial_function(t, omega1=8, omega2=4):
    """
    The trial function on which we will perform LSSA.
    
    This function is a superposition of two cosine waves with a Gaussian envelope.

    Args:
        t (np.ndarray): The time points.
        omega1 (float, optional): The frequency of the first cosine wave. Defaults to 8.
        omega2 (float, optional): The frequency of the second cosine wave. Defaults to 4.

    Returns:
        np.ndarray: The signal values.
    """
    y = (np.cos(omega1 * np.pi * t) + np.cos(omega2 * np.pi * t)) * np.exp(-t**2)
    return y

# --- Main Script ---

# 1. Generate the Signal
Nt = 400
t = np.linspace(-4.0, 4.0, Nt)
y = trial_function(t)

# Set up the plots
fig = plt.figure(dpi=200, figsize=(3, 5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Plot the original signal
ax1.plot(t, y, 'k', label='Truth')

# 2. Compute and Plot the FFT for comparison
Y = np.fft.fft(y) * (t[1] - t[0])
freq = np.fft.fftfreq(Nt, t[1] - t[0])
# Shift the FFT so that the zero frequency is in the center
Y = np.fft.fftshift(Y)
freq = np.fft.fftshift(freq)
ax2.plot(freq, np.abs(Y), 'k', label='FFT')

# 3. Perform LSSA for different numbers of basis functions
N_list = [45, 25]
for N in N_list:
    # a. Construct the basis matrix
    A, k = cos_basis_matrix(N, Nt, t)

    # b. Solve the normal equations (A.T * A) * theta = A.T * y for the coefficients theta
    ATA = np.dot(A.T, A)
    ATy = np.dot(A.T, y)
    theta = np.linalg.solve(ATA, ATy)

    # c. Plot the results
    # Reconstructed signal from the LSSA coefficients
    ax1.plot(t, np.dot(A, theta), ls='--', label=f'LSSA N = {N}')
    # The LSSA spectrum (the coefficients theta)
    # The factor of 4 is for scaling to match the FFT amplitude
    ax2.plot(k, 4 * theta, ls='--', label=f'LSSA N = {N}')

# --- Formatting the Plot ---
ax1.set_xlim(t[0], t[-1])
ax1.set_ylabel("y(t)")
ax1.set_xlabel("t")
ax2.set_ylabel("Y(k)")
ax2.set_xlabel("k (frequency)")
ax2.set_xlim(0.0, 8.0)
ax1.legend(frameon=False, fontsize=6)
ax2.legend(frameon=False, fontsize=6)

fig.tight_layout()
plt.show()
