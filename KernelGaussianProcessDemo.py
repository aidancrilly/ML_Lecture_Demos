"""
Demonstration of Gaussian Process (GP) Interpolation with a Gaussian Kernel.

This script provides a visual, step-by-step illustration of the fundamental concept 
behind Gaussian Process regression. It shows how a GP interpolates between two known 
data points by constructing the prediction as a weighted sum of kernel functions centered 
at the known data locations.

The script generates a sequence of plots to create an animation of this process:
1.  It starts with two fixed, known data points (x_known, y_known).
2.  It defines a Gaussian kernel, which measures the 'similarity' between points.
3.  It iterates through a series of 'prediction points' between the two known points.
4.  For each prediction point, it calculates:
    a. The correlations (kernel values) between the prediction point and the known points.
    b. The GP correlation matrix.
    c. The conditional mean (the GP's prediction), which is a weighted average of the 
       known y-values. The weights are determined by the kernel.
5.  Each plot shows the two kernel functions, the data points, the current prediction, 
    and the path of the interpolation so far.

This provides a clear visual intuition for how a GP uses kernels to make predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib to use LaTeX for text rendering for nice mathematical formulas
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def kernel(x1, x2, l=0.2):
    """
    Calculates the Gaussian kernel value between two points.

    The Gaussian kernel (or squared exponential kernel) is a measure of similarity. 
    It returns a high value if the points are close and a low value if they are far apart.

    Args:
        x1 (float): The first point.
        x2 (float): The second point.
        l (float, optional): The length scale of the kernel, which controls how quickly 
                         the correlation decays with distance. Defaults to 0.2.

    Returns:
        float: The kernel value (similarity) between x1 and x2.
    """
    return np.exp(-0.5 * ((x1 - x2) / l)**2)

# --- Main Script ---

# Define the two known data points
x_known = np.array([0.2, 0.8])
y_known = np.array([-1.0, 1.0])

# Create a dense grid for plotting the kernel functions
x_plot = np.linspace(-0.2, 1.2, 100)

# --- Animation Loop ---
# Iterate through a series of prediction points to build the interpolation path
Nx = 50
x_pos = np.linspace(x_known[0], x_known[1], Nx)
y_pred = []

for i, x0 in enumerate(x_pos):
    # x0 is the point where we want to make a prediction
    # x1 and x2 are the locations of our known data
    x1, x2 = x_known

    # Calculate the kernel values (correlations)
    r01 = kernel(x0, x1)  # Correlation between prediction point and first known point
    r02 = kernel(x0, x2)  # Correlation between prediction point and second known point
    r12 = kernel(x1, x2)  # Correlation between the two known points

    # Construct the full GP correlation matrix for the set {x0, x1, x2}
    R_matrix = np.array([[1.0, r01, r02],
                         [r01, 1.0, r12],
                         [r02, r12, 1.0]])

    # Partition the correlation matrix for calculating the conditional distribution
    # This follows the standard GP prediction equations
    R12 = R_matrix[1:, 0]   # K(x_known, x0)
    R22 = R_matrix[1:, 1:]   # K(x_known, x_known)
    # R21 = R_matrix[0, 1:] # K(x0, x_known)

    inv_R22 = np.linalg.inv(R22)

    # Calculate the conditional mean (the GP prediction at x0)
    # This is a weighted sum of the known y-values.
    # E[y0 | y_known] = mu_0 + K(x0, x_known) * K(x_known, x_known)^-1 * (y_known - mu_known)
    # Here, we assume a prior mean of zero (mu_0 = mu_known = 0).
    E_y0 = 0.0 + np.dot(R12, np.dot(inv_R22, y_known - 0.0))
    y_pred.append(E_y0)

    # --- Plotting for the Animation Frame ---
    fig = plt.figure(dpi=200, figsize=(4, 3))
    ax1 = fig.add_subplot(111)

    # Plot the two kernel functions, scaled by their respective y_known values
    ax1.plot(x_plot, y_known[0] * kernel(x_plot, x_known[0]), 'r')
    ax1.plot(x_plot, y_known[1] * kernel(x_plot, x_known[1]), 'g')

    # Plot the known data points
    ax1.plot(x_known[0], y_known[0], 'rD')
    ax1.plot(x_known[1], y_known[1], 'gD')
    ax1.set_xlim(x_plot[0], x_plot[-1])

    # Plot the current prediction point and the path of the interpolation so far
    ax1.plot(x0, E_y0, 'bo')
    ax1.plot(x_pos[:i + 1], y_pred, 'b')

    # Display the correlation matrix on the plot using LaTeX
    ax1.text(-0.15, 0.8, r'$\mathbf{R} = \begin{bmatrix} 1 & %.2f & %.2f \\ %.2f & 1 & %.2f \\ %.2f & %.2f & 1 \end{bmatrix}$' % (r01, r02, r01, r12, r02, r12))

    fig.tight_layout()
    # Save the figure for this frame of the animation
    fig.savefig(f"./Images/kernel_{i:02d}.png")
    plt.close(fig) # Close the figure to avoid displaying it directly
