"""
Demonstration of the Metropolis-Hastings MCMC Algorithm.

This script provides a simple implementation of the Metropolis-Hastings algorithm, a 
fundamental Markov Chain Monte Carlo (MCMC) method. MCMC methods are used to draw 
samples from a probability distribution that is difficult or impossible to sample from directly.

The algorithm works as follows:
1.  **Initialization**: Start at a random point in the parameter space.
2.  **Proposal**: Propose a new point by making a random jump from the current point, 
    using a 'proposal distribution' (here, a simple Gaussian jump).
3.  **Acceptance**: Calculate the 'acceptance probability', which depends on the ratio of 
    the target probability density at the new point versus the current point. 
4.  **Decision**: Accept the new point with this probability. If the new point is not 
    accepted, the chain stays at the current point for this step.
5.  **Iteration**: Repeat this process. The sequence of accepted points forms a 'Markov chain'.

After a 'burn-in' period, the samples in this chain will be distributed according to the 
target probability distribution. This script uses a 2D multivariate Gaussian as the 
target distribution and generates an animation showing the chain exploring the space and 
the resulting histograms of the samples, which approximate the target distribution.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    The target probability density function (PDF) that we want to sample from.

    This function is proportional to the PDF. For the Metropolis algorithm, we only need 
    the ratio of the function at two points, so any normalization constant cancels out.
    Here, the target distribution is a multivariate Gaussian.

    Args:
        x (np.ndarray): The point in the parameter space at which to evaluate the function.

    Returns:
        float: The value of the unnormalized PDF at x.
    """
    d = x - mu
    arg = np.dot(d.T, np.dot(inv_corr, d))
    return np.exp(-0.5 * arg)

def J(x):
    """
    The proposal distribution (or jump function).

    Given the current point `x`, this function proposes a new point `y` by making a 
    random jump. Here, we use a symmetric Gaussian jump, meaning the probability of 
    jumping from x to y is the same as jumping from y to x.

    Args:
        x (np.ndarray): The current point in the chain.

    Returns:
        np.ndarray: The proposed new point.
    """
    return np.random.multivariate_normal(mean=x, cov=jump_size * np.eye(x.size))

# --- Main Script ---

# 1. Define the Target Distribution (a 2D multivariate Gaussian)
mu = np.array([1.0, 1.0])
corr = np.array([[0.05, 0.02], [0.02, 0.05]])
inv_corr = np.linalg.inv(corr)

# 2. Define the Proposal Distribution
jump_size = 0.0025  # The variance of the Gaussian jump

# 3. Initialize the MCMC Chain
chain_length = 50000
x = 2 * np.random.rand(2)  # Start at a random point

x1_arr = np.array([])
x2_arr = np.array([])

# --- 4. Run the Metropolis Algorithm ---
accepted_count = 0
for i in range(chain_length):
    # Propose a new point
    y = J(x)
    
    # Calculate the acceptance probability (alpha)
    # For a symmetric proposal, alpha = min(1, f(y) / f(x))
    alpha = f(y) / f(x)

    # Decide whether to accept the new point
    u = np.random.rand() # Draw a random number from [0, 1]
    if u < alpha:
        # Accept the new point
        x = y
        accepted_count += 1
    # If not accepted, the chain stays at the old point `x`

    # Store the current point in the chain
    x1_arr = np.append(x1_arr, x[0])
    x2_arr = np.append(x2_arr, x[1])

    # --- 5. Plotting (for animation) ---
    # Periodically save a plot of the chain's progress
    if i % 1000 == 0 and i > 0:
        fig = plt.figure(dpi=200, figsize=(6, 3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(224)

        # Plot the 2D path of the chain
        ax1.plot(x1_arr, x2_arr, 'ko', alpha=0.1, mew=0, ms=2)
        
        # Plot the histograms of the samples for each dimension
        # These should start to approximate the marginal distributions of the target PDF
        ax2.hist(x1_arr, bins=100, density=True)
        ax3.hist(x2_arr, bins=100, density=True)

        # Formatting
        ax1.set_xlim(0.0, 2.0)
        ax1.set_ylim(0.0, 2.0)
        ax2.set_xlim(0.0, 2.0)
        ax3.set_xlim(0.0, 2.0)
        ax2.set_ylim(0.0, 3.0)
        ax3.set_ylim(0.0, 3.0)
        ax1.set_title('MCMC Chain Path')
        ax2.set_title('x1 marginal')
        ax3.set_title('x2 marginal')
        fig.tight_layout()
        fig.savefig(f'./Images/MCMC_{i:05d}.png')
        plt.close(fig)

# Print the acceptance rate
acceptance_rate = accepted_count / chain_length
print(f"Final acceptance rate: {acceptance_rate:.2f}")