"""
Demonstration of Laplace's Method for Bayesian Inference using JAX.

This script illustrates Laplace's method, a technique used to approximate a posterior 
distribution with a Gaussian distribution. This is a common approach in Bayesian machine 
learning to get a sense of the uncertainty in a model's parameters beyond a simple 
point estimate.

The script performs the following steps:
1.  **Model Definition**: Defines a simple model (a Gaussian peak) to be fitted to data.
2.  **Synthetic Data**: Generates noisy data based on the model with some 'true' parameters.
3.  **Log-Likelihood**: Defines the negative log-likelihood function. In a Bayesian context, 
    minimising this is equivalent to finding the Maximum a Posteriori (MAP) estimate, 
    assuming a uniform prior.
4.  **Automatic Differentiation**: Uses the JAX library to automatically compute the gradient 
    (Jacobian) and the Hessian of the negative log-likelihood function. This is a key 
    advantage of modern deep learning frameworks.
5.  **Optimisation**: Performs gradient descent to find the parameters that minimise the 
    negative log-likelihood (i.e., find the MAP estimate).
6.  **Posterior Approximation**: At the MAP estimate, it computes the Hessian of the negative 
    log-likelihood. The inverse of the Hessian is used as the covariance matrix for the 
    approximating Gaussian distribution.
7.  **Visualisation**: It samples from this approximate posterior distribution and plots the 
    resulting model predictions to visualise the parameter uncertainty.
"""
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# JAX requires an explicit random key for reproducibility
key = jax.random.PRNGKey(0)

def Gaussian(x, mu, sig, A, B):
    """
    A simple Gaussian function.

    Args:
        x (jnp.ndarray): Input coordinates.
        mu (float): The mean of the Gaussian.
        sig (float): The standard deviation of the Gaussian.
        A (float): The amplitude of the Gaussian.
        B (float): A constant offset.

    Returns:
        jnp.ndarray: The value of the Gaussian function at x.
    """
    return A * jnp.exp(-0.5 * ((x - mu) / sig)**2) + B

def create_data(x_arr, mu, sig, A, B, noise, key):
    """
    Creates synthetic data by evaluating the Gaussian function and adding noise.

    Args:
        x_arr (jnp.ndarray): Input coordinates.
        mu, sig, A, B (float): Parameters of the Gaussian.
        noise (float): The standard deviation of the Gaussian noise to add.
        key (jax.random.PRNGKey): JAX random key.

    Returns:
        jnp.ndarray: The noisy data.
    """
    return Gaussian(x_arr, mu, sig, A, B) + noise * jax.random.normal(key, shape=[x_arr.shape[0]])

def neg_log_likelihood(mu, sig, A, B, x_data, y_data, yerr):
    """
    Computes the negative log-likelihood of the model given the data.

    Assuming Gaussian noise, the likelihood is proportional to exp(-0.5 * chi-squared), 
    so the negative log-likelihood is proportional to the chi-squared value.

    Args:
        mu, sig, A, B (float): Parameters of the Gaussian model.
        x_data (jnp.ndarray): The x-coordinates of the data.
        y_data (jnp.ndarray): The y-coordinates of the data.
        yerr (float): The uncertainty (standard deviation) of the y-data.

    Returns:
        float: The negative log-likelihood.
    """
    y_model = Gaussian(x_data, mu, sig, A, B)
    return 0.5 * jnp.sum(((y_data - y_model) / yerr)**2)

# --- Automatic Differentiation with JAX ---

# Use jax.grad to create functions that compute the gradient of the negative log-likelihood
# with respect to each of the model parameters.
# `argnums` specifies which argument of the function to differentiate with respect to.
gradL_mu = jax.grad(neg_log_likelihood, argnums=0)
gradL_sig = jax.grad(neg_log_likelihood, argnums=1)
gradL_A = jax.grad(neg_log_likelihood, argnums=2)
gradL_B = jax.grad(neg_log_likelihood, argnums=3)

def Jacobian(mu, sig, A, B, x_data, y_data, yerr):
    """
    Constructs the Jacobian (gradient vector) of the negative log-likelihood.

    Returns:
        jnp.ndarray: The gradient of the negative log-likelihood with respect to the parameters.
    """
    J0 = gradL_mu(mu, sig, A, B, x_data, y_data, yerr)
    J1 = gradL_sig(mu, sig, A, B, x_data, y_data, yerr)
    J2 = gradL_A(mu, sig, A, B, x_data, y_data, yerr)
    J3 = gradL_B(mu, sig, A, B, x_data, y_data, yerr)
    return jnp.array([J0, J1, J2, J3])

def Hessian(mu, sig, A, B, x_data, y_data, yerr):
    """
    Constructs the Hessian matrix of the negative log-likelihood using jax.hessian.

    The Hessian is the matrix of second derivatives. For Laplace's method, the inverse 
    of the Hessian at the MAP estimate gives the covariance of the approximating Gaussian.

    Returns:
        jnp.ndarray: The Hessian matrix.
    """
    # jax.hessian can compute the Hessian with respect to multiple arguments at once.
    H = jax.hessian(neg_log_likelihood, argnums=(0, 1, 2, 3))(mu, sig, A, B, x_data, y_data, yerr)
    # The output of jax.hessian is a nested tuple, so we reshape it into a matrix.
    return jnp.array(H).reshape(4, 4)

# --- Main Script ---

# 1. Create Synthetic Data
mu_true, sig_true, A_true, B_true = 0.25, 0.5, 1.0, 0.0
noise = 0.05
x_arr = jnp.linspace(-2.0, 2.0, 100)
y_arr = create_data(x_arr, mu_true, sig_true, A_true, B_true, noise, key)

# 2. Find MAP Estimate via Gradient Descent
# Initial guess for the parameters
candidate_arr = jnp.array([0.0, 0.75, 1.0, 0.0])
learning_rate = 1e-5
Niterations = 25

print("Running gradient descent to find MAP estimate...")
for i in range(Niterations):
    # Calculate the gradient at the current parameter values
    gradients = Jacobian(*candidate_arr, x_arr, y_arr, noise)
    # Update the parameters by taking a small step in the direction of the negative gradient
    candidate_arr -= learning_rate * gradients
print("MAP estimate found:", candidate_arr)

# 3. Approximate Posterior with a Gaussian
# At the MAP estimate, the posterior is approximated by N(MAP, H^-1)
# where H is the Hessian of the negative log-likelihood.
print("Calculating Hessian and inverting for covariance...")
H = Hessian(*candidate_arr, x_arr, y_arr, noise)
covar = jnp.linalg.inv(H)
print("Covariance matrix:\n", covar)

# --- 4. Plotting and Visualisation ---
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharex=ax1)
x_plot = jnp.linspace(x_arr[0], x_arr[-1], 500)

ax1.plot(x_arr, y_arr, 'kx', label='Data')
ax2.plot(x_arr, y_arr, 'kx', label='Data')

# Sample from the approximate posterior distribution
Nsamples = 100
Laplace_samples = jax.random.multivariate_normal(key, candidate_arr, covar, shape=[Nsamples])

print(f"Drawing {Nsamples} samples from the approximate posterior...")
for i in range(Nsamples):
    Laplace_sample = Laplace_samples[i, :]
    # Plot the model for each parameter sample (visualising parameter uncertainty)
    ax1.plot(x_plot, Gaussian(x_plot, *Laplace_sample), 'orange', alpha=0.05)
    
    # Plot the model plus noise for each sample (visualising predictive uncertainty)
    key, subkey = jax.random.split(key)
    ax2.plot(x_plot, create_data(x_plot, *Laplace_sample, noise, subkey), 'orange', ls='', marker='o', alpha=0.01, markersize=2)

# Plot the best-fit (MAP) model
ax1.plot(x_plot, Gaussian(x_plot, *candidate_arr), 'r', lw=1, label='MAP Fit')
ax2.plot(x_plot, Gaussian(x_plot, *candidate_arr), 'r', lw=1, label='MAP Fit')

ax1.set_xlim(x_arr[0], x_arr[-1])
ax1.set_title("Posterior over Functions")
ax2.set_title("Posterior Predictive Distribution")
ax1.legend()
ax2.legend()

plt.show()