"""
Demonstration of Bayesian Optimisation.

This script provides a practical example of Bayesian Optimisation, a powerful technique for 
the global optimisation of black-box functions. The core idea is to build a probabilistic 
model of the objective function and use this model to make decisions about where to sample next.

In this demonstration, we use a Gaussian Process (GP) as the probabilistic model. The GP provides 
a posterior distribution over the objective function, which we can use to balance exploration 
(sampling in areas of high uncertainty) and exploitation (sampling in areas likely to yield a good result).

The script performs the following steps:
1. Defines a test function to be optimised (the Forrester function).
2. Implements several acquisition functions (Lower Confidence Bound, Expected Improvement) 
   that guide the optimisation process.
3. In a loop, it:
    a. Fits a GP to the currently known data points.
    b. Uses the GP to create an acquisition function.
    c. Finds the point that maximises the acquisition function.
    d. Samples the objective function at this new point and adds it to our dataset.
    e. Visualises the process, showing the true function, the GP model, the sampled points, 
       and the acquisition function.

This script relies on the GP implementation from the `GaussianProcessesdemo.py` file.
"""
from GaussianProcessesdemo import *
from scipy.special import erf

def f_opt(x):
    """ 
    Forrester function, a standard test function for optimisation algorithms.
    
    This function is a simple one-dimensional test function. It is multimodal, 
    with one global minimum, one local minimum, and a zero-gradient inflection point,
    making it a good challenge for optimisation algorithms.

    Args:
        x (float or np.ndarray): The input point(s).

    Returns:
        float or np.ndarray: The function value(s) at x.
    """
    return (6*x-2)**2*np.sin(12*x-4)

def LowerConfidenceBound(mu, sig, y_min, lamb=1.5):
    """
    Calculates the Lower Confidence Bound (LCB) acquisition function.

    LCB is an optimistic acquisition function that favors points with a low mean (exploitation) 
    and high uncertainty (exploration). The balance between exploration and exploitation is 
    controlled by the parameter lambda.

    Args:
        mu (np.ndarray): The mean of the GP posterior at the trial points.
        sig (np.ndarray): The standard deviation of the GP posterior at the trial points.
        y_min (float): The minimum value of the objective function observed so far.
        lamb (float, optional): The exploration-exploitation trade-off parameter. Defaults to 1.5.

    Returns:
        np.ndarray: The LCB value for each trial point.
    """
    return mu - lamb * sig

def ExpectedImprovement(mu, sig, y_min, lamb=0.0):
    """
    Calculates the Expected Improvement (EI) acquisition function.

    EI measures the expected amount of improvement over the best-observed value. 
    It is a popular and effective acquisition function.

    Args:
        mu (np.ndarray): The mean of the GP posterior at the trial points.
        sig (np.ndarray): The standard deviation of the GP posterior at the trial points.
        y_min (float): The minimum value of the objective function observed so far.
        lamb (float, optional): A small jitter term to prevent division by zero. Defaults to 0.0.

    Returns:
        np.ndarray: The negative EI value for each trial point (for minimisation).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = -(mu - y_min - lamb) / sig
        norm = np.exp(-0.5 * arg**2) / np.sqrt(2 * np.pi)
        CDF = 0.5 * (1 + erf(arg / np.sqrt(2)))
        EI = -(mu - y_min - lamb) * CDF + sig * norm
    return -EI

def TrainGP(X_known, Y_known):
    """
    Trains a Gaussian Process model on the given data.

    This function takes known (X, Y) data points, finds the optimal hyperparameter 
    (correlation length scale `theta`) for the GP by maximising the log-likelihood, 
    and returns a function that can be used to predict the mean and covariance at new points.

    Args:
        X_known (np.ndarray): The input data points where the function has been evaluated.
        Y_known (np.ndarray): The corresponding function values.

    Returns:
        function: A function that takes a new x position (or an array of positions) and 
                  returns the mean and covariance of the GP posterior at that position(s).
    """
    # Fit maximum likelihood of free parameter, correlation length scale, from known data
    res = minimize(neg_log_likelihood, 0.1, args=(X_known, Y_known))
    
    # Compute GP properties from maximum likelihood result
    theta_opt = res.x[0]
    corr_opt = correlation_matrix(X_known, theta_opt)
    sigma_opt = np.sqrt(sigma2_MLE(corr_opt, Y_known))
    
    # Create a lambda function for convenient prediction
    mu_sig_GP = lambda x_trial: conditional_distribution(x_trial, X_known, Y_known, sigma_opt, theta_opt)
    return mu_sig_GP

def CreateAcquisitionFunction(X, Y, afunc):
    """
    Creates a closure for the acquisition function.

    This function takes the current data (X, Y) and an acquisition function type (`afunc`),
    trains a GP on the data, and returns a callable acquisition function that only 
    takes the trial point `x_trial` as input. This is useful for passing to an optimiser.

    Args:
        X (np.ndarray): The input data points.
        Y (np.ndarray): The corresponding function values.
        afunc (function): The acquisition function to use (e.g., ExpectedImprovement).

    Returns:
        tuple: A tuple containing:
            - function: The acquisition function, ready to be optimised.
            - function: The trained GP model.
    """
    GP = TrainGP(X, Y)
    y_min = np.amin(Y)
    
    def AcquisitionFunction(x_trial):
        mu, sig_mat = GP(np.atleast_1d(x_trial))
        sig = np.sqrt(np.diag(sig_mat))
        return afunc(mu, sig, y_min)
        
    return AcquisitionFunction, GP

# --- Main Script ---

# Set a seed for reproducibility
np.random.seed(9)

# Initial data: sample N_start points from the function
N_start = 3
X = np.random.uniform(size=(N_start,))
Y = f_opt(X)

# Create a dense grid for plotting the true function and the GP
x_plot = np.linspace(0.0, 1.0, 200)
y_plot = f_opt(x_plot)

# Set up the plot
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# --- Bayesian Optimisation Loop ---
N_BO_iters = 6
N_opt_restarts = 10 # Number of restarts for the acquisition function optimisation

for iter in range(N_BO_iters):
    # Clear the axes for the new iteration's plot
    ax1.clear()
    ax2.clear()
    
    # Plot the true function and the points sampled so far
    ax1.plot(x_plot, y_plot, 'k', label='True Function')
    ax1.plot(X, Y, 'bo', ls='', label='Sampled Points')

    # Create the acquisition function and the GP model for this iteration
    afunc, GP = CreateAcquisitionFunction(X, Y, ExpectedImprovement)
    
    # Get the GP's prediction over the plotting grid
    mu, sig = GP(x_plot)

    # Plot the GP mean and confidence interval
    ax1.plot(x_plot, mu, 'b', label='GP Mean')
    ax1.plot(x_plot, mu + np.sqrt(np.diag(sig)), 'b--', label='GP Confidence Interval')
    ax1.plot(x_plot, mu - np.sqrt(np.diag(sig)), 'b--')
    
    # Plot the acquisition function
    ax2.plot(x_plot, -afunc(x_plot), 'g', label='Acquisition Function')

    # --- Optimise the Acquisition Function ---
    # We use a multi-start optimisation to find the global maximum of the acquisition function
    x_next = 0.0
    a_next = 1.0
    for iter_opt in range(N_opt_restarts):
        # We use L-BFGS-B, a quasi-Newton method that can handle bounds
        opt_res = minimize(afunc, np.random.uniform(), method='L-BFGS-B', bounds=[(0.0, 1.0)])
        a_res = opt_res.fun
        
        # If we found a better point, store it
        if a_res < a_next:
            x_next, a_next = opt_res.x, a_res
            y_next = f_opt(x_next)
            
    # Plot the next point to be sampled
    ax1.plot(x_next, y_next, 'ro', label='Next Sample')

    # --- Finalise Plot and Update Data ---
    ax1.legend()
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel('Acquisition function', c='g')
    ax2.yaxis.set_label_position("right")
    
    # Save the figure for this iteration
    fig.savefig(f'./Images/BO_{str(iter).zfill(2)}.png')

    # Add the new sample to our dataset for the next iteration
    X = np.append(X, x_next)
    Y = np.append(Y, y_next)

# Display the final plot
plt.show()