"""
This script demonstrates the use of a Physics-Informed Neural Network (PINN)
to solve the 1D diffusion equation: ∂u/∂t = D * ∂²u/∂x²

This is a common example in physics and engineering, and PINNs provide a way
to solve such differential equations using neural networks, by encoding the
equation itself into the loss function of the network.

Key concepts demonstrated:
- A neural network is used to approximate the solution u(x, t).
- The loss function is composed of two parts:
    1. Mean Squared Error (MSE) loss: This ensures the solution fits any
       available "ground truth" data points.
    2. Physics loss: This ensures the solution obeys the diffusion equation.
       This loss is calculated from the residual of the PDE.
- Automatic differentiation (as provided by JAX) is used to compute the
  derivatives of the neural network's output with respect to its inputs (x and t),
  which is essential for calculating the physics loss.

For an introduction to JAX and differentiable programming, see:
https://github.com/aidancrilly/MiniCourse-DifferentiableSimulation
"""

import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import matplotlib.pyplot as plt

# Diffusivity constant for the diffusion equation.
D = 1.0

def diffusion_solution(t, x):
    """
    Analytical solution to the 1D diffusion equation for a Dirac delta
    function initial condition at t=0. This serves as our ground truth for
    generating training data and for comparison.

    Args:
        t (jax.numpy.ndarray): Time coordinates.
        x (jax.numpy.ndarray): Spatial coordinates.

    Returns:
        jax.numpy.ndarray: The value of the solution u(x, t).
    """
    return jnp.exp(-x**2 / (4 * D * t)) / jnp.sqrt(4 * jnp.pi * D * t)

class PINN(eqx.Module):
    """
    The Physics-Informed Neural Network model.

    This is a simple Multi-Layer Perceptron (MLP) that takes spatial (x) and
    temporal (t) coordinates as input and outputs the predicted value of the
    solution u(x, t).
    """
    mlp: eqx.nn.MLP

    def __init__(self, in_size, out_size, width_size, depth, *, key, **kwargs):
        """
        Initializes the MLP.

        Args:
            in_size (int): Input size (2 for x and t).
            out_size (int): Output size (1 for u).
            width_size (int): Number of neurons in each hidden layer.
            depth (int): Number of hidden layers.
            key (jax.random.PRNGKey): JAX random key for initialization.
        """
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, x, t):
        """
        Performs a forward pass of the network.

        Args:
            x (float): Spatial coordinate.
            t (float): Temporal coordinate.

        Returns:
            float: The predicted value of the solution u(x, t).
        """
        # The inputs are concatenated to form a single input vector for the MLP.
        input_vec = jnp.array([x, t])
        y_PINN = self.mlp(input_vec)
        return y_PINN.reshape(())

def train_PINN(
    training_ts, training_xs, training_sol,
    plotting_ts, plotting_xs,
    lr_strategy=(1e-3,),
    steps_strategy=(2000,),
    width_size=32,
    depth=3,
    seed=5678,
    plot=True,
    print_every=50,
):
    """
    Trains the PINN model.

    Args:
        training_ts (jax.numpy.ndarray): Time coordinates for training data.
        training_xs (jax.numpy.ndarray): Spatial coordinates for training data.
        training_sol (jax.numpy.ndarray): Solution values at training points.
        plotting_ts (jax.numpy.ndarray): Time coordinates for plotting the results.
        plotting_xs (jax.numpy.ndarray): Spatial coordinates for plotting the results.
        lr_strategy (tuple): Tuple of learning rates for the optimizer.
        steps_strategy (tuple): Tuple of number of training steps for each learning rate.
        width_size (int): Width of the neural network.
        depth (int): Depth of the neural network.
        seed (int): Random seed.
        plot (bool): Whether to generate plots during training.
        print_every (int): How often to print loss values.

    Returns:
        PINN: The trained model.
    """
    key = jrandom.PRNGKey(seed)
    __, model_key = jrandom.split(key)

    model = PINN(2, 1, width_size, depth, key=model_key)

    tt, xx = jnp.meshgrid(plotting_ts, plotting_xs, indexing='ij')
    sol_y = diffusion_solution(tt.flatten(), xx.flatten())

    @eqx.filter_jit
    def PDE_loss(model, ti, xi):
        """
        Calculates the physics-based loss.

        This function computes the residual of the diffusion equation.
        The goal of the training is to minimize this residual, effectively
        forcing the neural network to satisfy the physics of the problem.
        """
        # Use jax.grad to compute the derivatives of the model's output.
        # `jax.vmap` is used to apply the function over the batch of inputs.
        # ∂u/∂t
        dydt = jax.vmap(jax.grad(model, argnums=1))
        # ∂²u/∂x²
        d2ydx2 = jax.vmap(jax.grad(jax.grad(model, argnums=0), argnums=0))

        # The residual of the PDE: ∂u/∂t - D * ∂²u/∂x²
        # For a perfect solution, this would be zero.
        g_PDE = dydt(xi, ti) - D * d2ydx2(xi, ti)

        # We return the mean squared residual.
        return jnp.mean(g_PDE**2)

    @eqx.filter_jit
    def MSE_loss(model, ti, xi, yi):
        """
        Calculates the Mean Squared Error (MSE) loss.

        This is the "data-driven" part of the loss. It measures how well the
        network's prediction matches the provided training data.
        """
        y_pred = jax.vmap(model)(xi, ti)
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, xi, yi):
        """
        Calculates the total loss and its gradient.

        The total loss is a sum of the MSE loss and the PDE loss. This is the
        core idea of a PINN: the model learns to satisfy both the data and the
        underlying physics simultaneously.
        """
        mse_loss = MSE_loss(model, ti, xi, yi)
        pde_loss = PDE_loss(model, ti, xi)
        return mse_loss + pde_loss

    @eqx.filter_jit
    def make_step(ti, xi, yi, model, opt_state):
        """
        Performs a single optimization step.
        """
        loss, grads = grad_loss(model, ti, xi, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    # Training loop
    count = 0
    for lr, steps in zip(lr_strategy, steps_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        for step in range(steps):
            count += 1
            loss, model, opt_state = make_step(training_ts, training_xs, training_sol, model, opt_state)

            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, MSE Loss: {MSE_loss(model, training_ts, training_xs, training_sol)}, PDE Loss: {PDE_loss(model, training_ts, training_xs)}")
                if plot:
                    fig = plt.figure(dpi=200)
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    model_y = jax.vmap(model)(xx.flatten(), tt.flatten())
                    ax1.pcolormesh(tt, xx, model_y.reshape(plotting_ts.shape[0], plotting_xs.shape[0]), vmin=0, vmax=jnp.amax(sol_y))
                    ax2.pcolormesh(tt, xx, sol_y.reshape(plotting_ts.shape[0], plotting_xs.shape[0]), vmin=0, vmax=jnp.amax(sol_y))
                    ax2.plot(training_ts, training_xs, 'kx')
                    ax1.set_xlabel("t")
                    ax1.set_ylabel("x")
                    ax1.set_title(f"PINN, step = {step}")
                    ax2.set_title("Truth/Data")
                    fig.tight_layout()
                    # Note: This assumes a 'Plots' directory exists.
                    fig.savefig(f"./Images/PINN_{str(count).zfill(4)}.png")
                    plt.close(fig)

    return model

# --- Main script execution ---

# Define the spatial and temporal domain for plotting.
extent = 4.0
plotting_ts = jnp.linspace(0.5, 1.5, 50)
plotting_xs = jnp.linspace(-extent, extent, 100)

# Generate training data.
# We create a set of random points in space and time and use the analytical
# solution to get the "ground truth" values at these points.
data_seed = 404
key = jrandom.PRNGKey(data_seed)
Ntrain = 100

training_xs = jax.random.uniform(key, shape=(Ntrain,), minval=-extent, maxval=extent)
__, key = jrandom.split(key)
training_ts = jax.random.uniform(key, shape=(Ntrain,), minval=0.5, maxval=1.5)

# Get the solution at the training points.
training_ys = diffusion_solution(training_ts, training_xs)

# Train the PINN model.
NDE_model = train_PINN(training_ts, training_xs, training_ys, plotting_ts, plotting_xs, plot=True)