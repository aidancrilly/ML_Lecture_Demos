import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

def Gaussian(x,mu,sig,A,B):
	""" Gaussian function with mean mu and standard deviation sig"""
	return A*jnp.exp(-0.5*((x-mu)/sig)**2)+B

def create_data(x_arr,mu,sig,A,B,noise,key):
	""" Creates some synthetic data which is a noisy Gaussian peak """
	return Gaussian(x_arr,mu,sig,A,B) + noise*jax.random.normal(key,shape=[x_arr.shape[0]])

def neg_log_likelihood(mu,sig,A,B,x_data,y_data,yerr):
	""" Function to compute the negative log likelihood of our fit to data """
	y_model = Gaussian(x_data,mu,sig,A,B)
	return 0.5*jnp.sum(((y_data-y_model)/yerr)**2)

""" Gradient of neg_log_likelihood function w.r.t. input parameters """
# Here I am doing each gradient explicitly, but it can be done vectorially
# See how Hessian is done
gradL_mu  = jax.grad(neg_log_likelihood,argnums=0)
gradL_sig = jax.grad(neg_log_likelihood,argnums=1)
gradL_A   = jax.grad(neg_log_likelihood,argnums=2)
gradL_B   = jax.grad(neg_log_likelihood,argnums=3)

# N.B. we are going to perform gradient descent so 
# the Jacobian of interest is the one with respect to the loss and not the model function
def Jacobian(mu,sig,A,B,x_data,y_data,yerr):
	""" Construct the Jacobian using the Automatic Differential gradient functions """
	J0 = gradL_mu(mu,sig,A,B,x_data,y_data,yerr)
	J1 = gradL_sig(mu,sig,A,B,x_data,y_data,yerr)
	J2 = gradL_A(mu,sig,A,B,x_data,y_data,yerr)
	J3 = gradL_B(mu,sig,A,B,x_data,y_data,yerr)
	J = jnp.array([J0,J1,J2,J3])
	return J

def Hessian(mu,sig,A,B,x_data,y_data,yerr):
    """ Construct the Hessian using the Automatic Differential gradient functions """
    H = jax.hessian(neg_log_likelihood,argnums=(0,1,2,3))(mu,sig,A,B,x_data,y_data,yerr)
    return jnp.array(H).reshape(4,4)

# Truth values of inputs
mu_true  = 0.25
sig_true = 0.5
A_true   = 1.0
B_true   = 0.0
# Add a small amount of noise
noise = 0.05
# Synthetic data to be fit
x_arr = jnp.linspace(-2.0,2.0,100)
y_arr = create_data(x_arr,mu_true,sig_true,A_true,B_true,noise,key)

# Find optimum via gradient descent
mu_candidate  = 0.0
sig_candidate = 0.75
A_candidate   = 1.0
B_candidate   = 0.0
candidate_arr = jnp.array([mu_candidate,sig_candidate,A_candidate,B_candidate])
learning_rate = 1e-5
Niterations   = 25
for iter in range(Niterations):
	gradients = Jacobian(*candidate_arr,x_arr,y_arr,noise)
	candidate_arr -= learning_rate*gradients
	
# Compute covariance matrix using Hessian
H = Hessian(*candidate_arr,x_arr,y_arr,noise)
covar = jnp.linalg.inv(H)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122,sharex=ax1)
x_plot = jnp.linspace(x_arr[0],x_arr[-1],500)

ax1.plot(x_arr,y_arr,'kx')

Nsamples = int(1e2)
Laplace_samples = jax.random.multivariate_normal(key,candidate_arr,covar,shape=[Nsamples])
for i in range(Nsamples):
    Laplace_sample = Laplace_samples[i,:]
    # Without noise
    ax1.plot(x_plot,Gaussian(x_plot,*Laplace_sample),'orange',alpha=0.05)
    # With noise
    key, subkey = jax.random.split(key)
    ax2.plot(x_plot,create_data(x_plot,*Laplace_sample,noise,subkey),'orange',ls='',marker='o',alpha=0.01,markersize=2)

ax1.plot(x_plot,Gaussian(x_plot,*candidate_arr),'r',lw=1)
ax2.plot(x_arr,y_arr,'kx')

ax1.set_xlim(x_arr[0],x_arr[-1])

ax1.set_title("Fit w uncertainties")
ax2.set_title("Fit w uncertainties & noise")

plt.show()