import numpy as np
import matplotlib.pyplot as plt

def f(x):
	""" Function for which we want to find the PDF """
	d = x - mu
	arg = np.dot(d.T,np.dot(inv_corr,d))
	return np.exp(-0.5*arg)

def J(x):
	""" Normal jump function, given input x suggest a new postion y, centred on x """
	y = np.random.multivariate_normal(mean=x,cov=jump_size*np.eye(x.size))
	return y

# f parameters, multivariate Gaussian
mu   = np.array([1.0,1.0])
corr = np.array([[0.05,0.02],[0.02,0.05]])
inv_corr = np.linalg.inv(corr)

# Jump function properties
jump_size = 0.0025

# Metropolis algorithm with 
chain_length = 50000
x = 2*np.random.rand(2)

x1_arr = np.array([])
x2_arr = np.array([])
u = 1.0
alpha = 0.0
for i in range(chain_length):
	# Acceptance criterion
	while(u > alpha): 
		y = J(x) # Suggest new location
		u = np.random.rand() # Create random number in range 0 to 1 
		alpha = f(y)/f(x) # Compute ratio of function at trial point and current point

	# When new trial point accepted, move to this new point
	x = y
	# Reset u and alpha such that the acceptance criterion is no longer satisfied for next trial point
	u = 1.0
	alpha = 0.0
	x1_arr = np.append(x1_arr,x[0])
	x2_arr = np.append(x2_arr,x[1])

	# Plotting script
	if(i % 1000 == 0):
		# Initialise figure
		fig = plt.figure(dpi=200,figsize=(6,3))
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(224)
		ax1.plot(x1_arr,x2_arr,'ko',alpha=0.1,mew=0,ms=2)
		ax2.hist(x1_arr,bins=100,density=True)
		ax3.hist(x2_arr,bins=100,density=True)
		ax1.set_xlim(0.0,2.0)
		ax1.set_ylim(0.0,2.0)
		ax2.set_xlim(0.0,2.0)
		ax3.set_xlim(0.0,2.0)
		ax2.set_ylim(0.0,3.0)
		ax3.set_ylim(0.0,3.0)
		fig.tight_layout()
		fig.savefig(f'./Images/MCMC_{i:05d}.png')