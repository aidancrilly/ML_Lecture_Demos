""" Animation script showing the change in the conditional probability from a bivariate normal distribution """
import numpy as np
import matplotlib.pyplot as plt

def bivariate_normal(v,mu,covar):
	""" Given 2D position v, mean mu and covariance matrix covar,
	computes the bivariate normal probability """
	covar_inv = np.linalg.inv(covar)
	d = v-mu
	const = 1/((2*np.pi)*np.sqrt(np.linalg.det(covar)))
	p = const*np.exp(-0.5*np.einsum('ijk, kl, ijl -> ij', d, covar_inv, d))
	return p

def cond_univariate_normal(x,y_data,mu,covar):
	""" Computes the conditional probability p(x|y) of a bivariate normal """
	mu_cond = mu[0] + covar[0,1]/covar[1,1]*(y_data-mu[1])
	sig2_cond = covar[0,0] - covar[0,1]*covar[1,0]/covar[1,1]
	return np.exp(-0.5*(x-mu_cond)**2/sig2_cond)/np.sqrt(2*np.pi*sig2_cond)

Nx = 200
x = np.linspace(-3.0,3.0,Nx)
y = np.copy(x)

xx,yy = np.meshgrid(x,y)
v = np.dstack([xx,yy])

mu = np.array([0.0,0.0])
sig2 = 1.0

Nrot = 30
for i in range(Nrot):
	# Cycle through various correlation values, r
	r = -0.9+1.8*(i/(Nrot-1))

	fig = plt.figure(dpi=200,figsize=(3,6))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212,sharex=ax1)

	corr = np.array([[1.0,r],[r,1.0]])
	covar = sig2*corr

	P = bivariate_normal(v,mu,covar)

	ax1.pcolormesh(xx,yy,P,cmap='Greys')
	ax1.set_aspect('equal')

	y_data = 0.5
	ax1.axhline(y_data,c='r',ls='--')

	p_cond = cond_univariate_normal(x,y_data,mu,covar)

	ax2.plot(x,p_cond)
	ax1.set_xlim(x[0],x[-1])
	ax2.set_xlabel("x")
	ax2.set_ylabel("P(x|y=0.5)")
	ax2.set_ylim(-0.05,0.95)

	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.set_title("P(x,y)")

	fig.tight_layout()

	fig.savefig(f'./Images/CondNormal_{i}.png')