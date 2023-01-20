""" Non-Linear Least Squares demo with fitting a peak-normalised Gaussian """

import numpy as np
import matplotlib.pyplot as plt

def Gaussian(x,mu,sig):
	return np.exp(-0.5*((x-mu)/sig)**2)

def Jacobian(x,mu,sig):
	J = np.zeros((x.shape[0],2))
	J[:,0] = (1.0/sig)*((x-mu)/sig)*Gaussian(x,mu,sig)
	J[:,1] = (1.0/sig)*((x-mu)/sig)**2*Gaussian(x,mu,sig)
	return J

mu_true  = 0.5
sig_true = 0.25

noise = 0.01

x_arr = np.linspace(-2.0,2.0,100)
y_arr = Gaussian(x_arr,mu_true,sig_true) + noise*np.random.normal(size=x_arr.shape[0])

mu_arr = np.linspace(-1.0,2.0,100)
sig_arr = np.linspace(1e-2,1.25,100)

mm,ss = np.meshgrid(mu_arr,sig_arr)

S_arr = np.zeros((mu_arr.shape[0],sig_arr.shape[0]))
for i in range(mu_arr.shape[0]):
	for j in range(sig_arr.shape[0]):
		S_arr[i,j] = np.sum((y_arr-Gaussian(x_arr,mu_arr[i],sig_arr[j]))**2)


# Newton-Raphson
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.pcolormesh(mm,ss,S_arr.T)

mu_guess  = 0.0
sig_guess = 1.0
ax1.scatter(mu_guess,sig_guess)
ax2.plot(x_arr,Gaussian(x_arr,mu_guess,sig_guess))
for i in range(3):
	J = Jacobian(x_arr,mu_guess,sig_guess)
	R = y_arr-Gaussian(x_arr,mu_guess,sig_guess)

	JTJ = np.matmul(J.T,J)
	print(np.linalg.cond(JTJ),JTJ)
	JTJ_inv = np.linalg.inv(JTJ)

	mdSdtheta = np.matmul(J.T,R)
	dtheta    = np.matmul(JTJ_inv,mdSdtheta)

	print(dtheta)

	mu_guess += dtheta[0]
	sig_guess += dtheta[1]

	ax1.scatter(mu_guess,sig_guess)
	ax2.plot(x_arr,Gaussian(x_arr,mu_guess,sig_guess))

ax1.scatter(mu_true,sig_true,c='k',marker='x')
ax2.scatter(x_arr,y_arr,c='k',marker='x')

ax2.set_xlim(x_arr[0],x_arr[-1])
ax1.set_xlabel(r"$\mu$")
ax1.set_ylabel(r"$\sigma$")

ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Gradient-Descent
fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.pcolormesh(mm,ss,S_arr.T)

mu_guess  = 0.0
sig_guess = 1.0
ax1.scatter(mu_guess,sig_guess)
ax2.plot(x_arr,Gaussian(x_arr,mu_guess,sig_guess))
lamb = 7e1
for i in range(4):
	J = Jacobian(x_arr,mu_guess,sig_guess)
	R = y_arr-Gaussian(x_arr,mu_guess,sig_guess)

	JTJ_inv = np.linalg.inv(np.matmul(J.T,J))

	mdSdtheta = np.matmul(J.T,R)
	dtheta   = mdSdtheta/lamb

	print(dtheta)

	mu_guess += dtheta[0]
	sig_guess += dtheta[1]

	ax1.scatter(mu_guess,sig_guess)
	ax2.plot(x_arr,Gaussian(x_arr,mu_guess,sig_guess))

ax1.scatter(mu_true,sig_true,c='k',marker='x')
ax2.scatter(x_arr,y_arr,c='k',marker='x')

ax2.set_xlim(x_arr[0],x_arr[-1])
ax1.set_xlabel(r"$\mu$")
ax1.set_ylabel(r"$\sigma$")

ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.show()