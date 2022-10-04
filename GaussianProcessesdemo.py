import numpy as np
import matplotlib.pyplot as plt

def f_true(x,lamb=2e0,omega=2e1):
	""" Function we will draw samples from """
	return np.cos(omega*x)*np.exp(-lamb*x)

def correlation_function(x1,x2,theta):
	""" Correlation function or kernal of the GP, we will use a Gaussian with length scale theta """
	return np.exp(-((x1-x2)/theta)**2)

def correlation_matrix(x,sigma,theta):
	""" Construct correlation matrix using correlation function and vertical length scale sigma """
	x1,x2 = np.meshgrid(x,x)
	corr = sigma**2*correlation_function(x1,x2,theta)
	return corr

def conditional_distribution(x_trial,x_known,y_known,sigma=0.5,theta=0.1):
	""" Calculate p(x_trial | x_known,y_known) for given values of sigma and theta """
	x_total = np.concatenate((x_trial,x_known),axis=0)
	N_trail = x_trial.shape[0]
	corr    = correlation_matrix(x_total,sigma,theta)
	corr_11 = corr[:N_trail,:N_trail]
	corr_12 = corr[:N_trail,N_trail:]
	corr_21 = corr_12.T
	corr_22 = corr[N_trail:,N_trail:]
	# Note unconditional mu = 0
	corr_22_inv = np.linalg.inv(corr_22)
	mu_cond  = np.dot(np.matmul(corr_12,corr_22_inv),y_known)
	sig_cond = corr_11 - np.matmul(corr_12,np.matmul(corr_22_inv,corr_21))
	return mu_cond,sig_cond

N_known = 8
X_known = np.random.rand(N_known)
Y_known = f_true(X_known)

X_trial = np.linspace(0.0,1.0,100)

mu_cond,sig_cond = conditional_distribution(X_trial,X_known,Y_known)

plt.scatter(X_known,Y_known,marker='D',label='Data')
plt.plot(X_trial,mu_cond,'b',label='GP mean')
plt.plot(X_trial,mu_cond+np.sqrt(np.diag(sig_cond)),'b--',label='GP confidence interval')
plt.plot(X_trial,mu_cond-np.sqrt(np.diag(sig_cond)),'b--')

plt.plot(X_trial,f_true(X_trial),'k',label='True function')

plt.legend()
plt.show()