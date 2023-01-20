import numpy as np
import matplotlib.pyplot as plt

def R(x1,x2):
	""" Response matrix, boxcar averaging """
	xx1,xx2 = np.meshgrid(x1,x2)
	R_matrix = np.zeros_like(xx1)
	R_matrix[np.abs(xx1-xx2) < 1.5] = 1.0
	# Normalise
	R_matrix = R_matrix/np.sum(R_matrix,axis=1)
	return R_matrix

Nx = 500
noise_amp = 0.01
lamb = 1e-4

# Setting up truth and signal with noise and response
x  = np.linspace(-4.0,4.0,Nx)
dx = x[1]-x[0]
truth = np.exp(-x**2)
IRF = R(x,x)
signal = np.dot(IRF,truth)+noise_amp*np.random.normal(size=Nx)

# Straight deconvolution
deconv_signal = np.dot(np.linalg.pinv(IRF),signal)
# Deconvolution with regularisation
RTR = np.dot(IRF.T,IRF)
RTsignal = np.dot(IRF.T,signal)
D_matrix = (-1*np.eye(Nx)+1*np.eye(Nx,k=1))/(dx)
DTD = np.dot(D_matrix.T,D_matrix)
regularised_inv = np.linalg.inv(RTR+lamb*DTD)
deconv_reg_signal = np.dot(regularised_inv,RTsignal)

fig = plt.figure(dpi=200,figsize=(3,4))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212,sharex=ax1,sharey=ax1)

ax1.plot(x,truth,'k',label='Truth')
ax1.plot(x,signal,'r',label='Signal')
ax2.plot(x,truth,'k')
ax2.plot(x,deconv_signal,'b',alpha=0.5,label='LA Deconv.')
ax2.plot(x,deconv_reg_signal,'g',label='Regularised Deconv.')

ax1.set_xlim(x[0],x[-1])
ax1.set_ylim(-0.1,1.6)

ax1.legend(frameon=False)
ax2.legend(frameon=False)
fig.tight_layout()

plt.show()