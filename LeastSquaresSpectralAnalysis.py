""" Performs Least Squares Spectral Analysis (LSSA) using the Vaníček method"""
import numpy as np
import matplotlib.pyplot as plt

def cos_basis_matrix(Nm,Nx,t):
	""" Builds a matrix which columns are the cosine basis functions evaluated at x"""
	A = np.zeros((Nx,Nm))
	k = 0.5*np.arange(Nm)/t[-1]
	for m in range(Nm):
		A[:,m] = np.cos(2*np.pi*k[m]*t)
	return A,k

def trial_function(t,omega1=8,omega2=4):
	""" Trial function on which we will perform LSSA """
	y = (np.cos(omega1*np.pi*t)+np.cos(omega2*np.pi*t))*np.exp(-t**2)
	return y

Nt = 400
t = np.linspace(-4.0,4.0,Nt)
y = trial_function(t)

fig = plt.figure(dpi=200,figsize=(3,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(t,y,'k',label='Truth')
# FFT
Y    = np.fft.fft(y)*(t[1]-t[0])
freq = np.fft.fftfreq(Nt, t[1] - t[0])
Y    = np.fft.fftshift(Y)
freq = np.fft.fftshift(freq)
ax2.plot(freq,np.abs(Y),'k',label='FFT')

N_list = [45,25]
for N in N_list:
	A,k = cos_basis_matrix(N,Nt,t)

	ATA = np.dot(A.T,A)
	ATy = np.dot(A.T,y)
	theta = np.linalg.solve(ATA,ATy)

	ax1.plot(t,np.dot(A,theta),ls='--',label=f'LSSA N = {N}')
	ax2.plot(k,4*theta,ls='--',label=f'LSSA N = {N}')

ax1.set_xlim(t[0],t[-1])
ax1.set_ylabel("y(t)")
ax1.set_xlabel("t")
ax2.set_ylabel("Y(k)")
ax2.set_xlabel("k")
ax2.set_xlim(0.0,8.0)
ax1.legend(frameon=False,fontsize=6)
ax2.legend(frameon=False,fontsize=6)

fig.tight_layout()
plt.show()
