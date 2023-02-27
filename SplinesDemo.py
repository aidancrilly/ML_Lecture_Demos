""" Demonstration of B-spline fitting with regularisation """
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def f_true(x,lamb=2e0,omega=2e1):
	""" Function we will draw samples from """
	return np.cos(omega*x)*np.exp(-lamb*x)

def B_matrix(spline_order,knots,x):
	""" Construct B-spline matrix at spline_order (k = 4 is cubic), 
	for knot positions, knots, which interpolate to higher resolution grid, x 

	Matrix B is constructed such that B.A gives a spline f(x) where A is a vector of spline amplitudes

	"""
    Nx     = x.shape[0]
    Nknots = knots.shape[0]
    Nc     = Nknots-spline_order
    B_k    = np.zeros((Nx,Nc))
    for i in range(Nc):
        c = np.zeros(Nc)
        c[i] = 1.0
        B = BSpline(knots, c, spline_order-1)
        B_k[:,i] = B(x)
    return B_k

# Total number of data points being fit to
Nx    = 40
# Number of data points to start with
Nx_start = 5

# Splines are piecewise polynomial functions of order n
# Knots are the points where the first n-1 derivatives agree
N_knots = 40
spline_order = 4
N_control = N_knots-spline_order
# Open-Uniform knot structure
# Additional knots are added at either end of the domain allows non-zero values for the spline
knot_vec = np.linspace(0.0,1.0,N_knots-2*spline_order)
knot_vec = np.insert(knot_vec,0,[knot_vec[0],]*spline_order)
knot_vec = np.insert(knot_vec,-1,[knot_vec[-1],]*spline_order)

x_grid = np.linspace(0.0,1.0,1000)
x = np.random.rand(Nx_start)
# Construct B-spline matrix
B_grid = B_matrix(spline_order,knot_vec,x_grid)
# Add some regularisation to prevent ill-conditioned solutions
regularisation = 1e-6
for ix in range(Nx_start,Nx):
	x = np.append(x,np.random.rand())
	y = f_true(x)

	B = B_matrix(spline_order,knot_vec,x)
	BTB = np.dot(B.T,B)
	I = np.linalg.inv(BTB+regularisation*np.eye(BTB.shape[0]))
	# Estimate best-fit amplitudes
	a = np.dot(I,np.dot(B.T,y))

	fig = plt.figure(dpi = 200, figsize=(4,4))
	plt.scatter(x,y,label='Samples')

	plt.plot(x_grid,np.dot(B_grid,a),label='Spline')
	plt.plot(x_grid,f_true(x_grid),'k--',label='Truth')
	plt.legend(frameon=False)

	plt.xlim(0.0,1.0)
	plt.ylim(-1.1,1.1)
	plt.ylabel("f(x)")
	plt.xlabel("x")
	fig.tight_layout()
	fig.savefig(f'./Images/Spline_ix_{ix-Nx_start:03d}.png')

