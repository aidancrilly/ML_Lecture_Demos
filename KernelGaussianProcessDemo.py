import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

def kernel(x1,x2,l=0.2):
	return np.exp(-0.5*((x1-x2)/l)**2)

x_known = np.array([0.2,0.8])
y_known = np.array([-1.0,1.0])
x_plot = np.linspace(-0.2,1.2,100)

Nx = 50
x_pos = np.linspace(x_known[0],x_known[1],Nx)
y_pred = []
for i,x0 in enumerate(x_pos):
	x1,x2 = x_known
	r01 = kernel(x0,x1)
	r02 = kernel(x0,x2)
	r12 = kernel(x1,x2)

	R_matrix = np.array([[1.0,r01,r02],
						 [r01,1.0,r12],
						 [r02,r12,1.0]])

	R12 = R_matrix[1:,0]
	R22 = R_matrix[1:,1:]
	R21 = R_matrix[0,1:]
	inv_R22 = np.linalg.inv(R22)

	E_y0 = 0.0+np.dot(R12,np.dot(inv_R22,y_known-0.0))
	y_pred.append(E_y0)

	fig = plt.figure(dpi=200,figsize=(4,3))
	ax1 = fig.add_subplot(111)

	ax1.plot(x_plot,-kernel(x_plot,x_known[0]),'r')
	ax1.plot(x_plot,kernel(x_plot,x_known[1]),'g')
	ax1.plot(x_known[0],y_known[0],'rD')
	ax1.plot(x_known[1],y_known[1],'gD')
	ax1.set_xlim(x_plot[0],x_plot[-1])

	ax1.plot(x0,E_y0,'bo')
	ax1.plot(x_pos[:i+1],y_pred,'b')

	ax1.text(-0.15,0.8,r'$\mathbf{{R}} = \begin{{bmatrix}} 1 & {0:.2} & {1:.2} \\ {0:.2} & 1 & {2:.2} \\ {1:.2} & {2:.2} & 1 \end{{bmatrix}}$'.format(r01,r02,r12))

	fig.tight_layout()
	fig.savefig(f"./Images/kernel_{i:02d}.png")
