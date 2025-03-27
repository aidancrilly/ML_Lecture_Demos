"""  Demonstration of Bayesian optimisation using the simple implementation of GPs in our Gaussian Processes demo """
from GaussianProcessesdemo import *
from scipy.special import erf

def f_opt(x):
    """ Forrester function 
    
    This function is a simple one-dimensional test function.
    It is multimodal, with one global minimum, one local minimum and a zero-gradient inflection point.

    """
    return (6*x-2)**2*np.sin(12*x-4)

def LowerConfidenceBound(mu,sig,y_min,lamb=1.5):
    return mu-lamb*sig

def ExpectedImprovement(mu,sig,y_min,lamb=0.0):
    arg = -(mu-y_min-lamb)/sig
    norm = np.exp(-0.5*arg**2)/np.sqrt(2*np.pi)
    CDF  = 0.5*(1+erf(arg/np.sqrt(2)))
    EI = -(mu-y_min-lamb)*CDF+sig*norm
    return -EI

def TrainGP(X_known,Y_known):
    """
    
    Given X,Y pairs, regress a GP to this data

    Create a function which takes a new x position and returns mu,sig of the GP
    
    """
    # Fit maximum likelihood of free parameter, correlation length scale, from known data
    res = minimize(neg_log_likelihood,0.1,args=(X_known,Y_known))
    # Compute GP properties from maximum likelihood result
    theta_opt = res.x[0]
    corr_opt  = correlation_matrix(X_known,theta_opt)
    sigma_opt = np.sqrt(sigma2_MLE(corr_opt,Y_known))
    mu_sig_GP = lambda x_trial : conditional_distribution(x_trial,X_known,Y_known,sigma_opt,theta_opt)
    return mu_sig_GP

def CreateAcquisitionFunction(X,Y,afunc):
    """
    Create a closure such that we only need to pass the optimiser the trial point
    """
    GP = TrainGP(X,Y)
    y_min = np.amin(Y)
    def AcquisitionFunction(x_trial):
        mu,sig_mat = GP(np.atleast_1d(x_trial))
        sig = np.sqrt(np.diag(sig_mat))
        return afunc(mu,sig,y_min)
    return AcquisitionFunction,GP

np.random.seed(9)
N_start = 3
X = np.random.uniform(size=(N_start,))
Y = f_opt(X)

x_plot = np.linspace(0.0,1.0,200)
y_plot = f_opt(x_plot)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

N_BO_iters = 6
N_opt_restarts = 10
for iter in range(N_BO_iters):
    ax1.clear()
    ax2.clear()
    ax1.plot(x_plot,y_plot,'k',label='Truth')
    ax1.plot(X,Y,'bo',ls='')

    afunc,GP = CreateAcquisitionFunction(X,Y,ExpectedImprovement)
    mu,sig = GP(x_plot)

    ax1.plot(x_plot,mu,'b',label='GP mean')
    ax1.plot(x_plot,mu+np.sqrt(np.diag(sig)),'b--',label='GP confidence interval')
    ax1.plot(x_plot,mu-np.sqrt(np.diag(sig)),'b--')
    ax2.plot(x_plot,-afunc(x_plot),'g',label='Acq. func.')

    x_next = 0.0
    a_next = 1.0
    for iter_opt in range(N_opt_restarts):
        opt_res  = minimize(afunc,np.random.uniform(),method='L-BFGS-B',bounds=[(0.0,1.0)])
        a_res = opt_res.fun
        if(a_res < a_next):
            x_next,a_next = opt_res.x,a_res
            y_next = f_opt(x_next)
    ax1.plot(x_next,y_next,'ro')

    ax1.legend()
    ax1.set_xlim(0.0,1.0)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel('Acquisition function',c='g')
    ax2.yaxis.set_label_position("right")
    fig.savefig(f'./Images/BO_{str(iter).zfill(2)}.png')

    X = np.append(X,x_next)
    Y = np.append(Y,y_next)


plt.show()