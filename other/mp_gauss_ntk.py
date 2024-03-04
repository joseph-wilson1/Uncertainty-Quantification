import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.optimize import newton
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.special as sp
from tqdm import tqdm
import pandas

# Utils

def illconditioned(dim,condition):
    sigma = np.sqrt(condition)
    halfish = np.floor(dim/2).astype(int)
    small = np.ones(halfish)
    diff = dim - halfish
    fake_small = np.pad(small,(0,diff),'constant')
    one = np.ones(dim)
    std = (1/sigma - sigma) * fake_small + sigma * one
    return std

def intspace(m0, m1, N):
    return [int(m) for m in np.floor(np.linspace(m0,m1,N))]

def intlogspace(m0, m1, N):
    return [int(m) for m in np.floor(np.logspace(m0,m1,N))]

def ndata_to_dims(alpha,ndatas,const):
    return [np.floor(const * dim ** alpha).astype(int) for dim in ndatas]

def bayes_entropy(gamma, lam, X, y, mean = 0, scale = 1):
    """Computes the negative log marginal likelihood for Bayesian linear regression
    at temperature gamma, and Gaussian distribution with variance lam. 
    
    - gamma, lam > 0
    - X is n x d data matrix of inputs
    - y is n x 1 data vector of labels
    """
    n, d = X.shape
    
    if d > n:
        jac = X @ X.T / d
        P = (gamma * lam) * np.eye(n) +  jac
        PinvY = np.linalg.solve(P, y)
        numer = -lam * np.dot(y, PinvY)/2
        _, denom = np.linalg.slogdet(P)
        denom = 0.5*denom
        const = 0.5 * n * np.log(lam / (2*np.pi))
        result = numer - denom + const
        
    else:
        jac = X.T @ X / d
        P = jac + lam * gamma * np.eye(d)
        PinvY = np.linalg.solve(P, X.T @ y)
        PinvY = np.dot(y, X @ PinvY)
        numer = -1/(2*gamma) * np.dot(y,y) + 1/(2*gamma*d) * PinvY
        _, denom = np.linalg.slogdet(P)
        denom = 0.5*denom
        const = 0.5 * d * np.log(lam) - 0.5 * n * np.log(2*np.pi) + 0.5*(d-n) * np.log(gamma)
        result = numer - denom + const
    
    return -result

def bayes_entropy_kernel(gamma,lam, K_gram, y,mean = 0,scale = 1):
    """Computes the negative log marginal likelihood for Bayesian kernel regression
    at temperature gamma, and Gaussian distribution with variance lam. 
    
    - gamma, lam > 0
    - K_gram is n x n data Gram matrix with K_gram^ij = k(X_i, X_j)
    - y is n x 1 data vector of labels
    """
    n, n1 = K_gram.shape
    assert n == n1 # Make sure gram matrix is square.
    P = (gamma * lam) * np.eye(n) + K_gram
    #PinvY = np.linalg.solve(P, y)  ## Why is this commented out? Shouldn't we use Y?
    #numer = -lam/2 * np.dot(y, PinvY)
    Pinv = np.linalg.inv(P)
    if np.inner(mean,mean) == 0.:
        numer = -scale * lam/2 * (np.trace(Pinv))
    else:
        numer = -lam/2 * (scale * np.trace(Pinv) + mean.T @ Pinv @ mean)
    _, denom = np.linalg.slogdet(P)
    denom = 0.5*denom
    const = 0.5 * n * np.log(lam / (2*np.pi))
    result = numer - denom + const
    
    return -result

def bayes_entropy_kernel_rbf(gamma,lam,k,X,y,ntk_kernel,mean = 0,scale = 1, ntk_init_fn = None):
    """Computes the negative log marginal likelihood for Bayesian kernel regression
    at temperature gamma, and Gaussian distribution with variance lam. 
    
    - gamma, lam > 0
    - k is a scalar-valued non-negative function
    - X is n x d data matrix of inputs
    - y is n x 1 data vector of labels
    """
    n,d = X.shape
    if ntk_kernel:
        # _, params = ntk_init_fn(X.shape) #if eNTK
        # K_gram = k(X,X,params) # if eNTK
        K_gram = k(X,X) # if infinite NTK
    else:
        if k == 'linear' or k =='exp':
            K_gram = X@X.T/d
            if k == 'exp':
                K_gram = np.exp(K_gram)
        else:
            dists = squareform(pdist(X, 'sqeuclidean')) + np.eye(n) ## matrix of sq euclid distance between points. 
            K_gram = k(dists / d) ## this matrix (divided by d) passed through exp(-x).
            np.fill_diagonal(K_gram, 1) ## fill diagonal with 1, why should this occur??
    return bayes_entropy_kernel(gamma, lam, K_gram, y, mean,scale)

def optimal_lambda(c, gamma):
    lambda_ast = np.sqrt((c - 1)**2 + 4*c*gamma*gamma)
    lambda_ast += (c + 1) * gamma
    lambda_ast /= c*(1 - gamma*gamma)
    return lambda_ast


def limit_entropy(alpha,beta,gamma,lam,c):
    z = (beta + gamma*lam)/alpha
    if c < 1:
        T = (c-1-c*z+((c*z+c+1)**2-4*c)**0.5)/(2*z)
        D = np.log(1+T/c)-T/(c+T)-c*np.log(T/c)
        term1 = lam/2*((1-c)/(beta+gamma*lam))
        term2 = lam/2*1/alpha*T
        term3 = -1/2*np.log(lam/(2*np.pi*alpha))
        term4 = 1/2*D
        term5 = 1/2*(1-c)*np.log(z)
        return term1+term2+term3+term4+term5
    else:
        T = (1-c-c*z+((c*z+c+1)**2-4*c)**0.5)/(2*z)
        D = c*np.log(1+T/c)-c*T/(c+T)-np.log(T)
        term1 = lam/(2*alpha)*T
        term2 = -1/2*np.log(lam/(2*np.pi*alpha))
        term3 = D/2
        return term1+term2+term3
    
def opt_lambda(alpha,beta,gamma,c):
    guess = alpha*((c+1)*gamma+((c-1)**2+4*c*gamma**2)**0.5)
    guess /= c*(1-gamma**2)
    obj = lambda x: limit_entropy(alpha,beta,gamma,x,c)
    sol = minimize(obj, guess)
    return sol.x[0]

def optimize_lambdas_cs(alpha,beta,gamma,dims,ndatas):
    return [ opt_lambda(alpha,beta,gamma,dims[idx]/ndatas[idx]) for idx in np.arange(len(dims))]

def optimize_lambdas(alpha,beta,gamma,dims,ndata):
    return [ opt_lambda(alpha,beta,gamma,dim/ndata) for dim in dims]

def entropies_by_dims(ndata, dims, lambdas, 
                  gamma=0.1, iterations=100, coeffs = None, rbf_kernel = None, ntk_kernel = None, ntk_init_fn = None, condition_number = None,scale = 1):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    entropies = np.zeros(len(dims)) # array of zeros of size (number of dimensions)
    entropies_ci = np.zeros(len(dims)) # array of zeros of size (number of dimensions)

    ## Welfords online algorithm
    for idx,dim in enumerate(dims): # iterate through the dimensions
        entropy = 0
        M2 = 0
        for idy in range(iterations): # iterate through the iteration numbers
            X = np.random.randn(ndata,dim) # matrix of random variables xi, size ndata x dim.
            if condition_number: # Always none
                stds = illconditioned(dim,condition_number)
                X = X * stds 
            mean = X @ coeffs[:dim,idx] # I think this is the mean function?
#            if rbf_kernel == None or rbf_kernel == 'exp' or rbf_kernel == 'linear':
#                mean = X @ coeffs[:dim,idx]
#            else:
#                mean = np.exp(-(X @ coeffs[:dim,idx])**2)
            y = np.sqrt(scale) * np.random.randn(ndata) + mean # Matrix of random variables yi, size ndata x 1. Can be scaled and shifted.
            if rbf_kernel is None:
                Fn = bayes_entropy(gamma,lambdas[idx],X,y,mean,scale)
            else: # rbf_kernel is passed to this method, decided in method.
                Fn = bayes_entropy_kernel_rbf(gamma,lambdas[idx],rbf_kernel,X,y,ntk_kernel, mean,scale, ntk_init_fn)
            # Update running average and variance
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci

def entropies_by_gamma(ndata, gammas, lambdas, 
                  dim=None, iterations=100, coeffs = None, rbf_kernel = None):
    if dim is None:
        dim = 2*ndata
    if coeffs is None:
        coeffs = np.zeros((dim,len(gammas)))
    entropies = np.zeros(len(gammas))
    entropies_ci = np.zeros(len(gammas))
    for idx,gamma in enumerate(gammas):
        entropy = 0
        M2 = 0
        for idy in range(iterations):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:,idx]
            y = np.random.randn(ndata) + mean
            if rbf_kernel is None:
                Fn = bayes_entropy(gamma,lambdas[idx],X,y)
            else:
                Fn = bayes_entropy_kernel_rbf(gamma,lambdas[idx],rbf_kernel,X,y)
            # Update running average and variance
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci

def entropies_by_data(ndatas, dims, lambdas, 
                  gamma=0.1, iterations=100, coeffs = None, rbf_kernel = None):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    entropies = np.zeros(len(dims))
    entropies_ci = np.zeros(len(dims))
    for idx,data in enumerate(ndatas):
        entropy = 0
        M2 = 0
        for idy in range(iterations):
            X = np.random.randn(data,dims[idx])
            mean = X @ coeffs[:dims[idx],idx]
            y = np.random.randn(data) + mean
            if rbf_kernel is None:
                Fn = bayes_entropy(gamma,lambdas[idx],X,y)
            else:
                Fn = bayes_entropy_kernel_rbf(gamma,lambdas[idx],rbf_kernel,X,y)
            # Update running average and variance
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci

##################

### POSTERIOR LOSS CODE START

##################

def posterior_predictive(gamma,lam,X,Y):
    """Returns the mean and covariance of the posterior
    predictive distribution"""
    n, d = X.shape
    P = X @ X.T / d + lam * gamma * np.eye(n)
    Pinv = np.linalg.inv(P)
    f = lambda x: x.T @ X.T @ Pinv @ Y / d
    Sigma = lambda x: x.T @ (np.eye(d) - X.T @ Pinv @ X / d ) @ x/ d
    return f, Sigma

def ppL2(gamma, lam, X, Y):
    """Estimating the expected posterior predictive L2 loss
    with x, y ~ N(0,1)."""
    n, d = X.shape
    P = X @ X.T / d + lam * gamma * np.eye(n)
    Pinv = np.linalg.inv(P)
    f_sq = np.sum((X.T @ Pinv @ Y / d)**2) + 1
    Sig = np.trace(np.eye(d) - X.T @ Pinv @ X / d) / d
    return f_sq + Sig / lam

def ppL2_optlam(gamma, X, Y,kernel = None):
    """Estimating the expected posterior predictive L2 loss
    with x, y ~ N(0,1)."""
    guess = -np.log(np.random.rand())/100
    if kernel == 'gauss':
        obj = lambda x: ppL2_kernel(gamma,x,lambda x: np.exp(-x),X,Y)
    else:
        obj = lambda x: ppL2(gamma,x,X,Y)
    sol = newton(obj, guess)#, method = 'Newton-CG', jac = '2-point')
    print(sol.x, sol.success)
    return sol.fun[0]

def ppL2_kernel(gamma, lam,k, X, Y,test = False):
    """Estimating the expected posterior predictive L2 loss
    with x, y ~ N(0,1)."""
    n,d = X.shape
    if k == 'linear':
        K_gram = X@X.T/d
    else:
        Xnew = np.random.randn(1,d)
        Xhat = np.append(X,Xnew,axis = 0)
        dists = squareform(pdist(Xhat, 'sqeuclidean')) + np.eye(n+1)
        K_g = k(dists / d)
        np.fill_diagonal(K_g, 1)
        k_x = K_g[-1,:-1]
        K_gram = K_g[:-1,:-1]
    P = K_gram + lam * gamma * np.eye(n)
    Pinv = np.linalg.inv(P)
    Q = k_x@Pinv
    f_sq = Q@Q.T + 1
    Sig = 1 - np.inner(k_x, Q)
    return f_sq + Sig / lam

def ppnll(gamma, lam, X, Y):
    """Estimating the expected posterior predictive negative
    log-likelihood with x, y ~ N(0,1)."""
    n, d = X.shape
    P = X @ X.T / d + lam * gamma * np.eye(n)
    Pinv = np.linalg.inv(P)
    f_sq = np.sum((X.T @ Pinv @ Y / d)**2) + 1
    Sig = np.trace(np.eye(d) - X.T @ Pinv @ X / d) / d
    return f_sq/(2*gamma) + Sig/(2*lam*gamma) + 0.5*np.log(2*np.pi*gamma)

def ppnll_kernel(gamma, lam, X, Y):
    """Estimating the expected posterior predictive negative
    log-likelihood with x, y ~ N(0,1)."""
    n, d = X.shape
    Xnew = np.random.randn(1,d)
    Xhat = np.append(X,Xnew,axis = 0)
    dists = squareform(pdist(Xhat, 'sqeuclidean')) + np.eye(n+1)
    K_g = np.exp(-dists / d)
    np.fill_diagonal(K_g, 1)
    k_x = K_g[-1,:-1]
    K_gram = K_g[:-1,:-1]
    P = K_gram + lam * gamma * np.eye(n)
    Pinv = np.linalg.inv(P)
    Q = k_x@Pinv
    f_sq = np.trace(np.outer(Q,Q)) + 1
    Sig = 1 - np.inner(k_x, Q)
    return f_sq/(2*gamma) + Sig/(2*lam*gamma) + 0.5*np.log(2*np.pi*gamma)

def ppnll_optlam(gamma, X, Y,kernel = False):
    """Estimating the expected posterior predictive L2 loss
    with x, y ~ N(0,1)."""
    guess = -np.log(np.random.rand())
    if kernel == 'gauss':
        obj = lambda x: ppnll_kernel(gamma,x,X,Y)
    else:    
        obj = lambda x: ppnll(gamma,x,X,Y)
    sol = minimize(obj, guess)
    return sol.fun
    
def ppnll_optgam(mu, X, Y):
    """Estimating the expected optimal posterior predictive
    negative log-likelihood with x, y ~ N(0,1)."""
    n, d = X.shape
    P = X @ X.T / d + mu * np.eye(n)
    Pinv = np.linalg.inv(P)
    f_sq = np.sum((X.T @ Pinv @ Y / d)**2) + 1
    Sig = np.trace(np.eye(d) - X.T @ Pinv @ X / d) / d
    return 0.5*(1 + np.log(2*np.pi*f_sq)) + Sig/(2*mu)

def ppnll_opt(X, Y):
    """Estimating the expected optimal posterior predictive
    negative log-likelihood with x, y ~ N(0,1)."""
    guess = -np.log(np.random.rand())
    obj = lambda x: ppnll_optgam(x, X, Y)
    sol = minimize(obj, guess)
    return sol.fun

def ppL2_by_dims(ndata, dims, gamma, lam = None, 
                  iterations=10, coeffs = None, rbf_kernel = None):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    losses = np.zeros(len(dims))
    losses_ci = np.zeros(len(dims))
    for idx,dim in enumerate(dims):
        loss = 0
        M2 = 0
        for idy in range(iterations):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:dim,idx]
            Y = np.random.randn(ndata) + mean
            if rbf_kernel is None:
                if lam is None:
                    L = ppL2_optlam(gamma,X,Y)
                else:
                    L = ppL2(gamma,lam, X,Y)
            else:
                L = ppL2_kernel(gamma,lam,rbf_kernel,X,Y)
            # Update running average and variance
            delta = L - loss
            loss += delta / (idy + 1)
            delta2 = L - loss
            M2 += delta * delta2
        losses[idx] = loss
        losses_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return losses, losses_ci

def ppL2_by_dims_vec(ndata, dims, gamma, lam = None, 
                  iterations=10, coeffs = None, rbf_kernel = None):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    losses = np.zeros(len(dims))
    losses_ci = np.zeros(len(dims))
    for idx,dim in enumerate(dims):
        loss = 0
        M2 = 0
        for idy in range(iterations):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:dim,idx]
            Y = np.random.randn(ndata) + mean
            if rbf_kernel is None:
                if lam is None:
                    L = ppL2_optlam(gamma,X,Y)
                else:
                    L = ppL2(gamma,lam[idx],X,Y)
            else:
                L = ppL2_kernel(gamma,lam[idx],rbf_kernel,X,Y,test = True)
            # Update running average and variance
            delta = L - loss
            loss += delta / (idy + 1)
            delta2 = L - loss
            M2 += delta * delta2
        losses[idx] = loss
        losses_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return losses, losses_ci

def ppnll_by_dims(ndata, dims, gamma = None, lam = None,
                  iterations=10, coeffs = None, rbf_kernel = None):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    entropies = np.zeros(len(dims))
    entropies_ci = np.zeros(len(dims))
    for idx,dim in enumerate(dims):
        entropy = 0
        M2 = 0
        for idy in range(iterations):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:dim,idx]
            Y = np.random.randn(ndata) + mean
            if rbf_kernel is None:
                if lam is None:
                    if gamma is None:
                        Fn = ppnll_opt(X, Y)
                    else:
                        Fn = ppnll_optlam(gamma,X,Y)
                elif gamma is None:
                    Fn = ppnll_optgam(lam,X,Y)
                else:
                    Fn = ppnll(gamma,lam,X,Y)
            else:
                Fn = bayes_entropy_kernel_rbf(gamma,lam[idx],rbf_kernel,X,Y)
            # Update running average and variance
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci

##################

### POSTERIOR LOSS CODE END

##################

def plot_entropies_log(groups, xdata, entropies, entropies_ci, clabel=None,ticks = None, param = '$\gamma$',xl = 'Dimension', yl = 'Mean Bayes Entropy', scale = 1, xs = 'log', ys = 'log'):
    v_cmap = mpc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=plt.cm.viridis.name, a=0, b=0.8),
        plt.cm.viridis(np.linspace(0, 0.8, 256)))
    norm = mpc.LogNorm(vmin=min(groups),vmax=max(groups))
    sm = plt.cm.ScalarMappable(cmap=v_cmap,
                               norm=norm)
    plt.figure(dpi=300,facecolor='white',figsize=[scale * 4, scale * 3])
    if ticks == None:
        xticks = [min(groups)]
        xticks += groups[::2]
        xticks.append(max(groups))
    else:
        xticks = ticks
    groups_norm = norm(np.array(groups))
    colors = sm.get_cmap()(groups_norm)
    
    def no_scientific(x,pos):
        return str(x)

    for idx, val in enumerate(groups):
        plt.plot(xdata, entropies[idx], label = "%s" % val, 
                 lw = 1, color=colors[idx])
        plt.fill_between(xdata,
                         entropies[idx]-entropies_ci[idx],
                         entropies[idx]+entropies_ci[idx],
                         alpha=0.25, label='_nolegend_', 
                         color=colors[idx])

    plt.ylabel(yl)
    plt.xscale(xs)
    cbar = plt.colorbar(sm, format=ticker.FuncFormatter(no_scientific),
                ticks=xticks, pad = -.025)
    cbar.ax.tick_params(labelsize=6) 
    cbar.ax.set_title(param,fontsize = 6)
    if clabel is not None:
        cbar.ax.set_title(clabel)
    plt.xlabel(xl)
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    plt.yscale(ys)

def matern(nu, sqdist, var = 1, rho = 1):
    return var * ( 2 ** (1 - nu)) / sp.gamma(nu) * (np.sqrt(2 * nu * sqdist) / rho) ** nu * sp.kv( nu, np.sqrt(2 * nu * sqdist) / rho)

def grad_matern(nu, sqdist, var = 1, rho = 1):
    first = var * nu * ( np.sqrt(2 * nu / sqdist)/ rho ) * ( 2 ** (- nu)) / sp.gamma(nu) * (np.sqrt(2 * nu * sqdist) / rho)** ( nu - 1 ) * sp.kv( nu, np.sqrt(2 * nu * sqdist) / rho)
    second =  var * ( np.sqrt(2 * nu / sqdist)/ rho ) *  2 ** (- nu ) / sp.gamma(nu) * (np.sqrt(2 * nu * sqdist) / rho)** nu 
    third = - 0.5 * (sp.kv( nu - 1, np.sqrt(2 * nu * sqdist) / rho) + sp.kv( nu + 1, np.sqrt(2 * nu * sqdist) / rho))
    return first + second * third


def entropies_by_dims_data(ndata, dims, lambdas, X_dat, Y_dat,
                  gamma=0.1, iterations=100, coeffs = None, rbf_kernel = None):
    total_n, total_d = X_dat.shape
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    entropies = np.zeros(len(dims))
    entropies_ci = np.zeros(len(dims))
    for idx,dim in enumerate(tqdm(dims)):
        entropy = 0
        M2 = 0
        for idy in range(iterations):
            indices_dat = np.random.choice(total_n, ndata, replace=False)
            #indices_dim = np.random.choice(total_d, dim, replace=False)
            X = X_dat[indices_dat, :]
            X = X[:, :dim]
            Y = Y_dat[indices_dat]
            if rbf_kernel is None:
                Fn = bayes_entropy(gamma,lambdas[idx],X,Y)
            else:
                Fn = bayes_entropy_kernel_rbf(gamma,lambdas[idx],rbf_kernel,X,Y)
            # Update running average and variance
            delta = Fn - entropy
            entropy += delta / (idy + 1)
            delta2 = Fn - entropy
            M2 += delta * delta2
        entropies[idx] = entropy
        entropies_ci[idx] = 1.96 * (M2 / (iterations - 1) / iterations)**0.5
    return entropies, entropies_ci


def ppL2_by_dims_data(ndata, dims, gamma, X, Y, lam = None, 
                  iterations=10, coeffs = None, rbf_kernel = None):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    losses = np.zeros(len(dims))
    losses_ci = np.zeros(len(dims))
    for idx,dim in enumerate(tqdm(dims)):
        loss = 0
        M2 = 0
        temp_iters = iterations
        if dim < ndata and lam is None:
            temp_iters *= 10
        for idy in range(temp_iters):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:dim,idx]
            Y = np.random.randn(ndata) + mean
            if rbf_kernel is None:
                if lam is None:
                    L = ppL2_optlam(gamma,X,Y)
                else:
                    L = ppL2(gamma,lam[idx],X,Y)
            else:
                L = bayes_entropy_kernel_rbf(gamma,lam[idx],rbf_kernel,X,Y)
            # Update running average and variance
            delta = L - loss
            loss += delta / (idy + 1)
            delta2 = L - loss
            M2 += delta * delta2
        losses[idx] = loss
        losses_ci[idx] = 1.96 * (M2 / (temp_iters - 1) / temp_iters)**0.5
    return losses, losses_ci

def ppL2_by_dims_special(ndata, dims, gamma, lam = None, 
                  iterations=10, coeffs = None, kernel = False):
    if coeffs is None:
        coeffs = np.zeros((dims[-1],len(dims)))
    losses = np.zeros(len(dims))
    losses_ci = np.zeros(len(dims))
    for idx,dim in enumerate(dims):
        loss = 0
        M2 = 0
        temp_iters = iterations
        if dim < ndata and lam is None:
            temp_iters *= 10
        for idy in tqdm(range(temp_iters)):
            X = np.random.randn(ndata,dim)
            mean = X @ coeffs[:dim,idx]
            Y = np.random.randn(ndata) + mean
            if lam is None:
                L = ppL2_optlam(gamma,X,Y,kernel)
            else:
                L = ppL2(gamma,lam,X,Y)
            # Update running average and variance
            delta = L - loss
            loss += delta / (idy + 1)
            delta2 = L - loss
            M2 += delta * delta2
        losses[idx] = loss
        losses_ci[idx] = 1.96 * (M2 / (temp_iters - 1) / temp_iters)**0.5
    return losses, losses_ci