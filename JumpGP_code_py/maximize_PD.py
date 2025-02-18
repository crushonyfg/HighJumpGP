# % ***************************************************************************************
# %
# % This function implements Classification EM Algorithm for Jump GP
# % It is internally used by JumpGP_LD and JumpGP_QD
# %
# % Inputs:
# %       x: training inputs, y: training output
# %       xt: test inputs
# %       px: evaluations of boundary function basis psi(x) at training inputs (x)
# %       pxt: evaluations of boundary function basis psi(x) at test inputs (xt)
# %       w: parameters of boundary function
# %       logtheta: parameters of covariance function
# %       cv: covariance model
# %       bVerbose: whether printing out detailed progression information
# %
# % Outputs:
# %       model: fitted JumpGP model
# %
# % Copyright ©2022 reserved to Chiwoo Park (cpark5@fsu.edu) 
# % ***************************************************************************************


import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky
from scipy.stats import norm

from .lik.loglikelihood import loglikelihood
from .calculate_gx import calculate_gx
from .cov.covSum import covSum
from .cov.covSEard import covSEard
from .cov.covNoise import covNoise
from .lik.loglikelihood import loglikelihood
# from .lik.loglikelihood import negloglikelihood

def compute_kernel_matrix(x1, x2):
    """
    Compute the kernel matrix between two sets of points (X_star and X_test),
    with each feature scaled by a corresponding value in ell.
    
    X_star: (n, d) matrix of n points with d features
    X_test: (k, d) matrix of k points with d features
    ell: (d,) vector of scaling factors for each feature
    
    Returns:
    K: (n, k) kernel matrix
    """
    
    # Compute pairwise squared Euclidean distances between X_star and X_test, with scaling by ell
    dist_squared = np.sqrt(np.maximum(np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T), 0))
    # dist_squared = np.sqrt(np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T))
    # print(dist_squared.shape)
    return dist_squared
def cal1(logtheta, x1, x2, type=1):
    s2, ell, sf2 =np.exp(2*logtheta[0]), np.exp(logtheta[1:-1]), np.exp(2*logtheta[-1])
    B = sf2 * np.exp(-compute_kernel_matrix( x1 @ np.diag(1./ell), x2 @ np.diag(1./ell)) / 2)
    # print(x1.shape, x2.shape, B.shape)
    if type==1:
        return B
    else: 
        return B+np.dot(s2,np.eye(x1.shape[0]))

def maximize_PD(x, y, xt, px, pxt, w, logtheta, cv, bVerbose=False):
    nw = np.linalg.norm(w)
    w = w / nw
    nIter = 100

    phi_xt = np.dot(np.hstack(([1], pxt[0])), w) #phi_xt shape (1,Nt)
    w = w * np.sign(phi_xt)
    gx, phi_x = calculate_gx(px, w)
    
    r = gx >= 0
    # if r.sum() < 1:
    #     gx = -gx
    #     w = -w
    #     r = ~r
    # r = np.squeeze(r)
    
    if bVerbose:
        print("Current is maximize_PD func, Initial boundary visualization")

    err_flag = False
    for k in range(10):
        r1 = r.flatten()
        ms = np.mean(y[r1]).item()
        try:
            logtheta = minimize(loglikelihood, logtheta, args=(covSum, [covSEard, covNoise], x[r1,:], y[r1] - ms),method='L-BFGS-B', options={'maxiter': nIter}).x
            # logtheta = minimize(loglikelihood, logtheta, args=(covSum, [covSEard, covNoise], x[r1,:], y[r1] - ms),method='L-BFGS-B').x
        except:
            print(f"maximize_PD func, we fail at iteration {k}")
            err_flag = True

        # 20250110
        K = covSum([covSEard, covNoise], logtheta, x[r1,:])
        _, Kt = covSum([covSEard, covNoise], logtheta, x[r1,:], x)
        K += 1e-8 * np.eye(K.shape[0])
        L = cholesky(K, lower=True)
        Ly = np.linalg.solve(L, y[r1] - ms)
        LK = np.linalg.solve(L, Kt)
        # K = cal1(logtheta, x[r1,:], x[r1,:], type=2)
        # Kt = cal1(logtheta, x, x[r1,:])
        # L = cholesky(K, lower=True)
        # Ly = np.linalg.solve(L, y[r1] - ms)
        # LK = np.linalg.solve(L, Kt.T)

        # fs = np.dot(LK.T, Ly) + ms
        fs = LK.T @ Ly + ms
        
        sigma = np.sqrt(np.mean((y[r1] - fs[r1]) ** 2))
        if sigma==0: 
            sigma = 1e-6
        
        like = norm.pdf(y, loc=fs, scale=sigma)
        RR = norm.pdf(2.5 * sigma, loc=0, scale=sigma)
        prior_z = 1 / (1 + np.exp(-0.05 * nw * gx))
        prior_z = prior_z.reshape(-1,1)
        pos_z = prior_z * like / (prior_z * like + (1 - prior_z) * RR)
        
        r = pos_z >= 0.5

        # prevent all r are False(need to be modified)
        # if not r.any():  # r 中没有任何 True 值
        #     # 找到 pos_z 中最大值的位置
        #     max_index = np.argmax(pos_z)
        #     # 将该位置的 r 值设为 True
        #     r[max_index] = True
        # r1 = r.flatten()
        
        def wfun(wo):
            eps = 1e-10  # 设定一个极小的数，防止log(0)
            phi_w = np.dot(phi_x, wo)
            sigmoid_phi_w = 1 / (1 + np.exp(-phi_w))
            return -np.sum(r.T * np.log(np.clip(sigmoid_phi_w, eps, 1)) + 
                        (1 - r).T * np.log(np.clip(1 - sigmoid_phi_w, eps, 1)))

            # phi_w = np.dot(phi_x, wo)
            # return -np.sum(r.T * np.log(1 / (1 + np.exp(-phi_w))) + (1 - r).T * np.log(1 - 1 / (1 + np.exp(-phi_w))) )

        w_flattened = w.ravel()
        from scipy.optimize import LinearConstraint
        lc = LinearConstraint(-np.array([1, *pxt.flatten()]), ub=0)

        # constraints = {'type': 'ineq', 'fun': lambda wo: np.dot(np.array([1, *pxt.flatten()]), wo)}  # -[1, pxt]
        w_new = minimize(wfun, w_flattened, constraints=lc, options={'disp': False}).x
        # constraints = {'type': 'ineq', 'fun': lambda wo: -np.array([1, *wo])}  # -[1, pxt]
        # w_new = minimize(wfun, w_flattened, constraints=constraints, options={'disp': False}).x
        # w_new = minimize(wfun, w_flattened, bounds=[(-np.inf, np.inf)], method='SLSQP').x
        # w_new = w_new.reshape(w.shape)
        conv_crit = np.linalg.norm(w_new / np.linalg.norm(w_new) - w / np.linalg.norm(w))
        if conv_crit < 1e-3:
            break
        
        w = w_new
        nw = np.linalg.norm(w)
        w = w / nw
        gx, phi_x = calculate_gx(px, w)
        
        if err_flag:
            break
    
    r1 = r.flatten()
    # 20250110
    K = covSum([covSEard, covNoise], logtheta, x[r1,:])
    Ktt, Kt = covSum([covSEard, covNoise], logtheta, x[r1,:], xt)
    K += 1e-8 * np.eye(K.shape[0])
    L = cholesky(K, lower=True)
    Ly = np.linalg.solve(L, y[r1] - ms)
    LK = np.linalg.solve(L, Kt)
    # K = cal1(logtheta, x[r1,:], x[r1,:], type=2)
    # # Kt = cal1(logtheta, x, x[r1,:])
    # Ktt = cal1(logtheta, xt, x[r1,:])
    # K0 = cal1(logtheta, xt, xt)
    # L = cholesky(K, lower=True)
    # Ly = np.linalg.solve(L, y[r1] - ms)
    # LK = np.linalg.solve(L, Ktt.T)

    # fs = np.dot(LK.T, Ly) + ms
    fs = LK.T @ Ly + ms
    
    model = {
        'x': x,
        'y': y,
        'RR': RR,
        'fs': fs,
        'sigma': sigma,
        'xt': xt,
        'px': px,
        'pxt': pxt,
        'nll': loglikelihood(logtheta, covSum, [covSEard, covNoise], x[r1,:], y[r1]) / np.sum(r1),
        'r': r,
        'gamma': pos_z,
        'nw': nw,
        'w': w,
        'ms': ms,
        'logtheta': logtheta,
        'cv': [covSum, [covSEard, covNoise]],
        'mu_t': fs,
        'sig2_t': Ktt - np.sum(LK.T**2, axis=1)
        # 'sig2_t': K0 - np.sum(LK.T**2, axis=1)
    }
    
    return model
