import autograd.numpy as np
from autograd import grad
from autograd.scipy.linalg import cho_solve, cho_factor
from numpy.linalg import slogdet, inv, solve
from scipy.stats import multivariate_normal
from scipy.special import expit  # sigmoid

# ------------------------------
# Helper functions
# ------------------------------

def sigmoid(x):
    return expit(x)

def compute_K_A(X, sigma_a, sigma_q_value):
    # Compute the RBF kernel for GP mapping over test inputs X (shape: [T, D])
    T = X.shape[0]
    X1 = np.expand_dims(X, axis=1)  # (T,1,D)
    X2 = np.expand_dims(X, axis=0)  # (1,T,D)
    dists = np.sum((X1 - X2)**2, axis=2)  # (T,T)
    return sigma_a**2 * np.exp(-dists / (2 * sigma_q_value**2))

def log_normal_pdf(x, mean, var):
    return -0.5 * np.log(2*np.pi*var) - 0.5 * ((x - mean)**2 / var)

def log_multivariate_normal_pdf(x, mean, cov):
    diff = x - mean
    sign, logdet = slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-6*np.eye(len(x))
        sign, logdet = slogdet(cov)
    return -0.5 * (diff.T @ inv(cov) @ diff + logdet + len(x)*np.log(2*np.pi))

def compute_jumpGP_kernel(zeta, logtheta):
    # zeta: (M, Q)
    # logtheta: vector of length Q+2; first Q: log(ell_d), next: log(s_f), last: log(sigma_n)
    M, Q = zeta.shape
    ell = np.exp(logtheta[:Q])
    s_f = np.exp(logtheta[Q])
    sigma_n = np.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell  # shape (M, Q)
    diff = zeta_scaled[:, None, :] - zeta_scaled[None, :, :]  # (M, M, Q)
    dists_sq = np.sum(diff**2, axis=2)  # (M, M)
    K = s_f**2 * np.exp(-0.5 * dists_sq)
    K += (sigma_n**2) * np.eye(M)
    return K

def transform(A_t, x):
    # A_t: (Q, D); x: (D,)
    return np.dot(A_t, x)

# ------------------------------
# Log prior for A_t given A_{-t} using GP conditional (for each latent dimension & coordinate)
# ------------------------------

def log_prior_A_t(A_t_candidate, t, A, X_test, sigma_a, sigma_q):
    T, Q, D = A.shape
    logp = 0.0
    for q in range(Q):
        K = compute_K_A(X_test, sigma_a, sigma_q[q])  # (T, T)
        for d in range(D):
            A_qd = A[:, q, d]  # values for all test points for latent dimension q, coordinate d
            idx = [i for i in range(T) if i != t]
            A_minus = A_qd[idx]  # shape: (T-1,)
            k_tt = K[t, t]
            k_t_other = K[t, idx]  # shape: (T-1,)
            K_minus = K[np.ix_(idx, idx)]  # (T-1, T-1)
            sol = solve(K_minus, A_minus)
            cond_mean = k_t_other.dot(sol)
            cond_var = k_tt - k_t_other.dot(solve(K_minus, k_t_other.T))
            cond_var = np.maximum(cond_var, 1e-6)
            a_val = A_t_candidate[q, d]
            logp += log_normal_pdf(a_val, cond_mean, cond_var)
    return logp

# ------------------------------
# Log likelihood for A_t (local likelihood for f_t and r)
# ------------------------------

def log_likelihood_A_t(A_t_candidate, t, neighborhoods, jump_gp_results):
    # Retrieve neighborhood data and current JumpGP parameters for test point t
    neigh = neighborhoods[t]
    X_neighbors = neigh['X_neighbors']   # shape: (M, D)
    result = jump_gp_results[t]
    # Instead of "z_t", use "r" from the model (flatten to 1D)
    model = result['jumpGP_model']
    r = model['r'].flatten()  # shape: (M,)
    f_t = result['f_t']       # vector of length M
    w = model['w']            # vector of length (1+Q)
    ms = model['ms']          # scalar (GP mean)
    logtheta = model['logtheta']  # vector of length Q+2
    M = X_neighbors.shape[0]
    
    # Compute transformed representations for each neighbor
    zeta = np.array([transform(A_t_candidate, X_neighbors[j]) for j in range(M)])  # (M, Q)
    
    # Logistic likelihood for r (binary indicators)
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + np.dot(w[1:], zeta[j])
        p = sigmoid(g)
        p = np.clip(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r += np.log(p)
        else:
            loglik_r += np.log(1 - p)
    
    # Multivariate normal likelihood for f_t with kernel computed from zeta and logtheta
    K_jump = compute_jumpGP_kernel(zeta, logtheta)  # (M, M)
    mean_vec = ms * np.ones(M)
    loglik_f = log_multivariate_normal_pdf(f_t, mean_vec, K_jump)
    
    return loglik_r + loglik_f

# ------------------------------
# Combined log probability for A_t
# ------------------------------

def log_prob_A_t(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
    lp_prior = log_prior_A_t(A_t_candidate, t, A, X_test, sigma_a, sigma_q)
    lp_likelihood = log_likelihood_A_t(A_t_candidate, t, neighborhoods, jump_gp_results)
    return lp_prior + lp_likelihood

# ------------------------------
# HMC update for A_t
# ------------------------------

def hmc_sample_A_t(current_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size, num_steps):
    current = current_A_t.flatten()
    shape = current_A_t.shape  # (Q, D)
    
    def log_prob_flat(A_t_flat):
        A_t_candidate = A_t_flat.reshape(shape)
        return log_prob_A_t(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
    
    grad_log_prob = grad(log_prob_flat)
    
    current_momentum = np.random.randn(*current.shape)
    momentum = current_momentum.copy()
    
    grad_current = grad_log_prob(current)
    momentum = momentum + 0.5 * step_size * grad_current
    
    new_position = current.copy()
    for i in range(num_steps):
        new_position = new_position + step_size * momentum
        grad_new = grad_log_prob(new_position)
        if i != num_steps - 1:
            momentum = momentum + step_size * grad_new
    momentum = momentum + 0.5 * step_size * grad_new
    momentum = -momentum
    
    current_U = -log_prob_flat(current)
    current_K = np.sum(current_momentum**2) / 2.0
    proposed_U = -log_prob_flat(new_position)
    proposed_K = np.sum(momentum**2) / 2.0
    
    log_accept_ratio = current_U - proposed_U + current_K - proposed_K
    if np.log(np.random.rand()) < log_accept_ratio:
        return new_position.reshape(shape), True
    else:
        return current_A_t, False

# ------------------------------
# Gibbs update for A using HMC for each A_t
# ------------------------------

def gibbs_update_A_HMC(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size=0.01, num_steps=10):
    T, Q, D = A.shape
    A_new = A.copy()
    accept_count = 0
    for t in range(T):
        current_A_t = A[t].copy()
        new_A_t, accepted = hmc_sample_A_t(current_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size, num_steps)
        if accepted:
            A_new[t] = new_A_t
            A[t] = new_A_t  # update global A for subsequent conditional updates
            accept_count += 1
    print("A HMC update acceptance rate:", accept_count / T)
    return A_new

# ------------------------------
# Example usage:
# (Assume that the following global variables have been initialized:)
#   A: array of shape (T, Q, D)
#   X_test: array of shape (T, D)
#   neighborhoods: list of T dictionaries with neighborhood data
#   jump_gp_results: list of T dictionaries with keys 'f_t' and 'jumpGP_model' (which includes 'w', 'ms', 'logtheta', and 'r')
#   sigma_a: scalar hyperparameter for A's GP
#   sigma_q: array of length Q for A's GP
# ------------------------------

# Perform one Gibbs sweep to update A using HMC
A = gibbs_update_A_HMC(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size=0.01, num_steps=10)
