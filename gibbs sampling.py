import autograd.numpy as np
from autograd import grad
from autograd.scipy.linalg import cho_factor, cho_solve
from scipy.special import expit  # sigmoid function from SciPy
import numpy.random as npr

# -------------------------------
# 1. PROBABILITY CALCULATION FUNCTIONS
# -------------------------------

def sigmoid(x):
    """Sigmoid (logistic) function."""
    return 1.0 / (1.0 + np.exp(-x))

def log_mvn_pdf(x, mean, cov):
    """
    Log density of a multivariate normal.
    
    x, mean: 1d arrays of length d.
    cov: (d,d) covariance matrix.
    """
    d = x.shape[0]
    L = np.linalg.cholesky(cov)
    diff = x - mean
    sol = np.linalg.solve(L, diff)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (np.dot(sol, sol) + logdet + d * np.log(2 * np.pi))

def compute_K_A(X, sigma_a, sigma_q):
    """
    Compute the covariance matrix K_A for a GP mapping,
    using a squared–exponential kernel.
    
    X: array of shape (T, D) (here the test inputs)
    sigma_a: overall scale
    sigma_q: length–scale for the qth latent dimension
    """
    # Compute pairwise squared Euclidean distances.
    dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    return sigma_a**2 * np.exp(-dists / (2 * sigma_q**2))

def log_p_A(A, X, sigma_a, sigma_q_vec):
    """
    Prior log–density for the latent mapping A.
    
    A: array of shape (T, Q, D)
       (each test point t has a latent mapping matrix A[t] of shape (Q,D))
    X: array of test inputs of shape (T, D)
    sigma_a: scale parameter
    sigma_q_vec: length Q array (each entry is sigma_q for that latent row)
    """
    T, Q, D = A.shape
    logp = 0.0
    for q in range(Q):
        # Compute covariance over test points for the q-th row.
        K = compute_K_A(X, sigma_a, sigma_q_vec[q])
        for d in range(D):
            vec = A[:, q, d]
            logp += log_mvn_pdf(vec, np.zeros(T), K)
    return logp

def transform(A_t, x):
    """
    Given a latent mapping A_t (shape (Q, D)) for a test point and a neighbor’s input x (length D),
    return the transformed feature vector in the latent space (length Q).
    """
    return np.dot(A_t, x)  # (Q,)

def log_p_z_i(z, A_t, x_i, w_t):
    """
    Log–density for a single binary indicator z (0 or 1) for neighbor i.
    
    A_t: latent mapping for the test point (Q, D)
    x_i: neighbor’s input (D,)
    w_t: parameter vector (of length Q+1). The first entry multiplies the bias.
    """
    zeta = transform(A_t, x_i)  # (Q,)
    features = np.concatenate(([1.0], zeta))  # (Q+1,)
    g = np.dot(w_t, features)
    p = sigmoid(g)
    # Add a small constant to avoid log(0).
    if z == 1:
        return np.log(p + 1e-10)
    else:
        return np.log(1 - p + 1e-10)

def log_p_z(z, A_t, X_neighbors, w_t):
    """
    Log–density for all the latent indicators for one test point.
    
    z: binary vector of length M.
    X_neighbors: array of shape (M, D) for the M neighbors.
    w_t: parameter vector (length Q+1).
    """
    logp = 0.0
    M = X_neighbors.shape[0]
    for i in range(M):
        logp += log_p_z_i(z[i], A_t, X_neighbors[i], w_t)
    return logp

def compute_C(zeta, theta):
    """
    Compute the covariance matrix for the latent function f.
    
    zeta: array of shape (M, Q), where each row is the transformed neighbor input.
    theta: dictionary with kernel hyper–parameters:
           theta['sigma_f'] and theta['l'] (length–scale).
    """
    M = zeta.shape[0]
    sigma_f = theta['sigma_f']
    l = theta['l']
    # Compute squared distances between rows.
    dists = np.sum((zeta[:, None, :] - zeta[None, :, :])**2, axis=2)
    C = sigma_f**2 * np.exp(-dists / (2 * l**2))
    # Add a little jitter for numerical stability.
    return C + 1e-6 * np.eye(M)

def log_p_f(f, m, C):
    """
    Log–density of the latent function f.
    
    f: array of length M.
    m: mean (a scalar; the GP mean is m * ones)
    C: covariance matrix (M, M).
    """
    mean = m * np.ones(f.shape)
    return log_mvn_pdf(f, mean, C)

def log_p_y(y, f, z, sigma2, U):
    """
    Log–likelihood for the observed responses y in the neighborhood.
    
    For each neighbor:
      if z[i]==1, then y[i] ~ N(f[i], sigma2);
      else, the likelihood is a constant U.
    
    y, f, z: arrays of length M.
    sigma2: noise variance.
    U: constant likelihood for outliers.
    """
    logp = 0.0
    for i in range(len(y)):
        if z[i] == 1:
            logp += -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((y[i] - f[i])**2) / sigma2
        else:
            logp += np.log(U + 1e-10)
    return logp

def log_joint_t(f, z, A_t, y, X_neighbors, w_t, m, theta, sigma2, U):
    """
    Joint log–density for one test point’s neighborhood.
    
    f: latent function values (M,)
    z: binary indicators (M,)
    A_t: latent mapping for the test point (Q, D)
    y: responses (M,)
    X_neighbors: neighbor inputs (M, D)
    w_t: parameter vector for z (length Q+1)
    m: GP mean (scalar)
    theta: dictionary for GP kernel hyper–parameters (for f)
    sigma2: noise variance for y
    U: constant outlier likelihood
    """
    M = X_neighbors.shape[0]
    # Compute the transformed features for all neighbors:
    zeta = np.array([transform(A_t, X_neighbors[i]) for i in range(M)])  # shape (M, Q)
    C = compute_C(zeta, theta)
    lp_z = log_p_z(z, A_t, X_neighbors, w_t)
    lp_f = log_p_f(f, m, C)
    lp_y = log_p_y(y, f, z, sigma2, U)
    return lp_z + lp_f + lp_y

def joint_log_prob(A, f_all, z_all, X_test, test_data, hyperparams):
    """
    Overall joint log–density for the entire model.
    
    A: array of shape (T, Q, D) for the latent mappings (for T test points)
    f_all: list of length T; each element is an array (M,) of latent function values for that test point’s neighborhood.
    z_all: list of length T; each element is a binary vector (M,) for that test point.
    X_test: array of shape (T, D) of test–point inputs (used to compute p(A|X)).
    test_data: list of dictionaries (length T) with keys: 
               'X_neighbors' (M×D), 'y' (M,), 'w', 'm', 'theta', 'sigma2', 'U'
    hyperparams: dictionary with keys for A’s prior, for example: 'sigma_a' and 'sigma_q' (an array of length Q)
    """
    T = X_test.shape[0]
    lp = 0.0
    lp += log_p_A(A, X_test, hyperparams['sigma_a'], hyperparams['sigma_q'])
    for t in range(T):
        A_t = A[t]           # (Q, D)
        f_t = f_all[t]       # (M,)
        z_t = z_all[t]       # (M,)
        data_t = test_data[t]
        lp += log_joint_t(f_t, z_t, A_t, data_t['y'], data_t['X_neighbors'],
                          data_t['w'], data_t['m'], data_t['theta'],
                          data_t['sigma2'], data_t['U'])
    return lp

# -------------------------------
# 2. HMC SAMPLER
# -------------------------------

def hmc_sample(current, log_prob, grad_log_prob, step_size, num_steps):
    """
    A basic HMC sampler.
    
    current: current value (a NumPy array, can be multidimensional)
    log_prob: function that returns the log–density at a given point
    grad_log_prob: function returning the gradient of log_prob
    step_size: leapfrog step size
    num_steps: number of leapfrog steps
    """
    x = current.copy()
    momentum = npr.randn(*x.shape)
    current_momentum = momentum.copy()
    
    # Half-step for momentum
    grad_x = grad_log_prob(x)
    momentum = momentum + 0.5 * step_size * grad_x
    
    # Leapfrog integration
    for i in range(num_steps):
        x = x + step_size * momentum
        grad_x = grad_log_prob(x)
        if i != num_steps - 1:
            momentum = momentum + step_size * grad_x
    # Final half-step
    momentum = momentum + 0.5 * step_size * grad_x
    # Negate momentum for reversibility.
    momentum = -momentum
    
    # Compute energies
    current_U = -log_prob(current)
    current_K = np.sum(current_momentum**2) / 2.0
    proposed_U = -log_prob(x)
    proposed_K = np.sum(momentum**2) / 2.0
    
    # Metropolis–Hastings acceptance step
    if npr.rand() < np.exp(current_U - proposed_U + current_K - proposed_K):
        return x, True
    else:
        return current, False

# -------------------------------
# 3. GIBBS SAMPLING (with HMC for continuous updates)
# -------------------------------

def gibbs_sampler(initial_A, initial_f_all, initial_z_all, X_test, test_data, hyperparams,
                  num_iterations, hmc_params):
    """
    Gibbs sampling that alternates between updating the discrete z and
    sampling the continuous f and A (using HMC).
    
    initial_A: array (T, Q, D) for latent mapping matrices.
    initial_f_all: list (length T) with each element an array (M,) for latent f.
    initial_z_all: list (length T) with each element a binary vector (M,).
    X_test: array (T, D) of test inputs.
    test_data: list (length T) of dictionaries (each with keys 'X_neighbors', 'y', 'w', 'm', 'theta', 'sigma2', 'U')
    hyperparams: dictionary for the A–prior (e.g. 'sigma_a' and 'sigma_q')
    num_iterations: number of Gibbs iterations.
    hmc_params: dictionary with keys 'step_size_f', 'num_steps_f', 'step_size_A', 'num_steps_A'
    """
    T = X_test.shape[0]
    samples_A = []
    samples_f_all = []
    samples_z_all = []
    
    A = initial_A.copy()                # shape (T, Q, D)
    f_all = [f.copy() for f in initial_f_all]  # list of T arrays, each of shape (M,)
    z_all = [z.copy() for z in initial_z_all]  # list of T arrays, each of length M
    
    for it in range(num_iterations):
        # --- Update discrete latent indicators z ---
        for t in range(T):
            data_t = test_data[t]
            X_neighbors = data_t['X_neighbors']  # shape (M, D)
            y_t = data_t['y']                    # (M,)
            w_t = data_t['w']                    # vector of length Q+1
            sigma2 = data_t['sigma2']
            U = data_t['U']
            A_t = A[t]                           # (Q, D)
            # For each neighbor update z
            for i in range(len(y_t)):
                # Compute the probability for z_i = 1:
                g = np.dot(w_t, np.concatenate(([1.0], transform(A_t, X_neighbors[i]))))
                p_z1 = sigmoid(g)
                # Likelihood if z_i==1:
                like1 = np.exp(-0.5 * np.log(2 * np.pi * sigma2) -
                               0.5 * ((y_t[i] - f_all[t][i])**2) / sigma2)
                # Likelihood if z_i==0 is U (a constant)
                like0 = U
                # Full conditional:
                prob = (p_z1 * like1) / (p_z1 * like1 + (1 - p_z1) * like0 + 1e-10)
                z_all[t][i] = 1 if npr.rand() < prob else 0

        # --- Update latent function values f (continuous) using HMC ---
        for t in range(T):
            data_t = test_data[t]
            M = data_t['X_neighbors'].shape[0]
            # Define a function for the f–log–density at test point t:
            def log_prob_f(f):
                # Compute the transformed features for each neighbor.
                zeta = np.array([transform(A[t], X) for X in data_t['X_neighbors']])
                C = compute_C(zeta, data_t['theta'])
                lp_f = log_p_f(f, data_t['m'], C)
                lp_y = log_p_y(data_t['y'], f, z_all[t], data_t['sigma2'], data_t['U'])
                return lp_f + lp_y
            grad_log_prob_f = grad(log_prob_f)
            current_f = f_all[t]
            new_f, accepted = hmc_sample(current_f, log_prob_f, grad_log_prob_f,
                                         hmc_params['step_size_f'], hmc_params['num_steps_f'])
            f_all[t] = new_f

        # --- Update latent mapping matrices A (continuous) using HMC ---
        for t in range(T):
            data_t = test_data[t]
            # A[t] is a matrix (Q, D). We flatten it.
            current_A_t = A[t]
            def log_prob_A(A_flat):
                A_mat = A_flat.reshape(current_A_t.shape)
                # Only the z–likelihood and the GP prior for f depend on A_t (via the transformed neighbors).
                lp = log_p_z(z_all[t], A_mat, data_t['X_neighbors'], data_t['w'])
                zeta = np.array([transform(A_mat, X) for X in data_t['X_neighbors']])
                C = compute_C(zeta, data_t['theta'])
                lp += log_p_f(f_all[t], data_t['m'], C)
                return lp
            grad_log_prob_A = grad(log_prob_A)
            A_flat = current_A_t.flatten()
            new_A_flat, accepted = hmc_sample(A_flat, log_prob_A, grad_log_prob_A,
                                              hmc_params['step_size_A'], hmc_params['num_steps_A'])
            A[t] = new_A_flat.reshape(current_A_t.shape)
        
        samples_A.append(A.copy())
        samples_f_all.append([f.copy() for f in f_all])
        samples_z_all.append([z.copy() for z in z_all])
        
        print("Iteration", it, "completed.")
    
    return samples_A, samples_f_all, samples_z_all

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == '__main__':
    # (Here we create some dummy data to show how the code might be used.
    #  In practice, you would replace these with your actual data.)
    
    npr.seed(42)
    
    # Suppose we have T test points (e.g., 3)
    T = 3
    D = 5      # dimension of the input space
    Q = 2      # latent dimension (each A_t is (Q, D))
    M = 10     # number of neighbors per test point
    
    # Create dummy test inputs X_test (T, D)
    X_test = npr.randn(T, D)
    
    # For each test point, create dummy neighborhood data
    test_data = []
    for t in range(T):
        X_neighbors = npr.randn(M, D)
        y = npr.randn(M)
        # w_t is (Q+1,)
        w = npr.randn(Q + 1)
        m = 0.0
        theta = {'sigma_f': 1.0, 'l': 1.0}
        sigma2 = 0.5
        U = 0.1  # constant likelihood for outliers
        test_data.append({'X_neighbors': X_neighbors, 'y': y, 'w': w,
                          'm': m, 'theta': theta, 'sigma2': sigma2, 'U': U})
    
    # Hyper–parameters for p(A|X)
    hyperparams = {'sigma_a': 1.0, 'sigma_q': np.array([1.0, 1.0])}  # length Q=2
    
    # Initial values: for each test point, A is (Q, D)
    initial_A = npr.randn(T, Q, D)
    # For each test point, f (length M) and binary indicators z (length M)
    initial_f_all = [npr.randn(M) for _ in range(T)]
    initial_z_all = [npr.randint(0, 2, size=M) for _ in range(T)]
    
    # HMC parameters for the continuous updates
    hmc_params = {'step_size_f': 0.01,
                  'num_steps_f': 10,
                  'step_size_A': 0.005,
                  'num_steps_A': 10}
    
    # Run Gibbs sampling for a modest number of iterations:
    num_iterations = 100
    samples_A, samples_f_all, samples_z_all = gibbs_sampler(initial_A, initial_f_all, initial_z_all,
                                                            X_test, test_data, hyperparams,
                                                            num_iterations, hmc_params)
    
    # (You can now use the samples_A, samples_f_all, samples_z_all as approximate samples from the posterior.)
