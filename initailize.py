# %% [code]
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma

# Assuming the JumpGP_LD module is already imported
from JumpGP_code_py.JumpGP_LD import JumpGP_LD

# -------------------------------
# 1. Sample training and test data
# -------------------------------
np.random.seed(42)  # Ensure reproducibility

# Data dimensions
N = 100   # Number of training points
D = 2     # Input dimension (chosen as 2 for visualization)
T = 20    # Number of test points

# --- Generate training data ---
X_train = np.random.randn(N, D)
Y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.1 * np.random.randn(N)

# --- Generate test data ---
X_test = np.random.randn(T, D)

# -------------------------------
# 2. Set hyperparameters (or sample from prior distributions)
# -------------------------------
# For the GP prior of A:
#   Kernel function: K_A(x,x') = sigma_a^2 * exp( -||x-x'||^2/(2*sigma_q^2) )
sigma_a = gamma(a=2, scale=1).rvs()  # Overall scale
Q = 2  # Latent variable dimension
sigma_q = gamma(a=2, scale=1).rvs(size=Q)  # Length scale for each latent dimension

print("Sampled A-GP hyperparameters:")
print("  sigma_a =", sigma_a)
print("  sigma_q =", sigma_q)

# For the local Jump GP, we use fixed hyperparameters (modifiable if needed)
local_gp_params = {
    'm': 0.0,       # GP mean (mean of local function f)
    'sigma_f': 1.0, # Signal variance
    'l': 1.0        # Length scale (reserved)
}

# -------------------------------
# 3. Sample A matrix from GP prior
# -------------------------------
def compute_K_A(X, sigma_a, sigma_q_value):
    """
    Compute the GP mapping kernel matrix (RBF kernel) for a given input X:
        K(i,j) = sigma_a^2 * exp( -||x_i - x_j||^2/(2*sigma_q_value^2) )
    """
    T = X.shape[0]
    X1 = np.expand_dims(X, axis=1)  # Shape (T,1,D)
    X2 = np.expand_dims(X, axis=0)  # Shape (1,T,D)
    dists = np.sum((X1 - X2)**2, axis=2)
    return sigma_a**2 * np.exp(-dists / (2 * sigma_q_value**2))

# Sample A for each test point: for each test point, A[t] is a (Q x D) matrix
A = np.zeros((T, Q, D))
for q in range(Q):
    K_q = compute_K_A(X_test, sigma_a, sigma_q[q])
    jitter = 1e-6 * np.eye(T)
    K_q += jitter
    for d in range(D):
        A[:, q, d] = multivariate_normal.rvs(mean=np.zeros(T), cov=K_q)

# -------------------------------
# 4. Construct neighborhood for each test point: select M nearest training points
# -------------------------------
def get_neighborhoods(X_test, X_train, Y_train, M):
    """
    For each test point, select its M nearest neighbors from the training data.
    Returns a list where each element is a dictionary containing:
        'X_neighbors': Shape (M, D)
        'y_neighbors': Shape (M,)
        'indices': Indices of nearest neighbors
    """
    neighborhoods = []
    T = X_test.shape[0]
    for t in range(T):
        test_point = X_test[t]
        dists = np.sum((X_train - test_point) ** 2, axis=1)
        indices = np.argsort(dists)[:M]
        X_neighbors = X_train[indices]
        y_neighbors = Y_train[indices]
        neighborhoods.append({
            'X_neighbors': X_neighbors,
            'y_neighbors': y_neighbors,
            'indices': indices
        })
    return neighborhoods

M = 20
neighborhoods = get_neighborhoods(X_test, X_train, Y_train, M)

# -------------------------------
# 5. Define function to transform input using A_t
# -------------------------------
def transform(A_t, x):
    """
    Given A_t (shape (Q, D)) for a test point and an input x (shape (D,)),
    return the transformed feature vector (shape (Q,)).
    Uses a simple linear mapping: A_t @ x.
    """
    return np.dot(A_t, x)

# -------------------------------
# 6. Define Jump GP kernel function and sample f_t for all points in the neighborhood
# -------------------------------
def compute_jumpGP_kernel(zeta, logtheta):
    """
    Compute the Jump GP kernel matrix K given transformed inputs zeta (shape (M, Q)) 
    and hyperparameters logtheta (length Q+2), following:
        K(i,j) = s_f^2 * exp(-0.5 * sum_{d=1}^{Q} ((zeta[i,d] - zeta[j,d])^2 / ell_d^2))
    with:
      - ell_d = exp(logtheta[d]), for d=0,...,Q-1
      - s_f = exp(logtheta[Q])
      - Noise std sigma_n = exp(logtheta[Q+1])
    """
    M, Q = zeta.shape
    ell = np.exp(logtheta[:Q])
    s_f = np.exp(logtheta[Q])
    sigma_n = np.exp(logtheta[Q+1])
    
    # Normalize each dimension
    zeta_scaled = zeta / ell  # (M, Q)
    # Compute squared Euclidean distances
    diff = zeta_scaled[:, None, :] - zeta_scaled[None, :, :]  # (M, M, Q)
    dists_sq = np.sum(diff**2, axis=2)  # (M, M)
    
    # Compute kernel matrix
    K = s_f**2 * np.exp(-0.5 * dists_sq)
    # Add noise term
    K += sigma_n**2 * np.eye(M)
    return K

# -------------------------------
# 7. Compute JumpGP model parameters and sample f_t for neighborhood points
# -------------------------------
jump_gp_results = []

for t in range(T):
    A_t = A[t]
    
    neigh = neighborhoods[t]
    X_neigh = neigh['X_neighbors']
    y_neigh = neigh['y_neighbors']
    
    zeta_t = np.array([transform(A_t, X_neigh[i]) for i in range(M)])
    x_t_test = transform(A_t, X_test[t])
    
    mu_t, sig2_t, model, h = JumpGP_LD(zeta_t,
                                       y_neigh.reshape(-1, 1),
                                       x_t_test.reshape(1, -1),
                                       'CEM',
                                       True)
    
    mean_f = np.full(M, model['ms'])
    K_jump = compute_jumpGP_kernel(zeta_t, model['logtheta'])
    
    f_t = multivariate_normal.rvs(mean=mean_f, cov=K_jump)
    
    jump_gp_results.append({
        'test_index': t,
        'x_test': X_test[t],
        'A_t': A_t,
        'zeta_t': zeta_t,
        'x_t_test': x_t_test,
        'y_neigh': y_neigh,
        'jumpGP_model': model,
        'K_jump': K_jump,
        'f_t': f_t
    })

# -------------------------------
# 8. Visualization for a selected test point
# -------------------------------
selected = jump_gp_results[0]
test_index = selected['test_index']

plt.figure(figsize=(12, 5))
plt.suptitle(f"Test point {test_index} and its neighborhood")

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, label='Training points')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='^', s=100, label='Test points')
plt.scatter(X_test[test_index, 0], X_test[test_index, 1], c='red', marker='^', s=150)
neigh_indices = neighborhoods[test_index]['indices']
plt.scatter(X_train[neigh_indices, 0], X_train[neigh_indices, 1], edgecolors='black', facecolors='none', s=150, linewidths=2, label=f'{M} Nearest neighbors')
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)

plt.show()
