import torch
import math

# Set default dtype and device if needed
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")  # 或 "cuda"（若可用）

# ------------------------------
# Helper functions (using torch)
# ------------------------------

def sigmoid(x):
    return torch.sigmoid(x)

def compute_K_A_torch(X, sigma_a, sigma_q_value):
    # X: tensor of shape (T, D)
    T = X.shape[0]
    X1 = X.unsqueeze(1)  # (T, 1, D)
    X2 = X.unsqueeze(0)  # (1, T, D)
    dists = torch.sum((X1 - X2)**2, dim=2)  # (T, T)
    return sigma_a**2 * torch.exp(-dists / (2 * sigma_q_value**2))

def log_normal_pdf_torch(x, mean, var):
    return -0.5 * torch.log(2 * math.pi * var) - 0.5 * ((x - mean)**2 / var)

def log_multivariate_normal_pdf_torch(x, mean, cov):
    # cov: tensor of shape (n, n)
    cov = cov.to(torch.float64)
    diff = x - mean
    sign, logdet = torch.slogdet(cov)
    if sign.item() <= 0:
        cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
        sign, logdet = torch.slogdet(cov)
    return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))

def compute_jumpGP_kernel_torch(zeta, logtheta):
    # zeta: (M, Q); logtheta: tensor of shape (Q+2,)
    # 强制转换 logtheta 为 float64（若尚未转换）
    logtheta = logtheta.to(torch.float64)
    # zeta 也确保为 float64
    zeta = zeta.to(torch.float64)
    M, Q = zeta.shape
    ell = torch.exp(logtheta[:Q])
    s_f = torch.exp(logtheta[Q])
    sigma_n = torch.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell  # 广播，每列除以对应的 ell
    diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)  # (M, M, Q)
    dists_sq = torch.sum(diff**2, dim=2)  # (M, M)
    K = s_f**2 * torch.exp(-0.5 * dists_sq)
    K = K + (sigma_n**2) * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
    return K

def transform_torch(A_t, x):
    # A_t: tensor of shape (Q, D); x: tensor of shape (D,)
    return A_t @ x

# ------------------------------
# Log prior for A_t given A_{-t} (GP conditional) 
# ------------------------------

def log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q):
    # A: tensor of shape (T, Q, D), X_test: (T, D)
    T, Q, D = A.shape
    logp = 0.0
    for q in range(Q):
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])  # (T, T)
        for d in range(D):
            A_qd = A[:, q, d]  # (T,)
            # 取除 t 外的索引
            idx = [i for i in range(T) if i != t]
            idx_tensor = torch.tensor(idx, dtype=torch.long, device=A.device)
            A_minus = A_qd[idx_tensor]  # (T-1,)
            k_tt = K[t, t]
            k_t_other = K[t, idx_tensor]  # (T-1,)
            K_minus = K[idx_tensor][:, idx_tensor]  # (T-1, T-1)
            sol = torch.linalg.solve(K_minus, A_minus)
            cond_mean = (k_t_other @ sol).item()  # 标量
            cond_var = k_tt - (k_t_other @ torch.linalg.solve(K_minus, k_t_other))
            cond_var = torch.clamp(cond_var, min=1e-6)
            a_val = A_t_candidate[q, d]
            logp = logp + log_normal_pdf_torch(a_val, cond_mean, cond_var)
    return logp

# ------------------------------
# Log likelihood for A_t (local likelihood for f_t and r)
# ------------------------------

def log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results):
    # neighborhoods: list of dictionaries，每个字典中 key "X_neighbors" 为 tensor (M, D)
    # jump_gp_results: list of dictionaries，每个字典中 key "jumpGP_model" 包含 'r', 'w', 'ms', 'logtheta'
    neigh = neighborhoods[t]
    X_neighbors = neigh["X_neighbors"]  # tensor (M, D)
    result = jump_gp_results[t]
    model = result["jumpGP_model"]
    r = model["r"].flatten()  # tensor shape (M,); r 为布尔型，此处转换为 float
    r = r.to(torch.float64)
    f_t = result["f_t"]  # tensor (M,)
    w = model["w"]       # tensor of shape (1+Q,)
    ms = model["ms"]     # scalar
    logtheta = model["logtheta"]  # tensor of shape (Q+2,)
    M = X_neighbors.shape[0]
    # 计算每个邻域点经 A_t_candidate 变换后的表示 zeta (M, Q)
    zeta = torch.stack([transform_torch(A_t_candidate, X_neighbors[j]) for j in range(M)], dim=0)
    # Logistic likelihood for r
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        p = sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r = loglik_r + torch.log(p)
        else:
            loglik_r = loglik_r + torch.log(1 - p)
    # f_t 的多元正态似然：均值向量 ms，协方差由 zeta 和 logtheta 得到
    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)  # (M, M)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    return loglik_r + loglik_f

# ------------------------------
# Combined log probability for A_t
# ------------------------------

def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
    lp_prior = log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q)
    lp_likelihood = log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results)
    return lp_prior + lp_likelihood

# ------------------------------
# HMC update for A_t using torch.autograd
# ------------------------------

def hmc_sample_A_t_torch(current_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size, num_steps):
    # current_A_t: tensor (Q, D)
    current = current_A_t.flatten().clone().detach().requires_grad_(True)
    shape = current_A_t.shape  # (Q, D)
    
    def log_prob_flat(A_t_flat):
        A_t_candidate = A_t_flat.reshape(shape)
        return log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
    
    # 利用 torch.autograd.grad 计算梯度
    def grad_log_prob(x):
        return torch.autograd.grad(log_prob_flat(x), x, create_graph=False)[0]
    
    current_momentum = torch.randn_like(current)
    momentum = current_momentum.clone()
    
    grad_current = grad_log_prob(current)
    momentum = momentum + 0.5 * step_size * grad_current
    
    new_position = current.clone()
    for i in range(num_steps):
        new_position = new_position + step_size * momentum
        grad_new = grad_log_prob(new_position)
        if i != num_steps - 1:
            momentum = momentum + step_size * grad_new
    momentum = momentum + 0.5 * step_size * grad_new
    momentum = -momentum  # Reverse momentum for symmetry
    
    current_U = -log_prob_flat(current)
    current_K = torch.sum(current_momentum**2) / 2.0
    proposed_U = -log_prob_flat(new_position)
    proposed_K = torch.sum(momentum**2) / 2.0
    
    log_accept_ratio = current_U - proposed_U + current_K - proposed_K
    if torch.log(torch.rand(1, device=current.device)) < log_accept_ratio:
        return new_position.reshape(shape).detach(), True
    else:
        return current_A_t, False

# ------------------------------
# Gibbs update for A using HMC for each A_t (using torch)
# ------------------------------

def gibbs_update_A_HMC_torch(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size=0.01, num_steps=10):
    # A: tensor of shape (T, Q, D)
    T, Q, D = A.shape
    A_new = A.clone()
    accept_count = 0
    for t in range(T):
        current_A_t = A[t].clone()
        new_A_t, accepted = hmc_sample_A_t_torch(current_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size, num_steps)
        if accepted:
            A_new[t] = new_A_t
            A[t] = new_A_t  # 更新全局 A，便于后续条件更新
            accept_count += 1
    print("A HMC update acceptance rate:", accept_count / T)
    return A_new

# ------------------------------
# Example usage:
# 假设以下全局变量已经初始化为 torch.Tensor：
#   A: tensor of shape (T, Q, D)
#   X_test: tensor of shape (T, D)
#   neighborhoods: list of T dictionaries，其中每个字典的 "X_neighbors" 为 tensor (M, D)
#   jump_gp_results: list of T dictionaries，其中 "jumpGP_model" 包含 'w', 'ms', 'logtheta', 'r'，且 "f_t" 为 tensor (M,)
#   sigma_a: scalar tensor
#   sigma_q: tensor of shape (Q,)
# ------------------------------

# 例如：
# A = torch.randn(T, Q, D, device=device)
# X_test = torch.randn(T, D, device=device)
# 其他变量请确保均为 torch.Tensor，并存储在相应的结构中。

# Perform one Gibbs sweep to update A using HMC
A = gibbs_update_A_HMC_torch(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q, step_size=0.01, num_steps=10)
