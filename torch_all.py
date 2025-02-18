import torch
import math
import matplotlib.pyplot as plt
import numpy as np  # 用于转换显示
from scipy.stats import gamma
import torch
import pyro
import pyro.infer.mcmc as mcmc
import pyro.distributions as dist

from JumpGP_code_py.JumpGP_LD import JumpGP_LD

# =============================================================================
# 0. 定义 jumpgp_ld_wrapper，用于包装 JumpGP_LD 的输入输出
# =============================================================================
def jumpgp_ld_wrapper(zeta_t_torch, y_t_torch, xt_torch, mode, flag, device=torch.device("cpu")):
    """
    Wrapper for JumpGP_LD:
      - Converts input torch.Tensors (zeta: (M, Q), y: (M,1), xt: (Nt,d))
        to numpy arrays.
      - Calls JumpGP_LD.
      - Reconstructs the returned model dictionary, converting numeric items to torch.Tensors.
    
    Returns:
      mu_t: torch.Tensor (e.g. shape (1,1))
      sig2_t: torch.Tensor
      model: dict containing keys 'w', 'ms', 'logtheta', 'r' as torch.Tensors
      h: auxiliary output (保持不变)
    """
    zeta_np = zeta_t_torch.detach().cpu().numpy()
    y_np = y_t_torch.detach().cpu().numpy()
    xt_np = xt_torch.detach().cpu().numpy()
    
    # 调用原 JumpGP_LD 函数（不修改）
    mu_t_np, sig2_t_np, model_np, h = JumpGP_LD(zeta_np, y_np, xt_np, mode, flag)
    
    mu_t = torch.from_numpy(mu_t_np).to(device)
    sig2_t = torch.from_numpy(sig2_t_np).to(device)
    
    keys_to_keep = ['w', 'ms', 'logtheta', 'r']
    model_torch = {}
    for key in keys_to_keep:
        if key in model_np:
            val = model_np[key]
            if isinstance(val, np.ndarray):
                model_torch[key] = torch.from_numpy(val).to(device)
            elif isinstance(val, (int, float)):
                model_torch[key] = torch.tensor(val, dtype=torch.float64, device=device)
            else:
                model_torch[key] = val
    return mu_t, sig2_t, model_torch, h

# =============================================================================
# 1. Data Generation and A's GP Prior Sampling
# =============================================================================

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# 设置维度
N = 100      # 训练点数
D = 2        # 输入维度
T = 20       # 测试点数
M = 20       # 每个测试点的邻域大小
Q = 2        # 潜在空间维度

# 生成训练数据与测试数据（使用 torch）
X_train = torch.randn(N, D, device=device)
Y_train = torch.sin(X_train[:, 0]) + torch.cos(X_train[:, 1]) + 0.1 * torch.randn(N, device=device)
X_test = torch.randn(T, D, device=device)

# 采样 A 的 GP 超参数（使用 Gamma 分布）
gamma_dist = torch.distributions.Gamma(2.0, 1.0)
sigma_a = gamma_dist.sample().to(device)
sigma_q = gamma_dist.sample((Q,)).to(device)
print("Sampled A-GP hyperparameters:")
print(" sigma_a =", sigma_a.item())
print(" sigma_q =", sigma_q)

# 定义 RBF 核（GP 映射），torch 版本
def compute_K_A_torch(X, sigma_a, sigma_q_value):
    T = X.shape[0]
    X1 = X.unsqueeze(1)  # (T, 1, D)
    X2 = X.unsqueeze(0)  # (1, T, D)
    dists = torch.sum((X1 - X2)**2, dim=2)  # (T, T)
    return sigma_a**2 * torch.exp(-dists / (2 * sigma_q_value**2))

# 根据 GP 先验采样 A: 形状 (T, Q, D)
A = torch.zeros(T, Q, D, device=device)
for q in range(Q):
    K_q = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
    jitter = 1e-6 * torch.eye(T, device=device, dtype=K_q.dtype)
    K_q = K_q + jitter
    for d in range(D):
        mean_vec = torch.zeros(T, device=device)
        mvn = torch.distributions.MultivariateNormal(mean_vec, covariance_matrix=K_q)
        A[:, q, d] = mvn.sample()

# 可视化原始数据与 A 的均值
X_train_np = X_train.cpu().numpy()
X_test_np = X_test.cpu().numpy()
A_mean = A.mean(dim=1).cpu().numpy()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c='blue', alpha=0.7, label='Training X')
plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c='red', marker='^', s=100, label='Test X')
plt.title("Training and Test Inputs (Original Space)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.scatter(A_mean[:, 0], A_mean[:, 1], c='magenta', marker='s', s=100)
for t in range(T):
    plt.text(A_mean[t, 0] + 0.02, A_mean[t, 1] + 0.02, f"{t}", fontsize=9)
plt.title("Mean Latent Mapping A for Test Points")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# 2. Construct Neighborhoods for Each Test Point
# =============================================================================

# 使用 torch.cdist 计算距离并选取邻域
dists = torch.cdist(X_test, X_train)  # (T, N)
_, indices = torch.sort(dists, dim=1)
indices = indices[:, :M]  # (T, M)
neighborhoods = []
for t in range(T):
    idx = indices[t]
    X_neigh = X_train[idx]  # (M, D)
    y_neigh = Y_train[idx]  # (M,)
    neighborhoods.append({
        "X_neighbors": X_neigh,
        "y_neighbors": y_neigh,
        "indices": idx
    })

# =============================================================================
# 3. Jump GP: Transformation, Kernel, and f_t Sampling (using jumpgp_ld_wrapper)
# =============================================================================

def transform_torch(A_t, x):
    return A_t @ x

def compute_jumpGP_kernel_torch(zeta, logtheta):
    logtheta = logtheta.to(torch.float64)
    zeta = zeta.to(torch.float64)
    M, Q = zeta.shape
    ell = torch.exp(logtheta[:Q])
    s_f = torch.exp(logtheta[Q])
    sigma_n = torch.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell
    diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)
    dists_sq = torch.sum(diff**2, dim=2)
    K = s_f**2 * torch.exp(-0.5 * dists_sq)
    K = K + (sigma_n**2) * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
    return K

# 这里调用 jumpgp_ld_wrapper 对 JumpGP_LD 进行包装转换
jump_gp_results = []
for t in range(T):
    A_t = A[t]  # (Q, D)
    neigh = neighborhoods[t]
    X_neigh = neigh["X_neighbors"]  # (M, D)
    y_neigh = neigh["y_neighbors"]    # (M,)
    # 计算邻域中每个点经过 A_t 变换后的表示 zeta_t (M, Q)
    zeta_t = torch.stack([transform_torch(A_t, X_neigh[i]) for i in range(M)], dim=0)
    # 对测试点进行变换，得到 xt (形状 (1, D) -> (1, Q))
    x_t_test = transform_torch(A_t, X_test[t])
    # 调用包装器转换输入输出
    mu_t, sig2_t, model, _ = jumpgp_ld_wrapper(zeta_t, y_neigh.view(-1, 1), x_t_test.view(1, -1), mode="CEM", flag=True, device=device)
    mean_f = model["ms"] * torch.ones(M, device=A.device, dtype=A.dtype)
    K_jump = compute_jumpGP_kernel_torch(zeta_t, model["logtheta"])
    mvn = torch.distributions.MultivariateNormal(mean_f, covariance_matrix=K_jump)
    f_t = mvn.sample()
    jump_gp_results.append({
        "test_index": t,
        "x_test": X_test[t],
        "A_t": A_t,
        "zeta_t": zeta_t,
        "x_t_test": x_t_test,
        "y_neigh": y_neigh,
        "jumpGP_model": model,
        "K_jump": K_jump,
        "f_t": f_t
    })

# 可视化选定测试点及其邻域
selected = jump_gp_results[0]
t_sel = selected["test_index"]
plt.figure(figsize=(12, 5))
plt.suptitle(f"Test point {t_sel} and its neighborhood")
plt.subplot(1, 2, 1)
plt.scatter(X_train.cpu()[:, 0], X_train.cpu()[:, 1], c="blue", alpha=0.5, label="Training points")
plt.scatter(X_test.cpu()[:, 0], X_test.cpu()[:, 1], c="red", marker="^", s=100, label="Test points")
plt.scatter(X_test.cpu()[t_sel, 0], X_test.cpu()[t_sel, 1], c="red", marker="^", s=150)
neigh_idx = neighborhoods[t_sel]["indices"].cpu().numpy()
plt.scatter(X_train.cpu()[neigh_idx, 0], X_train.cpu()[neigh_idx, 1],
            edgecolors="black", facecolors="none", s=150, linewidths=2, label=f"{M} Nearest neighbors")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
zeta_t_np = selected["zeta_t"].cpu().numpy()
plt.scatter(zeta_t_np[:, 0], zeta_t_np[:, 1], c="green", label="Transformed neighbors")
x_t_test_np = selected["x_t_test"].cpu().numpy()
plt.scatter(x_t_test_np[0], x_t_test_np[1], c="magenta", marker="^", s=150, label="Transformed test point")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# 4. HMC Update for A using torch.autograd (不修改 JumpGP_LD)
# =============================================================================

def log_normal_pdf_torch(x, mean, var):
    return -0.5 * torch.log(2 * math.pi * var) - 0.5 * ((x - mean)**2 / var)

def log_multivariate_normal_pdf_torch(x, mean, cov):
    cov = cov.to(torch.float64)
    diff = x - mean
    sign, logdet = torch.slogdet(cov)
    if sign.item() <= 0:
        cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
        sign, logdet = torch.slogdet(cov)
    return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))

def compute_jumpGP_kernel_torch(zeta, logtheta):
    logtheta = logtheta.to(torch.float64)
    zeta = zeta.to(torch.float64)
    M, Q = zeta.shape
    ell = torch.exp(logtheta[:Q])
    s_f = torch.exp(logtheta[Q])
    sigma_n = torch.exp(logtheta[Q+1])
    zeta_scaled = zeta / ell
    diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)
    dists_sq = torch.sum(diff**2, dim=2)
    K = s_f**2 * torch.exp(-0.5 * dists_sq)
    K = K + (sigma_n**2) * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
    return K

def transform_torch(A_t, x):
    return A_t @ x

def log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q):
    T, Q, D = A.shape
    logp = 0.0
    for q in range(Q):
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        for d in range(D):
            A_qd = A[:, q, d]
            idx = [i for i in range(T) if i != t]
            idx_tensor = torch.tensor(idx, dtype=torch.long, device=A.device)
            A_minus = A_qd[idx_tensor]
            k_tt = K[t, t]
            k_t_other = K[t, idx_tensor]
            K_minus = K[idx_tensor][:, idx_tensor]
            sol = torch.linalg.solve(K_minus, A_minus)
            cond_mean = k_t_other @ sol
            cond_var = k_tt - k_t_other @ torch.linalg.solve(K_minus, k_t_other)
            cond_var = torch.clamp(cond_var, min=1e-6)
            a_val = A_t_candidate[q, d]
            logp = logp + log_normal_pdf_torch(a_val, cond_mean, cond_var)
    return logp

def log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results):
    neigh = neighborhoods[t]
    X_neighbors = neigh["X_neighbors"]
    result = jump_gp_results[t]
    model = result["jumpGP_model"]
    r = model["r"].flatten()
    r = r.to(torch.float64)
    f_t = result["f_t"]
    w = model["w"]
    ms = model["ms"]
    logtheta = model["logtheta"]
    M = X_neighbors.shape[0]
    zeta = torch.stack([transform_torch(A_t_candidate, X_neighbors[j]) for j in range(M)], dim=0)
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        p = torch.sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r = loglik_r + torch.log(p)
        else:
            loglik_r = loglik_r + torch.log(1 - p)
    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    return loglik_r + loglik_f

def log_prob_A_t_torch(A_t_candidate, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q):
    lp_prior = log_prior_A_t_torch(A_t_candidate, t, A, X_test, sigma_a, sigma_q)
    lp_likelihood = log_likelihood_A_t_torch(A_t_candidate, t, neighborhoods, jump_gp_results)
    return lp_prior + lp_likelihood

import torch
import pyro
from pyro.infer.mcmc import NUTS, MCMC

def sample_A_t_HMC(initial_A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
                   step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    利用 Pyro 的 NUTS（HMC 的变种，支持自动步长调节）对索引 t 对应的 A_t 进行采样更新。
    
    参数:
      initial_A_t: 用于初始化 HMC 采样的 A[t]（深拷贝，避免修改原数据）
      t: 当前更新的索引
      A: 当前完整的参数张量，包含已经更新的部分（保证 Gibbs 更新）
      X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q:
          模型中所需的其他变量（确保 log_prob_A_t_torch 可用且可微）
      step_size: NUTS 的初始步长（会在 warmup 阶段自动调整）
      num_steps: NUTS 的最大树深度（对应每次采样的步数）
      num_samples: 采样的样本数
      warmup_steps: 预热步数
    返回:
      new_A_t: 采样得到的新的 A[t]（深拷贝，防止后续修改）
    """
    def potential_fn(params):
        # 从参数字典中提取当前待更新的 A_t
        A_t = params["A_t"]
        # 计算负的 log 概率（保证 log_prob_A_t_torch 对 A_t 可微分）
        return -log_prob_A_t_torch(A_t, t, A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q)
    
    # 用传入的初始 A_t 作为初始值（使用 clone() 防止引用原始张量）
    init_params = {"A_t": initial_A_t.clone()}
    
    # 使用 NUTS 内核，自动调整步长。通过 step_size 和 max_tree_depth 控制采样过程。
    nuts_kernel = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=0.8,   # 目标接受率，可根据需要调整
        step_size=step_size,      # 初始步长
        max_tree_depth=num_steps  # 最大树深度，控制每次采样的步数
    )
    
    # 构造 MCMC 对象，传入初始参数（以字典形式），指定采样数量和预热步数
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    
    # 开始采样
    mcmc.run()
    samples = mcmc.get_samples()
    
    # 提取采样得到的第一个样本作为更新后的 A_t，并使用 clone() 确保数据独立
    new_A_t = samples["A_t"][0].clone()
    return new_A_t

def gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
                             step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    对参数 A 中的每个 t 进行 Gibbs 更新，每次更新时都使用最新的 A_new，从而保证 Gibbs 效果。
    
    参数:
      A: 原始参数张量，形状为 [T, Q, D]
      其他参数: 与模型相关的变量
      step_size, num_steps, num_samples, warmup_steps: 控制 NUTS 采样的参数
    返回:
      A_new: Gibbs 更新后的 A
    """
    T, Q, D = A.shape
    # 对 A 进行深拷贝，防止直接修改原始 A
    A_new = A.clone()
    
    # 对每个 t 进行依次更新，更新时传入当前最新的 A_new
    for t in range(T):
        # 当前 A_new[t] 作为初始值进行采样（使用 clone() 确保独立性）
        initial_A_t = A_new[t].clone()
        # 采样更新 A[t]，此处传入 A_new 保证使用最新的 Gibbs 更新结果
        new_A_t = sample_A_t_HMC(initial_A_t, t, A_new, X_test, neighborhoods, jump_gp_results,
                                 sigma_a, sigma_q, step_size, num_steps, num_samples, warmup_steps)
        # 将更新后的 A[t] 存入 A_new（clone() 确保赋值的是一个独立的张量）
        A_new[t] = new_A_t.clone()
    print("Pyro NUTS (HMC) update performed for all A_t.")
    return A_new

# =============================================================================
# 示例：使用 Pyro NUTS 更新 A
# =============================================================================

# 假设 A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q 均已定义，
# 且 A 的形状为 [T, Q, D]（例如 T 个 [2, 2] 的矩阵）
A = gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
                             step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50)


import torch
import math
import pyro
import pyro.infer.mcmc as mcmc
import pyro.distributions as dist

# 假设 A: tensor of shape (T, Q, D)
#         X_test: tensor of shape (T, D)
# 已经由之前代码生成
# 例如，前面的代码已经生成了 A, X_test, … 
# 这里直接使用这些变量

# 超参数先验参数（InvGamma 参数）
alpha_a = 2.0
beta_a = 1.0
alpha_q = 2.0
beta_q = 1.0

def compute_K_A_torch(X, sigma_a, sigma_q_value):
    # X: tensor of shape (T, D)
    T = X.shape[0]
    X1 = X.unsqueeze(1)
    X2 = X.unsqueeze(0)
    dists = torch.sum((X1 - X2)**2, dim=2)
    return sigma_a**2 * torch.exp(-dists / (2 * sigma_q_value**2))

def potential_fn_theta(theta, A, X_test):
    """
    theta: tensor of shape (Q+1,), where theta[0] = log(sigma_a), theta[1:] = log(sigma_q)
    A: tensor of shape (T, Q, D)
    X_test: tensor of shape (T, D)
    
    返回负的 log 后验（潜在 GP 超参数的目标函数，不含常数项）
    """
    T, Q, D = A.shape
    sigma_a = torch.exp(theta[0])
    sigma_q = torch.exp(theta[1:])  # 长度 Q
    
    log_lik = 0.0
    for q in range(Q):
        K = compute_K_A_torch(X_test, sigma_a, sigma_q[q])
        # 加上 jitter
        jitter = 1e-6 * torch.eye(T, device=X_test.device, dtype=X_test.dtype)
        K = K + jitter
        sign, logdet = torch.slogdet(K)
        if sign.item() <= 0:
            K = K + 1e-6 * torch.eye(T, device=X_test.device, dtype=X_test.dtype)
            sign, logdet = torch.slogdet(K)
        invK = torch.inverse(K)
        for d in range(D):
            vec = A[:, q, d]
            quad = vec @ (invK @ vec)
            log_lik = log_lik - 0.5 * (logdet + quad)
    # log prior for theta
    # 对于 sigma_a: log p(theta_a) = - (alpha_a+1)*theta_a - beta_a/exp(theta_a) + theta_a
    log_prior = - (alpha_a + 1) * theta[0] - beta_a / torch.exp(theta[0]) + theta[0]
    for q in range(Q):
        log_prior = log_prior - (alpha_q + 1) * theta[1+q] - beta_q / torch.exp(theta[1+q]) + theta[1+q]
    log_post = log_lik + log_prior
    return -log_post  # potential function

def sample_theta_HMC(A, X_test, initial_theta, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    利用 Pyro 的 NUTS 对 theta 进行采样
    initial_theta: tensor of shape (Q+1,)
    A: tensor of shape (T, Q, D)
    X_test: tensor of shape (T, D)
    返回更新后的 theta（tensor of shape (Q+1,)）
    """
    def potential_fn(params):
        theta = params["theta"]
        return potential_fn_theta(theta, A, X_test)
    
    init_params = {"theta": initial_theta.clone()}
    nuts_kernel = mcmc.NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                            step_size=step_size, max_tree_depth=num_steps)
    mcmc_run = mcmc.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    mcmc_run.run()
    samples = mcmc_run.get_samples()
    new_theta = samples["theta"][0].clone()
    return new_theta

def gibbs_update_hyperparams(A, X_test, initial_theta, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    对 GP 超参数 theta = [log(sigma_a), log(sigma_q[0]), ..., log(sigma_q[Q-1])] 进行采样更新。
    A: (T, Q, D)
    X_test: (T, D)
    initial_theta: tensor of shape (Q+1,)
    返回更新后的 theta 以及 sigma_a, sigma_q
    """
    new_theta = sample_theta_HMC(A, X_test, initial_theta, step_size, num_steps, num_samples, warmup_steps)
    sigma_a_new = torch.exp(new_theta[0])
    sigma_q_new = torch.exp(new_theta[1:])
    return new_theta, sigma_a_new, sigma_q_new

# =============================================================================
# Example usage for hyperparameters update:
# =============================================================================

# 初始化 theta 为当前 sigma_a 和 sigma_q 的对数
initial_theta = torch.zeros(Q+1, device=device)
initial_theta[0] = torch.log(sigma_a)
initial_theta[1:] = torch.log(sigma_q)

new_theta, sigma_a_new, sigma_q_new = gibbs_update_hyperparams(A, X_test, initial_theta,
                                                                step_size=0.01, num_steps=10, 
                                                                num_samples=1, warmup_steps=50)

print("Updated theta:", new_theta)
print("Updated sigma_a:", sigma_a_new.item())
print("Updated sigma_q:", sigma_q_new)

import torch
import math
import pyro
import pyro.infer.mcmc as mcmc
import pyro.distributions as dist

# 假设之前已经定义了如下函数，并且 jump_gp_results 中每个元素的 "jumpGP_model" 包含字段:
# "w": tensor of shape (1+Q,)
# "ms": scalar (local GP mean m_t)
# "logtheta": tensor of shape (Q+2,)
# "r": tensor of shape (M,1)（布尔类型）
# 同时 zeta_t 为邻域中每个点经过 transform_torch(A_t, x) 得到的 tensor, 形状 (M, Q)
# 以及 f_t 为采样得到的局部潜变量向量，形状 (M,)

# 定义局部目标函数，用于采样 φₜ = [w, logtheta] 的联合更新
def potential_local(phi, t, jump_gp_results):
    """
    phi: tensor of shape (2Q+3,), 其中 phi[:1+Q] 对应 w, phi[1+Q:] 对应 logtheta.
    t: 测试点索引
    jump_gp_results: 列表，每个元素为一个字典，其中 "jumpGP_model" 包含:
         - "r": tensor of shape (M, 1), 布尔型
         - "f_t": tensor of shape (M,)
         - "zeta_t": tensor of shape (M, Q)
         - "ms": scalar, 即 m_t
    返回：潜在函数值 (标量)，即负的目标 log 概率（不含常数项）。
    """
    # 从 phi 中拆分出 w 和 logtheta
    Q = jump_gp_results[t]["zeta_t"].shape[1]
    M = jump_gp_results[t]["zeta_t"].shape[0]
    w = phi[:1+Q]
    logtheta = phi[1+Q:]
    
    model = jump_gp_results[t]["jumpGP_model"]
    # 取 r, f_t, zeta_t, ms
    r = model["r"].flatten().to(torch.float64)  # shape (M,)
    f_t = jump_gp_results[t]["f_t"].to(torch.float64)  # shape (M,)
    zeta = jump_gp_results[t]["zeta_t"].to(torch.float64)  # shape (M, Q)
    ms = model["ms"]
    
    # 计算 membership likelihood
    loglik_r = 0.0
    for j in range(M):
        g = w[0] + torch.dot(w[1:], zeta[j])
        p = torch.sigmoid(g)
        p = torch.clamp(p, 1e-10, 1-1e-10)
        if r[j]:
            loglik_r = loglik_r + torch.log(p)
        else:
            loglik_r = loglik_r + torch.log(1 - p)
    
    # 计算 f_t 的似然
    # 计算局部 GP 核矩阵，利用 zeta 和 logtheta
    def compute_jumpGP_kernel_torch(zeta, logtheta):
        logtheta = logtheta.to(torch.float64)
        zeta = zeta.to(torch.float64)
        M, Q = zeta.shape
        ell = torch.exp(logtheta[:Q])
        s_f = torch.exp(logtheta[Q])
        sigma_n = torch.exp(logtheta[Q+1])
        zeta_scaled = zeta / ell
        diff = zeta_scaled.unsqueeze(1) - zeta_scaled.unsqueeze(0)
        dists_sq = torch.sum(diff**2, dim=2)
        K = s_f**2 * torch.exp(-0.5 * dists_sq)
        K = K + (sigma_n**2) * torch.eye(M, device=zeta.device, dtype=zeta.dtype)
        return K

    K_jump = compute_jumpGP_kernel_torch(zeta, logtheta)
    mean_vec = ms * torch.ones(M, device=K_jump.device, dtype=K_jump.dtype)
    
    def log_multivariate_normal_pdf_torch(x, mean, cov):
        cov = cov.to(torch.float64)
        diff = x - mean
        sign, logdet = torch.slogdet(cov)
        if sign.item() <= 0:
            cov = cov + 1e-6 * torch.eye(len(x), device=cov.device, dtype=cov.dtype)
            sign, logdet = torch.slogdet(cov)
        return -0.5 * (diff @ torch.inverse(cov) @ diff + logdet + len(x) * math.log(2 * math.pi))
    
    loglik_f = log_multivariate_normal_pdf_torch(f_t, mean_vec, K_jump)
    
    # 定义先验项：
    # 对 w: 假设 p(w) ~ N(0, I)，则 log p(w) = -0.5 * ||w||^2 (忽略常数)
    log_prior_w = -0.5 * torch.sum(w**2)
    # 对 logtheta: 假设 p(logtheta) ~ N(0, I)
    log_prior_logtheta = -0.5 * torch.sum(logtheta**2)
    
    log_prior = log_prior_w + log_prior_logtheta
    
    # 目标 log 后验为： loglik_r + loglik_f + log_prior
    log_post = loglik_r + loglik_f + log_prior
    return -log_post  # 返回负的 log 后验，作为 potential 函数

# 使用 Pyro 的 NUTS 对局部超参数进行采样更新
def sample_local_hyperparams(initial_phi, t, jump_gp_results, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    对测试点 t 的局部超参数 φₜ = [w, logtheta] 进行采样更新，使用 Pyro 的 NUTS。
    
    参数：
      initial_phi: tensor of shape (2Q+3,), 初始值（w 和 logtheta 的扁平化向量）
      t: 测试点索引
      jump_gp_results: 列表，包含每个测试点的 jump GP 结果（其中 jumpGP_model 包含 w, ms, logtheta, r）
      其他参数为 NUTS 的采样参数。
    
    返回：
      new_phi: tensor of shape (2Q+3,), 更新后的局部超参数
    """
    def potential_fn(params):
        phi = params["phi"]
        return potential_local(phi, t, jump_gp_results)
    
    init_params = {"phi": initial_phi.clone()}
    nuts_kernel = mcmc.NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                            step_size=step_size, max_tree_depth=num_steps)
    mcmc_run = mcmc.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=init_params)
    mcmc_run.run()
    samples = mcmc_run.get_samples()
    new_phi = samples["phi"][0].clone()
    return new_phi

def gibbs_update_local_hyperparams(jump_gp_results, step_size=0.01, num_steps=10, num_samples=1, warmup_steps=50):
    """
    对每个测试点的 jump GP 局部超参数更新：更新 jumpGP_model 中的 'w' 和 'logtheta'
    使用 Pyro 的 NUTS 采样器对联合参数 φₜ = [w, logtheta] 进行采样。
    """
    for t in range(len(jump_gp_results)):
        model = jump_gp_results[t]["jumpGP_model"]
        Q = jump_gp_results[t]["zeta_t"].shape[1]
        # 初始值 φₜ = [w, logtheta], w: shape (1+Q,), logtheta: shape (Q+2,)
        initial_phi = torch.cat((model["w"].flatten(), model["logtheta"].flatten()), dim=0)
        new_phi = sample_local_hyperparams(initial_phi, t, jump_gp_results,
                                           step_size=step_size, num_steps=num_steps,
                                           num_samples=num_samples, warmup_steps=warmup_steps)
        # 更新 jumpGP_model 中对应的参数
        new_w = new_phi[:1+Q]
        new_logtheta = new_phi[1+Q:]
        model["w"] = new_w.clone()
        model["logtheta"] = new_logtheta.clone()
        jump_gp_results[t]["jumpGP_model"] = model
    print("Local hyperparameters updated for all test points via Pyro NUTS.")
    return jump_gp_results

# =============================================================================
# 示例：更新单个测试点的局部超参数
# =============================================================================

# 对 jump_gp_results 中每个测试点对应的局部超参数进行 Gibbs 更新
jump_gp_results = gibbs_update_local_hyperparams(jump_gp_results,
                                                  step_size=0.01,
                                                  num_steps=10,
                                                  num_samples=1,
                                                  warmup_steps=50)

# 打印更新后的第 0 个测试点的局部超参数
model_updated = jump_gp_results[0]["jumpGP_model"]
print("Updated local hyperparameters for test point 0:")
print(" w:", model_updated["w"])
print(" logtheta:", model_updated["logtheta"])


def update_r_for_test_point(jump_result, U=0.1):
    """
    对单个测试点 t 的邻域 membership 指示变量 r 进行更新。
    
    输入 jump_result 是一个字典，其中包含：
      - "y_neigh": tensor of shape (M,) —— 观测响应
      - "f_t": tensor of shape (M,) —— 局部 GP 潜变量
      - "zeta_t": tensor of shape (M, Q) —— 经 A_t 变换后的邻域特征
      - "jumpGP_model": 字典，至少包含：
             "w": tensor of shape (1+Q,), 局部 membership 参数
             "logtheta": tensor of shape (Q+2,), 局部 GP 核超参数（其中最后一个元素用于噪声标准差）
             "r": tensor of shape (M, 1) —— 当前的 membership 指示变量（将被更新）
    U: 离群点似然常数
    返回更新后的 jump_result，其中 jumpGP_model["r"] 已更新为新的指示变量。
    """
    y = jump_result["y_neigh"]   # (M,)
    f = jump_result["f_t"]       # (M,)
    zeta = jump_result["zeta_t"] # (M, Q)
    model = jump_result["jumpGP_model"]
    w = model["w"]               # (1+Q,)
    logtheta = model["logtheta"] # (Q+2,)
    # 根据要求，噪声标准差 sigma = exp(logtheta[-1])
    sigma = torch.exp(logtheta[-1])
    sigma2 = sigma**2
    M = y.shape[0]
    new_r = torch.zeros(M, dtype=torch.bool, device=y.device)
    for j in range(M):
        # 计算 g = w[0] + dot(w[1:], zeta[j])
        g = w[0] + torch.dot(w[1:], zeta[j])
        p_mem = torch.sigmoid(g)
        # 计算正态似然 L1 = N(y_j | f_j, sigma^2)
        L1 = (1.0 / torch.sqrt(2 * math.pi * sigma2)) * torch.exp(-0.5 * ((y[j] - f[j])**2 / sigma2))
        numerator = p_mem * L1
        denominator = numerator + (1 - p_mem) * U
        p_z = numerator / (denominator + 1e-10)
        new_r[j] = (torch.rand(1, device=y.device) < p_z).bool().item()
    new_r = new_r.unsqueeze(1)  # 转换为 (M,1)
    model["r"] = new_r.clone()
    jump_result["jumpGP_model"] = model
    return jump_result

def update_all_r(neighborhoods, jump_gp_results, U=0.1):
    """
    对所有测试点更新其邻域 membership 指示变量 r。
    
    参数:
      neighborhoods: 列表，每个元素为字典，包含 "y_neighbors" 等
      jump_gp_results: 列表，每个元素为 jump_result，包含 "jumpGP_model"、"f_t"、"zeta_t"、"y_neigh" 等
      U: 离群点似然常数
    返回:
      更新后的 jump_gp_results
    """
    new_results = []
    for t in range(len(jump_gp_results)):
        new_result = update_r_for_test_point(jump_gp_results[t], U)
        new_results.append(new_result)
    return new_results

# 示例：更新所有测试点的 r
jump_gp_results = update_all_r(neighborhoods, jump_gp_results, U=0.1)

# 输出第 0 个测试点更新后的 r
print("Updated r for test point 0:")
print(jump_gp_results[0]["jumpGP_model"]["r"])

def sample_f_t_torch(jump_result):
    """
    对单个测试点 t 的 f^t 进行逐坐标 Gibbs 更新：
      对每个邻域点 j，固定其他 f 的值，计算 f_j 的后验分布，然后从中采样更新 f_j。
    
    输入 jump_result 为字典，必须包含：
      - "jumpGP_model": 字典，其中 "ms" 为 m_t, "logtheta" 为局部 GP 超参数（形状 (Q+2,)），
           其中最后一个元素用于噪声标准差 sigma_t = exp(logtheta[-1]).
           另外，"r" 为布尔张量，形状 (M,1)。
      - "zeta_t": tensor of shape (M, Q) —— 经过 A_t 变换后的邻域输入
      - "y_neigh": tensor of shape (M,) —— 邻域响应
      - "f_t": tensor of shape (M,) —— 当前 f^t 值
    返回更新后的 f^t (tensor of shape (M,))
    """
    model = jump_result["jumpGP_model"]
    m_t = model["ms"]  # scalar, GP 均值
    logtheta = model["logtheta"]  # tensor of shape (Q+2,)
    sigma_t = torch.exp(logtheta[-1])
    sigma2 = sigma_t**2
    # r: (M,1) 转换为 1d float mask (1 if True, 0 if False)
    r = model["r"].flatten().to(torch.float64).float()  # shape (M,)
    y = jump_result["y_neigh"].to(torch.float64)  # (M,)
    zeta = jump_result["zeta_t"].to(torch.float64)  # (M, Q)
    f_current = jump_result["f_t"].to(torch.float64)  # (M,)
    M = zeta.shape[0]
    
    # 计算协方差矩阵 C（先验协方差）：
    # C = compute_jumpGP_kernel_torch(zeta, logtheta)
    # 此处假设 compute_jumpGP_kernel_torch 已定义，返回 (M, M) 的核矩阵
    C = compute_jumpGP_kernel_torch(zeta, logtheta)
    
    # 为了便于计算逐坐标条件分布，我们对每个 j 分块计算：
    f_new = f_current.clone()
    for j in range(M):
        # 令 I = {0,...,M-1} \ {j}
        idx = [i for i in range(M) if i != j]
        idx_tensor = torch.tensor(idx, dtype=torch.long, device=zeta.device)
        # 从 C 中提取相关块：
        C_jj = C[j, j]  # 标量
        C_jI = C[j, idx_tensor]  # (M-1,)
        C_II = C[idx_tensor][:, idx_tensor]  # (M-1, M-1)
        # 先验条件分布：
        # f_j | f_{-j} ~ N(m_prior, v_prior)
        # 其中 m_prior = m_t + C_jI * inv(C_II) * (f_current[idx] - m_t)
        f_minus = f_new[idx_tensor]
        invC_II = torch.inverse(C_II)
        m_prior = m_t + (C_jI @ (invC_II @ (f_minus - m_t * torch.ones_like(f_minus))))
        v_prior = C_jj - (C_jI @ (invC_II @ C_jI.T))
        # 如果 r[j]==1，则引入观测似然：
        if r[j] > 0.5:
            v_post = 1.0 / (1.0 / v_prior + 1.0 / sigma2)
            m_post = v_post * (m_prior / v_prior + y[j] / sigma2)
        else:
            v_post = v_prior
            m_post = m_prior
        # 采样 f_j ~ N(m_post, v_post)
        f_new[j] = m_post + torch.sqrt(v_post) * torch.randn(1, device=zeta.device, dtype=zeta.dtype)
    return f_new

def gibbs_update_f_all(jump_gp_results):
    """
    对 jump_gp_results 中每个测试点更新其 f^t（局部 GP 潜变量），采用逐坐标 Gibbs 更新。
    更新后将 f_t 写回 jump_gp_results。
    """
    for t in range(len(jump_gp_results)):
        new_f = sample_f_t_torch(jump_gp_results[t])
        jump_gp_results[t]["f_t"] = new_f.clone()
    print("Gibbs update of f_t completed for all test points.")
    return jump_gp_results

# =============================================================================
# 示例：更新所有测试点的 f^t
# =============================================================================
jump_gp_results = gibbs_update_f_all(jump_gp_results)

# 输出第 0 个测试点更新后的 f^t
print("Updated f_t for test point 0:")
print(jump_gp_results[0]["f_t"])



'''
def sample_f_t_torch(jump_result):
    """
    对单个测试点 t 的局部 GP 潜变量 f^t 进行采样更新。
    
    输入 jump_result 为字典，必须包含：
      - "jumpGP_model": 字典，其中包含：
             "ms": scalar, 表示 m_t
             "logtheta": tensor of shape (Q+2,), 用于计算核矩阵
             "r": tensor of shape (M,1)，指示哪些邻域点被认为是“正常”的（取值 True/False）
      - "zeta_t": tensor of shape (M, Q)，表示经过 A_t 变换后的邻域输入
      - "y_neigh": tensor of shape (M,), 邻域中对应的观测响应 y^t
    返回：
      f_sample: tensor of shape (M,), 表示采样得到的新的 f^t
    """
    model = jump_result["jumpGP_model"]
    ms = model["ms"]  # scalar, m_t
    logtheta = model["logtheta"]  # tensor, shape (Q+2,)
    # 根据题目要求，噪声标准差 sigma_t = exp(logtheta[-1])
    sigma_t = torch.exp(logtheta[-1])
    sigma_t2 = sigma_t**2
    # r: (M,1) -> (M,), 转换为 float (0 or 1)
    r = model["r"].flatten().to(torch.float64)
    # 观测响应
    y = jump_result["y_neigh"].to(torch.float64)
    # 经过 A_t 变换后的邻域输入 zeta
    zeta = jump_result["zeta_t"].to(torch.float64)  # (M, Q)
    M = zeta.shape[0]
    # 计算核矩阵 K = compute_jumpGP_kernel_torch(zeta, logtheta)
    K = compute_jumpGP_kernel_torch(zeta, logtheta)  # (M, M)
    invK = torch.inverse(K)
    # 构造对角矩阵 D，其中对于 j, 如果 r_j 为 True，则 D_jj = 1/sigma_t^2，否则为 0
    D_diag = r / sigma_t2  # (M,)
    D_matrix = torch.diag(D_diag)
    # 后验精度矩阵 Lambda = invK + D_matrix
    Lambda = invK + D_matrix
    # 计算 b = invK * (m_t * 1_M) + (r/sigma_t^2) .* y
    ones_vec = torch.ones(M, device=y.device, dtype=y.dtype)
    b = (invK @ (ms * ones_vec)) + (D_diag * y)
    # 后验协方差
    Cov = torch.inverse(Lambda)
    # 后验均值
    mu = Cov @ b
    # 采样：f_sample = mu + L * eps, 其中 L 为 Cov 的 Cholesky 分解
    L = torch.linalg.cholesky(Cov)
    eps = torch.randn(M, device=y.device, dtype=y.dtype)
    f_sample = mu + L @ eps
    return f_sample

def gibbs_update_f(jump_gp_results):
    """
    对所有测试点更新 f^t。
    
    对 jump_gp_results 中的每个元素，更新其 "f_t" 字段。
    """
    for t in range(len(jump_gp_results)):
        new_f = sample_f_t_torch(jump_gp_results[t])
        jump_gp_results[t]["f_t"] = new_f.clone()
    print("Updated f_t for all test points.")
    return jump_gp_results

# 示例：更新所有测试点的 f^t
jump_gp_results = gibbs_update_f(jump_gp_results)

# 输出第 0 个测试点采样得到的 f^t
print("Sampled f_t for test point 0:")
print(jump_gp_results[0]["f_t"])

'''


