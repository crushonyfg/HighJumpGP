import torch
import math
import matplotlib.pyplot as plt
import numpy as np  # 用于转换显示
from scipy.stats import gamma
import torch
import pyro
import pyro.infer.mcmc as mcmc
from pyro.infer.mcmc import NUTS, MCMC
import pyro.distributions as dist
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


from JumpGP_code_py.JumpGP_LD import JumpGP_LD
from utils import *

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


pyro_update_params = {"step_size": 0.01, "num_steps": 10, "num_samples": 1, "warmup_steps": 10}
local_params = {"local_step_size": 0.01, "local_num_steps": 10, "local_num_samples": 1, "local_warmup_steps": 10}
hyper_update_params = {"step_size": 0.01, "num_steps": 10, "num_samples": 1, "warmup_steps": 10}

num_iterations = 2

# 外层 Gibbs 采样
K_cache_A = {q: compute_K_A_torch(X_test, sigma_a, sigma_q[q]) for q in range(len(sigma_q))}
A, jump_gp_results, sigma_a, sigma_q, _ = gibbs_sampling(A, X_test, neighborhoods, jump_gp_results, K_cache_A, 
                                                       sigma_a, sigma_q,
                                                       num_iterations=num_iterations,
                                                       pyro_update_params=pyro_update_params,
                                                       local_params=local_params,
                                                       hyper_update_params=hyper_update_params,
                                                       U=0.1,
                                                       num_workers=4)

print("\nFinal update for test point 0:")
model_updated = jump_gp_results[0]["jumpGP_model"]
print(" Updated local hyperparameters (w):", model_updated["w"])
print(" Updated local hyperparameters (logtheta):", model_updated["logtheta"])
print(" Updated membership r:", model_updated["r"])
print(" Updated f_t:", jump_gp_results[0]["f_t"])
