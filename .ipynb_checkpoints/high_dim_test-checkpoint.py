import torch
import math
import matplotlib.pyplot as plt
import numpy as np  # 用于转换显示
from scipy.stats import gamma
import pyro
import pyro.infer.mcmc as mcmc
from pyro.infer.mcmc import NUTS, MCMC
import pyro.distributions as dist
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from JumpGP_code_py.JumpGP_LD import JumpGP_LD
from utils import *  # 包含 transform_torch, compute_K_A_torch, jumpgp_ld_wrapper, gibbs_sampling 等

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# =============================================================================
# 1. 数据生成（高维输入，但内在结构为2维）
# =============================================================================
# 设置维度
N = 1000      # 训练点数
D = 20        # 高维输入（外显维度）
T = 20        # 测试点数
M = 100       # 每个测试点的邻域大小
latent_dim = 2  # 内在2维
Q = 2         # 潜在空间维度

# 生成内在低维变量 Z_train 和 Z_test
Z_train = torch.randn(N, latent_dim, device=device)
Z_test = torch.randn(T, latent_dim, device=device)

# 生成一个随机投影矩阵，将2维映射到D维（可保持同一个投影矩阵使训练和测试在同一流形上）
proj = torch.randn(latent_dim, D, device=device)

# 得到高维输入 X_train 和 X_test
X_train = Z_train @ proj   # (N, D)
X_test  = Z_test  @ proj   # (T, D)

# 根据内在低维生成目标变量 Y（保持2维内在函数关系）
Y_train = torch.sin(Z_train[:, 0]) + torch.cos(Z_train[:, 1]) + 0.1 * torch.randn(N, device=device)
Y_test  = torch.sin(Z_test[:, 0]) + torch.cos(Z_test[:, 1]) + 0.1 * torch.randn(T, device=device)

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

# 可视化（仅选取前2个维度显示，以便观察）
X_train_np = X_train.cpu().numpy()
X_test_np = X_test.cpu().numpy()
A_mean = A.mean(dim=1).cpu().numpy()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c='blue', alpha=0.7, label='Training X')
plt.scatter(X_test_np[:, 0], X_test_np[:, 1], c='red', marker='^', s=100, label='Test X')
plt.title("Training and Test Inputs (First 2 dims)")
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
# 2. 构造每个测试点的邻域
# =============================================================================
# 这里基于高维的 X_train 和 X_test 计算距离
dists = torch.cdist(X_test, X_train)  # (T, N)
_, indices = torch.sort(dists, dim=1)
indices = indices[:, :M]  # 选取 M 个最近邻
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
# 3. Jump GP: 局部映射与局部 GP 采样
# =============================================================================
jump_gp_results = []
for t in range(T):
    A_t = A[t]  # (Q, D)
    neigh = neighborhoods[t]
    X_neigh = neigh["X_neighbors"]  # (M, D)
    y_neigh = neigh["y_neighbors"]    # (M,)
    # 计算邻域中每个点经过 A_t 变换后的表示 zeta_t (M, Q)
    zeta_t = torch.stack([transform_torch(A_t, X_neigh[i]) for i in range(M)], dim=0)
    # 对测试点进行变换，得到 x_t_test (形状 (Q,))
    x_t_test = transform_torch(A_t, X_test[t])
    # 调用 jumpgp_ld_wrapper 进行局部 GP 建模，获得 mu_t, sig2_t, 模型参数等
    mu_t, sig2_t, model, _ = jumpgp_ld_wrapper(zeta_t, y_neigh.view(-1, 1), x_t_test.view(1, -1), mode="CEM", flag=True, device=device)
    # 使用 model 中的 ms 作为均值构造 prior，计算局部核矩阵，并采样 f_t
    mean_f = model["ms"] * torch.ones(M, device=A.device, dtype=A.dtype)
    K_jump = compute_jumpGP_kernel_torch(zeta_t, model["logtheta"])
    mvn = torch.distributions.MultivariateNormal(mean_f, covariance_matrix=K_jump)
    f_t = mvn.sample()
    jump_gp_results.append({
        "test_index": t,
        "x_test": X_test[t],
        "A_t": A[t],
        "zeta_t": zeta_t,
        "x_t_test": x_t_test,
        "y_neigh": y_neigh,
        "jumpGP_model": model,
        "K_jump": K_jump,
        "f_t": f_t
    })

# =============================================================================
# 4. 定义预测函数（利用局部 GP 预测测试点的输出）
# =============================================================================
def predict_test_point(jump_result):
    """
    利用局部 GP 模型预测测试点的输出值。
    这里使用 GP 标准预测公式： m_test = m + k_*^T K^{-1}(f - m)
    其中 m = ms, f 为邻域 f_t, k_* 为测试点与邻域的核向量。
    """
    model = jump_result["jumpGP_model"]
    zeta = jump_result["zeta_t"]  
    y_neigh = jump_result["y_neigh"]
    x_test = jump_result["x_t_test"] 

    mu_t, sig2_t, model, _ = jumpgp_ld_wrapper(zeta, y_neigh.view(-1, 1), x_test.view(1, -1), mode="CEM", flag=True, device=device)
    return mu_t.item()

# =============================================================================
# 5. 外层 Gibbs 采样，并在每次迭代后计算测试集预测的 RMSE
# =============================================================================
pyro_update_params = {"step_size": 0.01, "num_steps": 10, "num_samples": 1, "warmup_steps": 50}
local_params = {"local_step_size": 0.01, "local_num_steps": 10, "local_num_samples": 1, "local_warmup_steps": 50}
hyper_update_params = {"step_size": 0.01, "num_steps": 10, "num_samples": 1, "warmup_steps": 50}

num_iterations = 20
rmse_trace = []  # 用于保存每次迭代测试集 RMSE

for it in range(num_iterations):
    print(f"\nGibbs iteration {it+1}/{num_iterations}")
    # 1. 更新全局 A 以及同步更新 jump_gp_results 中的 A_t, zeta_t, x_t_test 和 K_jump
    A = gibbs_update_A_with_pyro(A, X_test, neighborhoods, jump_gp_results, sigma_a, sigma_q,
                                 **(pyro_update_params or {}))
    T_total = A.shape[0]
    for t in range(T_total):
        jump_gp_results[t]["A_t"] = A[t].clone()
        X_neighbors = neighborhoods[t]["X_neighbors"]
        zeta_t = torch.stack([transform_torch(A[t], x) for x in X_neighbors], dim=0)
        jump_gp_results[t]["zeta_t"] = zeta_t
        jump_gp_results[t]["x_t_test"] = transform_torch(A[t], X_test[t])
        logtheta = jump_gp_results[t]["jumpGP_model"]["logtheta"]
        K_jump = compute_jumpGP_kernel_torch(zeta_t, logtheta)
        jump_gp_results[t]["K_jump"] = K_jump

    print(f"Iteration {it+1}: Global A updated and synchronized.")

    # 2. 并行更新所有测试点的局部参数（局部超参数、membership 指示、f_t、ms等）
    jump_gp_results = parallel_update_all_test_points(jump_gp_results, U=0.1, **(local_params or {}), num_workers=4)
    print(f"Iteration {it+1}: Local parameters updated for all test points.")

    # 3. 更新 GP 超参数 sigma_a, sigma_q
    Q_val = sigma_q.shape[0]
    initial_theta = torch.zeros(Q_val+1, device=X_test.device)
    initial_theta[0] = torch.log(sigma_a)
    initial_theta[1:] = torch.log(sigma_q)
    new_theta, sigma_a_new, sigma_q_new = gibbs_update_hyperparams(A, X_test, initial_theta,
                                                                   **(hyper_update_params or {}))
    sigma_a = sigma_a_new
    sigma_q = sigma_q_new
    print(f"Iteration {it+1}: Global GP hyperparameters updated: sigma_a = {sigma_a.item():.4f}, sigma_q = {sigma_q}")

    # 4. 对所有测试点进行预测，并计算 RMSE
    preds = []
    for t in range(T):
        pred = predict_test_point(jump_gp_results[t])
        preds.append(pred)
    preds = torch.stack(preds)  # (T,)
    rmse = torch.sqrt(torch.mean((preds - Y_test)**2))
    rmse_trace.append(rmse.item())
    print(f"Iteration {it+1}: Test RMSE = {rmse.item():.4f}")

# 保存 RMSE 结果（例如保存为 numpy 文件）
np.save("rmse_trace.npy", np.array(rmse_trace))
print("\nFinal RMSE trace:", rmse_trace)

# =============================================================================
# 6. 可视化部分（例如展示最后一次的测试点预测与真实值对比）
# =============================================================================
plt.figure(figsize=(8, 5))
plt.plot(rmse_trace, marker='o')
plt.xlabel("Gibbs Iteration")
plt.ylabel("Test RMSE")
plt.title("RMSE Trace over Gibbs Sampling Iterations")
plt.grid(True)
plt.show()

# 对某个测试点（例如 t=0），对比预测值与真实 Y_test
t_sel = 0
pred_t = predict_test_point(jump_gp_results[t_sel]).item()
true_t = Y_test[t_sel].item()
print(f"Test point {t_sel}: Predicted Y = {pred_t:.4f}, True Y = {true_t:.4f}")
