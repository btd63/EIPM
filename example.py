from __future__ import annotations

import torch


def h0_t(T_train: torch.Tensor, t: torch.Tensor, a_h: float, k: int) -> torch.Tensor:
    T_tr = T_train.view(-1, 1)
    T_q = t.view(1, -1)
    dist = torch.abs(T_tr - T_q)
    dist_sorted, _ = torch.sort(dist, dim=0)
    k_eff = min(k, T_tr.shape[0] - 1)
    h_knn = a_h * dist_sorted[k_eff, :]
    return h_knn.view(-1, 1)


def rbf(X: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(X, X) ** 2 / (2.0 * sigma**2))


def eipm_loss_vectorized(X, T, s_val, a_sigma, a_h, k_nn):
    n = X.shape[0]
    h_T = h0_t(T, T, a_h=a_h, k=k_nn).view(-1, 1)

    s_max = torch.max(s_val)
    W_s = torch.exp(s_val - s_max)

    T_flat = T.view(-1, 1)
    dist_sq_T = (T_flat - T_flat.t()) ** 2
    H = (h_T @ h_T.t()).clamp_min(1e-8)
    K_T = torch.exp(-0.5 * dist_sq_T / H)

    d_med = torch.median(torch.cdist(X, X).flatten()[1:])
    sigma = float(a_sigma) * float(d_med)
    K_X = rbf(X, sigma)

    denom = K_T @ W_s.view(-1, 1)
    W_mat = (K_T * W_s.view(1, -1)) / (denom.view(-1, 1) + 1e-8)

    term1 = torch.sum((W_mat @ K_X) * W_mat, dim=1)
    term2 = torch.mean(K_X)
    term3 = 2.0 * torch.mean(W_mat @ K_X, dim=1)

    return torch.mean(term1 - term3 + term2)


def eipm_loss_loop(X, T, s_val, a_sigma, a_h, k_nn):
    n = X.shape[0]

    # h_i
    h_T = h0_t(T, T, a_h=a_h, k=k_nn).view(-1)

    # K_T(i,j)
    K_T = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            denom = float(h_T[i] * h_T[j])
            denom = max(denom, 1e-8)
            K_T[i, j] = torch.exp(-0.5 * (T[i] - T[j]) ** 2 / denom)

    # K_X(j,k)
    d_med = torch.median(torch.cdist(X, X).flatten()[1:])
    sigma = float(a_sigma) * float(d_med)
    K_X = torch.zeros(n, n)
    for j in range(n):
        for k in range(n):
            diff = X[j] - X[k]
            K_X[j, k] = torch.exp(-torch.dot(diff, diff) / (2.0 * sigma**2))

    # W_s and W_ij
    s_max = torch.max(s_val)
    W_s = torch.exp(s_val - s_max)

    W_mat = torch.zeros(n, n)
    for i in range(n):
        denom = 0.0
        for l in range(n):
            denom += K_T[i, l] * W_s[l]
        for j in range(n):
            W_mat[i, j] = (K_T[i, j] * W_s[j]) / (denom + 1e-8)

    # terms
    term1 = torch.zeros(n)
    term3 = torch.zeros(n)

    for i in range(n):
        # term1_i = sum_j (W_i * (K_X @ W_i))_j
        # compute (W_i @ K_X) as vector
        Wi = W_mat[i]
        Wi_KX = Wi @ K_X  # (n,)
        term1[i] = torch.sum(Wi_KX * Wi)
        term3[i] = 2.0 * torch.mean(Wi_KX)

    term2 = torch.mean(K_X)
    return torch.mean(term1 - term3 + term2)


def main() -> None:
    torch.manual_seed(0)

    X = torch.tensor([[0.0, 0.5], [1.0, -0.5], [0.25, 0.25]], dtype=torch.float32)
    T = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float32)
    s_val = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)

    a_sigma = 1.0
    a_h = 1.0
    k_nn = 1

    loss_vec = eipm_loss_vectorized(X, T, s_val, a_sigma, a_h, k_nn)
    loss_loop = eipm_loss_loop(X, T, s_val, a_sigma, a_h, k_nn)

    print(f"loss_vectorized={loss_vec.item():.8f}")
    print(f"loss_loop     ={loss_loop.item():.8f}")
    print(f"close? {torch.allclose(loss_vec, loss_loop, atol=1e-6)}")


if __name__ == "__main__":
    main()
