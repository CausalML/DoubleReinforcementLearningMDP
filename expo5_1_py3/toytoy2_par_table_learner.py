# Full GPU-Accelerated Simulation with 15 Estimators Using PyTorch
import torch
import numpy as np
import time
import argparse
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from torch.cuda.amp import autocast, GradScaler
from sklearn.ensemble import ExtraTreesRegressor
from cuml.ensemble import RandomForestRegressor as cuRF
import cuml
import cupy as cp



# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SmallMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

def fit_mlp(X, y, epochs=10, lr=1e-2):
    model = SmallMLP(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-0.1 * x))

def behav_dens(s, a, beta):
    b = beta * sigmoid(s) + beta * 0.5
    # Convert a_val to tensor if it's a scalar
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=device, dtype=torch.float32)
    # Ensure a is the same shape as b for comparison
    if a.dim() == 0:
        a = a.expand_as(b)
    return torch.where(a == 1.0, b, 1.0 - b)

def eval_dens(s, a, alpha):
    b = alpha * sigmoid(s) + (1 - alpha) * 0.5
    # Convert a_val to tensor if it's a scalar
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, device=device, dtype=torch.float32)
    # Ensure a is the same shape as b for comparison
    if a.dim() == 0:
        a = a.expand_as(b)
    return torch.where(a == 1.0, b, 1.0 - b)

def linear_regression(X, y):
    beta_hat = torch.linalg.lstsq(X, y).solution
    return beta_hat

def regress_q(s, a, r, alpha, squared=False):
    # Added alpha parameter to fix the scope issue
    N, T = s.shape
    q_weights = []
    V_next = torch.zeros(N, device=device)
    for t in reversed(range(T)):
        s_t = s[:, t]**2 if squared else s[:, t]
        sa = torch.stack([s_t, s_t * a[:, t], torch.ones(N, device=device)], dim=1)
        y = r[:, t] if t == T - 1 else r[:, t] + V_next
        w = linear_regression(sa, y)
        q_weights.insert(0, w)
        # V for next step
        V_next = 0
        for a_val in [0.0, 1.0]:
            s_a = torch.stack([s_t, s_t * a_val, torch.ones(N, device=device)], dim=1)
            # Create a tensor of a_val with the same size as s[:, t]
            a_val_tensor = torch.full_like(s[:, t], a_val)
            # Pass a_val_tensor to eval_dens
            prob = eval_dens(s[:, t], a_val_tensor, alpha)
            V_next += prob * (s_a @ w).squeeze()
    return q_weights

def regress_mu(s, a, w_, squared=False, method='linear'):
    N, T = s.shape
    mu_models = []
    for t in range(T):
        s_t = s[:, t]**2 if squared else s[:, t]
        sa = torch.stack([s_t, s_t * a[:, t], torch.ones(N, device=device)], dim=1)
        y = w_[:, t]

        if method == 'linear':
            mu_models.append(linear_regression(sa, y))
        elif method == 'mlp':
            mu_models.append(fit_mlp(sa.float(), y.float()))
        elif method == 'rf':
            sa_cp = cp.asarray(sa)  # Convert torch.Tensor to CuPy array
            y_cp = cp.asarray(y)

            rf = cuRF(
                n_estimators=10,         # Low for speed; increase if accuracy is poor
                max_depth=4,             # Slightly deeper trees than depth=3 to capture interactions
                min_samples_split=10,    # Avoid overfitting on small batches
                min_samples_leaf=5,      # Avoid overly deep trees with tiny leaves
                max_features=1.0,        # Try all features since you only have 3 — avoid randomness
                n_streams=8              # Parallel GPU streams (leave as-is for A100)
            )

            rf.fit(sa_cp, y_cp)
            mu_models.append(rf)

        else:
            raise ValueError("Unknown method for mu regression")

    return mu_models

def eval_dm(s, q0, alpha, squared=False):
    # Added alpha parameter to fix the scope issue
    N = s.shape[0]
    s0 = s[:, 0]**2 if squared else s[:, 0]
    V = torch.zeros(N, device=device)
    for a_val in [0.0, 1.0]:
        sa = torch.stack([s0, s0 * a_val, torch.ones(N, device=device)], dim=1)
        # Create a tensor of a_val with the same size as s[:, 0]
        a_val_tensor = torch.full_like(s[:, 0], a_val)
        # Pass a_val_tensor to eval_dens
        V += eval_dens(s[:, 0], a_val_tensor, alpha) * (sa @ q0).squeeze()
    return V.mean()

def eval_ipw(r, w):
    return (r * w).sum(dim=1).mean()

def eval_mis(mu_weights, s, a, r, squared=False):
    total = 0
    for t, w in enumerate(mu_weights):
        s_t = s[:, t]**2 if squared else s[:, t]
        sa = torch.stack([s_t, s_t * a[:, t], torch.ones(len(s), device=device)], dim=1)

        if isinstance(w, torch.Tensor):
            pred = (sa @ w).squeeze()
        elif isinstance(w, nn.Module):
            pred = w(sa).squeeze()
        else:
            pred = torch.tensor(w.predict(sa.cpu().numpy()), device=device)
        total += pred * r[:, t]
        # total += (sa @ w).squeeze() * r[:, t]
    return total.mean()

def eval_dr(q_weights1, q_weights2, s1, s2, a1, a2, r1, r2, w1, w2, alpha, squared=False):
    # Added alpha parameter to fix the scope issue
    def compute_half(qw, s, a, r, w):
        total = 0
        for t in range(s.shape[1]):
            s_t = s[:, t]**2 if squared else s[:, t]
            sa = torch.stack([s_t, s_t * a[:, t], torch.ones(len(s), device=device)], dim=1)
            V_t = torch.zeros(len(s), device=device)
            for a_val in [0.0, 1.0]:
                sa_val = torch.stack([s_t, s_t * a_val, torch.ones(len(s), device=device)], dim=1)
                # Create a tensor of a_val with the same size as s[:, t]
                a_val_tensor = torch.full_like(s[:, t], a_val)
                # Pass a_val_tensor to eval_dens
                V_t += eval_dens(s[:, t], a_val_tensor, alpha) * (sa_val @ qw[t]).squeeze()
            V_w = V_t if t == 0 else V_t * w[:, t - 1]
            total += (r[:, t] * w[:, t] - (sa @ qw[t]).squeeze() * w[:, t] + V_w).mean()
        return total
    return (compute_half(q_weights2, s1, a1, r1, w1) + compute_half(q_weights1, s2, a2, r2, w2)) / 2

def eval_dr2(q1, q2, mu1, mu2, s1, s2, a1, a2, r1, r2, alpha, squared_q=False, squared_mu=False):
    def compute_half(qw, mw, s, a, r):
        total = 0
        for t in range(s.shape[1]):
            sq = s[:, t]**2 if squared_q else s[:, t]
            sm = s[:, t]**2 if squared_mu else s[:, t]
            sa_q = torch.stack([sq, sq * a[:, t], torch.ones(len(s), device=device)], dim=1)
            sa_m = torch.stack([sm, sm * a[:, t], torch.ones(len(s), device=device)], dim=1)
            
            # Handle different model types for pred_mu
            if isinstance(mw[t], torch.Tensor):
                pred_mu = (sa_m @ mw[t]).squeeze()
            elif isinstance(mw[t], nn.Module):
                pred_mu = mw[t](sa_m).squeeze()
            else:
                pred_mu = torch.tensor(mw[t].predict(sa_m.cpu().numpy()), device=device)
            
            V_t = torch.zeros(len(s), device=device)
            for a_val in [0.0, 1.0]:
                sq_val = sq
                sa_val = torch.stack([sq_val, sq_val * a_val, torch.ones(len(s), device=device)], dim=1)
                a_val_tensor = torch.full_like(s[:, t], a_val)
                V_t += eval_dens(s[:, t], a_val_tensor, alpha) * (sa_val @ qw[t]).squeeze()
            
            if t > 0:
                sm_prev = s[:, t - 1]**2 if squared_mu else s[:, t - 1]
                sa_m_prev = torch.stack([sm_prev, sm_prev * a[:, t - 1], torch.ones(len(s), device=device)], dim=1)
                
                # Handle different model types for the previous mu weights
                if isinstance(mw[t-1], torch.Tensor):
                    pred_mu_prev = (sa_m_prev @ mw[t-1]).squeeze()
                elif isinstance(mw[t-1], nn.Module):
                    pred_mu_prev = mw[t-1](sa_m_prev).squeeze()
                else:
                    pred_mu_prev = torch.tensor(mw[t-1].predict(sa_m_prev.cpu().numpy()), device=device)
                
                V_t *= pred_mu_prev
                
            total += (pred_mu * r[:, t] - pred_mu * (sa_q @ qw[t]).squeeze() + V_t).mean()
        return total
    
    return (compute_half(q2, mu2, s1, a1, r1) + compute_half(q1, mu1, s2, a2, r2)) / 2
    
def run_single_repetition(N, T, beta, alpha, method='linear'):
    # Generate trajectories
    s = torch.zeros(N, T, device=device)
    a = torch.zeros(N, T, device=device)
    r = torch.zeros(N, T, device=device)
    w = torch.ones(N, T, device=device)

    s[:, 0] = torch.normal(0.5, 0.2, size=(N,), device=device)
    p = beta * sigmoid(s[:, 0]) + beta * torch.rand(N, device=device)
    a[:, 0] = torch.bernoulli(p)
    r[:, 0] = torch.normal(0.9 * s[:, 0] + 0.3 * a[:, 0], 0.2)

    for t in range(1, T):
        s[:, t] = torch.normal(0.02 * (t % 2) + s[:, t - 1] - 0.3 * (a[:, t - 1] - 0.5), 0.2)
        p = beta * sigmoid(s[:, t]) + beta * torch.rand(N, device=device)
        a[:, t] = torch.bernoulli(p)
        w[:, t] = eval_dens(s[:, t], a[:, t], alpha) / behav_dens(s[:, t], a[:, t], beta) * w[:, t - 1]
        r[:, t] = torch.normal(0.9 * s[:, t] + 0.3 * a[:, t] - 0.02 * (t % 2), 0.2)

    # Split data for cross-fitting
    s1, s2 = s.chunk(2)
    a1, a2 = a.chunk(2)
    r1, r2 = r.chunk(2)
    w1, w2 = w.chunk(2)

    # Regression for q-functions and mu-functions
    # Pass alpha to all functions that use it
    q1 = regress_q(s1, a1, r1, alpha)
    q2 = regress_q(s2, a2, r2, alpha)
    q1_sq = regress_q(s1, a1, r1, alpha, squared=True)
    q2_sq = regress_q(s2, a2, r2, alpha, squared=True)

    mu1 = regress_mu(s1, a1, w1, method=method)
    mu2 = regress_mu(s2, a2, w2, method=method)
    mu1_sq = regress_mu(s1, a1, w1, squared=True, method=method)
    mu2_sq = regress_mu(s2, a2, w2, squared=True, method=method)

    # Calculate all estimators
    # Pass alpha to all functions that use it
    return {
        'ipw': eval_ipw(r, w).item(),
        'dm': ((eval_dm(s1, q1[0], alpha) + eval_dm(s2, q2[0], alpha)) / 2).item(),
        'ipw2': ((eval_mis(mu1, s2, a2, r2) + eval_mis(mu2, s1, a1, r1)) / 2).item(),
        'dr': eval_dr(q1, q2, s1, s2, a1, a2, r1, r2, w1, w2, alpha).item(),
        'dr2': eval_dr2(q1, q2, mu1, mu2, s1, s2, a1, a2, r1, r2, alpha).item(),
        'ipw_mis_q': eval_ipw(r, w).item(),
        'dm_mis_q': ((eval_dm(s1, q1_sq[0], alpha, squared=True) + eval_dm(s2, q2_sq[0], alpha, squared=True)) / 2).item(),
        'ipw2_mis_q': ((eval_mis(mu1, s2, a2, r2) + eval_mis(mu2, s1, a1, r1)) / 2).item(),
        'dr_mis_q': eval_dr(q1_sq, q2_sq, s1, s2, a1, a2, r1, r2, w1, w2, alpha, squared=True).item(),
        'dr2_mis_q': eval_dr2(q1_sq, q2_sq, mu1, mu2, s1, s2, a1, a2, r1, r2, alpha, squared_q=True).item(),
        'ipw_mis_mu': eval_ipw(r, w).item(),
        'dm_mis_mu': ((eval_dm(s1, q1[0], alpha) + eval_dm(s2, q2[0], alpha)) / 2).item(),
        'ipw2_mis_mu': ((eval_mis(mu1_sq, s2, a2, r2, squared=True) + eval_mis(mu2_sq, s1, a1, r1, squared=True)) / 2).item(),
        'dr_mis_mu': eval_dr(q1, q2, s1, s2, a1, a2, r1, r2, w1, w2, alpha).item(),
        'dr2_mis_mu': eval_dr2(q1, q2, mu1_sq, mu2_sq, s1, s2, a1, a2, r1, r2, alpha, squared_mu=True).item(),
    }

def main():
    parser = argparse.ArgumentParser(description='Run GPU-accelerated RL estimators')
    parser.add_argument('--N', type=int, default=1500, help='Number of trajectories')
    parser.add_argument('--T', type=int, default=30, help='Time horizon')
    parser.add_argument('--beta', type=float, default=0.2, help='Behavior policy parameter')
    parser.add_argument('--alpha', type=float, default=0.9, help='Evaluation policy parameter')
    parser.add_argument('--reps', type=int, default=1500, help='Number of repetitions')
    parser.add_argument('--mu_method', type=str, default='linear', choices=['linear', 'mlp', 'rf'], help='W-function learner')
    args = parser.parse_args()

    print(f"Running with parameters: N={args.N}, T={args.T}, beta={args.beta}, alpha={args.alpha}, reps={args.reps}, mu_method={args.mu_method}")

    sample_N = min(100, args.N)
    sample_res = run_single_repetition(sample_N, args.T, args.beta, args.alpha, args.mu_method)
    results = {k: [] for k in sample_res.keys()}

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    start_time = time.time()
    for i in range(args.reps):
        rep_start_time = time.time()
        print(f"\nStarting repetition {i+1}/{args.reps}")
        try:
            res = run_single_repetition(args.N, args.T, args.beta, args.alpha, args.mu_method)
            for k in results:
                results[k].append(res[k])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rep_time = time.time() - rep_start_time
            print(f"Repetition {i+1} completed in {rep_time:.2f} seconds")
            if (i+1) % 10 == 0 or i == args.reps-1:
                for k in results:
                    np.savez(f"estimator_list_{k}_{args.mu_method}_{args.N}", a=np.array(results[k]))
                print(f"Saved checkpoint after {i+1} repetitions")
        except Exception as e:
            print(f"Error in repetition {i+1}: {e}")
    for k in results:
        np.savez(f"estimator_list_{k}_{args.mu_method}_{args.N}", a=np.array(results[k]))
    total_time = time.time() - start_time
    print(f"All {args.reps} repetitions completed in {total_time:.2f} seconds")

if __name__ == '__main__':
    import sys
    main()