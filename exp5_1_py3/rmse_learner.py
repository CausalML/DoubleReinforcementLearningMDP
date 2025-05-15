import numpy as np
import pandas as pd
import os
import argparse
from joblib import Parallel, delayed
import multiprocessing as mp
import time

# Estimators from your outputs
ESTIMATORS = [
    'ipw', 'dm', 'ipw2', 'dr', 'dr2',
    'ipw_mis_q', 'dm_mis_q', 'ipw2_mis_q', 'dr_mis_q', 'dr2_mis_q',
    'ipw_mis_mu', 'dm_mis_mu', 'ipw2_mis_mu', 'dr_mis_mu', 'dr2_mis_mu'
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-0.1 * x))

def estimate_true_value_single(seed, T=30, alpha=0.9):
    np.random.seed(seed)
    s = np.random.normal(0.5, 0.2)
    total_r = 0
    for j in range(T):
        a_prob = alpha * sigmoid(s) + (1 - alpha) * np.random.uniform(0, 1)
        a = np.random.binomial(1, a_prob)
        r = np.random.normal(0.9 * s + 0.3 * a - 0.02*(j%2), 0.2)
        total_r += r
        if j < T-1:
            s = np.random.normal(0.02*(j%2) + s - 0.3*(a-0.5), 0.2)
    return total_r

def estimate_true_value(n_episodes=100000, T=30, alpha=0.9, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    returns = Parallel(n_jobs=n_jobs)(delayed(estimate_true_value_single)(i, T, alpha) for i in range(n_episodes))
    mean_return = np.mean(returns)
    std_error = np.std(returns) / np.sqrt(n_episodes)
    print(f"Estimated true value: {mean_return:.6f} ± {std_error:.6f}")
    return mean_return

def load_estimates(N, mu_method):
    estimates = {}
    for est in ESTIMATORS:
        fname = f"estimator_list_{est}_{mu_method}_{N}.npz"
        if os.path.exists(fname):
            data = np.load(fname)
            estimates[est] = data['a'] if 'a' in data else data[list(data.keys())[0]]
        else:
            print(f"Warning: {fname} not found.")
            estimates[est] = np.array([])
    return estimates

def calculate_rmse(estimates, true_value):
    results = []
    for name, values in estimates.items():
        if values.size == 0:
            continue
        bias = np.mean(values) - true_value
        rmse = np.sqrt(np.mean((values - true_value)**2))
        se = np.std((values - true_value)**2) / (2 * rmse * np.sqrt(len(values)))
        results.append({'Estimator': name, 'RMSE': rmse, 'Bias': bias, 'SE': se, 'Std': np.std(values)})
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1500)
    parser.add_argument('--true_value', type=float, default=None)
    parser.add_argument('--n_episodes', type=int, default=100000)
    parser.add_argument('--output', type=str, default='rmse_results.csv')
    parser.add_argument('--mu_method', type=str, default='linear', choices=['linear', 'mlp', 'rf'], help='W-function learner')


    args = parser.parse_args()

    # Step 1: Estimate true value if not provided
    if args.true_value is None:
        print("No true value provided. Estimating...")
        true_value = estimate_true_value(n_episodes=args.n_episodes)
    else:
        true_value = args.true_value
        print(f"Using provided true value: {true_value}")

    # Step 2: Load estimates
    estimates = load_estimates(args.N, args.mu_method)

    # Step 3: Calculate RMSE
    results = calculate_rmse(estimates, true_value)
    results = results.sort_values(by='RMSE')

    # Step 4: Save and display
    results.to_csv(args.output, index=False)
    print("\n=== RMSE Results ===")
    print(results)

if __name__ == "__main__":
    main()
