import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from joblib import Parallel, delayed
import argparse

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-0.1 * x))

def estimate_true_value_single(seed, T=30, alpha=0.9, beta=0.2):
    """
    Run a single episode using the evaluation policy to estimate the true value.
    """
    np.random.seed(seed)
    
    # Initial state
    s = np.random.normal(0.5, 0.2)
    
    # Total return for this episode
    episode_return = 0.0
    
    # Run the episode
    for j in range(T):
        # Use evaluation policy directly
        a_prob = alpha * sigmoid(s) + (1 - alpha) * np.random.uniform(0.0, 1.0)
        a = np.random.binomial(1, a_prob, 1)[0]
        
        # Get reward
        r = np.random.normal(0.9 * s + 0.3 * a - 0.02 * (j % 2), 0.2)
        episode_return += r
        
        # State transition (if not the last step)
        if j < T-1:
            s = np.random.normal(0.02 * ((j+1) % 2) + s * 1.0 - 0.3 * (a - 0.5), 0.2)
    
    return episode_return

def estimate_true_value(n_episodes=100000, T=30, alpha=0.9, beta=0.2, n_jobs=-1):
    """
    Estimate the true expected return of the evaluation policy.
    """
    start_time = time.time()
    print(f"Estimating true value using {n_episodes} direct episodes...")
    
    # Use joblib for parallelization
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # Create batches to show progress
    batch_size = min(10000, n_episodes)
    n_batches = (n_episodes + batch_size - 1) // batch_size
    
    all_returns = []
    
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, n_episodes)
        current_batch_size = end_idx - start_idx
        
        print(f"Running batch {batch+1}/{n_batches} ({start_idx+1}-{end_idx} of {n_episodes})...")
        
        # Run episodes in parallel
        batch_returns = Parallel(n_jobs=n_jobs)(
            delayed(estimate_true_value_single)(
                seed=start_idx+i, 
                T=T, 
                alpha=alpha, 
                beta=beta
            ) for i in range(current_batch_size)
        )
        
        all_returns.extend(batch_returns)
        
        # Show intermediate results
        current_mean = np.mean(all_returns)
        current_se = np.std(all_returns) / np.sqrt(len(all_returns))
        
        print(f"  Completed {len(all_returns)}/{n_episodes} episodes")
        print(f"  Current estimate: {current_mean:.6f} ± {current_se:.6f}")
    
    # Calculate final estimate and standard error
    true_value = np.mean(all_returns)
    std_error = np.std(all_returns) / np.sqrt(n_episodes)
    
    # Calculate time taken
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\nEstimation complete in {minutes}m {seconds}s")
    print(f"Final true value estimate: {true_value:.6f} ± {std_error:.6f}")
    print(f"Based on {n_episodes} episodes with evaluation policy (α={alpha})")
    
    return true_value, std_error

def load_npz_file(file_path):
    """Load a single NPZ file and return its contents."""
    try:
        data = np.load(file_path)
        # Check for different array names - first 'a' (old format) then 'data' (new format)
        if 'a' in data:
            return data['a']
        elif 'data' in data:
            return data['data']
        else:
            # Try to get the first array in the file
            array_keys = list(data.keys())
            if array_keys:
                return data[array_keys[0]]
            else:
                print(f"Warning: No valid arrays found in {file_path}")
                return np.array([])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def load_estimator_results(directory, sample_sizes):
    """
    Load all estimator results from NPZ files with the new naming convention.
    
    Parameters:
    -----------
    directory : str or list
        Directory or list of directories where NPZ files are stored
    sample_sizes : list
        List of sample sizes to load
        
    Returns:
    --------
    dict
        Dictionary with sample sizes as keys and dictionaries of estimators as values
    """
    all_estimators = {}
    
    # Estimator type mapping for nice display names
    estimator_display_names = {
        'ipw': 'IPW',
        'dr': 'DRL(M₁)',
        'dm': 'DM',
        'ipw2': 'IPW₂',
        'dr2': 'DRL(M₂)',
        'ipw_mis_q': 'IPW (q mis.)',
        'dr_mis_q': 'DRL(M₁) (q mis.)',
        'dm_mis_q': 'DM (q mis.)',
        'ipw2_mis_q': 'IPW₂ (q mis.)',
        'dr2_mis_q': 'DRL(M₂) (q mis.)',
        'ipw_mis_mu': 'IPW (μ mis.)',
        'dr_mis_mu': 'DRL(M₁) (μ mis.)',
        'dm_mis_mu': 'DM (μ mis.)',
        'ipw2_mis_mu': 'IPW₂ (μ mis.)',
        'dr2_mis_mu': 'DRL(M₂) (μ mis.)'
    }
    
    # Track if we found any files
    found_any_files = False
    
    for N in sample_sizes:
        all_estimators[N] = {}
        
        # Check for multiple possible file pattern formats
        possible_patterns = [
            f"_{N}.npz",              # New format: estimator_list_ipw_1500.npz
            f"_0default_{N}.npz",     # Old format: estimator_list_ipw_0default_1500.npz 
            f"_%d0default_{N}.npz",   # Old format with %d: estimator_list_ipw_%d0default_1500.npz
            f"_n{N}.npz",             # Alternative format: estimator_list_ipw_n1500.npz
            f"_gpu_{N}.npz",          # GPU format: gpu_ipw_1500.npz
            f"_{N}_gpu.npz"           # Another GPU format: ipw_1500_gpu.npz
        ]
        
        print(f"\nLooking for results with N = {N}:")
        
        # Check if directory is a list of possible directories
        if isinstance(directory, list):
            search_dirs = directory
        else:
            search_dirs = [directory]
            
        for search_dir in search_dirs:
            # Find all NPZ files for this sample size
            for pattern in possible_patterns:
                npz_files = [f for f in os.listdir(search_dir) if f.endswith('.npz') and pattern in f]
                
                if npz_files:
                    print(f"  Found {len(npz_files)} files with pattern {pattern}")
                    found_any_files = True
                    
                    for npz_file in npz_files:
                        try:
                            # Extract estimator name using different patterns
                            estimator_key = None
                            
                            # Try different naming conventions
                            if "estimator_list_" in npz_file:
                                # Extract part between "estimator_list_" and the pattern
                                estimator_part = npz_file.replace("estimator_list_", "")
                                for p in possible_patterns:
                                    if p in estimator_part:
                                        estimator_key = estimator_part.split(p.replace(".npz", ""))[0]
                                        # Remove trailing underscore if present
                                        estimator_key = estimator_key.rstrip('_')
                                        break
                            elif "gpu_" in npz_file:
                                # Format like gpu_ipw_1500.npz
                                parts = npz_file.split('_')
                                if len(parts) > 1:
                                    estimator_key = parts[1]
                            else:
                                # Last resort: try to extract from filename
                                parts = npz_file.split('_')
                                if len(parts) > 0:
                                    estimator_key = parts[0]
                            
                            # If we couldn't determine the estimator key, use the filename without extension
                            if not estimator_key:
                                estimator_key = os.path.splitext(npz_file)[0]
                                
                            # Load the NPZ file
                            values = load_npz_file(os.path.join(search_dir, npz_file))
                            
                            if len(values) > 0:
                                # Use display name if available
                                display_name = estimator_display_names.get(estimator_key, estimator_key)
                                all_estimators[N][display_name] = values
                                print(f"    Loaded {display_name} from {npz_file}: {len(values)} values")
                        except Exception as e:
                            print(f"    Error processing {npz_file}: {e}")
    
    if not found_any_files:
        print("\nWARNING: No NPZ files found matching the expected patterns!")
        print(f"Searched in: {directory}")
        print(f"Looking for sample sizes: {sample_sizes}")
        print("\nPlease check that your files are in the correct location and named correctly.")
        # List all files in the directory for debugging
        if isinstance(directory, str) and os.path.exists(directory):
            print("\nFiles found in the directory:")
            for f in os.listdir(directory):
                if f.endswith('.npz'):
                    print(f"  {f}")
    
    return all_estimators

def calculate_rmse(estimators, true_value):
    """
    Calculate RMSE, standard error, and bias for each estimator
    
    Parameters:
    -----------
    estimators : dict
        Dictionary with estimator names as keys and arrays of values as values
    true_value : float
        The true parameter value being estimated
        
    Returns:
    --------
    DataFrame
        DataFrame with RMSE, std errors, and bias for each estimator
    """
    results = []
    
    for name, values in estimators.items():
        if len(values) > 0:
            # Calculate RMSE
            squared_errors = np.square(np.array(values) - true_value)
            rmse = np.sqrt(np.mean(squared_errors))
            
            # Calculate standard error of RMSE
            # Based on the delta method approximation
            se_rmse = np.std(squared_errors) / (2 * rmse * np.sqrt(len(values)))
            
            # Calculate bias
            bias = np.mean(values) - true_value
            
            # Calculate mean and standard deviation
            mean = np.mean(values)
            std = np.std(values)
            
            results.append({
                'Estimator': name,
                'RMSE': rmse,
                'SE': se_rmse,
                'Bias': bias,
                'Mean': mean,
                'Std': std,
                'n_samples': len(values)
            })
    
    # Handle empty results
    if not results:
        print("WARNING: No data available to calculate RMSE!")
        # Return empty DataFrame with the expected columns
        return pd.DataFrame(columns=['Estimator', 'RMSE', 'SE', 'Bias', 'Mean', 'Std', 'n_samples'])
        
    return pd.DataFrame(results)

def create_rmse_table(all_estimators, true_value):
    """
    Create RMSE table for all sample sizes
    
    Parameters:
    -----------
    all_estimators : dict
        Dictionary with sample sizes as keys and dictionaries of estimators as values
    true_value : float
        The true parameter value
        
    Returns:
    --------
    tuple
        Tuple containing (rmse_table, se_table, bias_table, all_results)
    """
    all_results = []
    
    for N, estimators in all_estimators.items():
        if estimators:  # Check if there are any estimators for this N
            results = calculate_rmse(estimators, true_value)
            if not results.empty:
                results['N'] = N
                all_results.append(results)
        else:
            print(f"No estimators found for N={N}")
    
    # Handle case where no results were calculated
    if not all_results:
        print("WARNING: No valid results found to create RMSE table!")
        empty_df = pd.DataFrame(columns=['Estimator', 'RMSE', 'SE', 'Bias', 'Mean', 'Std', 'n_samples', 'N'])
        return empty_df, empty_df, empty_df, empty_df
    
    # Combine results from all sample sizes
    combined_results = pd.concat(all_results, ignore_index=True)
    
    if 'Estimator' not in combined_results.columns or 'N' not in combined_results.columns:
        print("WARNING: Missing required columns in results!")
        print(f"Available columns: {combined_results.columns.tolist()}")
        empty_df = pd.DataFrame(columns=['Estimator', 'RMSE', 'SE', 'Bias'])
        return empty_df, empty_df, empty_df, combined_results
    
    # Create pivot tables
    rmse_table = combined_results.pivot(index='Estimator', columns='N', values='RMSE')
    se_table = combined_results.pivot(index='Estimator', columns='N', values='SE')
    bias_table = combined_results.pivot(index='Estimator', columns='N', values='Bias')
    
    return rmse_table, se_table, bias_table, combined_results

def create_latex_table(rmse_table, se_table, output_file):
    """
    Create a LaTeX table with RMSE values and standard errors
    
    Parameters:
    -----------
    rmse_table : DataFrame
        DataFrame with RMSE values
    se_table : DataFrame
        DataFrame with standard error values
    output_file : str
        Output file path
    """
    # Skip if tables are empty
    if rmse_table.empty or se_table.empty:
        print(f"Skipping LaTeX table creation because data tables are empty")
        return
        
    with open(output_file, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{RMSE of Estimators (with standard errors in parentheses)}\n")
        f.write("\\begin{tabular}{l" + "c" * len(rmse_table.columns) + "}\n")
        f.write("\\hline\n")
        
        # Header row
        f.write("Estimator & " + " & ".join([f"N={n}" for n in rmse_table.columns]) + " \\\\\n")
        f.write("\\hline\n")
        
        # Data rows
        for estimator in rmse_table.index:
            row = f"{estimator}"
            for n in rmse_table.columns:
                if n in rmse_table.columns and n in se_table.columns:
                    rmse = rmse_table.loc[estimator, n]
                    se = se_table.loc[estimator, n]
                    row += f" & {rmse:.4f} ({se:.4f})"
                else:
                    row += " & -"
            row += " \\\\\n"
            f.write(row)
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        
    print(f"LaTeX table created: {output_file}")

def create_visualizations(rmse_table, se_table, combined_results, output_dir):
    """
    Create visualizations for RMSE results
    
    Parameters:
    -----------
    rmse_table : DataFrame
        Pivot table with RMSE values
    se_table : DataFrame
        Pivot table with standard error values
    combined_results : DataFrame
        Combined results DataFrame
    output_dir : str
        Directory to save visualizations
    """
    # Skip if tables are empty
    if rmse_table.empty or se_table.empty or combined_results.empty:
        print(f"Skipping visualization creation because data tables are empty")
        return
        
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['savefig.dpi'] = 300
    
    try:
        # 1. RMSE by sample size for each estimator
        plt.figure(figsize=(12, 8))
        for estimator in rmse_table.index:
            plt.plot(rmse_table.columns, rmse_table.loc[estimator], marker='o', linewidth=2, label=estimator)
        
        plt.xlabel('Sample Size (N)', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.title('RMSE by Sample Size for Each Estimator', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rmse_by_sample_size.png'))
        plt.close()
        
        # 2. Bar plot for each sample size
        for N in rmse_table.columns:
            plt.figure(figsize=(14, 8))
            
            # Sort estimators by RMSE
            sorted_estimators = rmse_table[N].sort_values().index
            
            # Plot RMSE bars
            ax = plt.bar(range(len(sorted_estimators)), rmse_table.loc[sorted_estimators, N])
            
            # Add error bars
            plt.errorbar(
                x=range(len(sorted_estimators)),
                y=rmse_table.loc[sorted_estimators, N],
                yerr=se_table.loc[sorted_estimators, N],
                fmt='none', capsize=5, color='black', elinewidth=1.5
            )
            
            plt.title(f'RMSE for Each Estimator (N = {N})', fontsize=16)
            plt.ylabel('RMSE', fontsize=14)
            plt.xticks(range(len(sorted_estimators)), sorted_estimators, rotation=45, ha='right', fontsize=12)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'rmse_n{N}.png'))
            plt.close()
        
        # 3. Heatmap of RMSE values
        plt.figure(figsize=(10, 8))
        sns.heatmap(rmse_table, annot=True, cmap='YlGnBu', fmt='.4f')
        plt.title('RMSE Heatmap by Estimator and Sample Size', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rmse_heatmap.png'))
        plt.close()
        
        # 4. Bias comparison if bias data is available
        if 'Bias' in combined_results.columns and 'N' in combined_results.columns:
            bias_table = combined_results.pivot(index='Estimator', columns='N', values='Bias')
            
            plt.figure(figsize=(12, 8))
            for estimator in bias_table.index:
                plt.plot(bias_table.columns, bias_table.loc[estimator], marker='o', linewidth=2, label=estimator)
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)  # Add zero line for reference
            plt.xlabel('Sample Size (N)', fontsize=14)
            plt.ylabel('Bias', fontsize=14)
            plt.title('Bias by Sample Size for Each Estimator', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bias_by_sample_size.png'))
            plt.close()
            
        print(f"All visualizations created in {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def find_npz_directories(base_dir):
    """Find directories containing NPZ files"""
    npz_dirs = []
    
    # First check the base directory
    if any(f.endswith('.npz') for f in os.listdir(base_dir)):
        npz_dirs.append(base_dir)
    
    # Then check subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if any(f.endswith('.npz') for f in os.listdir(item_path)):
                npz_dirs.append(item_path)
    
    return npz_dirs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate RMSE for reinforcement learning estimators')
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[100, 1500, 3000, 4500], 
                        help='Sample sizes to analyze')
    parser.add_argument('--true-value', type=float, default=None, 
                        help='Known true value (if not provided, will be estimated)')
    parser.add_argument('--n-episodes', type=int, default=100000,
                        help='Number of episodes to use for estimating true value')
    parser.add_argument('--output-dir', type=str, default='rmse_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--list-files', action='store_true',
                        help='List all NPZ files found in the directories')
    
    args = parser.parse_args()
    
    # Find the current directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Look for directories containing NPZ files
    npz_dirs = find_npz_directories(current_dir)
    
    if not npz_dirs:
        print("No directories with NPZ files found! Using current directory.")
        directory = current_dir
    else:
        print(f"Found {len(npz_dirs)} directories with NPZ files:")
        for i, d in enumerate(npz_dirs):
            print(f"  {i+1}. {d}")
        
        directory = npz_dirs  # Search in all found directories
    
    # List files if requested
    if args.list_files:
        print("\nListing all NPZ files in the directories:")
        for d in (npz_dirs if npz_dirs else [current_dir]):
            print(f"\nFiles in {d}:")
            npz_files = [f for f in os.listdir(d) if f.endswith('.npz')]
            for f in sorted(npz_files):
                print(f"  {f}")
    
    # Create output directory
    output_dir = os.path.join(current_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Get the true value
    if args.true_value is not None:
        # Use provided true value
        true_value = args.true_value
        print(f"Using provided true value: {true_value}")
    else:
        # Check if we already have the true value saved
        true_value_file = os.path.join(current_dir, "true_value_estimate.txt")
        if os.path.exists(true_value_file):
            try:
                with open(true_value_file, "r") as f:
                    lines = f.readlines()
                    true_value = float(lines[0].split(":")[1].strip())
                    print(f"Using existing true value: {true_value}")
            except Exception as e:
                print(f"Error reading true value file: {e}")
                print("Estimating new true value...")
                true_value, std_error = estimate_true_value(n_episodes=args.n_episodes)
        else:
            # Estimate the true value
            true_value, std_error = estimate_true_value(n_episodes=args.n_episodes)
            
            # Save the result
            with open(os.path.join(output_dir, "true_value_estimate.txt"), "w") as f:
                f.write(f"Estimated true value: {true_value}\n")
                f.write(f"Standard error: {std_error}\n")
                f.write(f"Parameters: alpha=0.9, beta=0.2, T=30, episodes={args.n_episodes}\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 2: Load all estimator results
    all_estimators = load_estimator_results(directory, args.sample_sizes)
    
    # Step 3: Calculate RMSE for all estimators
    rmse_table, se_table, bias_table, combined_results = create_rmse_table(all_estimators, true_value)
    
    # Check if we have any results
    if rmse_table.empty:
        print("\nNo valid results were found to analyze. Please check your file paths and naming conventions.")
        return
    
    # Step 4: Display results
    print("\nRMSE Results:")
    print(rmse_table)
    
    print("\nStandard Errors:")
    print(se_table)
    
    print("\nBias Values:")
    print(bias_table)
    
    # Step 5: Save results to CSV
    rmse_table.to_csv(os.path.join(output_dir, "rmse_table.csv"))
    se_table.to_csv(os.path.join(output_dir, "se_table.csv"))
    bias_table.to_csv(os.path.join(output_dir, "bias_table.csv"))
    combined_results.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)
    
    # Step 6: Create LaTeX table for publication
    create_latex_table(rmse_table, se_table, os.path.join(output_dir, "rmse_table.tex"))
    
    # Step 7: Create visualizations
    create_visualizations(rmse_table, se_table, combined_results, output_dir)
    
    print(f"\nAll analysis results saved to {output_dir}/")

if __name__ == "__main__":
    main()