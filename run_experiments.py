"""
run_experiments.py
==================
Reusable experiment functions for seed node identification in Erdős-Rényi graphs.

Import this module from demo.ipynb (or any notebook) to run experiments:

    from run_experiments import run_single_experiment, run_benchmark, save_raw_data_to_csv

No code runs at import time — all logic is inside functions.
"""

import csv
import time

import numpy as np
import networkx as nx
import torch

from simulation.simulator import simulate_si_multistep, get_theta
from algorithms.bp import estimate_prior_EM, run_BP
from algorithms.baseline import netfill
from utils.metrics import extract_metrics
from algorithms.ilp import run_ILP
import traceback

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def _generate_data(N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE, device):
    """
    Build an Erdős-Rényi graph and run one SI simulation step.

    Parameters
    ----------
    N          : int   – number of nodes
    NUM_SEEDS  : int   – number of seed (initially infected) nodes
    BETA       : float – SI infection probability per step
    P_NOISE    : float – observation noise probability
    AVG_DEGREE : float – target average degree for the ER graph
    device     : torch.device – target device for tensors

    Returns
    -------
    adj_csr          : scipy.sparse.csr_matrix  – adjacency matrix (sparse)
    adjacency_matrix : torch.sparse_coo_tensor  – adjacency on `device`
    measurement_matrix : torch.sparse_coo_tensor – A + I on `device`
    x_true           : torch.Tensor  – ground-truth infection vector at t=0
    y_obs            : torch.Tensor  – noisy observation vector
    """
    # --- Graph ---------------------------------------------------------------
    theta = get_theta(N, avg_degree=AVG_DEGREE)
    er_graph = nx.erdos_renyi_graph(n=N, p=theta)
    adj_csr = nx.adjacency_matrix(er_graph).tocsr()

    # --- Sparse adjacency tensor (A) -----------------------------------------
    adj_coo = adj_csr.tocoo()
    indices_a = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long()
    values_a  = torch.from_numpy(adj_coo.data).float()
    adjacency_matrix = (
        torch.sparse_coo_tensor(indices_a, values_a, (N, N))
        .to(device)
        .coalesce()
    )

    # --- Measurement matrix (A + I) ------------------------------------------
    indices_diag = torch.arange(N).unsqueeze(0).repeat(2, 1)
    values_diag  = torch.ones(N)
    indices_m    = torch.cat((indices_a, indices_diag), dim=1)
    values_m     = torch.cat((values_a,  values_diag))
    measurement_matrix = (
        torch.sparse_coo_tensor(indices_m, values_m, (N, N))
        .to(device)
        .coalesce()
    )

    # --- SI simulation -------------------------------------------------------
    seeds = np.random.choice(range(N), size=NUM_SEEDS, replace=False)
    history, y_obs = simulate_si_multistep(
        adjacency_matrix, seeds, BETA, T=1, p_noise=P_NOISE
    )
    x_true = history[0]

    return adj_csr, adjacency_matrix, measurement_matrix, x_true, y_obs


# ---------------------------------------------------------------------------
# Single Experiment
# ---------------------------------------------------------------------------

def run_single_experiment(N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE):
    """
    Run one trial of the benchmark and return metrics + runtimes.

    Parameters
    ----------
    N          : int   – number of nodes in the ER graph
    NUM_SEEDS  : int   – true number of seed nodes
    BETA       : float – SI infection probability
    P_NOISE    : float – observation noise probability
    AVG_DEGREE : float – average node degree for the ER graph

    Returns
    -------
    metrics : dict  {'NetFill': {...}, 'BP': {...}, 'ILP': {...}}
    times   : dict  {'NetFill': float, 'BP': float}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BENCHMARK STARTING")
    print(f"  Device     : {device}")
    print(f"  Nodes      : {N} | Seeds: {NUM_SEEDS} | Beta: {BETA} | Noise: {P_NOISE}")
    print(f"  Avg Degree : {AVG_DEGREE}")
    print("-" * 60)

    # --------------------------------------------------
    # 1. Data Generation
    # --------------------------------------------------
    print("[1] Generating data (ER graph + SI simulation)...")
    adj_csr, adjacency_matrix, measurement_matrix, x_true, y_obs = \
        _generate_data(N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE, device)
    print(f"    Data ready. Observed infected nodes: {int(y_obs.sum().item())}")
    print("-" * 60)

    # --------------------------------------------------
    # 2. Belief Propagation (BP)
    # --------------------------------------------------
    print("\n[2] Running algorithm: Belief Propagation (BP)...")
    t_start_bp = time.perf_counter()
    bp_pred_final = torch.zeros(N).to(device)

    try:
        optimal_pi = estimate_prior_EM(
            measurement_matrix, y_obs, BETA, P_NOISE,
            max_em_iters=20, tol=1e-5
        )
        bp_pred_final = run_BP(measurement_matrix, y_obs, BETA, P_NOISE, prior=optimal_pi)
        print(f"    Optimal pi: {optimal_pi:.5f}")
    except Exception as e:
        print(f"    BP failed: {e}")
        traceback.print_exc()

    t_bp = time.perf_counter() - t_start_bp
    print(f"   >> BP Finished in {t_bp:.4f}s")

    # --------------------------------------------------
    # 3. NetFill
    # --------------------------------------------------
    print("\n[3] Running algorithm: NetFill...")
    t_start_nf = time.perf_counter()
    nf_pred_final = torch.zeros(N).to(device)

    try:
        y_obs_numpy = y_obs.cpu().numpy().astype(int)
        y_obs_indices = np.where(y_obs_numpy == 1)[0]
        nf_pred_numpy, _ = netfill(adj_csr, y_obs_indices, beta=BETA)
        nf_pred_final = torch.from_numpy(nf_pred_numpy).float().to(device)
    except Exception as e:
        print(f"    NetFill failed: {e}")
        traceback.print_exc()

    t_nf = time.perf_counter() - t_start_nf
    print(f"   >> NETFILL Finished in {t_nf:.4f}s")

    # --------------------------------------------------
    # 4. ILP
    # --------------------------------------------------
    print("\n[4] Running algorithm: ILP (MAP estimation)...")
    t_start_ilp = time.perf_counter()
    try:
        ilp_pred_final = run_ILP(measurement_matrix, y_obs, BETA, P_NOISE, time_limit=120)
    except Exception as e:
        print(f"    ILP failed: {e}")
        ilp_pred_final = torch.zeros(N).to(device)
        traceback.print_exc()
    t_ilp = time.perf_counter() - t_start_ilp
    print(f"   >> ILP Finished in {t_ilp:.4f}s")

    # --------------------------------------------------
    # Collect results
    # --------------------------------------------------
    y_true_np    = x_true.cpu().numpy()
    bp_pred_np   = bp_pred_final.cpu().numpy()
    nf_pred_np   = nf_pred_final.cpu().numpy()
    ilp_pred_np  = ilp_pred_final.cpu().numpy()

    metrics = {
        'NetFill': extract_metrics(y_true_np, nf_pred_np),
        'BP':      extract_metrics(y_true_np, bp_pred_np),
        'ILP':     extract_metrics(y_true_np, ilp_pred_np),
    }
    times = {
        'NetFill': t_nf,
        'BP':      t_bp,
        'ILP':   t_ilp,  
    }
    return metrics, times


# ---------------------------------------------------------------------------
# Multi-run Benchmark Loop (convenience wrapper for notebooks)
# ---------------------------------------------------------------------------

def run_benchmark(N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE, num_runs=5,
                  csv_filename="raw.csv", algorithms=('NetFill', 'BP')):
    """
    Run the experiment `num_runs` times and aggregate results into a list
    of row-dicts ready for save_raw_data_to_csv() or pandas.

    Parameters
    ----------
    N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE : see run_single_experiment()
    num_runs      : int  – number of independent trials
    csv_filename  : str  – path to save the CSV; set to None to skip saving
    algorithms    : list – which algorithm keys to collect ('NetFill', 'BP', 'ILP')

    Returns
    -------
    raw_data_list : list of dicts (Run, Algorithm, Precision, Recall, F1-Score, Runtime)
    """
    raw_data_list = []

    for run in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run} / {num_runs}")
        print(f"{'='*60}")
        metrics, times = run_single_experiment(N, NUM_SEEDS, BETA, P_NOISE, AVG_DEGREE)

        for algo in algorithms:
            if algo not in metrics:
                continue
            row = {
                'Run':       run,
                'Algorithm': algo,
                'Precision': metrics[algo]['Precision'],
                'Recall':    metrics[algo]['Recall'],
                'F1-Score':  metrics[algo]['F1-Score'],
                'Runtime':   times.get(algo, float('nan')),
            }
            raw_data_list.append(row)

    if csv_filename:
        save_raw_data_to_csv(raw_data_list, filename=csv_filename)

    return raw_data_list


# ---------------------------------------------------------------------------
# CSV Serialisation
# ---------------------------------------------------------------------------

def save_raw_data_to_csv(all_results, filename="raw.csv"):
    """Write a list of result dicts to a CSV file."""
    fieldnames = ['Run', 'Algorithm', 'Precision', 'Recall', 'F1-Score', 'Runtime']
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"💾 Results saved → {filename}")
