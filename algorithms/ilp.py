"""
ILP (Integer Linear Programming) solver for seed node identification.
"""

import math
import numpy as np
import torch
import pulp

from algorithms.bp import decide_node_state


def estimate_seed_sparsity(measurement_matrix, y_obs, beta, p_noise):
    """
    Estimates the prior probability (pi) of a node being a seed using the Method of Moments.

    Formula:
        Estimated_Seeds = Observed_Infected / ( (1 - p_noise) * (1 + beta * avg_degree) )
        pi = Estimated_Seeds / N

    Args:
        measurement_matrix (torch.sparse.Tensor): The measurement matrix (Phi = A + I).
        y_obs (torch.Tensor): Binary observation vector.
        beta (float): Infection rate.
        p_noise (float): False negative rate.

    Returns:
        float: Estimated sparsity pi (clamped between 1/N and 0.5).
    """
    N = y_obs.shape[0]

    indices = measurement_matrix.coalesce().indices()
    
    mask_non_self_loops = (indices[0] != indices[1])
    num_actual_edges = mask_non_self_loops.sum().item()
    
    avg_degree = num_actual_edges / N

    num_observed = y_obs.sum().item()
    if num_observed == 0:
        return 1.0 / N

    prob_observe = max(1 - p_noise, 1e-6)
    estimated_latent_infected = num_observed / prob_observe

    spread_factor = 1 + (beta * avg_degree)
    estimated_k = estimated_latent_infected / spread_factor
    pi_est = estimated_k / N

    return pi_est


def run_ILP(measurement_matrix, y_obs, beta, p_noise, time_limit=300):
    """
    Solves the Seed Identification problem using Integer Linear Programming (ILP).
    Automates the regularization weight (W_seed) estimation using Method of Moments.

    Args:
        measurement_matrix (torch.sparse.Tensor): The measurement matrix    .
        y_obs (torch.Tensor): Binary observation vector (size N).
        beta (float): Infection rate (0 < beta <= 1).
        p_noise (float): Probability of false negative (0 < p < 1).
        time_limit (int): Solver time limit in seconds.

    Returns:
        torch.Tensor: Binary prediction vector of size N.
    """
    device = y_obs.device
    N = y_obs.shape[0]

    adj_coo = measurement_matrix.coalesce().indices().cpu().numpy()
    rows = adj_coo[0]
    cols = adj_coo[1]

    y_numpy = y_obs.cpu().numpy().astype(int)
    pos_indices = np.where(y_numpy == 1)[0]
    neg_indices = np.where(y_numpy == 0)[0]

    pi_est = estimate_seed_sparsity(measurement_matrix, y_obs, beta, p_noise)
    print(f"   [ILP Info] Estimated seed sparsity (pi): {pi_est:.5f}")

    epsilon = 1e-9
    w_trans = -math.log(beta + epsilon)
    w_fail = -math.log(1 - beta + epsilon)
    w_noise = -math.log(p_noise + epsilon)
    w_seed = math.log((1 - pi_est) / pi_est)

    prob = pulp.LpProblem("Seed_Identification_MAP", pulp.LpMinimize)

    x_vars = {j: pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(N)}
    z_vars = {i: pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) for i in range(N)}

    edge_list = list(zip(rows, cols))
    edge_list_no_self = [(j, i) for j, i in edge_list if j != i]

    e_vars = {
        (j, i): pulp.LpVariable(f"e_{j}_{i}", cat=pulp.LpBinary)
        for j, i in edge_list_no_self
    }

    # Objective Function
    obj_prior = [w_seed * x_vars[j] for j in range(N)]

    cost_diff = w_trans - w_fail
    obj_trans = []
    for j, i in edge_list_no_self:
        obj_trans.append(w_fail * x_vars[j])
        obj_trans.append(cost_diff * e_vars[(j, i)])

    obj_noise = [w_noise * z_vars[i] for i in neg_indices]

    prob += pulp.lpSum(obj_prior + obj_trans + obj_noise)

    # Constraints
    for j, i in edge_list_no_self:
        prob += e_vars[(j, i)] <= x_vars[j]

    for i in range(N):
        prob += z_vars[i] >= x_vars[i]

    incoming_edges = {i: [] for i in range(N)}
    for j, i in edge_list_no_self:
        incoming_edges[i].append((j, i))

    for i in range(N):
        potential_causes = [e_vars[e] for e in incoming_edges[i]]
        potential_causes.append(x_vars[i])

        for cause in potential_causes:
            prob += z_vars[i] >= cause

        prob += z_vars[i] <= pulp.lpSum(potential_causes)

    for i in pos_indices:
        prob += z_vars[i] == 1

    # Solve
    solver = pulp.SCIP_PY(msg=True, timeLimit=time_limit)
    prob.solve(solver)

    x_pred_numpy = np.zeros(N, dtype=int)

    if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible', 'Not Solved']:
        try:
            x_values = [pulp.value(x_vars[j]) if pulp.value(x_vars[j]) is not None else 0.0 for j in range(N)]
            x_tensor = torch.tensor(x_values, dtype=torch.float32, device=device)
            return decide_node_state(x_tensor, threshold=0.5)
        except Exception as e:
            print(f"⚠️ Error extracting variables: {e}")
    else:
        print(f"⚠️ ILP Solver Failed: {pulp.LpStatus[prob.status]}")

    return torch.from_numpy(x_pred_numpy).float().to(device)
