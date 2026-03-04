"""
NetFill / NetSleuth baseline algorithms for seed node identification.
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def compute_laplacian(adj_matrix):
    """
    Computes the Combinatorial Graph Laplacian matrix L = D - A using sparse arithmetic.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): The sparse adjacency matrix of the graph (NxN).

    Returns:
        scipy.sparse.csr_matrix: The sparse Laplacian matrix L (NxN).
    """
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - adj_matrix
    return L


def get_eigen_centrality(sub_L):
    """
    Computes the eigen-centrality score for the infected subgraph.

    Args:
        sub_L (scipy.sparse.csr_matrix): The Laplacian sub-matrix corresponding
                                         to the current set of infected nodes.

    Returns:
        np.ndarray: A 1D array of scores (absolute values of the principal eigenvector).
    """
    N = sub_L.shape[0]
    if N <= 1:
        return np.ones(N)
    try:
        vals, vecs = eigsh(sub_L, k=1, sigma=1e-5, which='LM')
        smallest_eigenvector = vecs[:, 0]
        return np.abs(smallest_eigenvector)
    except Exception:
        return np.ones(N)


# ---------------------------------------------------------------------------
# MDL cost utilities
# ---------------------------------------------------------------------------

def log2_n_choose_k(n, k):
    """
    Compute log2 of n choose k: log2(C(n, k)).
    Uses gammaln to avoid overflow with large n.
    """
    if k < 0 or k > n:
        return np.inf
    if k == 0 or k == n:
        return 0.0
    ln_val = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return ln_val / np.log(2)


def L_N(n):
    """
    Universal code for integers (L_N) cost in bits.
    Exact implementation per NetSleuth paper Eq (cite: 91).
    """
    if n <= 0:
        return 0.0
    cost = 1.51857
    current_val = n
    while current_val > 0:
        log_val = np.log2(current_val)
        if log_val <= 1e-9:
            break
        cost += log_val
        current_val = log_val
    return cost


def calculate_negative_log_likelihood(m_d, f_d, p_d):
    """
    Calculates the encoding cost for a specific degree group 'd'
    EXACTLY according to Equation (2) in the NetSleuth paper.

    Arguments:
        m_d (int): Number of infected nodes (actual).
        f_d (int): Total number of nodes in frontier group (trials).
        p_d (float): Infection probability.

    Returns:
        float: Cost in bits.
    """
    p_d = np.clip(p_d, 1e-9, 1.0 - 1e-9)

    log_binom = log2_n_choose_k(f_d, m_d)
    log_p_success = m_d * np.log2(p_d)
    log_p_fail = (f_d - m_d) * np.log2(1.0 - p_d)
    term_A = log_binom + log_p_success + log_p_fail

    term_B = 0.0
    if f_d > 0:
        ratio = m_d / f_d
        if m_d > 0:
            term_B += m_d * np.log2(ratio)
        if m_d < f_d:
            term_B += (f_d - m_d) * np.log2(1.0 - ratio)

    return -(term_A + term_B)


def calculate_mdl_cost(current_seeds, observed_infected, adj_matrix,
                       missing_nodes=None, false_positive_nodes=None, beta=0.1):
    """
    Calculates MDL cost L(D, S, R) according to NetSleuth/NetFill.
    Uses Greedy Likelihood Ripple optimization (Section IV-E of NetSleuth paper).
    """
    N = adj_matrix.shape[0]
    S = np.unique(current_seeds)
    D = np.unique(observed_infected)
    C_minus = np.unique(missing_nodes) if missing_nodes is not None else np.array([], dtype=int)
    C_plus = np.unique(false_positive_nodes) if false_positive_nodes is not None else np.array([], dtype=int)

    I_union = np.union1d(D, C_minus)
    I_true = np.setdiff1d(I_union, C_plus)

    num_S, num_I = len(S), len(I_true)
    if num_S == 0:
        return np.inf

    cost_seed = L_N(num_S) + log2_n_choose_k(N, num_S)

    infected_set = set(S)
    cost_R = 0.0
    time_steps_T = 0

    while len(infected_set) < num_I:
        current_infected = list(infected_set)
        all_neighbors = np.unique(adj_matrix[current_infected].indices)
        F = np.setdiff1d(all_neighbors, current_infected)

        if F.size == 0:
            break

        d_vector = np.array(adj_matrix[current_infected][:, F].sum(axis=0)).flatten()
        valid_candidates_mask = np.isin(F, I_true)
        unique_d = np.unique(d_vector)
        newly_infected = []

        for d in unique_d:
            if d == 0:
                continue
            d_mask = (d_vector == d)
            f_d = np.sum(d_mask)
            candidates_d = F[d_mask & valid_candidates_mask]
            k_available = len(candidates_d)

            p_d = max(1e-9, 1 - (1 - beta)**d)
            if p_d >= 1:
                p_d = 1.0 - 1e-9

            mode_val = int(np.floor((f_d / beta + 1) * p_d))
            m_d = min(k_available, mode_val)

            if k_available > 0 and m_d == 0:
                m_d = 1

            if m_d > 0:
                chosen_indices = np.random.choice(candidates_d, m_d, replace=False)
                newly_infected.extend(chosen_indices)

            cost_R += calculate_negative_log_likelihood(m_d, f_d, p_d)

        if not newly_infected:
            break
        else:
            infected_set.update(newly_infected)
            time_steps_T += 1

    cost_R += L_N(time_steps_T)

    num_covered = len(infected_set)
    num_missed = num_I - num_covered
    cost_Unexplained = 0.0
    if num_missed > 0:
        cost_Unexplained = L_N(num_missed) + log2_n_choose_k(N - num_covered, num_missed)

    cost_Noise = 0.0
    if len(C_minus) > 0:
        cost_Noise += L_N(len(C_minus)) + log2_n_choose_k(num_I, len(C_minus))
    if len(C_plus) > 0:
        cost_Noise += L_N(len(C_plus)) + log2_n_choose_k(N - num_I, len(C_plus))

    return cost_seed + cost_R + cost_Noise + cost_Unexplained


# ---------------------------------------------------------------------------
# NetSleuth
# ---------------------------------------------------------------------------

def netsleuth(adj_matrix, infected_indices):
    """
    Task A: NetSleuth - A heuristic algorithm for seed identification.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): The global sparse adjacency matrix (NxN).
        infected_indices (np.ndarray): A 1D numpy array containing the indices of
                                       all currently considered infected nodes (Observed + Missing).

    Returns:
        set: A set of integer indices representing the identified seed nodes.
    """
    if not isinstance(adj_matrix, sp.csr_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)

    L_G = compute_laplacian(adj_matrix)
    S = set()
    current_nodes = np.array(infected_indices, dtype=int)
    previous_cost = calculate_mdl_cost(list(S), infected_indices, adj_matrix)

    while current_nodes.size > 0:
        L_A = L_G[current_nodes, :][:, current_nodes]
        eigen_centrality = get_eigen_centrality(L_A)
        local_idx = np.argmax(eigen_centrality)
        next_node = current_nodes[local_idx]

        temp_S = S.union({next_node})
        new_cost = calculate_mdl_cost(list(temp_S), infected_indices, adj_matrix)

        if new_cost >= previous_cost:
            break

        S.add(next_node)
        previous_cost = new_cost
        current_nodes = np.delete(current_nodes, local_idx)

    return S


# ---------------------------------------------------------------------------
# NetFill helpers
# ---------------------------------------------------------------------------

def find_seeds(adj_matrix, D, C_minus):
    """
    Task A: Identifies the most likely seed nodes given the currently estimated
    set of all infected nodes (observed + missing).

    Args:
        adj_matrix (scipy.sparse.csr_matrix): The sparse adjacency matrix of the graph (NxN).
        D (np.ndarray): 1D array of indices representing observed infected nodes.
        C_minus (np.ndarray): 1D array of indices representing estimated missing infected nodes.

    Returns:
        set: A set of integer indices representing the identified seed nodes.
    """
    I_current = np.union1d(D, C_minus)
    S = netsleuth(adj_matrix, I_current)
    return S


def get_frontier_set(adj_matrix, D_indices):
    """
    Identifies the frontier set of the observed infected nodes (D).

    Args:
        adj_matrix (scipy.sparse.csr_matrix): The sparse adjacency matrix of the graph (NxN).
        D_indices (np.ndarray): A 1D numpy array containing indices of observed infected nodes.

    Returns:
        np.ndarray: A 1D numpy array containing the sorted indices of the frontier nodes.
    """
    if D_indices.size == 0:
        return np.array([], dtype=int)

    sub_adj = adj_matrix[D_indices, :]
    all_neighbors = np.unique(sub_adj.indices)
    frontier_indices = np.setdiff1d(all_neighbors, D_indices)
    return frontier_indices


def find_missing_nodes(adj_matrix, S, D, C_prev, beta):
    """
    Implements Task B: Find missing nodes given seeds (Algorithm 1 & 2 wrapper).
    Strictly follows the NetFill paper logic.

    Args:
        adj_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix (NxN).
        S (list/set): Current set of seed nodes.
        D (np.ndarray): Observed infected nodes.
        C_prev (np.ndarray): Previously estimated missing nodes.
        beta (float): Infection probability (for MDL calculation).

    Returns:
        np.ndarray: Updated array of missing nodes (C_best).
    """
    C_best = np.array([], dtype=int)
    BATCH_SIZE = 10

    current_mdl = calculate_mdl_cost(list(S), D, adj_matrix, missing_nodes=C_best, beta=beta)
    scores = find_node_scores(adj_matrix, D, C_prev, list(S))

    mask = np.ones(adj_matrix.shape[0], dtype=bool)
    mask[D] = False

    candidate_indices = np.where((scores > 1e-9) & mask)[0]
    sorted_candidates = candidate_indices[np.argsort(-scores[candidate_indices])]

    idx = 0
    while idx < len(sorted_candidates):
        next_batch = sorted_candidates[idx: idx + BATCH_SIZE]
        if len(next_batch) == 0:
            break

        C_trial = np.union1d(C_best, next_batch)
        trial_mdl = calculate_mdl_cost(list(S), D, adj_matrix, missing_nodes=C_trial, beta=beta)

        if trial_mdl < current_mdl:
            C_best = C_trial
            current_mdl = trial_mdl
            idx += BATCH_SIZE
        else:
            break

    return C_best


def find_node_scores(adj_matrix, D, C_prev, S):
    """
    Implements Function FINDNODESCORES from Algorithm 1 [cite: 779-793].
    Calculates the appropriateness of adding a node to C- based on eigenvector centrality
    relative to the seeds, using 'Exoneration' to handle multiple seeds.

    Args:
        adj_matrix: Global adjacency matrix.
        D: Observed infected nodes.
        C_prev: Previously estimated missing nodes.
        S: List of seed nodes.

    Returns:
        np.ndarray: Z-scores for all nodes in the graph.
    """
    N = adj_matrix.shape[0]
    if isinstance(S, set):
        S = list(S)

    Z_scores_list = []
    seeds_loop = S if len(S) > 0 else [None]
    current_ignored_seeds = []

    for i in range(len(seeds_loop)):
        current_I_nodes = np.union1d(D, C_prev)

        if len(current_ignored_seeds) > 0:
            current_I_nodes = np.setdiff1d(current_I_nodes, current_ignored_seeds)

        if seeds_loop[i] is not None:
            current_ignored_seeds.append(seeds_loop[i])

        if current_I_nodes.size <= 1:
            Z_scores_list.append(np.zeros(N))
            continue

        sub_adj = adj_matrix[current_I_nodes, :][:, current_I_nodes]
        degrees = np.array(sub_adj.sum(axis=1)).flatten()
        sub_D = sp.diags(degrees)
        sub_L = sub_D - sub_adj

        try:
            vals, vecs = eigsh(sub_L, k=1, sigma=1e-5, which='LM')
            u_i_sub = np.abs(vecs[:, 0])
        except Exception:
            u_i_sub = np.zeros(current_I_nodes.size)

        u_i_full = np.zeros(N)
        u_i_full[current_I_nodes] = u_i_sub

        if len(C_prev) > 0:
            u_i_full[C_prev] = 0.0

        Z_n_i = adj_matrix.dot(u_i_full)
        Z_scores_list.append(Z_n_i)

    if not Z_scores_list:
        return np.zeros(N)

    final_scores = np.max(np.array(Z_scores_list), axis=0)
    return final_scores


# ---------------------------------------------------------------------------
# NetFill main algorithm
# ---------------------------------------------------------------------------

def netfill(adj_matrix, y_obs_indices, beta):
    """
    Algorithm 2: NetFill
    
    Args:
        adj_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix (NxN).
        y_obs_indices (np.ndarray): 1D array of observed infected node indices (D).
        beta (float): Infectivity parameter.

    Returns:
        tuple:
            - nf_pred_final (np.ndarray): Binary vector (N,) where 1 indicates a Seed.
            - C_minus (np.ndarray): Array of indices of the missing nodes.
    """
    N = adj_matrix.shape[0]
    D = np.array(y_obs_indices, dtype=int)

    C_minus = get_frontier_set(adj_matrix, D)
    S = find_seeds(adj_matrix, D, C_minus)

    current_mdl = calculate_mdl_cost(list(S), D, adj_matrix, missing_nodes=C_minus)

    max_iter = 20

    for i in range(max_iter):
        prev_mdl = current_mdl
        S_prev = S.copy() if isinstance(S, set) else S
        C_prev_iter = C_minus.copy()

        S = find_seeds(adj_matrix, D, C_minus)

        C_input_for_finding = C_minus.copy()
        C_minus = find_missing_nodes(adj_matrix, S, D, C_input_for_finding, beta)

        new_mdl = calculate_mdl_cost(list(S), D, adj_matrix, missing_nodes=C_minus)

        print(f"Iter {i}: Old C size = {len(C_prev_iter)}, New C size = {len(C_minus)}")
        if new_mdl < prev_mdl:
            current_mdl = new_mdl
        else:
            S = S_prev
            C_minus = C_prev_iter
            break

    nf_pred_final = np.zeros(N)
    if len(S) > 0:
        seed_list = list(S) if isinstance(S, set) else S
        nf_pred_final[seed_list] = 1.0

    return nf_pred_final, C_minus
