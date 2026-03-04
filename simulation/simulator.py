"""
SI (Susceptible-Infected) epidemic simulation utilities.
"""

import torch


def simulate_si_multistep(adj, seeds_idx, beta, T, p_noise):
    """
    Simulates the propagation of the SI (Susceptible-Infected) epidemic model
    over a graph for T time steps and generates a noisy observation via a Z-Channel.

    Parameters
    ----------
    adj : torch.Tensor
        The adjacency matrix of the graph. Shape: (N, N).
    seeds_idx : list or np.array or torch.Tensor
        Indices of the initial source nodes (seeds) infected at t=0.
    beta : float
        Infection probability per infected neighbor per step.
    T : int
        Number of time steps to simulate.
    p_noise : float
        Z-Channel noise.

    Returns
    -------
    Xt_history : list of torch.Tensor
        List containing infection states from t=0 to t=T.
    Y_observed : torch.Tensor
        Noisy observation of the final state at t=T.
    """
    # 1. Initialize state at t=0
    Xt = torch.zeros(adj.shape[0], device=adj.device)
    Xt[seeds_idx] = 1.0

    Xt_history = [Xt.clone()]

    # 2. Loop through T steps
    for step in range(T):
        neighbor_infected_counts = torch.mv(adj, Xt)
        prob_new_infection = 1 - (1 - beta)**neighbor_infected_counts
        prob_combined = torch.clamp(prob_new_infection + Xt, max=1.0)
        Xt = torch.bernoulli(prob_combined)
        Xt_history.append(Xt.clone())

    # 3. Generate observation noise
    observation_prob = Xt * (1 - p_noise)
    Y_observed = torch.bernoulli(observation_prob)

    return Xt_history, Y_observed


def get_theta(N, avg_degree):
    """Return the edge probability for an Erdős-Rényi graph with given average degree."""
    return avg_degree / (N - 1)
