"""
Belief Propagation (BP) algorithm for seed node identification
in Erdős-Rényi graphs via Group Testing.
"""

import time
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Likelihood Function
# ---------------------------------------------------------------------------

def compute_likelihood(Y_edges, prod_fail_to_infect_others, beta_edges, p_noise):
    prob_i_is_sick_given_j_is_healthy = 1.0 - prod_fail_to_infect_others
    prob_i_is_sick_given_j_is_infected = 1.0 - (prod_fail_to_infect_others * (1.0 - beta_edges))

    # Case 1: Y = 1
    i_think_j_is_normal_if_i_is_sick = (1.0 - p_noise) * prob_i_is_sick_given_j_is_healthy
    i_think_j_is_sick_if_i_is_sick   = (1.0 - p_noise) * prob_i_is_sick_given_j_is_infected

    # Case 2: Y = 0
    i_think_j_is_normal_if_i_is_normal = (p_noise * prob_i_is_sick_given_j_is_healthy) + \
                                         (1.0 * (1.0 - prob_i_is_sick_given_j_is_healthy))

    i_think_j_is_sick_if_i_is_normal   = (p_noise * prob_i_is_sick_given_j_is_infected) + \
                                         (1.0 * (1.0 - prob_i_is_sick_given_j_is_infected))

    # Selection based on observation Y
    likelihood_j_is_healthy = torch.where(
        Y_edges == 1,
        i_think_j_is_normal_if_i_is_sick,
        i_think_j_is_normal_if_i_is_normal
    )

    likelihood_j_is_infected = torch.where(
        Y_edges == 1,
        i_think_j_is_sick_if_i_is_sick,
        i_think_j_is_sick_if_i_is_normal
    )

    return likelihood_j_is_healthy, likelihood_j_is_infected


# ---------------------------------------------------------------------------
# BPDecoder
# ---------------------------------------------------------------------------

class BPDecoder(nn.Module):
    def __init__(self, adj_matrix, beta, p, prior_prob, max_iters=50, damping=0.5):
        super().__init__()
        self.adj = adj_matrix
        self.beta_val = beta
        self.p = p
        self.prior = prior_prob
        self.max_iters = max_iters
        self.damping = damping
        self.epsilon = 1e-10

        coo = self.adj.coalesce()
        self.idx_i = coo.indices()[0]
        self.idx_j = coo.indices()[1]

        self.num_edges = self.idx_i.shape[0]
        self.num_tests = self.adj.shape[0]
        self.num_patients = self.adj.shape[1]

        self.beta_edges = torch.where(
            self.idx_i == self.idx_j,
            torch.tensor(1.0, device=self.adj.device),
            torch.tensor(self.beta_val, device=self.adj.device)
        )

    def forward(self, Y_obs):
        device = self.adj.device
        Y_edges = Y_obs[self.idx_i]

        msg_j_to_i = torch.full((self.num_edges,), self.prior, device=device)
        old_msg_i_to_j0 = torch.zeros(self.num_edges, device=device)
        old_msg_i_to_j1 = torch.zeros(self.num_edges, device=device)

        node_beliefs = torch.full((self.num_patients,), self.prior, device=device)

        for it in range(self.max_iters):

            # Step 1: Variable -> Factor
            prob_infect_fail_edge = 1.0 - (self.beta_edges * msg_j_to_i)
            log_prob_fail = torch.log(prob_infect_fail_edge + self.epsilon)
            total_log_fail = torch.zeros(self.num_tests, device=device)
            total_log_fail.index_add_(0, self.idx_i, log_prob_fail)
            prod_fail_all = torch.exp(total_log_fail[self.idx_i])

            # Step 2: Factor -> Variable
            prod_fail_others = prod_fail_all / (prob_infect_fail_edge + self.epsilon)
            new_msg_i_to_j0, new_msg_i_to_j1 = compute_likelihood(
                Y_edges, prod_fail_others, self.beta_edges, self.p
            )

            # Damping
            if it > 0:
                msg_i_to_j0 = self.damping * new_msg_i_to_j0 + (1 - self.damping) * old_msg_i_to_j0
                msg_i_to_j1 = self.damping * new_msg_i_to_j1 + (1 - self.damping) * old_msg_i_to_j1
            else:
                msg_i_to_j0 = new_msg_i_to_j0
                msg_i_to_j1 = new_msg_i_to_j1

            old_msg_i_to_j0 = msg_i_to_j0.clone()
            old_msg_i_to_j1 = msg_i_to_j1.clone()

            norm = msg_i_to_j0 + msg_i_to_j1 + self.epsilon
            msg_i_to_j0 /= norm
            msg_i_to_j1 /= norm

            # Step 3: Variable Aggregation
            log_mi_to_j0 = torch.log(msg_i_to_j0 + self.epsilon)
            log_mi_to_j1 = torch.log(msg_i_to_j1 + self.epsilon)

            total_log_mi_to_j0 = torch.zeros(self.num_patients, device=device)
            total_log_mi_to_j1 = torch.zeros(self.num_patients, device=device)

            total_log_mi_to_j0.index_add_(0, self.idx_j, log_mi_to_j0)
            total_log_mi_to_j1.index_add_(0, self.idx_j, log_mi_to_j1)

            log_belief_0 = torch.log(torch.tensor(1.0 - self.prior)) + total_log_mi_to_j0
            log_belief_1 = torch.log(torch.tensor(self.prior)) + total_log_mi_to_j1

            # Step 4: Update Beliefs & Prepare Next Messages
            current_posterior = 1.0 / (1.0 + torch.exp(log_belief_0 - log_belief_1))
            node_beliefs = current_posterior.detach()

            log_msg_next_0 = log_belief_0[self.idx_j] - log_mi_to_j0
            log_msg_next_1 = log_belief_1[self.idx_j] - log_mi_to_j1

            msg_j_to_i = 1.0 / (1.0 + torch.exp(log_msg_next_0 - log_msg_next_1))

        return node_beliefs


# ---------------------------------------------------------------------------
# Decision Helper
# ---------------------------------------------------------------------------

def decide_node_state(posterior_probs, threshold=0.5):
    """
    Categorizes nodes based on posteriors.

    Returns:
        states (Tensor): 1 (Fixed Seed), 0 (Fixed Healthy)
    """
    states = torch.zeros_like(posterior_probs)
    states[posterior_probs >= threshold] = 1.0
    states[posterior_probs < threshold] = 0.0
    return states


# ---------------------------------------------------------------------------
# run_BP
# ---------------------------------------------------------------------------

def run_BP(measurement_matrix, Y_obs, beta, p_noise, prior):
    start_time = time.time()
    N = measurement_matrix.shape[0]
    device = measurement_matrix.device

    print(f"🚀 Starting BP  on {device}...")

    print(f"🔹  Running Final BP with Prior ={prior:.2f}...")
    bp_decoder = BPDecoder(measurement_matrix, beta, p_noise, prior, max_iters=100).to(device)
    bp_posteriors = bp_decoder(Y_obs)

    node_states = decide_node_state(bp_posteriors)
    num_seeds = (node_states == 1).sum().item()

    print(f"   [Decision] Selected {num_seeds} seeds.")

    total_seeds = (node_states == 1).sum().item()
    print(f">>> Total Seeds Found: {total_seeds}")

    return node_states


# ---------------------------------------------------------------------------
# estimate_prior_EM
# ---------------------------------------------------------------------------

def estimate_prior_EM(measurement_matrix, y_obs, beta, p_noise,
                      max_em_iters=20, tol=1e-4):
    """
    Estimate the prior infection probability pi via Expectation-Maximisation.

    Parameters
    ----------
    measurement_matrix : torch.sparse_coo_tensor – A + I on the target device
    y_obs              : torch.Tensor            – observed infection vector
    beta               : float                   – SI infection probability
    p_noise            : float                   – observation noise probability
    max_em_iters       : int                     – maximum EM iterations
    tol                : float                   – convergence tolerance on pi

    Returns
    -------
    current_pi : float – estimated prior probability of a node being a seed
    """
    device = measurement_matrix.device
    bp_model = BPDecoder(measurement_matrix, beta, p_noise, prior_prob=0.1, max_iters=10).to(device)

    y_mean = y_obs.float().mean().item()
    current_pi = torch.rand(1).item() * y_mean
    if current_pi == 0:
        current_pi = 0.01

    print(f"    EM prior estimation starting. Init pi: {current_pi:.5f}")

    for i in range(max_em_iters):
        prev_pi = current_pi

        bp_model.prior = current_pi
        current_beliefs = bp_model(y_obs)

        current_pi = current_beliefs.mean().item()
        current_pi = max(1e-6, min(1.0 - 1e-6, current_pi))

        diff = abs(current_pi - prev_pi)

        if diff < tol:
            print(f"    EM converged at iter {i + 1}. Optimal pi = {current_pi:.5f}")
            break

    return current_pi
