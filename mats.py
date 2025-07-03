import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart
import random
from collections import defaultdict
import os


def find_max_dim(filepath):
    max_dim = -1
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            for p in parts[2:]:
                i, _ = p.split(":")
                dim = int(i)
                if dim > max_dim:
                    max_dim = dim
    return max_dim


def load_dataset(filepath):
    max_dim = find_max_dim(filepath)
    d = max_dim + 1
    arms = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            arm = int(parts[1].split(":")[1])
            vector = np.zeros(d)
            for p in parts[2:]:
                i, v = p.split(":")
                vector[int(i)] = float(v)
            arms[arm].append(vector)
    return {k: np.array(v) for k, v in arms.items()}, d


def niw_sample(data, mu0, kappa0, psi0, nu0):
    n = len(data)
    if n == 0:
        Sigma = invwishart.rvs(df=nu0, scale=psi0)
        mu = np.random.multivariate_normal(mu0, Sigma / kappa0)
        return mu, Sigma

    x_bar = np.mean(data, axis=0)
    S = np.cov(data.T) * (n - 1) if n > 1 else np.zeros((data.shape[1], data.shape[1]))
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    diff = (x_bar - mu0).reshape(-1, 1)
    psi_n = psi0 + S + (kappa0 * n / kappa_n) * (diff @ diff.T)

    Sigma = invwishart.rvs(df=nu_n, scale=psi_n)
    mu = np.random.multivariate_normal(mu_n, Sigma / kappa_n)
    return mu, Sigma


def build_chain_coordination_graph(n_groups):
    graph = {e: [] for e in range(n_groups)}
    for e in range(n_groups):
        if e > 0:
            graph[e].append(e - 1)
        if e < n_groups - 1:
            graph[e].append(e + 1)
    return graph


def variable_elimination_chain(mu_hat, arms, graph):
    n_agents = len(mu_hat)
    messages = [{} for _ in range(n_agents)]
    backtrack = [{} for _ in range(n_agents)]

    for i in reversed(range(n_agents)):
        for a_i in arms:
            local_reward = np.linalg.norm(mu_hat[i][a_i])  # Convert vector to scalar
            if i == n_agents - 1:
                messages[i][a_i] = local_reward
            else:
                best_score = -np.inf
                best_a_next = None
                for a_next in arms:
                    score = local_reward + messages[i + 1][a_next]
                    if score > best_score:
                        best_score = score
                        best_a_next = a_next
                messages[i][a_i] = best_score
                backtrack[i][a_i] = best_a_next

    joint_action = []
    a_i = max(messages[0], key=messages[0].get)
    joint_action.append(a_i)
    for i in range(1, n_agents):
        a_i = backtrack[i - 1][a_i]
        joint_action.append(a_i)

    return joint_action


def run_hd_mats_with_elimination(dataset, n_groups, d, T, threshold=3):
    arm_keys = list(dataset.keys())
    k = len(arm_keys)
    mu0 = np.zeros(d)
    kappa0 = 1.0
    nu0 = d + 2
    psi0 = np.diag((np.ptp(np.vstack(list(dataset.values())), axis=0) / 2) ** 2 + 1e-6)

    group_buffers = [{arm: [] for arm in arm_keys} for _ in range(n_groups)]
    true_mu = {arm: np.mean(dataset[arm], axis=0) for arm in arm_keys}

    best_arm_true = max(true_mu, key=lambda a: np.linalg.norm(true_mu[a]))
    arm_choices = np.zeros((n_groups, T), dtype=int)
    per_round_group_regrets = np.zeros((n_groups, T))
    graph = build_chain_coordination_graph(n_groups)

    for t in range(T):
        print("rounds", t)
        mu_hat = [{} for _ in range(n_groups)]
        useful_agents = [e for e in range(n_groups) if sum(len(group_buffers[e][a]) for a in arm_keys) >= threshold]
        if not useful_agents:
            useful_agents = list(range(n_groups))

        mu_hat_filtered = [{} for _ in useful_agents]
        for idx, e in enumerate(useful_agents):
            for arm in arm_keys:
                data = np.array(group_buffers[e][arm])
                mu, _ = niw_sample(data, mu0, kappa0, psi0, nu0)
                mu_hat_filtered[idx][arm] = mu

        subgraph = {e: [n for n in graph[e] if n in useful_agents] for e in useful_agents}
        joint_action = variable_elimination_chain(mu_hat_filtered, arm_keys, subgraph)

        for e in range(n_groups):
            if e in joint_action:
                a = joint_action[e]
            else:
                a = random.choice(arm_keys)
            x = dataset[a][np.random.randint(0, len(dataset[a]))]
            group_buffers[e][a].append(x)
            arm_choices[e, t] = a
            print(np.sum(arm_choices == a), t)
            mu_est = np.mean(np.array(group_buffers[e][a]), axis=0)
            per_round_group_regrets[e, t] = np.linalg.norm(true_mu[best_arm_true] - mu_est)

    cumulative_regret = np.cumsum(per_round_group_regrets, axis=1)
    avg_regret = np.mean(cumulative_regret, axis=0)
    return avg_regret, arm_choices


def plot_results(cumulative_regret_averaged_agents, arm_choices, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(cumulative_regret_averaged_agents, color='blue')
    plt.title("Cumulative Regret (Averaged Groups)")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")

    plt.subplot(1, 2, 2)
    plt.imshow(arm_choices, aspect='auto', cmap='tab20', interpolation='nearest')
    plt.title("Group Arm Assignments")
    plt.xlabel("Round")
    plt.ylabel("Group")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/averaged_cumulative_regret.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.txt", help="Path to dataset")
    parser.add_argument("--agents", type=int, default=10, help="Number of groups (agents)")
    parser.add_argument("--rounds", type=int, default=5000, help="Simulation rounds")
    args = parser.parse_args()

    dataset, d = load_dataset(args.data)
    regrets, choices = run_hd_mats_with_elimination(dataset, n_groups=args.agents, d=d, T=args.rounds)
    plot_results(regrets, choices)