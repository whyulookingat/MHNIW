import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart
import random
from collections import defaultdict, Counter
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

def niw_posterior(data, mu0, kappa0, psi0, nu0):
    n = len(data)
    if n == 0:
        return mu0, psi0

    x_bar = np.mean(data, axis=0)
    S = np.cov(data.T) * (n - 1) if n > 1 else np.zeros((data.shape[1], data.shape[1]))
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    diff = (x_bar - mu0).reshape(-1, 1)
    psi_n = psi0 + S + (kappa0 * n / kappa_n) * (diff @ diff.T)

    # Sample Sigma ~ Inv-Wishart(psi_n, nu_n)
    Sigma = invwishart.rvs(df=nu_n, scale=psi_n)

    # Sample mu ~ N(mu_n, Sigma / kappa_n)
    mu = np.random.multivariate_normal(mu_n, Sigma / kappa_n)

    return mu, Sigma

def run_simulation(dataset, n_agents, d, T, sigma_sq=10.0, k=1.00):
    arm_keys = list(dataset.keys())
    k_arms = len(arm_keys)
    print(k_arms)

    if k_arms < 1:
        raise ValueError("The dataset must contain at least one arm.")

    S = int(np.ceil(np.log(n_agents)))
    mu0 = np.zeros(d)
    kappa0 = 1
            
    nu0 = d + 2
    psi0 = np.diag((np.ptp(np.vstack(list(dataset.values())), axis=0) / 2) ** 2 + 1e-6)

    buffers = {arm: [] for arm in arm_keys}
    est_mu = {arm: np.zeros(d) for arm in arm_keys}
    true_mu = {arm: np.mean(dataset[arm], axis=0) for arm in arm_keys}

    a_i = {}

    # Agent 1 (index 0) plays each arm once
    for arm in arm_keys:
        x = dataset[arm][np.random.randint(0, len(dataset[arm]))]
        buffers[arm].append(x)
    a_i[0] = arm_keys[-1]  # initialize to any arm (last one)

    # All other agents choose a random arm to start
    for i in range(1, n_agents):
        rand_arm = random.choice(arm_keys)
        a_i[i] = rand_arm
        x = dataset[rand_arm][np.random.randint(0, len(dataset[rand_arm]))]
        buffers[rand_arm].append(x)

    # Initial estimate from prior pulls
    for arm in arm_keys:
        if buffers[arm]:
            mu, _ = niw_posterior(np.array(buffers[arm]), mu0, kappa0, psi0, nu0)
            est_mu[arm] = mu

    arm_choices = np.zeros((n_agents, T), dtype=int)
    per_round_agent_regrets = np.zeros((n_agents, T))

    best_arm_true = max(true_mu, key=lambda arm: np.linalg.norm(true_mu[arm]))
    print(f"True best arm: {best_arm_true}")

    for t in range(T):
        print('round:', t)
        arm_counts = Counter(a_i.values())
        #best_arm = arm_counts.most_common(1)[0][0]
        best_arm = max(est_mu, key=lambda arm: np.linalg.norm(est_mu[arm]))
        
        for _ in range(S):
            for i in np.random.permutation(n_agents):
                j = random.choice(arm_keys)
                Ei = np.linalg.norm(est_mu[a_i[i]] - est_mu[best_arm])
                Ej = np.linalg.norm(est_mu[j] - est_mu[best_arm])
                diff = (1 / Ej**k) / (1 / Ei**k)
                print([len(buffers[arm]) for arm in arm_keys], len(buffers[best_arm_true]), diff, t+1)
                if random.random() < min(1, diff):
                    a_i[i] = j
                print(a_i[i])

        for i in range(n_agents):
            arm = a_i[i]
            x = dataset[arm][np.random.randint(0, len(dataset[arm]))]
            buffers[arm].append(x)
            arm_choices[i, t] = arm

        for arm in arm_keys:
            if buffers[arm]:
                mu, _ = niw_posterior(np.array(buffers[arm]), mu0, kappa0, psi0, nu0)
                est_mu[arm] = mu

        for i in range(n_agents):
            per_round_agent_regrets[i, t] = np.linalg.norm(true_mu[best_arm_true] - est_mu[a_i[i]])

    cumulative_regret_per_agent = np.cumsum(per_round_agent_regrets, axis=1)
    cumulative_regret_averaged_agents = np.mean(cumulative_regret_per_agent, axis=0)

    return cumulative_regret_averaged_agents, arm_choices



def plot_results(cumulative_regret_averaged_agents, arm_choices, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # Plot cumulative regret averaged over agents
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_regret_averaged_agents, color='blue')
    plt.title("Cumulative Regret (Averaged Agents)")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")

    # Arm choices heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(arm_choices, aspect='auto', cmap='tab20', interpolation='nearest')
    plt.title("Agent Arm Assignments")
    plt.xlabel("Round")
    plt.ylabel("Agent")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/averaged_cumulative_regret.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.txt", help="Path to the training dataset file")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=4000, help="Number of simulation rounds")
    args = parser.parse_args()

    dataset, d = load_dataset(args.data)
    regrets, arm_choices = run_simulation(dataset, n_agents=args.agents, d=d, T=args.rounds)
    plot_results(regrets, arm_choices)

