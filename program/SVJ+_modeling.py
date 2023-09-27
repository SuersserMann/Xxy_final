import numpy as np
from scipy.stats import norm, gamma
from program.Function import *

pd.set_option('expand_frame_repr', False)

# Parameters
mu = 0.0506
r = 0.0201
d = 0.0174
kappa = 5.9859
theta = 0.0358
sigma = 0.5423
rho = -0.8015
_lambda = 0.9658
mu_z = -0.0209
sigma_z = 0.0677
eta = 0.0000
lambda_Q = 3.4969
mu_z_Q = -0.1767

T = 1  # Months to years
dt = T / 252  # Time step
n_paths = 25000
n_timesteps = 252
S0 = 100
Ks_call = [96, 98, 100, 102, 104, 106, 108]
Ks_put = [92, 94, 96, 98, 100, 102, 104]

# Initialize arrays
S_P = np.zeros((n_timesteps + 1, n_paths))
S_Q = np.zeros((n_timesteps + 1, n_paths))
V = np.zeros((n_timesteps + 1, n_paths))
S_P[0] = S0
S_Q[0] = S0

# Simulate paths
# np.random.seed(42)
for t in range(1, n_timesteps + 1):
    dW1_P = np.random.normal(0, np.sqrt(dt), n_paths)
    dW1_Q = rho * dW1_P + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), n_paths)
    dW2_P = np.random.normal(0, np.sqrt(dt), n_paths)
    dW2_Q = np.random.normal(0, np.sqrt(dt), n_paths)
    dNt_P = np.random.poisson(_lambda * dt, n_paths)
    dNt_Q = np.random.poisson(lambda_Q * dt, n_paths)
    Z_P = np.random.normal(mu_z, sigma_z, n_paths)
    Z_Q = np.random.normal(mu_z_Q, sigma_z, n_paths)
    mu_bar = np.exp(mu_z + 0.5 * sigma_z ** 2) - 1
    mu_bar_Q = np.exp(mu_z_Q + 0.5 * sigma_z ** 2) - 1

    dSt_P = (mu + r - d) * S_P[t - 1] * dt + S_P[t - 1] * np.sqrt(V[t - 1]) * dW1_P + (np.exp(Z_P) - 1) * S_P[t - 1] * dNt_P - _lambda * mu_bar * S_P[t - 1] * dt
    dSt_Q = (r - d) * S_Q[t - 1] * dt + S_Q[t - 1] * np.sqrt(V[t - 1]) * dW1_Q + (np.exp(Z_Q) - 1) * S_Q[t - 1] * dNt_Q - lambda_Q * mu_bar_Q * S_Q[t - 1] * dt
    dVt_P = kappa * (theta - V[t - 1]) * dt + sigma * np.sqrt(V[t - 1]) * dW2_P
    dVt_Q = (kappa * (theta - V[t - 1]) - eta * V[t - 1]) * dt + sigma * np.sqrt(V[t - 1]) * dW2_Q

    S_P[t] = S_P[t - 1] + dSt_P
    S_Q[t] = S_Q[t - 1] + dSt_Q
    V[t] = np.maximum(V[t - 1] + dVt_P, 0)  # Ensure variance remains non-negative

# Calculate expected gross returns
expected_gross_returns = []
for K in Ks_call:
    numerator = np.mean(np.maximum(S_P[-1] - K, 0))
    denominator = np.mean(np.maximum(S_Q[-1] - K, 0)) * np.exp(-r * T)
    expected_gross_return = numerator / denominator - 1
    expected_gross_returns.append(expected_gross_return)

# for K in Ks_put:
#     numerator = np.mean(np.maximum(-S_P[-1] + K, 0))
#     denominator = np.mean(np.maximum(-S_Q[-1] + K, 0)) * np.exp(-r * T)
#     print(numerator)
#     print(denominator)
#     expected_gross_return = numerator / denominator - 1
#     expected_gross_returns.append(expected_gross_return)

# Print results
for K, egr in zip(Ks_call, expected_gross_returns):
    print(f"Expected gross return for strike price {K}: {egr:.5f}")

# # Print results
# for K, egr in zip(Ks_put, expected_gross_returns):
#     print(f"Expected gross return for strike price {K}: {egr:.5f}")
