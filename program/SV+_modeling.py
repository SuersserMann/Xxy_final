import numpy as np
import matplotlib.pyplot as plt


def terminal_values(S0, r, d, mu, eta, kappa, theta, sigma, rho, T, dt, num_paths, risk_neutral=False):
    num_steps = int(T / dt)

    Z1 = np.random.normal(size=(num_steps, num_paths))
    Z2 = np.random.normal(size=(num_steps, num_paths))
    W1 = np.sqrt(dt) * Z1
    W2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

    S = np.zeros((num_steps + 1, num_paths))
    V = np.zeros((num_steps + 1, num_paths))
    S[0] = S0
    V[0] = theta

    for t in range(1, num_steps + 1):
        if risk_neutral:
            drift = (r - d) * S[t - 1]
            dVt = (kappa * (theta - V[t - 1]) - eta * V[t - 1]) * dt + sigma * np.sqrt(V[t - 1]) * W2[t - 1]
        else:
            drift = (mu + r - d) * S[t - 1]
            dVt = kappa * (theta - V[t - 1]) * dt + sigma * np.sqrt(V[t - 1]) * W2[t - 1]

        dSt = drift * dt + np.sqrt(V[t - 1]) * S[t - 1] * W1[t - 1]
        S[t] = S[t - 1] + dSt
        V[t] = np.maximum(V[t - 1] + dVt, 0)  # Ensure variance is non-negative
        # V[t] = V[t - 1] + dVt

    return S[-1]


# Model parameters
S0 = 100
mu = 0.0506
r = 0.0201
d = 0.014  # 0.0174
eta = -5.1470 * 10/10  # -4.3470
kappa = 0.1054
theta = 0.0363
sigma = 0.5472
rho = -0.7944
T = 1
dt = 1 / 252
num_paths = 25000
Ks_call = [94, 96, 98, 100, 102, 104, 106, 108]
Ks_put = [92, 94, 96, 98, 100, 102, 104, 106]

# Simulate option prices under the physical measure (P)
# np.random.seed(0)
terminal_values_P = terminal_values(S0, r, d, mu, eta, kappa, theta, sigma, rho, T, dt, num_paths, risk_neutral=False)

# Simulate option prices under the risk-neutral measure (Q)
# np.random.seed(0)
terminal_values_Q = terminal_values(S0, r, d, mu, eta, kappa, theta, sigma, rho, T, dt, num_paths, risk_neutral=True)

# Calculate option returns
mean_option_returns = []
mid_option_returns = []

for K in Ks_call:
    payoff_P = np.maximum(terminal_values_P - K, 0)
    payoff_Q = np.maximum(terminal_values_Q - K, 0)

    expected_payoff_P = np.mean(payoff_P)
    print(expected_payoff_P)
    option_price_Q = np.exp(-r * T) * np.mean(payoff_Q)

    mean_option_return = expected_payoff_P / option_price_Q - 1
    mean_option_returns.append(mean_option_return)
# for K in Ks_put:
#     payoff_P = np.maximum(-terminal_values_P + K, 0)
#     payoff_Q = np.maximum(-terminal_values_Q + K, 0)
#
#     expected_payoff_P = np.mean(payoff_P)
#     print(expected_payoff_P)
#     option_price_Q = np.exp(-r * T) * np.mean(payoff_Q)
#
#     mean_option_return = expected_payoff_P / option_price_Q - 1
#     mean_option_returns.append(mean_option_return)

    # Print option returns
for K, option_return in zip(Ks_call, mean_option_returns):
    print(f"Option return for strike price {K}: {option_return:.4f}")

#     # Print option returns
# for K, option_return in zip(Ks_put, mean_option_returns):
#     print(f"Option return for strike price {K}: {option_return:.4f}")

# # Plot histograms
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# #
# # # Histogram for the physical measure (P)
# # ax1.hist(terminal_values_P, bins=100, alpha=0.75, color='blue', edgecolor='black')
# # ax1.set_title('Terminal Values Under Physical Measure (P)')
# # ax1.set_xlabel('Underlying Price')
# # ax1.set_ylabel('Frequency')
# #
# # # Histogram for the risk-neutral measure (Q)
# # ax2.hist(terminal_values_Q, bins=100, alpha=0.75, color='green', edgecolor='black')
# # ax2.set_title('Terminal Values Under Risk-neutral Measure (Q)')
# # ax2.set_xlabel('Underlying Price')
# #
# # plt.show()
