import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(x_i, x_j, sigma):
    # Calculate the squared Euclidean distance
    distance = np.linalg.norm(x_i - x_j) ** 2

    # Compute the RBF kernel value
    kernel_value = np.exp(-distance / (2 * sigma ** 2))

    return kernel_value


def plot_rbf_kernel(sigma):
    x = np.linspace(-5, 5, 100)  # Generate points on x-axis
    y = np.zeros_like(x)  # Initialize y-axis values

    # Calculate RBF kernel values for each x_i
    for i, x_i in enumerate(x):
        y[i] = rbf_kernel(x_i, 0, sigma)

    # Plot the RBF kernel function
    plt.plot(x, y)
    plt.title("RBF Kernel: sigma = {}".format(sigma))
    plt.xlabel("x")
    plt.ylabel("Kernel Value")
    plt.grid(True)
    plt.show()


# Example usage
sigma = 1.0
# plot_rbf_kernel(sigma)

# import numpy as np
#
# W = np.array([[1, 0, 0],
#               [0, 1, 0],
#               [1/3, 2/3, 0]])
#
# # Calculate the degree matrix D
# D = np.diag(W.sum(axis=1))
#
# # Calculate the transition probability matrix P
# P = np.linalg.inv(D).dot(W)
#
# # Extract P_UL and P_UU
# P_UL = P[2, :2]
# P_UU = P[2, 2]
#
# Y_L = np.array([1, 0])
#
# # Calculate Y_hat
# Y_hat = (1 - P_UU) ** (-1) * P_UL.dot(Y_L)
#
# print("Y_hat:", Y_hat)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data from a Gaussian distribution
np.random.seed(42)
true_mean = 5
true_std = 2
data = np.random.normal(true_mean, true_std, 100)

# Prior distribution parameters
prior_mean = 0
prior_std = 5

# Likelihood function parameters
likelihood_std = 2

# Compute the posterior distribution
posterior_mean = (prior_mean / prior_std**2 + np.mean(data) / likelihood_std**2) / \
    (1 / prior_std**2 + len(data) / likelihood_std**2)
posterior_std = np.sqrt(1 / (1 / prior_std**2 + len(data) / likelihood_std**2))

# Plot the prior, likelihood, and posterior distributions
x = np.linspace(0, 10, 100)
prior = norm.pdf(x, prior_mean, prior_std)
likelihood = norm.pdf(x, np.mean(data), likelihood_std)
posterior = norm.pdf(x, posterior_mean, posterior_std)

plt.plot(x, prior, label='Prior')
plt.plot(x, likelihood, label='Likelihood')
plt.plot(x, posterior, label='Posterior')
plt.xlabel('Parameter')
plt.ylabel('Probability Density')
plt.title('Prior, Likelihood, and Posterior Distributions')
plt.legend()
plt.show()