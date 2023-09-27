# ======================================================================================================================
#                                       Inputs for the feature extraction module
# ======================================================================================================================
# Path to the root directory of the Event Generation project.
project_path = r"""C:\Users\ntaghip1\PycharmProjects\Event_Generation_private\PSMLEI"""

# Path to the directory where generated event data will be stored.
generated_events_path = r"""C:\Users\ntaghip1\PycharmProjects\Event_Generation_private\PSMLEI\results\generated_events"""

# The first event number to be considered for data generation.
first_event = 1

# The last event number to be considered for data generation.
last_event = 5

# The index of the first sample in the signal data.
first_sample = 30

# The index of the last sample in the signal data.
last_sample = 330

# The total number of PMUs used in the simulation.
n_pmus = 95

# The number of subset of PMUs with the largest energy of the signal
n_pmus_prime = 20

# Rank approximation of the Hankel matrix used in matrix pencil method for mode decomposition.
Rank = 6

# Sampling rate of PMU data in seconds.
Ts = 1 / 30

# Number of modes used in the mode decomposition via the matrix pencil method.
p = 6

# Number of distinct modes (considering only one of the complex conjugate pairs).
p_prime = 3

# Decimal tolerance used for rounding numerical values.
decimal_tol = 5
