import numpy as np
from scipy.optimize import minimize
import pandas as pd
data = pd.read_csv("data/data-setpoint-berubah-2.csv")

# Number of time steps for MHE
N = 5
num_steps = len(data["Flow Measured Value"])     # Number of time steps
waktu = data["Time"]
# Generate noisy measurements
true_velocity = 2  # True velocity (m/s)

# True speed of water flow (for simulation)
true_speeds = data["Flow Set Point"]
measurements = data["Flow Measured Value"]
# Simulated noisy measurements
np.random.seed(0)  # For reproducibility


# Process noise (assumed to be small)
process_noise_std = 0.1

# Initial estimate
initial_estimate = 1.0

def mhe_objective(estimate, measurements, N):
    estimate = np.asarray(estimate)
    process_error = np.diff(estimate)
    measurement_error = estimate - measurements[:len(estimate)]
    
    # Objective function to minimize
    return np.sum(process_error**2) + np.sum(measurement_error**2)

# Sliding window MHE
estimated_speeds = []
for t in range(len(true_speeds) - N):
    window_measurements = measurements[t:t+N]
    if t == 0:
        estimate = np.full(N, initial_estimate)
    else:
        estimate = np.full(N, estimated_speeds[-1])

    result = minimize(mhe_objective, estimate, args=(window_measurements, N))
    estimated_speeds.append(result.x[0])

# Append the last estimates for completeness
estimated_speeds += list(result.x[1:])
print(len(measurements))
# Plot the results
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot(true_speeds, label='Set Point',linestyle='dotted')
    plt.plot(measurements, label='Flow Rate',color='red')
    plt.plot(estimated_speeds, label='MHE',color="green")
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Speed')
    plt.title('Moving Horizon Estimation of Water Flow Speed')
    plt.axis((0,600,0,2.5))
    plt.show()
