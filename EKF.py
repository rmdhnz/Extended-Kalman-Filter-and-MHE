import numpy as np
import matplotlib.pyplot as plt
from tes import data
import time
start_ekf = time.time()
# Parameter model
dt = 1.0  # Time step (seconds)
Q = 0.01  # Process noise covariance
R = 0.1   # Measurement noise covariance

# Non-linear state transition and measurement functions
def f(x):
    return x  # State transition model (x_k = x_{k-1})

def h(x):
    return x  # Measurement model (z_k = x_k)

# Jacobian of the state transition and measurement functions
def F(x):
    return np.array([[1]])  # Derivative of f with respect to x

def H(x):
    return np.array([[1]])  # Derivative of h with respect to x

# Initialization
x = np.array([[0]])  # Initial state (velocity in m/s)
P = np.array([[1]])  # Initial state covariance
num_steps = len(data["Flow Measured Value"])     # Number of time steps
waktu = data["Time"]
set_point = data["Flow Set Point"]
# Generate noisy measurements
true_velocity = 2  # True velocity (m/s)
measurements = data["Flow Measured Value"]

# Storage for estimated velocities
estimated_velocity = np.zeros((num_steps, 1))

# Extended Kalman Filter process
for k in range(num_steps):
    # Prediction
    x_pred = f(x)
    P_pred = F(x).dot(P).dot(F(x).T) + Q

    # Measurement
    z = measurements[k]

    # Kalman Gain
    K = P_pred.dot(H(x_pred).T).dot(np.linalg.inv(H(x_pred).dot(P_pred).dot(H(x_pred).T) + R))

    # Update
    x = x_pred + K.dot(z - h(x_pred))
    P = (np.eye(1) - K.dot(H(x_pred))).dot(P_pred)

    # Store the estimated velocity
    estimated_velocity[k] = x
end_ekf = time.time()
# Plotting results
if __name__ == '__main__':
    plt.figure()
    plt.plot(waktu,set_point,'g--',label="Set Point")
    plt.plot(waktu, measurements,'r', label='Measurements')
    plt.plot(waktu, estimated_velocity, 'b-', label='Estimasi EKF')
    plt.xlabel('Time')
    plt.ylabel('Water Flow Velocity (m/s)')
    plt.legend()
    plt.title('Water Flow Velocity Estimation using Extended Kalman Filter')
    plt.grid(True)
    plt.axis((0,60,0,2))
    plt.show()

