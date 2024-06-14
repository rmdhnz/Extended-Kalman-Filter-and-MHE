import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# Deklarasi Parameter
m = 0.2  # massa pendulum (kg)
M = 0.5  # massa cart (kg)
L = 0.3  # Panjang pendulum (m)
g = 9.81 # grafitasi (m/s^2)
d = 0.1  # damping coefficient (N/m/s)

# Define the dynamics of the pendulum system
def pendulum_dynamics(state, t, u, m, M, L, g, d):
    x, x_dot, theta, theta_dot = state
    Sx = np.sin(theta)
    Cx = np.cos(theta)
    D = M + m * Sx**2
    
    dxdt = x_dot
    dx_dotdt = (1/D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * theta_dot**2 * Sx - d * x_dot)) + m * L**2 * (1/D) * u
    dthetadt = theta_dot
    dtheta_dotdt = (1/D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * theta_dot**2 * Sx - d * x_dot)) - m * L * Cx * (1/D) * u
    
    return [dxdt, dx_dotdt, dthetadt, dtheta_dotdt]

# Generate training data
def generate_training_data(num_samples, t, initial_state):
    data = []
    output = []
    
    for _ in range(num_samples):
        initial_state = [0, 0, np.pi + (np.random.rand() - 0.5), 0]
        t_span = np.linspace(0, 10, len(t))
        control = 0
        
        solution = odeint(pendulum_dynamics, initial_state, t_span, args=(control, m, M, L, g, d))
        final_state = solution[-1]
        
        data.append(final_state)
        output.append(-10 * final_state[2]) # simple control law
    
    return np.array(data), np.array(output)

# Define membership functions
def gbellmf(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a)**(2 * b))

def gaussmf(x, sigma, c): 
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
# Define Fuzzy Inference System
class FIS:
    def __init__(self):
        self.rules = []
        self.params = []

    def add_rule(self, mf_params, output):
        self.rules.append((mf_params, output))
        self.params.extend(mf_params)

    def evaluate(self, x):
        num = 0
        den = 0
        for mf_params, output in self.rules:
            mf_values = [gbellmf(xi, *params) for xi, params in zip(x, mf_params)]
            weight = np.prod(mf_values)
            num += weight * output
            den += weight
        return num / den if den != 0 else 0 #untuk menghindari zero division error

# deklarasi kelas ANFIS
class ANFIS:
    def __init__(self, data, output):
        self.data = data
        self.output = output
        self.fis = FIS()

    def train(self, epochs=100):
        initial_params = np.random.rand(len(self.fis.params))
        res = minimize(self.loss_function, initial_params, method='L-BFGS-B', options={'maxiter': epochs})
        self.fis.params = res.x

    def loss_function(self, params):
        self.fis.params = params
        predictions = np.array([self.fis.evaluate(x) for x in self.data])
        return np.mean((predictions - self.output) ** 2)

# Simulate the system with the ANFIS controller
def simulate_anfis(anfis_model, initial_state, t):
    states = []
    controls = []
    state = initial_state

    for _ in t:
        control = anfis_model.fis.evaluate(state)
        t_span = [0, dt]
        state = odeint(pendulum_dynamics, state, t_span, args=(control, m, M, L, g, d))[-1]
        states.append(state)
        controls.append(control)
    
    return np.array(states), np.array(controls)


if __name__ == "__main__": 
    # Main script
  num_samples = 1000
  dt = 0.01
  t = np.linspace(0, 10, int(10/dt))
  initial_state = [0, 0, np.pi + 0.1, 0]

  data, output = generate_training_data(num_samples, t, initial_state)
  anfis_model = ANFIS(data, output)

  # Add rules to the FIS (example with random parameters, should be optimized)
  for _ in range(5):
      anfis_model.fis.add_rule([(1, 2, 0), (1, 2, np.pi), (1, 2, 0), (1, 2, 0)], 1)

  anfis_model.train(epochs=100)

  states, controls = simulate_anfis(anfis_model, initial_state, t)

  # Plot results
  plt.figure()
  plt.subplot(3, 1, 1)
  plt.plot(t, states[:, 0])
  plt.xlabel('Time (s)')
  plt.ylabel('Cart Position (m)')
  plt.title('Cart Position')
  plt.grid(True)

  plt.subplot(3, 1, 2)
  plt.plot(t, states[:, 2])
  plt.xlabel('Time (s)')
  plt.ylabel('Pendulum Angle (rad)')
  plt.title('Pendulum Angle')
  plt.grid(True)

  plt.subplot(3, 1, 3)
  plt.plot(t, controls)
  plt.xlabel('Time (s)')
  plt.ylabel('Control Force (N)')
  plt.title('Control Force')
  plt.grid(True)

  plt.tight_layout()
  plt.show()