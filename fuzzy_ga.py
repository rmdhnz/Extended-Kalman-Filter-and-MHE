import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# Parameters
m = 0.2  # mass of the pendulum (kg)
M = 0.5  # mass of the cart (kg)
L = 0.3  # length of the pendulum (m)
g = 9.81 # gravitational acceleration (m/s^2)
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
        output.append(-10 * final_state[2])  # simple control law

    return np.array(data), np.array(output)

# Define membership functions
def gbellmf(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a)**(2 * b))

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
        param_idx = 0
        for mf_params, output in self.rules:
            mf_values = [gbellmf(xi, *self.params[param_idx:param_idx+3]) for xi, _ in zip(x, mf_params)]
            param_idx += 3
            weight = np.prod(mf_values)
            num += weight * output
            den += weight
        return num / den if den != 0 else 0

# Define Genetic Algorithm for optimizing FIS
def fis_ga_optimization(fis, data, output, generations=100, population_size=20):
    def fitness(params):
        fis.params = params
        predictions = np.array([fis.evaluate(x) for x in data])
        return np.mean((predictions - output) ** 2)
    
    bounds = [(0, 10)] * len(fis.params)  # Define bounds for the GA

    # Callback function to print the progress
    def callback(xk, convergence):
        fis.params = xk
        predictions = np.array([fis.evaluate(x) for x in data])
        current_fitness = np.mean((predictions - output) ** 2)
        print(f"Current fitness: {current_fitness:.6f}")
        print(f"Current parameters: {xk}")

    result = differential_evolution(fitness, bounds, maxiter=generations, popsize=population_size, callback=callback)
    fis.params = result.x
    print(f"Final fitness: {result.fun:.6f}")
    print(f"Optimized parameters: {result.x}")

# Simulate the system with the FIS controller
def simulate_fis(fis, initial_state, t):
    states = []
    controls = []
    state = initial_state

    for _ in t:
        control = fis.evaluate(state)
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
    fis = FIS()

    # Add rules to the FIS (example with random parameters, should be optimized)
    for _ in range(5):
        fis.add_rule([(1, 2, 0), (1, 2, np.pi), (1, 2, 0), (1, 2, 0)], 1)

    fis_ga_optimization(fis, data, output, generations=100, population_size=20)

    states, controls = simulate_fis(fis, initial_state, t)

    # Plot results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, states[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Cart Position (m)')
    plt.title('Cart Position')

    plt.subplot(3, 1, 2)
    plt.plot(t, states[:, 2])
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.title('Pendulum Angle')

    plt.subplot(3, 1, 3)
    plt.plot(t, controls)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Force (N)')
    plt.title('Control Force')

    plt.tight_layout()
    plt.show() 