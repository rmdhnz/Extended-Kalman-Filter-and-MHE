import numpy as np
from scipy.integrate import odeint
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from deap import base, creator, tools, algorithms

# Define the inverted pendulum system
def inverted_pendulum(y, t, u):
    m = 0.15  # mass of the pendulum
    M = 0.5   # mass of the cart
    L = 0.5   # length of the pendulum
    g = 9.81  # acceleration due to gravity
    d = 0.1   # damping coefficient
    
    Sy = np.sin(y[2])
    Cy = np.cos(y[2])
    
    dydt = np.zeros(4)
    dydt[0] = y[1]
    dydt[1] = (u + m * Sy * (L * y[3]**2 + g * Cy) - d * y[1]) / (M + m * Sy**2)
    dydt[2] = y[3]
    dydt[3] = (-u * Cy - m * L * y[3]**2 * Sy * Cy - (M + m) * g * Sy + d * y[1] * Cy) / (L * (M + m * Sy**2))
    
    return dydt

# Create a fuzzy logic controller
theta = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 5), 'theta')
theta_dot = ctrl.Antecedent(np.linspace(-10, 10, 5), 'theta_dot')
force = ctrl.Consequent(np.linspace(-10, 10, 5), 'force')

theta.automf(3)
theta_dot.automf(3)
force.automf(3)

rule1 = ctrl.Rule(theta['poor'] & theta_dot['poor'], force['poor'])
rule2 = ctrl.Rule(theta['poor'] & theta_dot['average'], force['poor'])
rule3 = ctrl.Rule(theta['poor'] & theta_dot['good'], force['average'])
rule4 = ctrl.Rule(theta['average'] & theta_dot['poor'], force['poor'])
rule5 = ctrl.Rule(theta['average'] & theta_dot['average'], force['average'])
rule6 = ctrl.Rule(theta['average'] & theta_dot['good'], force['good'])
rule7 = ctrl.Rule(theta['good'] & theta_dot['poor'], force['average'])
rule8 = ctrl.Rule(theta['good'] & theta_dot['average'], force['good'])
rule9 = ctrl.Rule(theta['good'] & theta_dot['good'], force['good'])

fis = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
controller = ctrl.ControlSystemSimulation(fis)

# Test the FIS
controller.input['theta'] = 2
controller.input['theta_dot'] = -1
controller.compute()
print(f'Force: {controller.output["force"]}')

# Define genetic algorithm functions for optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_individual(icls, fis):
    ind = icls(fis)
    return ind

def eval_fis(individual):
    tspan = np.linspace(0, 10, 1000)
    y0 = [0, 0, np.pi, 0]
    
    def control_law(y, t):
        controller.input['theta'] = y[2]
        controller.input['theta_dot'] = y[3]
        controller.compute()
        return controller.output['force']
    
    y = odeint(inverted_pendulum, y0, tspan, args=(control_law,))
    
    fitness = -np.sum((y[:, 2] - np.pi)**2 + y[:, 3]**2)
    return fitness,

toolbox = base.Toolbox()
toolbox.register("individual", init_individual, creator.Individual, fis)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_fis)

def optimize_fuzzy_with_ga(fis):
    population_size = 20
    generations = 50
    mutation_rate = 0.01
    
    pop = toolbox.population(n=population_size)
    
    # Use the built-in DEAP algorithms for the GA process
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=mutation_rate, ngen=generations, 
                        stats=None, halloffame=None, verbose=True)
    
    # Return the best individual
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind

optimized_fis = optimize_fuzzy_with_ga(fis)

# Simulasi sistem dengan kontroler fuzzy yang sudah dioptimasi
tspan = np.linspace(0, 10, 1000)
y0 = [0, 0, np.pi, 0]  # Initial state: [cart position, cart velocity, pendulum angle, pendulum angular velocity]

def control_law_optimized(y, t):
    controller.input['theta'] = y[2]
    controller.input['theta_dot'] = y[3]
    controller.compute()
    return controller.output['force']

y = odeint(inverted_pendulum, y0, tspan, args=(control_law_optimized,))

# Plot hasil simulasi
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tspan, y[:, 2])
plt.xlabel('Time (s)')
plt.ylabel('Pendulum Angle (rad)')
plt.title('Pendulum Angle vs Time')

plt.subplot(2, 1, 2)
plt.plot(tspan, y[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Cart Position (m)')
plt.title('Cart Position vs Time')

plt.show()
