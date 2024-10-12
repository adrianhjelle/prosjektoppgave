from matplotlib import pyplot as plt
import numpy as np
from pyomo.environ import *
from scipy.stats import norm

# Define the model
model = ConcreteModel()

# Parameters
Q = 1000  # Total electricity available to sell
L = 550   # Maximum electricity that can be sold per time period
num_time_periods = 10  # Number of time periods (easily scalable)
num_samples = 100  # Number of Monte Carlo samples

# Decision variables for each time period (1 to num_time_periods)
model.x = Var(range(1, num_time_periods + 1), within=NonNegativeReals, bounds=(0, L))

# Continuous price distributions (mean, standard deviation) for each time period
price_distributions = {
    1: {'mean': 50, 'std': 10},
    2: {'mean': 30, 'std': 5},
    3: {'mean': 70, 'std': 15},
    4: {'mean': 60, 'std': 8},
    5: {'mean': 80, 'std': 20},
    6: {'mean': 50, 'std': 10},
    7: {'mean': 30, 'std': 5},
    8: {'mean': 70, 'std': 15},
    9: {'mean': 60, 'std': 8},
    10: {'mean': 80, 'std': 20},
}

# Monte Carlo sampling: Generate random price samples from normal distributions
price_samples = {
    t: norm.rvs(loc=price_distributions[t]['mean'], scale=price_distributions[t]['std'], size=num_samples)
    for t in range(1, num_time_periods + 1)
}

# Constraint: Total electricity sold cannot exceed the available amount Q
model.total_electricity_sold = Constraint(expr=sum(model.x[t] for t in range(1, num_time_periods + 1)) <= Q)

# Objective function: Maximize expected profit using Monte Carlo samples
def expected_profit_rule(model):
    total_profit = 0
    
    # Iterate through all Monte Carlo samples to approximate the expected profit
    for i in range(num_samples):
        scenario_profit = sum(price_samples[t][i] * model.x[t] for t in range(1, num_time_periods + 1))
        total_profit += scenario_profit

    # Divide by the number of samples to get the average (expected) profit
    return total_profit / num_samples

model.profit = Objective(rule=expected_profit_rule, sense=maximize)

# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model, tee=True)

# Display the results
model.display()

# Extract the results for optimal electricity sold in each time period
optimal_x = [value(model.x[t]) for t in range(1, num_time_periods + 1)]

# Plot the results
def plot(optimal_x, num_time_periods):
    fig, ax = plt.subplots()

    # Boxplot data for each time step using the Monte Carlo price samples
    price_sample_data = [price_samples[t] for t in range(1, num_time_periods + 1)]

    # Create the boxplot
    boxprops = dict(color="royalblue", linewidth=2)
    medianprops = dict(color="darkblue", linewidth=2)
    whiskerprops = dict(color="skyblue", linewidth=2)
    capprops = dict(color="skyblue", linewidth=2)

    ax.boxplot(price_sample_data, labels=[f'Time {t}' for t in range(1, num_time_periods + 1)],
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops,
               patch_artist=True, notch=False)

    # Scatter points for average prices
    average_prices = [np.mean(price_sample_data[t-1]) for t in range(1, num_time_periods + 1)]
    ax.scatter(range(1, num_time_periods + 1), average_prices, color='red', s=100, zorder=3, label='Average Prices')

    # Add the optimal solution as a line graph
    ax2 = ax.twinx()
    ax2.plot(range(1, num_time_periods + 1), optimal_x, color='green', marker='o', markersize=10, linewidth=2, label='Optimal Electricity Sold')

    # Customizing labels, titles, and adding grid for better readability
    ax.set_title('Electricity Prices and Optimal Selling Strategy (Continuous Distribution)', fontsize=14, fontweight='bold', color="darkblue")
    ax.set_ylabel('Price', fontsize=12, color="royalblue")
    ax2.set_ylabel('Electricity Sold', fontsize=12, color="green")

    ax.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(False)

    # Adding legends with customized positioning and styling
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Display the improved plot
    plt.show()

# Call the plotting function
plot(optimal_x, num_time_periods)
