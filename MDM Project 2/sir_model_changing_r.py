import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Given parameters
gamma = 1/7  # Recovery rate 
N = 1e6        # Total population
I0 = 102.6    # Initial infected individuals
R0 = 6666         # Initial recovered individuals
S0 = N - I0    # Initial susceptible individuals

# Define a 1D array for beta(t)
data_raw = pd.read_csv(r'/Users/jacksonbyrne/Desktop/SEMT10002_2024-main/MDM_1/MDM Project 2/data/Covid_effective_reporduction_full_time.csv')
data_raw = data_raw.to_numpy(dtype = str)  #convert to np.array with floats
# Convert column to NumPy array
R_data = np.array(data_raw[:, 2], dtype=float)
'''beta_array=R_data*gamma'''
'''
Takes raw data of covid numbers and then muliplies that with gamma to find the beta array
average beta over time = 3 - 15
dExposed = beta/time * (Infected * Susceptible) - inv(Time) * Exposed
'''
def new_beta():
    # Parameters
    beta_0 = 0.3  # Initial infection rate (during lockdown)
    beta = 0.6   # Pre-lockdown infection rate
    k = 30    # Weeks before people ignore lockdown (adjustable)

    D = 0.5       # Controls the steepness of transition
    t_l = 20      # Transition midpoint (weeks)

    def beta_t_exp(t, beta_0, beta, k):
        return beta_0 + (beta - beta_0) * (1 - np.exp(- (t / k) ** 4))

    def beta_t_full(t, beta_0, beta, D, t_l):
        return ((beta - beta_0)/2) * (1 + np.tanh(D * (t_l - t))) + beta_0 + (beta - beta_0) * (1 - np.exp(-(t / (k)) ** 4))



    t_values = np.linspace(0, 52, 100)  # Time in weeks
    beta_values_full = beta_t_full(t_values, beta_0, beta, D, t_l)
    return beta_values_full

beta_array = new_beta()

#Define a 1D array for real I(t)
data_raw = pd.read_csv(r'/Users/jacksonbyrne/Desktop/SEMT10002_2024-main/MDM_1/MDM Project 2/data/Covid_cases_brazil_full_time.csv')
data_raw = data_raw.to_numpy(dtype = str)  #convert to np.array with floats
# Convert column to NumPy array
infected_data = np.array(data_raw[:, 2], dtype=float)  

# Time points
t_data = np.linspace(0, len(beta_array)-1, len(beta_array))

# Function to interpolate beta at any given time t
def get_beta(t):
    t_index = int(min(max(t, 0), len(beta_array) - 1))  # Ensure within bounds
    return beta_array[t_index]

# Define the SIR model with time-varying beta
def sir_model(t, y):
    S, I, R = y
    beta_t = get_beta(t)  # Get the time-dependent beta value
    dSdt = -beta_t * S * I/N
    dIdt = beta_t * S * I/N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Solve the SIR model with time-varying beta
y0 = [S0, I0, R0]  # Initial conditions
solution = solve_ivp(sir_model, (0, len(t_data) - 1), y0, t_eval=t_data)

# Extract results
S_simulated, I_simulated, R_simulated = solution.y

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t_data, I_simulated, label="Infected I(t)", color="red")
'''plt.plot(t_data,infected_data,label='Real Value of Infected I(t)', color='blue')'''
# plt.plot(t_data, S_simulated, label="Susceptible S(t)", color="blue", linestyle="dashed")
# plt.plot(t_data, R_simulated, label="Recovered R(t)", color="green", linestyle="dotted")
plt.xlabel("Time (days)")
plt.ylabel("Number of Individuals")
plt.title("SIR Model with Time-Varying Beta")
plt.legend()
plt.grid()
plt.show()

# Plot the beta values over time
plt.figure(figsize=(10, 3))
plt.plot(t_data, beta_array, label="Beta(t) - Time-varying Transmission Rate", color="purple")
plt.xlabel("Time (days)")
plt.ylabel("Beta(t)")
plt.title("Time-Varying Beta Values")
plt.legend()
plt.grid()
plt.show()