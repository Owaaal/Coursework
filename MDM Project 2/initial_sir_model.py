import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
# SIR model equations
def SIR_model(y, t, B_data, gamma):
    beta=B_data[t]
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Read in data
data_raw = pd.read_csv(r'/Users/karandama/Documents/School stuff/Uni Shit/Python Scripts/MDM SCRIPTS/MDM Project 2/data/estimate-of-the-effective-reproduction-rate-r-of-covid-19.csv')
data_raw = data_raw.to_numpy(dtype = str)  #convert to np.array with floats
# Convert column to NumPy array
R_data = np.array(data_raw[:, 2], dtype=float)  
B_data=R_data*1/5
print(B_data)
"""
Initial conditions (such as S0, I0, and R0) are not to be random but I hardcoded them with specific values. These choices are typically made based on the characteristics of the disease being modeled and the context of the simulation. Initial condition are set such that S0 = 99%, which indicates the proportion of susceptible individuals when the simulation starts. I0 is set to 1%, which indicates proportion of infected individuals to be 1% when the simulation starts. R0 is set to 0% which is expected that there are are no recovered individuals when the simulations start.
"""
S0 = 0.99992
I0 = 8.39*(10**-5)
R0 = 2.20*(10**-3)
y0 = [S0, I0, R0]

# Parameters

gamma = 0.2 

# Time vector
t_data = np.linspace(0, 199, 200)  # Simulate for 200 days

# Solve the SIR model equations using odeint()
solution = solve_ivp(SIR_model, (0, len(t)-1), y0, t_eval=t_data, args=(B_data, gamma))

# Extract results
S, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()