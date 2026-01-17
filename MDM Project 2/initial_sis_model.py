import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
N = 208660842  # Total population
I0 = 1         # Initial number of infected individuals
S0 = N - I0    # Initial number of susceptible individuals
beta = 0.393   # Transmission rate per day
gamma = 0.143  # Recovery rate per day
days = 300     # Simulation period in days

# The SIS model differential equations
def deriv(y, t, N, beta, gamma):
    S, I = y
    dSdt = -beta * S * I / N + gamma * I
    dIdt = beta * S * I / N - gamma * I
    return dSdt, dIdt

# Initial conditions vector
y0 = S0, I0

# Time points (in days)
t = np.linspace(0, days, days)

# Integrate the SIS equations over the time grid, t
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I = ret.T

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.title('SIS Model Simulation of COVID-19 in Brazil (Early 2020)')
plt.legend()
plt.grid(True)
plt.show()
    