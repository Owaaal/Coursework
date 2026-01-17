import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd

# Read in data
data_raw = pd.read_csv(r'/Users/jacksonbyrne/Desktop/SEMT10002_2024-main/MDM_1/MDM Project 2/data/estimate-of-the-effective-reproduction-rate-r-of-covid-19.csv')
data_raw = data_raw.to_numpy(dtype = str)  #convert to np.array with floats
# Convert column to NumPy array
I_data = np.array(data_raw[:, 2], dtype=float)  

# Given parameters
gamma = 1 / 5  # Recovery rate
N = 1e6        # Total population
I0 = I_data[0]  # Initial infected cases
R0 = 0
S0 = N - I0

# Ensure time points match data length
t_data = np.linspace(0, len(I_data) - 1, len(I_data))

# Define the SIR model
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to simulate SIR model
def simulate_SIR(beta):
    beta = beta[0] if isinstance(beta, np.ndarray) else beta  # Ensure beta is a scalar
    y0 = [S0, I0, R0]
    solution = integrate.odeint(sir_model, y0, t_data, args=(beta, gamma, N))
    return solution # Return only I(t)

# Define loss function
def loss_function(beta):
    beta = beta[0]  # Extract scalar from array
    I_simulated = simulate_SIR(beta)
    min_length = min(len(I_simulated), len(I_data))  # Ensure same length
    return np.sum((I_simulated[:min_length] - I_data[:min_length]) ** 2)

# Optimize beta
beta_initial_guess = [0.2]  # Pass as a list to avoid shape issues
result = optimize.minimize(loss_function, beta_initial_guess, bounds=[(0, 1)])

# Get the fitted beta value
beta_fitted = result.x[0]

# Simulate SIR with fitted beta
I_fitted = simulate_SIR(0.2)

# Plot results
plt.scatter(t_data, I_data, label="Observed Data", color="red", alpha=0.6)
plt.plot(t_data, I_fitted, label=f"Fitted Model (β ≈ {beta_fitted:.3f})", color="blue")
plt.xlabel("Time (days)")
plt.ylabel("Infected I(t)")
plt.legend()
plt.title("Fitting β in SIR Model")
plt.show()

print(f"Estimated β: {beta_fitted:.3f}")