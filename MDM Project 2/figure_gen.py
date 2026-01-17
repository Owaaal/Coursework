import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta_0 = 0.1  # Initial infection rate (during lockdown)
beta = 0.6   # Pre-lockdown infection rate  
D = 0.2       # Controls the steepness of transition
t_l = 10      # Transition midpoint (weeks)
k = t_l + 80  # Weeks before people ignore lockdown (adjustable) needs to be greater than t_l

def beta_t_up(t, beta_0, beta, k):
    return beta_0 + (beta - beta_0) * (1 - np.exp(- (t / k) ** 4))

def beta_t_down(t, beta_0, beta, D, t_l):
    return ((beta - beta_0)/2) * (1 + np.tanh(D * (t_l - t))) + beta_0

def beta_t_full(t, beta_0, beta, D, t_l):
    return beta_0+((beta - beta_0)/2) * (1 + (np.tanh(D * (t_l - t)))) + (beta - beta_0)*(1 - np.exp(-(t / (k)) ** 4))

def beta_t_jump(t, beta_0, beta, D, t_l):
    return beta_0 + ((beta - beta_0)/2) * (1 + (np.tanh(D * (t_l - t)))) + ((beta - beta_0)/2) * (1 + (np.tanh(D * (t - k))))


t_values = np.linspace(0, 150, 50)  # Time in weeks
beta_values_full = beta_t_full(t_values, beta_0, beta, D, t_l)
beta_jump = beta_t_jump(t_values, beta_0, beta, D, t_l)
btu=beta_t_up(t_values, beta_0, beta, k)
btd=beta_t_down(t_values, beta_0, beta, D, t_l)

# Plot
plt.figure(figsize=(18, 6))
plt.rcParams.update({'font.size': 16})
plt.plot(t_values, beta_values_full, label=(r'$L_1$'), color='black', linestyle='-' )
plt.plot(t_values, beta_jump, label=(r'$L_2$'), color='red', linestyle='--')
plt.axhline(beta, linestyle='--', color='g', label='Inital β')
plt.xlabel('Time (Days)')
plt.ylabel('Infection rate β(t)')
plt.legend()
plt.grid()
plt.show()