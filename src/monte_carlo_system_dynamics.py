import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#differential equations
def model(y, t, sus_injury_rate, rec_injury_rate, rec_training_rate, rehab_rate, ret_rate):
    susceptible, injured, recovered, retired = y
    
    dsdt = - susceptible * sus_injury_rate
    didt = susceptible * sus_injury_rate + recovered * rec_injury_rate * rec_training_rate - \
           injured * rehab_rate - injured * ret_rate
    drdt = injured * rehab_rate - recovered * rec_injury_rate * rec_training_rate
    drretdt = injured * ret_rate
    
    return [dsdt, didt, drdt, drretdt]

#monte carlo parameters
simulations = 5
np.random.seed(42)  


#time steps
t = np.linspace(0, 20, 100)

#initializing arrays for the results
sus_average, inj_average, rec_average, ret_average = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

#monte carlo simulations
for _ in range(simulations):
    #random variations
    sus_injury_rate = 0.1 + 0.01 * np.random.randn()
    rec_injury_rate = 0.05 + 0.005 * np.random.randn()
    rec_training_rate = 0.2 + 0.02 * np.random.randn()
    rehab_rate = 0.1 + 0.01 * np.random.randn()
    ret_rate = 0.02 + 0.002 * np.random.randn()

    #initial conditions for each starting from zero recovered and zero retired
    initial_conditions = [0.9, 0.1, 0, 0]

    #solving the system
    result = odeint(model, initial_conditions, t, args=(sus_injury_rate, rec_injury_rate,
                                                       rec_training_rate, rehab_rate, ret_rate))

    #accumlating the results
    sus_average += result[:, 0]
    inj_average += result[:, 1]
    rec_average += result[:, 2]
    ret_average += result[:, 3]

#averaging the results over number of simulations (5)
sus_average /= simulations
inj_average /= simulations
rec_average /= simulations
ret_average /= simulations

#plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, sus_average, label='Susceptible', color='b')
plt.plot(t, inj_average, label='Injured', color='g')
plt.plot(t, rec_average, label='Recovered', color='r')
plt.plot(t, ret_average, label='Retired', color='c')

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Monte Carlo System Dynamics - Prediction of Low Extermity Injuries')
plt.legend()
plt.show()
