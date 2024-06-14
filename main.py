from MHE_algorithm import estimated_speeds
from EKF import estimated_velocity
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("data/data-setpoint-berubah-2.csv")
flow_rate = data["Flow Measured Value"]
set_point = data["Flow Set Point"]
plt.plot(set_point, label='Set Point',color="orange",linestyle='dotted')
plt.plot(flow_rate, label='Flowmeter',color='red')
plt.plot(estimated_speeds, label='MHE',color="green")
plt.plot(estimated_velocity, label='EKF',color="blue")
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Speed')
plt.title('MHSE vs EKF')
plt.axis((0,600,0,2.5))
plt.show()