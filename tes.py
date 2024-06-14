import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/data-setpoint-berubah-2.csv")
t = data["Time"]
flow = data["Flow Measured Value"]
set_point = data["Flow Set Point"]
plt.plot(t,flow,label="Flow rate")
plt.plot(t,set_point, label="Set point")
plt.legend(loc="lower right")
plt.axis((0,60,0,2.5))
plt.grid(True)
plt.show() 