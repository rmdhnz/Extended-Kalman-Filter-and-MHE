import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/data_p_1.2_i_0.8_d_0.2.csv")
t = data["Time"]
flow = data["Flow Measured Value"]
set_point = data["Flow Set Point"]
if __name__ == "__main__":
  plt.plot(t,flow,label="Flow rate")
  plt.plot(t,set_point, label="Set point")
  plt.legend(loc="lower right")
  plt.axis((0,60,0,2.5))
  plt.grid(True)
  plt.show() 