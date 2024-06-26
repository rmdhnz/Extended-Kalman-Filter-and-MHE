import pandas as pd
import matplotlib.pyplot as plt
import time
start = time.time()
data = pd.read_csv("data/sp1.5_pid_1.csv")
t = data["Time"]
flow = data["Flow Measured Value"]
set_point = data["Flow Set Point"]
if __name__ == "__main__":
  plt.plot(t,flow,label="Flow rate")
  plt.plot(t,set_point, label="Set point")
  plt.legend(loc="lower right")
  plt.axis((0,60,0,2))
  plt.grid(True)
  end = time.time()
  print("Waktu eksekusi :",(end-start)*10**3,"ms")
  plt.show() 