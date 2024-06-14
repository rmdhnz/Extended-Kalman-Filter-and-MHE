from MHE_algorithm import estimated_speeds
from EKF import estimated_velocity
import matplotlib.pyplot as plt
from tes import data
flow_rate = data["Flow Measured Value"]
waktu = data["Time"]
set_point = data["Flow Set Point"]
axis_need = (0,60,0,2.5)
def show_all() : 
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  plt.plot(waktu,estimated_velocity, label='EKF',color="blue")
  waktu.pop(len(waktu)-1)
  plt.plot(waktu,estimated_speeds, label='MHE',color="green")
  plt.legend()
  plt.title('MHSE vs EKF')
  plt.xlabel('Waktu (s)')
  plt.ylabel('Speed')
  plt.grid(True)
  plt.axis(axis_need)

def show_by_subplot(): 
  plt.subplot(2,1,1)
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  plt.plot(waktu,estimated_velocity, label='EKF',color="blue")
  plt.grid(True)
  plt.legend()
  plt.axis(axis_need)

  plt.subplot(2,1,2)
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  waktu.pop(len(waktu)-1)
  plt.plot(waktu,estimated_speeds, label='MHE',color="green")
  plt.grid(True)
  plt.legend()
  plt.axis(axis_need)

show_by_subplot()
# show_all()
plt.show()