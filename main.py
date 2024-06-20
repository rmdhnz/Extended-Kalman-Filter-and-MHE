from MHE_algorithm import estimated_speeds,start_mhe,end_mhe
from EKF import estimated_velocity,start_ekf,end_ekf
from NeuralNetworkz import pred as flow_pred
from NeuralNetworkz import start_nn,end_nn,test_set_mse
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
  plt.plot(waktu,flow_pred,label = "Neural Network",color="pink")
  waktu.pop(len(waktu)-1)
  plt.plot(waktu,estimated_speeds, label='MHE',color="green")
  plt.legend()
  plt.title('MHSE vs EKF')
  plt.xlabel('Waktu (s)')
  plt.ylabel('Speed')
  plt.grid(True)
  plt.axis(axis_need)

def show_by_subplot(): 
  plt.subplot(3,1,1)
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  plt.plot(waktu,estimated_velocity, label='EKF',color="blue")
  plt.grid(True)
  plt.legend()
  plt.axis(axis_need)
  
  plt.subplot(3,1,2)
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  plt.plot(waktu,flow_pred, label='Neural Network',color="pink")
  plt.grid(True)
  plt.legend()
  plt.axis(axis_need)

  plt.subplot(3,1,3)
  plt.plot(waktu,set_point, label='Set Point',color="orange",linestyle='dotted')
  plt.plot(waktu,flow_rate, label='Flowmeter',color='red')
  waktu.pop(len(waktu)-1)
  plt.plot(waktu,estimated_speeds, label='MHE',color="green")
  plt.grid(True)
  plt.legend()
  plt.axis(axis_need)



ekf_mse_val =  0
mhe_mse_val = 0
for i in range(min(len(flow_rate),len(estimated_velocity))): 
  ekf_mse_val+= (flow_rate[i]-estimated_velocity[i])**2
for i in range(min(len(flow_rate),len(estimated_speeds))):
  mhe_mse_val +=(flow_rate[i]-estimated_speeds[i])**2
print("MSE value for EKF : {}".format(ekf_mse_val[0]/min(len(flow_rate),len(estimated_velocity))))
print("MSE value for MHE : {}".format(mhe_mse_val/min(len(flow_rate),len(estimated_velocity))))
print("EKF Execution time : {} ms".format((end_ekf-start_ekf)*10**3))
print("MHE Execution time : {} ms".format((end_mhe-start_mhe)*10**3))
print("MSE Value for Neural Network : {}".format(test_set_mse))
print("Execution time for Neural Network : {} ms".format((end_nn-start_nn)*10**3))
# show_by_subplot()
show_all()
plt.show()