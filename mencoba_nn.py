from sklearn.neural_network import MLPRegressor
from tes import data
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
flow_csv = data["Flow Measured Value"]
set_point_csv = data["Flow Set Point"]
waktu = data["Time"]
# flow = np.array(flow_csv)
# set_point = np.array(set_point_csv)
flow = np.reshape(flow_csv,(1,554))
set_point = np.reshape(set_point_csv,(1,554))
# Instantiate MLPRegressor

nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(100, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=False
)

nn.fit(set_point, flow)
pred = nn.predict(set_point)
#
# Calculate accuracy and error metrics
#
test_set_rmse = np.sqrt(mean_squared_error(flow, pred))
#
# Print R_squared and RMSE value
#
print('RMSE: ', test_set_rmse)
pred = np.reshape(pred,(554,1))

plt.plot(waktu,set_point_csv,label="set point",color="orange",linestyle="dotted")
plt.plot(waktu,flow_csv,label="flow",color="red")
plt.plot(waktu,pred,label="prediksi",color="blue")
plt.legend()
plt.grid(True)
plt.show()