from sklearn.neural_network import MLPRegressor
from tes import data
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
start_nn = time.time()
flow_csv = data["Flow Measured Value"]
set_point_csv = data["Flow Set Point"]
waktu = data["Time"]
length_data = len(flow_csv)
x_train,x_test,y_train,y_test =  train_test_split(set_point_csv,flow_csv,test_size=0.8)
# print(y_train)
# exit()
flow = np.array(flow_csv)
set_point = np.array(set_point_csv)
flow = np.reshape(flow_csv,(1,length_data))
set_point = np.reshape(set_point_csv,(1,length_data))
# Instantiate MLPRegressor

nn = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(100, 100),
    alpha=0.001,
    random_state=20,
    early_stopping=False
)

nn.fit(x_train, y_train)
pred = nn.predict(y_test)
#
# Calculate accuracy and error metrics
#
test_set_mse = mean_squared_error(flow, pred)
#
# Print R_squared and RMSE value
#
pred = np.reshape(pred,(length_data,1))
end_nn = time.time()
if __name__ == "__main__":
    print('RMSE: ', test_set_mse)

    plt.plot(waktu,set_point_csv,label="set point",color="orange",linestyle="dotted")
    plt.plot(x_train,y_train,label="flow estimated",color="red")
    plt.plot(x_test,pred,label="prediksi",color="blue")
    plt.legend()
    plt.grid(True)
    plt.show()