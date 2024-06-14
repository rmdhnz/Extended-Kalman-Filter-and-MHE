import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("data/data_p_1.2_i_0.8_d_0.2.csv")
# Parameter model
true_velocity = 2  # True velocity (m/s)
num_steps = len(data["Flow Measured Value"])     # Number of time steps
waktu = data["Time"]
set_point = data["Flow Set Point"]   # Number of time steps
timesteps = 10     # Number of time steps to use as input for prediction

# Generate noisy measurements
np.random.seed(42)  # For reproducibility
measurements = data["Flow Measured Value"]

# Prepare dataset
X = []
y = []
for i in range(timesteps, num_steps):
    X.append(measurements[i-timesteps:i])
    y.append(measurements[i])
X, y = np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(timesteps, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform the test set for comparison
y_test = scaler.inverse_transform(y_test)

# Plotting results
plt.figure()
plt.plot(y_test, 'b-', label='True Velocity')
plt.plot(predictions, 'r-', label='Predicted Velocity')
plt.xlabel('Time')
plt.ylabel('Water Flow Velocity (m/s)')
plt.legend()
plt.title('Water Flow Velocity Estimation using Neural Network')
plt.grid(True)
plt.show()
