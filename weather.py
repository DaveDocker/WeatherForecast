import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample weather data
data = {
    'Day': np.arange(1, 21),
    'HighTemp': [30, 32, 33, 28, 25, 29, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 25, 28, 30, 31],
    'LowTemp': [20, 22, 23, 18, 15, 19, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 15, 18, 20, 21]
}
df = pd.DataFrame(data)

# Create the input data for the neural network
X = []
Y_high = []
Y_low = []

for i in range(5, len(df)):
    X.append(df.iloc[i-5:i, 1:3].values.flatten())
    Y_high.append(df.iloc[i, 1])
    Y_low.append(df.iloc[i, 2])

X = np.array(X)
Y_high = np.array(Y_high)
Y_low = np.array(Y_low)

# Build the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X, np.column_stack((Y_high, Y_low)), epochs=100, batch_size=1, verbose=1)

# Make predictions
predictions = model.predict(X)

# Create a DataFrame for analysis
df_results = df[5:].copy()
df_results['Predicted_High'] = predictions[:, 0]
df_results['Predicted_Low'] = predictions[:, 1]

# Print the DataFrame
print(df_results)
