import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess the dataset
file_path = 'XAU_USD Historical Data.csv'  # Update this path
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Price'] = data['Price'].str.replace(',', '').astype(float)
data.sort_values('Date', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data['Normalized Price'] = scaler.fit_transform(data[['Price']])
model_data = data[['Date', 'Normalized Price']].set_index('Date')

# Function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)]
        y = data.iloc[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Creating sequences and splitting the data
sequence_length = 60
X, y = create_sequences(model_data['Normalized Price'], sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Building the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.1)

# Evaluating the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Making predictions
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)
