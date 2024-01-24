# TWII-
Python- TWII trend analysis
# Install yfinance
!pip install yfinance

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Set stocknum and date for historical data
stock_code_historical = "^TWII"
start_date_historical = "2023-01-01"
end_date_historical = "2024-01-01"

# Download historical data
historical_data = yf.download(stock_code_historical, start=start_date_historical, end=end_date_historical)

# Print historical data
print(historical_data[['Open', 'High', 'Low', 'Close']])

# Set stocknum and date for LSTM model
stock_code_lstm = "^TWII"
start_date_lstm = "2023-01-01"
end_date_lstm = "2024-02-01"  # Extend the end date for future predictions

# Download data for LSTM model
stock_data = yf.download(stock_code_lstm, start=start_date_lstm, end=end_date_lstm)

# Select closing price as the target variable
data = stock_data[['Close']]

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create time series data
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        a = dataset[i:(i+time_steps), 0]
        X.append(a)
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_dataset(scaled_data, time_steps)

# Convert to 3D shape [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Implement early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Evaluate on the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
mse = mean_squared_error(y_test, test_predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# Create future dates for prediction
future_dates = pd.date_range(start=end_date_lstm, periods=30, freq='B')  # Assuming 30 business days in a month

# Make predictions for the next month
future_data = data.copy()
for _ in range(len(future_dates)):
    # Prepare input data for prediction
    input_data = future_data[-time_steps:]
    input_data = scaler.transform(input_data)
    input_data = np.reshape(input_data, (1, time_steps, 1))

    # Make the prediction
    next_month_prediction = model.predict(input_data)
    next_month_prediction = scaler.inverse_transform(next_month_prediction)

    # Append the prediction to the future data
    future_data = future_data.append(pd.DataFrame(next_month_prediction, columns=['Close'], index=[future_dates[_]]))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Actual Close Price', linewidth=2)
plt.plot(future_data.index, future_data['Close'], label='Predicted Close Price', linestyle='dashed', linewidth=2)
plt.title('Stock Price Prediction for the Next Month')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
