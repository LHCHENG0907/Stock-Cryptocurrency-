# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Set stock code and date ranges
stock_code = "stock or crypto or coin ticker"
start_date_historical = "start_date"
end_date_historical = "end_date"
end_date_lstm = "end_date"

# Download historical data
historical_data = yf.download(stock_code, start=start_date_historical, end=end_date_historical)

# Download data for LSTM model
stock_data = yf.download(stock_code, start=start_date_historical, end=end_date_lstm)

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

# Build LSTM model with adjustments
model = Sequential()
model.add(LSTM(units=25, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))  # Adding dropout layer
model.add(LSTM(units=25, return_sequences=True))
model.add(Dropout(0.2))  # Adding dropout layer
model.add(LSTM(units=25))
model.add(Dropout(0.2))  # Adding dropout layer
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Implement early stopping with increased patience
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model with more epochs
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Evaluate on the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
mse = mean_squared_error(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)
rmse = mean_squared_error(y_test, test_predictions, squared=False)

# Calculate percentage errors
mse_percentage = (mse / np.mean(y_test)) * 100
mae_percentage = (mae / np.mean(y_test)) * 100
rmse_percentage = (rmse / np.mean(y_test)) * 100

print(f'Mean Squared Error on Test Set: {mse}')
print(f'Mean Absolute Error on Test Set: {mae}')
print(f'Root Mean Squared Error on Test Set: {rmse}')
print(f'Mean Squared Error on Test Set (Percentage): {mse_percentage:.2f}%')
print(f'Mean Absolute Error on Test Set (Percentage): {mae_percentage:.2f}%')
print(f'Root Mean Squared Error on Test Set (Percentage): {rmse_percentage:.2f}%')

# Calculate accuracy percentage
accuracy_percentage = 100 - rmse_percentage
print(f'Accuracy: {accuracy_percentage:.2f}%')

# Generate predictions for the future period
future_data = data.copy()
for _ in range(len(pd.date_range(start=end_date_lstm, periods=30, freq='B'))):
    # Prepare input data for prediction
    input_data = future_data[-time_steps:]
    input_data = scaler.transform(input_data)
    input_data = np.reshape(input_data, (1, time_steps, 1))

    # Make the prediction
    next_month_prediction = model.predict(input_data)
    next_month_prediction = scaler.inverse_transform(next_month_prediction)

    # Append the prediction to the future data
    future_data = future_data.append(pd.DataFrame(next_month_prediction, columns=['Close'], index=[future_data.index[-1] + pd.Timedelta(days=1)]))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(historical_data.index, historical_data['Close'], label='Historical Close Price', linewidth=2)
plt.plot(future_data.index, future_data['Close'], label='Predicted Close Price', linestyle='dashed', linewidth=2)
plt.title('Stock Price Prediction for the Next Month')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
