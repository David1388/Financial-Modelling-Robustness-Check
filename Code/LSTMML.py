import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
from LSTM_model import logger, MyTransaction
import joblib
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop


def Read_Stock(txn):
    txn.df = pd.read_csv(
        "data/Adj_Close_data.csv", parse_dates=["Date"], index_col="Date"
    )
    logger.info(txn.df.head())
    return txn.df


def create_sequences(data, seq_length, date_index=None):
    X, y, time_index = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
        if date_index is not None:
            time_index.append(date_index[i + seq_length])
    return np.array(X), np.array(y), np.array(time_index)


def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(output_dim))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss="mse")
    return model


def training_model(txn):
    scaler = MinMaxScaler()
    txn.scaled_data = scaler.fit_transform(txn.df)
    X, y, time_index = create_sequences(txn.scaled_data, txn.SEQ_LEN, txn.df.index)

    txn.dates = np.array(time_index)
    txn.X = X
    txn.y = y

    X = X.reshape((X.shape[0], X.shape[1], txn.df.shape[1]))

    test_mask = np.array([(d.year, d.month) in txn.test_months for d in txn.dates])

    X_train = X[~test_mask]
    y_train = y[~test_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    model = build_model((txn.SEQ_LEN, txn.df.shape[1]), txn.df.shape[1])

    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)
    txn.y_rescaled = scaler.inverse_transform(txn.y)
    # model.save("saved_models/lstm_stock_modelV8.h5")
    # joblib.dump(scaler, "saved_models/scalerV8.pkl")

    return (
        predictions_rescaled,
        y_test_rescaled,
        txn.df.columns,
        history,
        test_mask,
        model,
        scaler,
    )


def forecast_future_prices(txn, model, scaler, future_days=79):
    last_sequence = txn.X[-1]
    future_preds_scaled = []

    for _ in range(future_days):
        pred = model.predict(last_sequence[np.newaxis, :, :])[0]
        future_preds_scaled.append(pred)
        last_sequence = np.vstack([last_sequence[1:], pred])

    future_preds = scaler.inverse_transform(future_preds_scaled)
    return future_preds


def plot_all_stocks_prediction(txn, predictions_rescaled, test_mask):
    all_dates = txn.dates
    full_prices = txn.y_rescaled
    stock_names = txn.df.columns

    for i, stock in enumerate(stock_names):
        full_price = full_prices[:, i]
        test_preds = predictions_rescaled[:, i]

        train_actual = np.full_like(full_price, np.nan)
        test_actual = np.full_like(full_price, np.nan)
        test_predicted = np.full_like(full_price, np.nan)

        train_actual[~test_mask] = full_price[~test_mask]
        test_actual[test_mask] = full_price[test_mask]
        test_predicted[test_mask] = test_preds

        plt.figure(figsize=(14, 6))
        plt.plot(
            all_dates,
            train_actual,
            label="Train Actual",
            linestyle="--",
            color="gray",
            alpha=0.5,
        )
        plt.plot(all_dates, test_actual, label="Test Actual", color="blue")
        plt.plot(all_dates, test_predicted, label="Test Predicted", color="orange")

        plt.title(f"{stock} - LSTM Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_future_predictions(txn, future_preds):
    future_dates = pd.date_range(
        start=txn.df.index[-1], periods=len(future_preds) + 1, freq="B"
    )[1:]
    for i, stock in enumerate(txn.df.columns):
        plt.figure(figsize=(12, 5))
        plt.plot(txn.df.index[-60:], txn.df.iloc[-60:, i], label="Recent Prices")
        plt.plot(future_dates, future_preds[:, i], label="Future Prediction")
        plt.title(f"{stock} - Future Price Prediction (Next 4 Months)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_loss_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_per_stock(predictions, y_true, stock_names):

    print("Evaluation of each stock on the test set:")
    for i, stock in enumerate(stock_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], predictions[:, i]))
        r2 = r2_score(y_true[:, i], predictions[:, i])
        print(f"{stock}: RMSE = {rmse:.4f}, R² = {r2:.4f}")


def main():
    txn = MyTransaction()
    Read_Stock(txn)
    (
        predictions_rescaled,
        y_test_rescaled,
        stock_names,
        history,
        test_mask,
        model,
        scaler,
    ) = training_model(txn)
    plot_all_stocks_prediction(txn, predictions_rescaled, test_mask)
    plot_loss_curve(history)
    evaluate_per_stock(predictions_rescaled, y_test_rescaled, stock_names)
    future_preds = forecast_future_prices(txn, model, scaler, future_days=79)
    plot_future_predictions(txn, future_preds)
    return txn


if __name__ == "__main__":
    txn = main()
