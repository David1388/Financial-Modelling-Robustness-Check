import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from LSTMML import create_sequences, evaluate_per_stock
from LSTM_model import MyTransaction


def evaluate(txn, scaler, df, model):
    scaled_data = scaler.transform(df)
    X, y, dates = create_sequences(scaled_data, txn.SEQ_LEN, df.index)
    X = X.reshape((X.shape[0], X.shape[1], df.shape[1]))

    test_mask = np.array([(d.year, d.month) in txn.test_months for d in dates])
    X_test = X[test_mask]
    y_test = y[test_mask]

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)

    evaluate_per_stock(predictions_rescaled, y_test_rescaled, df.columns)


def main():
    txn = MyTransaction()
    df = pd.read_csv("data/Adj_Close_data.csv", parse_dates=["Date"], index_col="Date")
    txn.df = df
    scaler = joblib.load("saved_models/scalerV6.pkl")
    model = load_model("saved_models/lstm_stock_modelV6.h5")
    evaluate(txn, scaler, df, model)
    return txn


if __name__ == "__main__":
    txn = main()
