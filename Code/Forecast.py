import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from LSTM_model import MyTransaction, logger
from LSTMML import create_sequences


MODEL_PATH = "saved_models/lstm_stock_modelV6.h5"
SCALER_PATH = "saved_models/scalerV6.pkl"
DATA_PATH = "data/Adj_Close_data.csv"


def load_data_and_prepare_txn(txn):
    txn = MyTransaction()
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    txn.df = df

    txn.stocks = df.columns.tolist()
    scaler = joblib.load(SCALER_PATH)
    scaled_data = scaler.transform(df)
    txn.scaled_data = scaled_data

    X, y, time_index = create_sequences(scaled_data, txn.SEQ_LEN, df.index)
    txn.X = X
    txn.y = y
    txn.dates = np.array(time_index)

    return txn, scaler


def forecast_future_prices(txn, model, scaler, future_days=79):
    last_sequence = txn.X[-1]
    future_preds_scaled = []

    for _ in range(future_days):
        pred = model.predict(last_sequence[np.newaxis, :, :], verbose=0)[0]
        future_preds_scaled.append(pred)
        last_sequence = np.vstack([last_sequence[1:], pred])

    txn.future_preds = scaler.inverse_transform(future_preds_scaled)
    return txn.future_preds


def Table_Stock(txn, future_dates):
    df_future = pd.DataFrame(txn.future_preds, index=future_dates, columns=txn.stocks)
    df_future.reset_index(inplace=True)
    df_future.rename(columns={"index": "Date"}, inplace=True)

    last_date = txn.df.index[-1]
    last_values = txn.df.iloc[-1][txn.stocks]
    last_row = pd.DataFrame(
        [[last_date] + list(last_values)], columns=["Date"] + txn.stocks
    )

    df_future = pd.concat([last_row, df_future], ignore_index=True)

    return df_future


def generate_future_trading_dates(start_date="2025-01-01", future_days=79):
    market_holidays = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-29",
            "2025-01-30",
            "2025-02-11",
            "2025-03-18",
            "2025-03-31",
            "2025-04-01",
            "2025-05-01",
            "2025-05-12",
            "2025-05-30",
            "2025-05-31",
            "2025-06-02",
            "2025-06-07",
            "2025-06-27",
            "2025-08-31",
            "2025-09-05",
            "2025-09-16",
            "2025-10-20",
            "2025-12-25",
        ]
    )
    start_date = pd.to_datetime(start_date)
    all_future_dates = pd.bdate_range(start=start_date, periods=future_days + 30)
    valid_future_dates = all_future_dates[~all_future_dates.isin(market_holidays)][
        :future_days
    ]
    logger.info(valid_future_dates)
    return valid_future_dates


def plot_future_predictions(txn, future_preds, future_dates):
    txn.df.index = pd.to_datetime(txn.df.index, dayfirst=True)

    for i, stock in enumerate(txn.stocks):
        plt.figure(figsize=(12, 5))
        plt.plot(txn.df.index[-60:], txn.df.iloc[-60:, i], label="Recent Prices")

        plt.plot(future_dates, future_preds[:, i], label="Future Prediction")

        plt.title(
            f"{stock} - Future Price Prediction (Next {len(future_dates)} Business Days)"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def calculate_portfolio_returns(df_future, pf_paths, txn):
    df_future = df_future.copy()
    df_future.set_index("Date", inplace=True)

    df_ret = df_future.pct_change().dropna()

    cutoff_date = pd.to_datetime("2025-02-19")
    before_df = df_ret[df_ret.index < cutoff_date]
    after_df = df_ret[df_ret.index >= cutoff_date]

    portfolio_returns = pd.DataFrame(index=df_ret.index)

    for path in pf_paths:
        strategy_name = path.split("/")[-1].replace(".csv", "")

        weights_df_before = pd.read_csv(path)[["stock_code", "optimal_weight"]]
        weights_df_before.set_index("stock_code", inplace=True)
        weights_before = weights_df_before.loc[df_ret.columns]["optimal_weight"].values
        port_ret_before = before_df.values @ weights_before
        port_ret_before_series = pd.Series(port_ret_before, index=before_df.index)

        if path in txn.updated_weights:
            weights_df_after = txn.updated_weights[path][
                ["stock_code", "optimal_weight"]
            ]
            weights_df_after.set_index("stock_code", inplace=True)

            common_stocks = weights_df_after.index.intersection(df_ret.columns)
            weights_after = weights_df_after.loc[common_stocks, "optimal_weight"].values
            port_ret_after = after_df[common_stocks].values @ weights_after
            port_ret_after_series = pd.Series(port_ret_after, index=after_df.index)

        else:
            port_ret_after_series = pd.Series(
                index=after_df.index, data=[0] * len(after_df)
            )

        full_ret_series = pd.concat([port_ret_before_series, port_ret_after_series])
        portfolio_returns[strategy_name] = full_ret_series
        txn.portfolio_returns[strategy_name] = full_ret_series

    return portfolio_returns


def evaluate_returns(portfolio_returns, start_date_str="2024-12-30", days=84):
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date + pd.Timedelta(days=days - 1)

    print(
        f"\nStrategy from {start_date_str} cumulative return for the next {days} days starting from now："
    )
    for strategy in portfolio_returns.columns:
        ret_series = portfolio_returns[strategy].loc[start_date:end_date]
        cumulative_return = (1 + ret_series).prod() - 1
        print(f"{strategy}: {cumulative_return:.2%}")
    return portfolio_returns


def plot_return(portfolio_returns):
    ((1 + portfolio_returns).cumprod() - 1).plot(
        figsize=(10, 6), title="Cumulative Portfolio Returns (Compound)"
    )
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()


def redistribute_weights(
    txn, pf_paths, stock_to_redistribute="5014.KL", tolerance=0.001
):
    updated_weights = {}

    for path in pf_paths:
        weights_df = pd.read_csv(path)
        logger.info(f"Missing value statistics for file {path}:")
        logger.info(weights_df.isnull().sum())

        if stock_to_redistribute in weights_df["stock_code"].values:
            stock_weight = weights_df.loc[
                weights_df["stock_code"] == stock_to_redistribute, "optimal_weight"
            ].values[0]
            logger.info(f"{stock_to_redistribute} Weight: {stock_weight}")

            remaining_weights_df = weights_df[
                weights_df["stock_code"] != stock_to_redistribute
            ]
            remaining_weight_sum = remaining_weights_df["optimal_weight"].sum()
            logger.info(f"Total weight of remaining stocks: {remaining_weight_sum}")
            logger.info(f"The number of remaining stocks: {len(remaining_weights_df)}")

            redistributed_weight = stock_weight / len(remaining_weights_df)
            logger.info(
                f"The weight assigned to each remaining stock: {redistributed_weight}"
            )

            remaining_weights_df.loc[:, "optimal_weight"] += redistributed_weight

            logger.info(
                f"Updated remaining total stock weight: {remaining_weights_df['optimal_weight'].sum()}"
            )

            updated_weights[path] = remaining_weights_df

        else:
            updated_weights[path] = weights_df

        total_weight = updated_weights[path]["optimal_weight"].sum()
        if abs(total_weight - 1) > tolerance:
            logger.warning(
                f"Warning: Total weight of file {path} is {total_weight:.4f}, not 1!"
            )

        txn.updated_weights[path] = updated_weights[path]

        # new_file_path = path.replace('.csv', '_updated.csv')
        # updated_weights[path].to_csv(new_file_path, index=False)
        # print(f"{new_file_path}")

    return updated_weights


def calculate_max_drawdown(portfolio_returns):
    max_drawdowns = {}
    for strategy in portfolio_returns.columns:
        cumulative_returns = (1 + portfolio_returns[strategy]).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdowns[strategy] = max_drawdown
        print(f"{strategy} maximum drawdown is: {max_drawdown:.2%}")

    return max_drawdowns


def plot_max_drawdown(portfolio_returns):
    for strategy in portfolio_returns.columns:
        cumulative_returns = (1 + portfolio_returns[strategy]).cumprod()

        peak = cumulative_returns.cummax()

        drawdown = (cumulative_returns - peak) / peak

        plt.figure(figsize=(10, 6))
        plt.plot(drawdown, label="Drawdown", color="red")
        plt.fill_between(drawdown.index, drawdown, color="red", alpha=0.3)
        plt.title(f"{strategy} maximum drawdown")
        plt.xlabel("Date")
        plt.ylabel("Retracement ratio")
        plt.legend()
        plt.show()


def main():
    txn = MyTransaction()
    txn, scaler = load_data_and_prepare_txn(txn)

    model = load_model(MODEL_PATH)

    future_preds = forecast_future_prices(txn, model, scaler, future_days=79)

    future_dates = generate_future_trading_dates(
        start_date="2025-01-01", future_days=79
    )

    plot_future_predictions(txn, future_preds, future_dates)

    df_future = Table_Stock(txn, future_dates)
    pf_paths = [
        "data/Min Variance.csv",
        "data/Max Sharpe.csv",
        "data/Steepest Descent.csv",
    ]
    redistribute_weights(txn, pf_paths, stock_to_redistribute="5014.KL")

    portfolio_returns = calculate_portfolio_returns(df_future, pf_paths, txn)
    evaluate_returns(portfolio_returns, start_date_str="2024-12-30", days=79)
    calculate_max_drawdown(portfolio_returns)

    plot_max_drawdown(portfolio_returns)
    plot_return(portfolio_returns)

    return txn, df_future


if __name__ == "__main__":
    txn = main()
