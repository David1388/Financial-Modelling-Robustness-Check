import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from models_task3 import logger


def Calculate_portfolio(xgt, txn):
    logger.info("Start calculating portfolio")
    tickers = txn.weights_df.index.tolist()
    df_prices = txn.df[tickers].copy()
    df_prices["Portfolio"] = txn.weighted_df["Price"]
    xgt.df_prices = df_prices
    logger.info(f"Portfolio calculation completed{df_prices}")
    return df_prices


def calculate_SMA(series, period):
    return series.rolling(window=period).mean()


def calculate_EMA(series, period):
    if series.isna().all():
        return pd.Series([], index=series.index)

    alpha = 2 / (period + 1)
    first_valid = series.dropna().iloc[0]
    ema = [first_valid]

    for price in series.dropna().iloc[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])

    return pd.Series(ema, index=series.dropna().index).reindex(series.index)


def calculate_RSI(series, period):
    if series.isna().all():
        return pd.Series([], index=series.index)

    delta = series.diff().fillna(0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = (
        pd.Series(gain, index=series.index).rolling(window=period, min_periods=1).mean()
    )
    avg_loss = (
        pd.Series(loss, index=series.index).rolling(window=period, min_periods=1).mean()
    )

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


def calculate_MACD(series, short_period, long_period, signal_period):
    if series.isna().all():
        return (
            pd.Series([], index=series.index),
            pd.Series([], index=series.index),
            pd.Series([], index=series.index),
        )

    ema_short = calculate_EMA(series, short_period).fillna(0)
    ema_long = calculate_EMA(series, long_period).fillna(0)
    macd = ema_short - ema_long
    signal_line = calculate_EMA(macd, signal_period).fillna(0)
    macd_histogram = macd - signal_line
    return macd, signal_line, macd_histogram


def Calculate_features(xgt):
    logger.info("Start calculating features")

    if xgt.df_prices.empty:
        logger.warning("xgt.df_prices empty！")

    xgt.df_prices["Portfolio"] = xgt.df_prices["Portfolio"].ffill()

    if not hasattr(xgt, "run_count"):
        xgt.run_count = 0

    sma_period = 4 if xgt.run_count == 0 else 24
    xgt.df_prices["SMA_24"] = calculate_SMA(xgt.df_prices["Portfolio"], sma_period)
    logger.info(f"Current run count: {xgt.run_count}")
    xgt.df_prices["EMA_3"] = calculate_EMA(xgt.df_prices["Portfolio"], 3)
    xgt.df_prices["EMA_11"] = calculate_EMA(xgt.df_prices["Portfolio"], 11)
    xgt.df_prices["RSI_2"] = calculate_RSI(xgt.df_prices["Portfolio"], 2)

    macd, signal_line, macd_histogram = calculate_MACD(
        xgt.df_prices["Portfolio"], 3, 11, 7
    )
    xgt.df_prices["MACD"] = macd
    xgt.df_prices["Signal_Line"] = signal_line
    xgt.df_prices["MACD_Histogram"] = macd_histogram

    xgt.df_prices.dropna(inplace=True)

    logger.info("Features calculation completed")
    return xgt.df_prices


def Constructing_training_data(xgt):
    logger.info("Constructing training data...")
    features = ["SMA_24", "RSI_2", "Signal_Line", "MACD_Histogram"]

    if xgt.df_prices.empty:
        logger.warning("xgt.df_prices empty")
        logger.info("df_prices:\n", xgt.df_prices.head())

    missing_features = [f for f in features if f not in xgt.df_prices.columns]
    if missing_features:
        logger.warning(f"loss features: {missing_features}")

    X = xgt.df_prices[features].copy()
    y = xgt.df_prices["Portfolio"].shift(-1).dropna()
    X = X.iloc[:-1]

    X.index = pd.to_datetime(X.index, errors="coerce")

    test_months = [
        (2022, 7),
        (2022, 11),
        (2023, 7),
        (2023, 2),
        (2023, 12),
        (2023, 5),
        (2024, 4),
        (2024, 2),
    ]

    mask = X.index.to_series().apply(
        lambda x: (x.year, x.month) in test_months if pd.notna(x) else False
    )
    test_indices = X.index[mask]

    X_test, y_test = X.loc[test_indices], y.loc[test_indices]
    X_train, y_train = X.drop(test_indices), y.drop(test_indices)

    xgt.X_test = X_test
    xgt.X_train = X_train
    xgt.y_test = y_test
    xgt.y_train = y_train
    xgt.y = y
    xgt.X = X

    logger.info("Constructing training data completed")
    return X_test, y_test, X_train, y_train, X, y


def Training_XGBoost(xgt):
    logger.info("Start training XGBoost model...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=1,
        gamma=0.2,
        subsample=0.65,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        reg_alpha=0.0,
        early_stopping_rounds=6,
        eval_metric="rmse",
        random_state=42,
    )

    model.fit(
        xgt.X_train, xgt.y_train, eval_set=[(xgt.X_test, xgt.y_test)], verbose=False
    )
    logger.info("XGBoost training completed")
    return model


def Prediction_test(xgt, model):
    y_pred = model.predict(xgt.X_test)
    xgt.y_pred = y_pred
    return y_pred


def Evaluate_model(xgt):
    logger.info("Start evaluating XGBoost prediction results")
    y_mean = np.mean(xgt.y)
    rmse = np.sqrt(mean_squared_error(xgt.y_test, xgt.y_pred))
    rmse_percentage = (rmse / y_mean) * 100

    y_test_series = xgt.y_test.copy()
    y_pred_series = pd.Series(xgt.y_pred, index=xgt.y_test.index)

    mse = mean_squared_error(xgt.y_test, xgt.y_pred)
    r2 = r2_score(xgt.y_test, xgt.y_pred)

    direction_accuracy = (
        np.mean(
            (np.sign(y_test_series.diff()) == np.sign(y_pred_series.diff())).dropna()
        )
        * 100
    )
    direction_diff = np.sign(y_test_series.diff()) - np.sign(y_pred_series.diff())
    print(f" Test RMSE: {rmse:.4f}")
    print(f" rmse_percentage: {rmse_percentage:.2f}%")
    print(f" Direction Accuracy: {direction_accuracy:.3f}%")
    print(f" R² Score: {r2:.4f}")
    print(f" MSE: {mse:.4f}")

    xgt.rmse_percentage = rmse_percentage
    xgt.mse = mse
    xgt.r2 = r2
    xgt.direction_accuracy = direction_accuracy
    xgt.direction_diff = direction_diff
    logger.info(
        f"rmse:{rmse},rmse_percentage:{rmse_percentage},mse:{mse},r2:{r2}, direction_accuracy:{direction_accuracy}, direction_diff:{direction_diff.value_counts()}"
    )

    return rmse_percentage, mse, r2, direction_accuracy, direction_diff


def Future_predictions(model, xgt):
    logger.info("Generate forecast results for the next 84 days")
    last_date = xgt.df_prices.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 85)]
    df_future = pd.DataFrame(index=future_dates, columns=xgt.X.columns)
    last_features = xgt.X.iloc[-1].copy()

    pred_prices = list(xgt.df_prices["Portfolio"].iloc[-max(24, 11) :])
    future_predictions = []

    for i in range(84):
        pred_price = model.predict(last_features.values.reshape(1, -1))[0]
        future_predictions.append(pred_price)
        pred_prices.append(pred_price)
        if len(pred_prices) > 24:
            pred_prices.pop(0)

        macd, signal_line, macd_histogram = calculate_MACD(
            pd.Series(pred_prices), 3, 11, 7
        )
        last_features["Signal_Line"] = (
            signal_line.iloc[-1] if len(signal_line) > 0 else 0
        )
        last_features["MACD_Histogram"] = (
            macd_histogram.iloc[-1] if len(macd_histogram) > 0 else 0
        )

        rsi_series = calculate_RSI(pd.Series(pred_prices[-2:]), 2)
        last_features["RSI_2"] = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50

        sma_period = 4 if xgt.run_count == 0 else 24
        sma_series = calculate_SMA(pd.Series(pred_prices), sma_period)
        last_features["SMA_24"] = (
            sma_series.iloc[-1] if len(sma_series) > 0 else pred_price
        )

        df_future.iloc[i] = last_features
    xgt.run_count += 1
    logger.info(f"Current run count: {xgt.run_count}")
    df_future["Predicted Portfolio"] = future_predictions
    xgt.df_future = df_future
    logger.info("Future prediction completed")
    return df_future


def xglog_return(xgt):
    logger.info("Calculate Log Return for the next 84 days")
    xgt.df_future["Log Return"] = np.log(
        xgt.df_future["Predicted Portfolio"]
        / xgt.df_future["Predicted Portfolio"].shift(1)
    )
    total_log_return = xgt.df_future["Log Return"].sum()
    print(f" Total Log Return after 84 days: {total_log_return:.6f}")

    xgt.total_log_return = total_log_return
    logger.info(f"Total Log Return for the next 84 days: {total_log_return:.6f}")
    return xgt.df_future, total_log_return




def Max_Drawdown(xgt):
    logger.info("Calculate the maximum drawdown (Max Drawdown)")
    xgt.df_future["Cumulative Return"] = (
        xgt.df_future["Predicted Portfolio"]
        / xgt.df_future["Predicted Portfolio"].iloc[0]
    ) - 1
    xgt.df_future["Cumulative Max"] = xgt.df_future["Cumulative Return"].cummax()
    xgt.df_future["Drawdown"] = (
        xgt.df_future["Cumulative Return"] - xgt.df_future["Cumulative Max"]
    )
    max_drawdown = xgt.df_future["Drawdown"].min()

    print(f" Max Drawdown: {max_drawdown:.6f}")
    logger.info(f"Max Drawdown: {max_drawdown:.6f}")
    xgt.max_drawdown = max_drawdown
    return xgt.df_future, max_drawdown


def Plot_training_testing(xgt, portfolio_file):
    xgt.y_train = xgt.y_train.sort_index()
    xgt.y_test = xgt.y_test.sort_index()
    xgt.y_pred = pd.Series(xgt.y_pred, index=xgt.y_test.index).sort_index()

    df_plot = pd.DataFrame(index=xgt.y_train.index.union(xgt.y_test.index))
    df_plot["Train Actual"] = xgt.y_train
    df_plot["Test Actual"] = xgt.y_test
    df_plot["Test Predicted"] = pd.Series(xgt.y_pred, index=xgt.y_test.index)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_plot.index,
        df_plot["Train Actual"],
        label="Train Actual",
        color="gray",
        linestyle="dashed",
        alpha=0.6,
    )
    plt.plot(df_plot.index, df_plot["Test Actual"], label="Test Actual", color="blue")
    plt.plot(
        df_plot.index, df_plot["Test Predicted"], label="Test Predicted", color="orange"
    )

    plt.legend()
    plt.title(f"{portfolio_file}Portfolio Price Prediction (Train & Test)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Price")
    plt.grid(True)

    plt.show()


def plot_importance_features(model):
    xgb.plot_importance(model)
    plt.show()


def polt_future(xgt, portfolio_file):
    plt.figure(figsize=(12, 6))

    plt.plot(
        xgt.df_future.index,
        xgt.df_future["Predicted Portfolio"],
        label="Future Predictions",
        color="red",
        linestyle="dashed",
    )

    plt.legend()
    plt.title(f"{portfolio_file} Future Portfolio Price Prediction (Next 84 Days)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Price")
    plt.grid(True)

    plt.show()


def Plot_Max_Drawdown(xgt, portfolio_file):
    plt.figure(figsize=(12, 6))
    plt.plot(
        xgt.df_future.index,
        xgt.df_future["Cumulative Return"],
        label="Cumulative Return",
        color="blue",
    )
    plt.fill_between(
        xgt.df_future.index,
        xgt.df_future["Drawdown"],
        color="red",
        alpha=0.3,
        label="Drawdown",
    )
    plt.plot(
        xgt.df_future.index,
        xgt.df_future["Cumulative Max"],
        linestyle="dashed",
        color="gray",
        label="Cumulative Max",
    )

    min_drawdown_index = xgt.df_future["Drawdown"].idxmin()
    plt.scatter(
        min_drawdown_index,
        xgt.df_future["Drawdown"].min(),
        color="red",
        label=f"Max Drawdown: {xgt.max_drawdown:.2%}",
    )

    plt.title(f"{portfolio_file} Future Portfolio Cumulative Return & Max Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    plt.show()
