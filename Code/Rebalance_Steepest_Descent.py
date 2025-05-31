import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Reb_model import MyTransaction, logger


def read(txn):
    price_data = pd.read_csv("data/Adj_Close_data.csv", index_col=0, parse_dates=True)
    if price_data.ndim == 1:
        price_data = price_data.to_frame()
    txn.price_data = price_data
    logger.info("Price data loaded.")
    return price_data


def log_return(txn):
    log_return = np.log(txn.price_data / txn.price_data.shift(1)).dropna()
    txn.log_return = log_return
    logger.info(f"Log Returns calculated:\n{log_return.head()}")
    return log_return


def cov_matrices(txn, start_idx, end_idx):
    cov = txn.log_return.iloc[start_idx:end_idx].cov()
    txn.cov_matrices = cov
    return cov


def mean_returns(txn, start_idx, end_idx):
    mean = txn.log_return.iloc[start_idx:end_idx].mean() * 252
    txn.mean_returns = mean
    return mean


def mean_variance_objective(w, mu, Sigma, risk_aversion):
    return -(mu.T @ w) + (risk_aversion / 2) * (w.T @ Sigma @ w)


def mean_variance_gradient(w, mu, Sigma, risk_aversion):
    return -mu + risk_aversion * (Sigma @ w)


def steepest_descent_mean_variance(
    mu, Sigma, risk_aversion, learning_rate=0.001, iterations=10000
):
    n_assets = len(mu)
    w = np.ones(n_assets) / n_assets
    w = w / np.sum(w)

    history = []

    for i in range(iterations):
        grad = mean_variance_gradient(w, mu, Sigma, risk_aversion)
        w = w - learning_rate * grad

        w = np.maximum(w, 0)
        w = w / np.sum(w)

        history.append(mean_variance_objective(w, mu, Sigma, risk_aversion))

    return w, history


def rebalancing(txn, optimal_weights):
    thresholds = np.linspace(0.0, 1.0, 101)
    last_weights = (
        txn.last_rebalanced_weights[-1]
        if txn.last_rebalanced_weights
        else np.array(optimal_weights)
    )

    for th in thresholds:
        if th not in txn.rebalanced_weights_by_threshold:
            txn.rebalanced_weights_by_threshold[th] = []
            txn.rebalanced_prices_by_threshold[th] = []
            txn.rebalance_count_by_threshold[th] = 0

        if len(txn.rebalanced_weights_by_threshold[th]) == 0:
            txn.rebalanced_weights_by_threshold[th].append(np.array(optimal_weights))
        else:
            last_weights = txn.rebalanced_weights_by_threshold[th][-1]

        weight_changes = np.abs(np.array(optimal_weights) - last_weights)
        triggered = np.any(weight_changes > th)

        if triggered:
            txn.rebalanced_weights_by_threshold[th].append(np.array(optimal_weights))
            txn.rebalance_count_by_threshold[th] += 1
        else:
            txn.rebalanced_weights_by_threshold[th].append(last_weights)

        current_day_index = len(txn.rebalanced_weights_by_threshold[th]) + 251
        if current_day_index < len(txn.price_data):
            today_price = txn.price_data.iloc[current_day_index]
            current_weights = txn.rebalanced_weights_by_threshold[th][-1]
            portfolio_price = np.dot(today_price, current_weights)
            txn.rebalanced_prices_by_threshold[th].append(portfolio_price)


def calculate_all_threshold_returns(txn):
    returns = {}
    for th, prices in txn.rebalanced_prices_by_threshold.items():
        if len(prices) >= 2:
            ret = (prices[-1] / prices[0]) - 1
            returns[th] = ret
            logger.info(f"Threshold {th:.2f}: Return = {ret:.4f}")
            print(f"Threshold {th:.2f}: Return = {ret:.4f}")
            print(
                f"Threshold {th:.2f}: Rebalance Count = {txn.rebalance_count_by_threshold[th]}"
            )
    return returns


def plot_all_threshold_prices(txn):
    for th in sorted(txn.rebalanced_prices_by_threshold.keys()):
        prices = txn.rebalanced_prices_by_threshold[th]
        if len(prices) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(prices, label=f"Threshold {th:.2f}")
            plt.title(f"Portfolio Price over Time (Threshold {th:.2f})")
            plt.xlabel("Time (Days)")
            plt.ylabel("Portfolio Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def plot_threshold_returns(returns):
    thresholds = sorted(returns.keys())
    rets = [returns[th] for th in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, rets, marker="o")
    plt.xlabel("Rebalancing Threshold")
    plt.ylabel("Total Return")
    plt.title("Return vs Rebalancing Threshold")
    plt.grid(True)
    plt.show()


def main():
    txn = MyTransaction()
    read(txn)
    log_return(txn)
    window = 252
    risk_aversion = 5
    learning_rate = 0.001
    iterations = 10000

    for i in range(window, len(txn.log_return)):
        start_idx, end_idx = i - window, i
        rolling_cov = cov_matrices(txn, start_idx, end_idx)
        rolling_mean = mean_returns(txn, start_idx, end_idx)

        mu = rolling_mean.values
        Sigma = rolling_cov.values

        optimal_weights, history = steepest_descent_mean_variance(
            mu, Sigma, risk_aversion, learning_rate=learning_rate, iterations=iterations
        )

        rebalancing(txn, optimal_weights)

    returns = calculate_all_threshold_returns(txn)
    # plot_all_threshold_prices(txn)
    plot_threshold_returns(returns)

    return txn


if __name__ == "__main__":
    txn = main()
