import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    cov_matrices = txn.log_return.iloc[start_idx:end_idx].cov()
    txn.cov_matrices = cov_matrices
    return cov_matrices


def mean_returns(txn, start_idx, end_idx):
    mean_returns = txn.log_return.iloc[start_idx:end_idx].mean() * 252
    txn.mean_returns = mean_returns
    return mean_returns


def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)


def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def get_constraints_and_bounds(num_assets):
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    return constraints, bounds


def optimize_portfolio(target_return, rolling_mean, rolling_cov, num_assets):
    constraints_sum, bounds = get_constraints_and_bounds(num_assets)
    constraints = [
        constraints_sum,
        {
            "type": "eq",
            "fun": lambda w: portfolio_return(w, rolling_mean) - target_return,
        },
    ]
    init_guess = np.array([1 / num_assets] * num_assets)
    result = minimize(
        portfolio_variance,
        init_guess,
        args=(rolling_cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def min_risk(mean_returns, cov_matrices, txn):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    frontier_returns, frontier_std, portfolio_weights = [], [], []
    num_assets = len(mean_returns)

    for target in target_returns:
        result = optimize_portfolio(target, mean_returns, cov_matrices, num_assets)
        if result.success:
            weights = result.x
            ret = portfolio_return(weights, mean_returns)
            std = np.sqrt(portfolio_variance(weights, cov_matrices))
            frontier_returns.append(ret)
            frontier_std.append(std)
            portfolio_weights.append(weights)
        else:
            frontier_returns.append(np.nan)
            frontier_std.append(np.nan)
            portfolio_weights.append(None)

    min_risk_idx = np.nanargmin(frontier_std)
    txn.optimal_weights.append(portfolio_weights[min_risk_idx])
    logger.info(f"Optimal weights (min risk): {portfolio_weights[min_risk_idx]}")
    return (
        portfolio_weights,
        frontier_returns,
        frontier_std,
        portfolio_weights[min_risk_idx],
        frontier_returns[min_risk_idx],
        frontier_std[min_risk_idx],
    )


def plot_efficient_frontier(
    frontier_std, frontier_returns, optimal_std, optimal_return
):
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_std, frontier_returns, label="Efficient Frontier", color="blue")
    plt.scatter(
        optimal_std,
        optimal_return,
        color="red",
        marker="o",
        label="Max Sharpe Ratio Portfolio",
    )
    plt.xlabel("Volatility (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def rebalancing(txn, optimal_weights):
    thresholds = np.round(np.arange(0.01, 1.01, 0.01), 2)
    last_weights = (
        txn.last_rebalanced_weights[-1]
        if txn.last_rebalanced_weights
        else np.array(optimal_weights)
    )

    if not hasattr(txn, "scale_factors_by_threshold"):
        txn.scale_factors_by_threshold = {}
    if not hasattr(txn, "rebalance_count_by_threshold"):
        txn.rebalance_count_by_threshold = {}
    if not hasattr(txn, "rebalance_dates_by_threshold"):
        txn.rebalance_dates_by_threshold = {}

    for th in thresholds:
        if th not in txn.rebalanced_weights_by_threshold:
            txn.rebalanced_weights_by_threshold[th] = []
            txn.rebalanced_prices_by_threshold[th] = []
            txn.scale_factors_by_threshold[th] = []
            txn.rebalance_count_by_threshold[th] = 0
            txn.rebalance_dates_by_threshold[th] = []

        if len(txn.rebalanced_weights_by_threshold[th]) == 0:
            txn.rebalanced_weights_by_threshold[th].append(np.array(optimal_weights))
            base_price = np.dot(txn.price_data.iloc[251], optimal_weights)
            txn.rebalanced_prices_by_threshold[th].append(base_price)
            txn.scale_factors_by_threshold[th].append(1.0)
            continue
        else:
            last_weights = txn.rebalanced_weights_by_threshold[th][-1]

        weight_changes = np.abs(np.array(optimal_weights) - last_weights)
        triggered = np.any(weight_changes > th)

        current_day_index = len(txn.rebalanced_weights_by_threshold[th]) + 251
        if current_day_index >= len(txn.price_data):
            continue

        today_price = txn.price_data.iloc[current_day_index]
        new_weights = np.array(optimal_weights) if triggered else last_weights
        txn.rebalanced_weights_by_threshold[th].append(new_weights)

        current_price = np.dot(today_price, new_weights)

        if triggered:
            last_price = txn.rebalanced_prices_by_threshold[th][-1]
            scale_factor = last_price / current_price

            txn.rebalance_count_by_threshold[th] += 1
            txn.rebalance_dates_by_threshold[th].append(
                txn.price_data.index[current_day_index]
            )
        else:
            scale_factor = txn.scale_factors_by_threshold[th][-1]

        adjusted_price = current_price * scale_factor
        txn.rebalanced_prices_by_threshold[th].append(adjusted_price)
        txn.scale_factors_by_threshold[th].append(scale_factor)


def calculate_all_threshold_returns(txn):
    returns = {}
    for th, prices in txn.rebalanced_prices_by_threshold.items():
        if len(prices) >= 2:
            ret = (prices[-1] / prices[0]) - 1
            returns[th] = ret
            logger.info(f"Threshold {th:.2f}: Return = {ret:.4f}")
            print(f"Threshold {th:.2f}: Return = {ret:.4f}")
    return returns


def calculate_all_threshold_variances(txn):
    variances = {}
    for th, prices in txn.rebalanced_prices_by_threshold.items():
        if len(prices) >= 2:
            daily_returns = np.diff(prices) / prices[:-1]
            var = np.var(daily_returns)
            variances[th] = var
            logger.info(f"Threshold {th:.2f}: Variance = {var:.6f}")
            print(f"Threshold {th:.2f}: Variance = {var:.6f}")
            print(
                f"Threshold {th:.2f}: Rebalance Count = {txn.rebalance_count_by_threshold[th]}"
            )
    return variances


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


def plot_threshold_returns(returns, variances):
    thresholds = sorted(returns.keys())
    rets = [returns[th] for th in thresholds]
    vars_ = [variances.get(th, np.nan) for th in thresholds]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Rebalancing Threshold")
    ax1.set_ylabel("Total Return", color=color)
    ax1.plot(thresholds, rets, marker="o", color=color, label="Total Return")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Variance of Daily Returns", color=color)
    ax2.plot(
        thresholds, vars_, marker="x", linestyle="--", color=color, label="Variance"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Return and Variance vs Rebalancing Threshold")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def main():
    txn = MyTransaction()
    read(txn)
    log_return(txn)
    window = 252

    for i in range(window, len(txn.log_return)):
        start_idx, end_idx = i - window, i
        rolling_cov = cov_matrices(txn, start_idx, end_idx)
        rolling_mean = mean_returns(txn, start_idx, end_idx)

        (
            all_weights,
            frontier_returns,
            frontier_std,
            optimal_weights,
            optimal_return,
            optimal_std,
        ) = min_risk(rolling_mean, rolling_cov, txn)

        rebalancing(txn, optimal_weights)
        # plot_efficient_frontier(frontier_std, frontier_returns, optimal_std, optimal_return)

    returns = calculate_all_threshold_returns(txn)
    variances = calculate_all_threshold_variances(txn)
    #plot_all_threshold_prices(txn)
    plot_threshold_returns(returns, variances)

    return txn


if __name__ == "__main__":
    txn = main()
