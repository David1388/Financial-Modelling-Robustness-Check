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
    logger.info("Successfully read price data")
    return price_data


def log_return(txn):
    log_return = np.log(txn.price_data / txn.price_data.shift(1)).dropna()
    logger.info("Logarithmic rate of return calculation completed")
    txn.log_return = log_return
    return log_return


def cov_matrices(txn, start_idx, end_idx):
    cov = txn.log_return.iloc[start_idx:end_idx].cov()
    txn.cov_matrices = cov
    return cov


def mean_returns(txn, start_idx, end_idx):
    mean = txn.log_return.iloc[start_idx:end_idx].mean() * 252
    txn.mean_returns = mean
    return mean


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


def max_Sharpe(mean_returns, cov_matrices, txn):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
    frontier_returns, frontier_std, portfolio_weights = [], [], []
    risk_free_rate = 0.02
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

    sharpe_ratios = (np.array(frontier_returns) - risk_free_rate) / np.array(
        frontier_std
    )
    max_sharpe_idx = np.nanargmax(sharpe_ratios)
    best_weights = portfolio_weights[max_sharpe_idx]

    txn.optimal_weights.append(best_weights)
    txn.last_rebalanced_weights.append(best_weights)

    logger.info(
        "Maximum Sharpe ratio portfolio weight:\n%s",
        np.array2string(best_weights, precision=6, suppress_small=True),
    )

    return (
        portfolio_weights,
        frontier_returns,
        frontier_std,
        best_weights,
        frontier_returns[max_sharpe_idx],
        frontier_std[max_sharpe_idx],
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
        label="Max Sharpe Portfolio",
    )
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    plt.show()


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


def plort_rebalanced(txn):
    plt.figure(figsize=(10, 6))
    plt.plot(txn.portfolio_prices, label="Non-Rebalanced", color="blue")
    plt.plot(txn.rebalanced_portfolio_price, label="Rebalanced", color="red")
    plt.title("Portfolio Prices: Rebalanced vs Non-Rebalanced")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_threshold_returns(returns):
    thresholds = list(returns.keys())
    rets = list(returns.values())
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, rets, marker="o")
    plt.xlabel("Rebalancing Threshold")
    plt.ylabel("Total Return")
    plt.title("Return vs Rebalancing Threshold")
    plt.grid(True)
    plt.show()


def plot_all_threshold_prices(txn):
    for th in sorted(txn.rebalanced_prices_by_threshold.keys()):
        prices = txn.rebalanced_prices_by_threshold[th]
        if len(prices) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(prices, label=f"Threshold {th:.2f}")
            plt.title(f"Portfolio Price (Threshold {th:.2f})")
            plt.xlabel("Time (Days)")
            plt.ylabel("Portfolio Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
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
        ) = max_Sharpe(rolling_mean, rolling_cov, txn)

        rebalancing(txn, optimal_weights)
        # plot_efficient_frontier(frontier_std, frontier_returns, optimal_std, optimal_return)

    returns = calculate_all_threshold_returns(txn)
    # plot_all_threshold_prices(txn)
    plot_threshold_returns(returns)

    return txn


if __name__ == "__main__":
    txn = main()
