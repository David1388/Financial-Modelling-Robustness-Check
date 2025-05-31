import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from KDE_model import MyTransaction, logger


def load_kde_and_generate(
    data: np.ndarray,
    bandwidth: float,
    n_days: int,
    n_paths: int,
    random_state: int = None,
) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
    kde = gaussian_kde(data, bw_method=bandwidth)
    samples = kde.resample(n_days * n_paths).reshape(n_paths, n_days)
    return samples


def load_all_and_generate(
    input_dir: str, n_days: int, n_paths: int, random_state: int = None
) -> dict:
    synth = {}
    for i, fname in enumerate(sorted(os.listdir(input_dir))):
        if not fname.endswith("_kde.pkl"):
            continue
        ticker = fname.replace("_kde.pkl", "")
        path = os.path.join(input_dir, fname)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        data = payload["data"]
        bandwidth = payload["bandwidth"]
        samples = load_kde_and_generate(
            data,
            bandwidth,
            n_days,
            n_paths,
            random_state=random_state + i if random_state is not None else None,
        )
        synth[ticker] = samples
    return synth


def convert_returns_to_prices(txn, synth_dict: dict, price_csv_path: str) -> dict:
    df = pd.read_csv(price_csv_path, index_col=0, parse_dates=True)
    last_prices = df.iloc[-1]
    price_dict = {}

    for ticker, return_paths in synth_dict.items():
        if ticker not in last_prices:
            logger.warning(f"Warning: {ticker} not found in close price.")
            continue

        P0 = last_prices[ticker]
        price_paths = P0 * np.cumprod(1 + return_paths, axis=1)
        price_dict[ticker] = price_paths
        txn.price_paths = price_paths
    return price_dict


def calculate_quantiles(price_paths):
    median_path = np.median(price_paths, axis=0)
    percentile_2_5_path = np.percentile(price_paths, 2.5, axis=0)
    percentile_97_5_path = np.percentile(price_paths, 97.5, axis=0)
    return median_path, percentile_2_5_path, percentile_97_5_path


def find_best_paths(
    price_paths, median_path, percentile_2_5_path, percentile_97_5_path
):
    def closest_path(target, paths):
        return min(
            paths, key=lambda path: np.linalg.norm(np.array(path) - np.array(target))
        )

    best_median = closest_path(median_path, price_paths)
    best_2_5 = closest_path(percentile_2_5_path, price_paths)
    best_97_5 = closest_path(percentile_97_5_path, price_paths)

    return best_median, best_2_5, best_97_5


def generate_trading_days(txn, start="2025-01-01", end="2025-04-30"):
    all_days = pd.date_range(start=start, end=end, freq="B")
    trading_days = all_days[~all_days.isin(txn.market_holidays)]
    return trading_days


def plot_mc_paths(
    price_paths, ticker, median_path, p2_5_path, p97_5_path, best_paths, trading_days
):
    plt.figure(figsize=(12, 6))
    for path in price_paths:
        plt.plot(trading_days[: len(path)], path, color="lightgray", linewidth=0.5)

    median_path, p2_5_path, p97_5_path = calculate_quantiles(price_paths)

    best_median_path, best_p2_5_path, best_p97_5_path = find_best_paths(
        price_paths, median_path, p2_5_path, p97_5_path
    )
    plt.plot(
        trading_days[: len(best_median_path)],
        best_median_path,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"{ticker}Closest to Median",
    )
    plt.plot(
        trading_days[: len(best_p97_5_path)],
        best_p97_5_path,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"{ticker} Closest to 97.5%",
    )
    plt.plot(
        trading_days[: len(best_p2_5_path)],
        best_p2_5_path,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"{ticker} Closest to 2.5%",
    )

    plt.title(f"Simulated price path of {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_portfolio_weights(files):
    portfolios = {}
    for file in files:
        name = file.split("/")[-1].replace(".csv", "").replace("_updated", "")
        df = pd.read_csv(file, index_col=0)
        portfolios[name] = df
    return portfolios


def calculate_portfolio_price_paths(txn, price_paths, portfolios):
    portfolio_paths = {}
    for name, weights in portfolios.items():
        common_tickers = list(set(weights.index).intersection(set(price_paths.keys())))
        logger.info(f"[{name}] common_tickers: {common_tickers}")
        logger.info(f"[{name}] weights.index: {list(weights.index)}")
        logger.info(f"[{name}] price_paths.keys(): {list(price_paths.keys())}")

        weights = weights.loc[common_tickers]
        weights = weights / weights.sum()
        logger.info(f"[{name}] normalized weights: {weights}")

        first_ticker = common_tickers[0]
        portfolio_sim = np.zeros_like(price_paths[first_ticker])

        for ticker in common_tickers:
            portfolio_sim += price_paths[ticker] * weights.loc[ticker, "optimal_weight"]

        portfolio_paths[name] = portfolio_sim
        txn.portfolio_paths = portfolio_paths
    return portfolio_paths


def calculate_portfolio_price_paths_with_rebalancing(
    txn,
    price_paths,
    before_portfolios,
    after_portfolios,
    trading_days,
    rebalance_date_str="2025-02-19",
):
    rebalance_date = pd.to_datetime(rebalance_date_str)
    try:
        rebalance_index = trading_days.get_loc(rebalance_date)
    except KeyError:
        raise ValueError(
            f"{rebalance_date_str} Not in the trading day list, please check holiday settings。"
        )

    portfolio_paths_combined = {}
    for name in before_portfolios:
        weights_before = before_portfolios[name]
        weights_after = after_portfolios[name]
        price_paths_before = {
            ticker: paths[:, :rebalance_index] for ticker, paths in price_paths.items()
        }
        price_paths_after = {
            ticker: paths[:, rebalance_index:] for ticker, paths in price_paths.items()
        }
        portfolio_before = calculate_portfolio_price_paths(
            txn, price_paths_before, {name: weights_before}
        )[name]
        portfolio_after = calculate_portfolio_price_paths(
            txn, price_paths_after, {name: weights_after}
        )[name]
        portfolio_paths_combined[name] = np.hstack((portfolio_before, portfolio_after))

    txn.portfolio_paths = portfolio_paths_combined
    txn.simulation_start_date = trading_days[0]
    logger.info("portfolio_paths_combined:")
    for name, paths in portfolio_paths_combined.items():
        logger.info(f"{name}: {paths}")
    return portfolio_paths_combined


def plot_single_portfolio_mc_paths(portfolio_name, portfolio_paths, trading_days):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for path in portfolio_paths:
        plt.plot(
            trading_days[: len(path)], path, color="lightgray", linewidth=0.5, alpha=0.3
        )

    median_path, p2_5_path, p97_5_path = calculate_quantiles(portfolio_paths)

    best_median_path, best_p2_5_path, best_p97_5_path = find_best_paths(
        portfolio_paths, median_path, p2_5_path, p97_5_path
    )

    plt.plot(
        trading_days[: len(best_median_path)],
        best_median_path,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"{portfolio_name} Closest to Median",
    )
    plt.plot(
        trading_days[: len(best_p97_5_path)],
        best_p97_5_path,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"{portfolio_name} Closest to 97.5%",
    )
    plt.plot(
        trading_days[: len(best_p2_5_path)],
        best_p2_5_path,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"{portfolio_name} Closest to 2.5%",
    )

    plt.title(f"Simulated Price Path of Portfolio: {portfolio_name}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_returns(price_paths):
    returns = (price_paths[:, -1] / price_paths[:, 0]) - 1
    return returns


def calculate_daily_returns(price_paths):
    daily_returns = price_paths[:, 1:] / price_paths[:, :-1] - 1
    return daily_returns


def calculate_var_cvar_from_daily_returns(daily_returns, alpha=0.05):
    flattened_returns = daily_returns.flatten()
    sorted_returns = np.sort(flattened_returns)
    index = int(np.floor(alpha * len(sorted_returns)))
    var = sorted_returns[index]
    cvar = sorted_returns[:index].mean()
    return var, cvar


def calculate_best_path_return(best_path_median):
    return (best_path_median[-1] / best_path_median[0]) - 1


def main():
    txn = MyTransaction()
    INPUT_DIR = "kde_models"
    PRICE_CSV = "data/Adj_Close_data.csv"
    trading_days = generate_trading_days(txn)
    NUM_DAYS = len(trading_days)
    NUM_PATHS = 1000

    synth_dict = load_all_and_generate(INPUT_DIR, NUM_DAYS, NUM_PATHS, random_state=42)
    price_dict = convert_returns_to_prices(txn, synth_dict, PRICE_CSV)

    for ticker, price_paths in price_dict.items():
        median_path, p2_5_path, p97_5_path = calculate_quantiles(price_paths)
        best_paths = find_best_paths(price_paths, median_path, p2_5_path, p97_5_path)
        plot_mc_paths(
            price_paths,
            ticker,
            median_path,
            p2_5_path,
            p97_5_path,
            best_paths,
            trading_days,
        )

    before_files = [
        "data/Min Variance.csv",
        "data/Max Sharpe.csv",
        "data/Steepest Descent.csv",
    ]
    after_files = [
        "data/Min Variance_updated.csv",
        "data/Max Sharpe_updated.csv",
        "data/Steepest Descent_updated.csv",
    ]
    before_portfolios = load_portfolio_weights(before_files)
    after_portfolios = load_portfolio_weights(after_files)

    portfolio_paths_combined = calculate_portfolio_price_paths_with_rebalancing(
        txn, price_dict, before_portfolios, after_portfolios, trading_days
    )

    for name, portfolio_paths in portfolio_paths_combined.items():
        plot_single_portfolio_mc_paths(name, portfolio_paths, trading_days)

        daily_returns = calculate_daily_returns(portfolio_paths)
        var_95, cvar_95 = calculate_var_cvar_from_daily_returns(
            daily_returns, alpha=0.05
        )

        total_returns = calculate_returns(portfolio_paths)
        var_total, cvar_total = calculate_var_cvar_from_daily_returns(
            total_returns, alpha=0.05
        )

        median_path, _, _ = calculate_quantiles(portfolio_paths)
        best_median_path, _, _ = find_best_paths(
            portfolio_paths, *calculate_quantiles(portfolio_paths)
        )
        best_return = calculate_best_path_return(best_median_path)

        print(f"\n[{name}] Portfolio Summary:")
        print(f"{name} | Daily VaR (95%): {var_95:.4f}, CVaR: {cvar_95:.4f}")
        print(f"{name} | Total VaR (95%): {var_total:.4f}, CVaR: {cvar_total:.4f}")
        print(f"  Best Path Median Return: {best_return:.4f}")
    return txn


if __name__ == "__main__":
    txn = main()
