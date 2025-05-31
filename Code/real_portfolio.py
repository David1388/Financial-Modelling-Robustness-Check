import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_combined_returns(df_future, before_files, after_files):
    cutoff_date = pd.to_datetime("2025-02-19")
    df_ret = df_future.pct_change().dropna()
    before_df = df_ret[df_ret.index < cutoff_date]
    after_df = df_ret[df_ret.index >= cutoff_date]

    all_returns = {}

    for before_file, after_file in zip(before_files, after_files):
        strategy_name = os.path.splitext(os.path.basename(before_file))[0]

        weights_df_before = pd.read_csv(before_file, index_col="stock_code")
        common_before = before_df.columns.intersection(weights_df_before.index)
        weights_before = weights_df_before.loc[common_before]["optimal_weight"].values
        port_ret_before = before_df[common_before].values @ weights_before
        port_ret_before = pd.Series(port_ret_before, index=before_df.index)

        weights_df_after = pd.read_csv(after_file, index_col="stock_code")
        common_after = after_df.columns.intersection(weights_df_after.index)
        weights_after = weights_df_after.loc[common_after]["optimal_weight"].values
        port_ret_after = after_df[common_after].values @ weights_after
        port_ret_after = pd.Series(port_ret_after, index=after_df.index)

        full_ret = pd.concat([port_ret_before, port_ret_after])
        df = pd.DataFrame({"Daily Return": full_ret})
        df["Log Return"] = np.log1p(df["Daily Return"])
        df["Cumulative Return"] = df["Log Return"].cumsum()
        df["Rolling Variance"] = df["Log Return"].rolling(window=20).var()

        all_returns[strategy_name] = df
        print(
            f"{strategy_name} Final Cumulative Return: {df['Cumulative Return'].iloc[-1]:.4f}"
        )
        print(f"{strategy_name} Return Variance: {df['Log Return'].var():.6f}")

    return all_returns


def plot_combined_return(all_returns, df_klse):
    plt.figure(figsize=(12, 6))
    for name, df in all_returns.items():

        df.loc[pd.Timestamp("2024-12-31")] = [0, 0, 0, np.nan]
        df = df.sort_index()

        df["Cumulative Return"] = df["Log Return"].cumsum()

        plt.plot(df.index, df["Cumulative Return"], label=name)

    df_klse.loc[pd.Timestamp("2024-12-31")] = [0, 0, 0, 0, 0, 0]
    df_klse = df_klse.sort_index()

    print(f"KLSE Final Cumulative Return: {df_klse['Cumulative Return'].iloc[-1]:.4f}")

    plt.plot(
        df_klse.index,
        df_klse["Cumulative Return"],
        label="KLSE Index",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Cumulative Log Return (2024-12-30 to 2025-04-30)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_combined_variance(all_returns):
    plt.figure(figsize=(12, 6))
    for name, df in all_returns.items():
        plt.plot(df.index, df["Rolling Variance"], label=name)
    plt.title("20-Day Rolling Variance of Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df_future = pd.read_csv(
        "data/Adj_Close_data_4m.csv", parse_dates=["Date"], index_col="Date"
    )

    if df_future.index.tz is not None:
        df_future.index = df_future.index.tz_convert(None)

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

    all_returns = calculate_combined_returns(df_future, before_files, after_files)

    df_klse = pd.read_csv(
        "data/KLCI_stooq_20241231_20250501.csv", parse_dates=["Date"], index_col="Date"
    )
    if df_klse.index.tz is not None:
        df_klse.index = df_klse.index.tz_convert(None)

    df_klse_ret = df_klse["Close"].pct_change().dropna()
    df_klse["Cumulative Return"] = np.log1p(df_klse_ret).cumsum()

    plot_combined_return(all_returns, df_klse)
    plot_combined_variance(all_returns)


if __name__ == "__main__":
    main()
