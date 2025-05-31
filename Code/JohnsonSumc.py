import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from models_task3 import logger, MyTransaction, XgTransaction
import seaborn as sns
from scipy.stats import johnsonsu
import os


def Read_Portfolio(txn, portfolio_file):
    logger.info(f"Reading portfolio price: {portfolio_file}")

    txn.weighted_df = pd.read_csv(
        portfolio_file, parse_dates=["Date"], index_col="Date"
    )
    txn.portfolio_name = portfolio_file.split("/")[-1].split("_")[0]
    logger.info(
        f"Successfully loaded portfolio price from {portfolio_file}, total {len(txn.weighted_df)} records"
    )
    return txn.weighted_df


def read_exp_parameters(file_path="data/MLE_Parameter_Estimation_Portfolio.csv"):
    logger.info(f"Reading parameter file: {file_path}")
    parameters_df = pd.read_csv(file_path, index_col=0)
    logger.info(f"Parameters loaded:\n{parameters_df}")
    return parameters_df


def JohnsonSU_MC_fixed(
    txn,
    portfolio_name,
    parameters_df,
    a_multiplier,
    b_multiplier,
    loc_multiplier,
    scale_multiplier,
):
    logger.info(
        f"Start Johnson SU + Monte Carlo simulation for {portfolio_name} with fixed parameters"
    )

    a = parameters_df.loc[portfolio_name, "a"] * a_multiplier
    b = parameters_df.loc[portfolio_name, "b"] * b_multiplier
    loc = parameters_df.loc[portfolio_name, "loc"] * loc_multiplier
    scale = parameters_df.loc[portfolio_name, "scale"] * scale_multiplier

    logger.info(
        f"Using parameters for {portfolio_name}: a={a}, b={b}, loc={loc}, scale={scale}"
    )

    S0 = txn.weighted_df["Price"].iloc[-1]

    num_simulations = 1000
    num_days = 84

    np.random.seed(42)
    simulated_returns = johnsonsu.rvs(
        a, b, loc=loc, scale=scale, size=(num_days, num_simulations)
    )

    S_paths = np.zeros((num_days, num_simulations))
    S_paths[0, :] = S0
    for t in range(1, num_days):
        S_paths[t, :] = S_paths[t - 1, :] * np.exp(simulated_returns[t, :])

    txn.S_paths = S_paths
    txn.su_params = (a, b, loc, scale)
    logger.info(f"JohnsonSU Monte Carlo simulation for {portfolio_name} completed.")
    return S_paths, (a, b, loc, scale)


def median_95(txn):
    median_price = np.median(txn.S_paths, axis=1)  # Median Price
    percentile_2_5 = np.percentile(txn.S_paths, 2.5, axis=1)  # 2.5% quantile
    percentile_97_5 = np.percentile(txn.S_paths, 97.5, axis=1)  # 97.5% quantile

    errors_median = np.abs(txn.S_paths - median_price[:, np.newaxis])
    errors_2_5 = np.abs(txn.S_paths - percentile_2_5[:, np.newaxis])
    errors_97_5 = np.abs(txn.S_paths - percentile_97_5[:, np.newaxis])

    total_errors_median = np.sum(errors_median, axis=0)
    total_errors_2_5 = np.sum(errors_2_5, axis=0)
    total_errors_97_5 = np.sum(errors_97_5, axis=0)

    # Find the path index with the smallest error
    best_index_median = np.argmin(total_errors_median)
    best_index_2_5 = np.argmin(total_errors_2_5)
    best_index_97_5 = np.argmin(total_errors_97_5)

    best_path_median = txn.S_paths[:, best_index_median]
    best_path_2_5 = txn.S_paths[:, best_index_2_5]
    best_path_97_5 = txn.S_paths[:, best_index_97_5]

    txn.best_path_median = best_path_median
    txn.best_path_2_5 = best_path_2_5
    txn.best_path_97_5 = best_path_97_5

    return best_path_median, best_path_2_5, best_path_97_5


def log_return(txn):
    logger.info("Calculate future returns (Log Return)")
    # Log return of 1000 results
    future_log_returns = np.log(txn.S_paths[1:, :] / txn.S_paths[:-1, :])
    log_return_mean = np.mean(future_log_returns, axis=1)
    All_log_return_mean = np.mean(log_return_mean)
    ##Log return of path_median
    log_returns_median = np.log(txn.best_path_median[1:] / txn.best_path_median[:-1])
    day_log_return = np.mean(log_returns_median)
    total_log_return = np.sum(log_returns_median)
    total_return = np.exp(total_log_return) - 1

    txn.future_log_returns = future_log_returns
    txn.All_log_return_mean = All_log_return_mean
    txn.day_log_return = day_log_return
    txn.total_log_return = total_log_return
    print(f" day Log Return: {day_log_return:.4%}")
    print(f" total Log Return: {total_log_return:.4%}")
    print(f" total Return: {total_return:.4%}")
    logger.info(
        f"Future Log Return calculation completed: All mean={All_log_return_mean:.4%},day_log_return={day_log_return:.4%} total={total_return:.4%}"
    )
    return (
        All_log_return_mean,
        day_log_return,
        total_log_return,
        total_return,
        future_log_returns,
    )


def VaR(txn):
    logger.info("Calculate VaR and CVaR")
    var_95_future = np.percentile(txn.future_log_returns, 5)
    cvar_95_future = txn.future_log_returns[
        txn.future_log_returns <= var_95_future
    ].mean()
    txn.var_95_future = var_95_future
    txn.cvar_95_future = cvar_95_future
    print(f" 95% VaR: {var_95_future:.4%}")
    print(f" 95% CVaR: {cvar_95_future:.4%}")
    logger.info(
        f"VaR calculation completed: 95% VaR={var_95_future:.4%}, 95% CVaR={cvar_95_future:.4%}"
    )
    return var_95_future, cvar_95_future


def plot_Simulation(txn):
    logger.info(f"Draw Monte Carlo simulation results of {txn.portfolio_name}")
    last_date = txn.weighted_df.index[-1]
    future_dates = pd.date_range(
        start=last_date, periods=txn.S_paths.shape[0] + 1, freq="B"
    )[1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        txn.weighted_df.index,
        txn.weighted_df["Price"],
        color="blue",
        lw=2,
        label="Historical Price",
    )
    ax.plot(future_dates, txn.S_paths, lw=0.8, alpha=0.5, color="gray")

    ax.plot(
        future_dates,
        txn.best_path_median,
        color="r",
        linestyle="dashed",
        label="Median Price",
    )
    ax.plot(
        future_dates,
        txn.best_path_2_5,
        color="g",
        linestyle="dashed",
        label="2.5% Quantile",
    )
    ax.plot(
        future_dates,
        txn.best_path_97_5,
        color="b",
        linestyle="dashed",
        label="97.5% Quantile",
    )

    ax.set_title(f"{txn.portfolio_name} johnsonsu + Monte Carlo Simulation (4 Months)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Price")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.grid(True)

    axins = inset_axes(
        ax,
        width="60%",
        height="60%",
        loc="upper left",
        bbox_to_anchor=(0.09, 0.15, 0.6, 0.6),  # (left, bottom, width, height)
        bbox_transform=ax.transAxes,
    )

    for i in range(txn.S_paths.shape[1]):
        axins.plot(future_dates, txn.S_paths[:, i], lw=0.8, alpha=0.5, color="gray")

    axins.plot(future_dates, txn.best_path_median, color="r", linestyle="dashed")
    axins.plot(future_dates, txn.best_path_2_5, color="g", linestyle="dashed")
    axins.plot(future_dates, txn.best_path_97_5, color="b", linestyle="dashed")

    axins.set_xlim(future_dates.min(), future_dates.max())

    axins.set_xticks([])
    axins.set_yticks([])

    plt.show()
    logger.info("Monte Carlo plotting completed")


def plot_VaR(txn):
    logger.info(f"Draw VaR histogram of {txn.portfolio_name}")
    txn.future_log_returns = np.array(txn.future_log_returns).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(
        txn.future_log_returns,
        bins=50,
        alpha=0.75,
        color="blue",
        edgecolor="black",
        density=True,
    )

    plt.axvline(
        txn.var_95_future,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"95% VaR: {txn.var_95_future:.2%}",
    )
    plt.axvline(
        txn.cvar_95_future,
        color="Orange",
        linestyle="dashed",
        linewidth=2,
        label=f"95% CVaR: {txn.cvar_95_future:.2%}",
    )
    plt.axvline(
        txn.day_log_return,
        color="#00FF00",
        linestyle="dashed",
        linewidth=2,
        label=f"day_log_return: {txn.day_log_return:.2%}",
    )

    plt.title(f"{txn.portfolio_name} 95% VaR and CVaR Distribution")
    plt.xlabel("Log Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.show()
    logger.info("VaR histogram drawing completed")


def plot_distribution(txn):
    for portfolio_file in txn.results.keys():
        if "GBM" in portfolio_file:
            base_name = portfolio_file.replace("_GBM", "")
            gbm_s_paths = txn.results[portfolio_file].get("S_paths", None)
            vol_s_paths = txn.results.get(f"{base_name}_Volatility", {}).get(
                "S_paths", None
            )

            if gbm_s_paths is None or vol_s_paths is None:
                print(f" No data found for {base_name}")
                continue

            gbm_log_returns = np.log(gbm_s_paths[:, 1:] / gbm_s_paths[:, :-1])
            vol_log_returns = np.log(vol_s_paths[:, 1:] / vol_s_paths[:, :-1])

            gbm_log_returns_flat = gbm_log_returns.flatten()
            vol_log_returns_flat = vol_log_returns.flatten()

            plt.figure(figsize=(10, 5))
            sns.kdeplot(
                gbm_log_returns_flat,
                label=f"{base_name} - GBM Log Return",
                fill=True,
                color="blue",
                bw_adjust=0.5,
            )
            sns.kdeplot(
                vol_log_returns_flat,
                label=f"{base_name} - Volatility Shocks Log Return",
                fill=True,
                color="red",
                bw_adjust=1,
            )

            plt.xlabel("Log Return")
            plt.ylabel("Density")
            plt.title(f"{base_name}: GBM vs Volatility Shocks Log Return Distribution")
            plt.legend()
            plt.show()


def main():
    txn = MyTransaction()
    xgt = XgTransaction()

    pf = [
        "data/Min Variance_portfolio_price.csv",
        "data/Max Sharpe_portfolio_price.csv",
        "data/Steepest Descent_portfolio_price.csv",
    ]
    parameters_df = read_exp_parameters("data/MLE_Parameter_Estimation_Portfolio.csv")
    for portfolio_file in pf:
        portfolio_name = os.path.splitext(os.path.basename(portfolio_file))[0].replace(
            "_portfolio_price", ""
        )
        # GBM
        logger.info(f"\n# ========== {portfolio_file} ==========")
        Read_Portfolio(txn, portfolio_file)
        print(f"\n ========== {txn.portfolio_name} ==========")
        logger.info("\n## ------ 'johnsonsu' ------")
        print("\n ------ 'johnsonsu' ------")
        JohnsonSU_MC_fixed(txn, portfolio_name, parameters_df, 1, 1, 1, 1)
        median_95(txn)
        logger.info("\n### ---- Return ----")
        print("\n ---- Return ----")
        log_return(txn)
        logger.info("\n### ---- Risk ----")
        print("\n ---- Risk ----")
        var_95_future, cvar_95_future = VaR(txn)
        txn.store_results(portfolio_file)
        plot_Simulation(txn)
        plot_VaR(txn)

        # economic downturns or volatility shocks
        print("\n ------ 'volatility shocks' ------")
        JohnsonSU_MC_fixed(txn, portfolio_name, parameters_df, 2.11, 1.07, 3.77, 2.4)
        median_95(txn)
        logger.info("\n### ---- Return ----")
        print("\n ---- Return ----")
        log_return(txn)
        logger.info("\n### ---- Risk ----")
        print("\n ---- Risk ----")
        var_95_future, cvar_95_future = VaR(txn)
        txn.volatility_results(portfolio_file)
        plot_Simulation(txn)
        plot_VaR(txn)

    plot_distribution(txn)
    print("\n ========== END ==========")
    return txn.results, txn, xgt, xgt.xgresults


if __name__ == "__main__":
    txn_xgt = main()
