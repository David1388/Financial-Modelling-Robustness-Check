import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sigmamodels import MyTransaction, logger


def read_data(txn):
    try:
        df = pd.read_csv("data/KLCI_20y.csv")
        df["date"] = pd.to_datetime(df["Date"])
        txn.df = df
        logger.info(
            f"Data read successfully from 'KLCI_20y.csv'. Total rows: {len(df)}."
        )
        return df
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise


def compute_sigma(txn):
    try:
        txn.df["log_return"] = np.log(txn.df["Close"] / txn.df["Close"].shift(1))
        txn.df["sigma"] = txn.df["log_return"].rolling(window=txn.window).std()
        txn.df = txn.df.dropna().reset_index(drop=True)
        logger.info("Sigma (volatility) computed successfully.")
        return txn.df
    except Exception as e:
        logger.error(f"Error computing sigma: {e}")
        raise


def analyze_sigma_events(txn):
    try:
        min_idx = argrelextrema(txn.df["sigma"].values, np.less, order=200)[0]
        max_idx = argrelextrema(txn.df["sigma"].values, np.greater, order=200)[0]

        i, j = 0, 0
        events_list = []

        while i < len(min_idx) and j < len(max_idx):
            while j < len(max_idx) and max_idx[j] < min_idx[i]:
                j += 1
            if j < len(max_idx) and i < len(min_idx):
                min_date = txn.df.loc[min_idx[i], "date"]
                max_date = txn.df.loc[max_idx[j], "date"]
                min_sigma = txn.df.loc[min_idx[i], "sigma"]
                max_sigma = txn.df.loc[max_idx[j], "sigma"]
                ratio = max_sigma / min_sigma

                event_df = pd.DataFrame(
                    [
                        {
                            "min_date": min_date,
                            "max_date": max_date,
                            "min_sigma": min_sigma,
                            "max_sigma": max_sigma,
                            "ratio": ratio,
                        }
                    ]
                )

                if not event_df.empty:
                    txn.events = txn.events.dropna(axis=1, how="all")
                    txn.events = pd.concat([txn.events, event_df], ignore_index=True)
                    events_list.append(event_df)

                i += 1
                j += 1
            else:
                break

        logger.info(f"Total sigma events found: {len(events_list)}.")
        return txn.events

    except Exception as e:
        logger.error(f"Error analyzing sigma events: {e}")
        raise


def Max_ratio(txn):
    try:
        max_ratio_row = txn.events.loc[txn.events["ratio"].idxmax()]
        txn.max_ratio_row = max_ratio_row
        logger.info("Max ratio event identified.")
        print("The maximum multiple interval:")
        print(
            f"The minimum Sigma appears in: {max_ratio_row['min_date'].date()}, values: {max_ratio_row['min_sigma']:.4f}"
        )
        print(
            f"Maximum Sigma occurs at: {max_ratio_row['max_date'].date()}, values: {max_ratio_row['max_sigma']:.4f}"
        )
        print(
            f"The maximum Sigma is {max_ratio_row['ratio']:.2f} times the minimum Sigma."
        )
        return txn.max_ratio_row
    except Exception as e:
        logger.error(f"Error finding max ratio event: {e}")
        raise


def plot_max_ratio(txn):
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(txn.df["date"], txn.df["sigma"], label="Rolling Sigma")
        for e in txn.events.itertuples():
            plt.axvline(e.min_date, color="green", linestyle="--", alpha=0.5)
            plt.axvline(e.max_date, color="red", linestyle="--", alpha=0.5)
        plt.axvspan(
            txn.max_ratio_row["min_date"],
            txn.max_ratio_row["max_date"],
            color="yellow",
            alpha=0.2,
            label="Max Ratio Interval",
        )
        plt.legend()
        plt.title("Rolling Sigma with Local Min/Max")
        plt.grid()
        plt.show()
        logger.info("Max ratio plot generated successfully.")
    except Exception as e:
        logger.error(f"Error generating max ratio plot: {e}")
        raise


def plot_sigma_price(txn):
    try:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price", color="blue")
        ax1.plot(txn.df["date"], txn.df["Close"], color="blue", label="Close Price")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Sigma", color="red")
        ax2.plot(
            txn.df["date"], txn.df["sigma"], color="red", label="Sigma (Volatility)"
        )
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title("Stock Price and Rolling Sigma (from Log Return)")
        plt.grid()
        plt.show()
        logger.info("Sigma-price plot generated successfully.")
    except Exception as e:
        logger.error(f"Error generating sigma-price plot: {e}")
        raise


def main():
    txn = MyTransaction()
    try:
        read_data(txn)
        compute_sigma(txn)
        analyze_sigma_events(txn)
        Max_ratio(txn)
        plot_max_ratio(txn)
        plot_sigma_price(txn)
        logger.info("Program completed successfully.")
    except Exception as e:
        logger.error(f"Program encountered an error: {e}")
        raise


if __name__ == "__main__":
    main()
