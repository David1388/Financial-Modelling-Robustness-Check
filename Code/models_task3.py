import logging
import pandas as pd
import numpy as np


class MyTransaction:
    def __init__(self):
        self.results = {}
        self.df = pd.DataFrame()
        self.mu = 0.0
        self.weights_df = pd.DataFrame()
        self.weighted_df = pd.DataFrame()
        self.sigma = 0.0
        self.S_paths = np.array([])
        self.best_path_median = np.array([])
        self.best_path_2_5 = np.array([])
        self.best_path_97_5 = np.array([])
        self.future_log_returns = np.array([])
        self.day_log_return = 0.0
        self.total_log_return = 0.0
        self.total_return = 0.0
        # self.future_log_returns = pd.DataFrame()
        self.var_95_future = 0.0
        self.cvar_95_future = 0.0
        self.All_log_return_mean = 0.0
        self.portfolio_name = {}

    def store_results_with_name(self, portfolio_file, name):
        self.results[f"{portfolio_file}_{name}"] = {
            "S_paths": self.S_paths,
            "mu": self.mu,
            "sigma": self.sigma,
            "best_path_median": self.best_path_median,
            "best_path_2_5": self.best_path_2_5,
            "best_path_97_5": self.best_path_97_5,
            "All_log_return_mean": self.All_log_return_mean,
            "day_log_return": self.day_log_return,
            "total_log_return": self.total_log_return,
            "var_95_future": self.var_95_future,
            "cvar_95_future": self.cvar_95_future,
        }

    def store_results(self, portfolio_file):
        self.store_results_with_name(portfolio_file, "GBM")

    def volatility_results(self, portfolio_file):
        self.store_results_with_name(portfolio_file, "Volatility")


class XgTransaction:
    def __init__(self):
        self.xgresults = {}
        self.df_adj_close = pd.DataFrame()
        self.df_weights = pd.DataFrame()
        self.df_prices = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_pred = pd.DataFrame()
        self.y = pd.DataFrame()
        self.X = pd.DataFrame()
        self.rmse_percentage = 0.0
        self.mse = 0.0
        self.r2 = 0.0
        self.direction_accuracy = 0.0
        self.direction_diff = pd.DataFrame()
        self.period = 0
        self.alpha_3 = 0.0
        self.alpha_11 = 0.0
        self.alpha_signal = 0.0
        self.direction_accuracy = 0.0
        self.df_future = pd.DataFrame()
        self.total_log_return = 0.0
        self.volatility = 0.0
        self.max_drawdown = 0.0

    def xg_results(self, portfolio_file):
        self.xgresults[portfolio_file] = {
            "direction_accuracy": self.direction_accuracy,
            "r2": self.r2,
            "mse": self.mse,
            "direction_diff": self.direction_diff.to_dict(),
            "rmse_percentage": self.rmse_percentage,
            "total_log_return": self.total_log_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
        }


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("algo.log", mode="w")],
)

logger = logging.getLogger("algo")
