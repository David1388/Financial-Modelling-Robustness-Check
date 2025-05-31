# models_task3
import logging
import pandas as pd


class MyTransaction:
    def __init__(self):
        self.optimal_weights = []
        self.price_data = pd.DataFrame()
        self.log_return = pd.DataFrame()
        self.cov_matrices = pd.DataFrame()
        self.mean_returns = pd.DataFrame()
        self.last_rebalanced_weights = []
        self.portfolio_prices = []
        self.rebalanced_portfolio_price = []
        self.portfolio_return = []
        self.portfolio_variance = []
        self.rebalance_threshold = []
        self.rebalanced_weights_by_threshold = {}
        self.rebalanced_prices_by_threshold = {}
        self.rebalanced_returns_by_threshold = {}
        self.rebalance_count_by_threshold = {}
        self.rebalance_dates_by_threshold = {}
        self.scale_factors_by_threshold = {}


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("algo_rebalanced.log", mode="w")],
)

logger = logging.getLogger("algo_rebalanced")
