import logging
import pandas as pd


class MyTransaction:
    def __init__(self):
        self.events = pd.DataFrame(
            columns=["min_date", "max_date", "min_sigma", "max_sigma", "ratio"]
        )
        self.df = []
        self.window = 252
        self.max_ratio_row = []


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("algo.log", mode="w")],
)

logger = logging.getLogger("algo")
