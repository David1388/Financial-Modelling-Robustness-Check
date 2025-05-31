import logging
import pandas as pd


class MyTransaction:
    def __init__(self):
        self.price_paths = pd.DataFrame()
        self.portfolio_paths = pd.DataFrame()
        self.market_holidays = pd.to_datetime(
            [
                "2025-01-01",  # New Year's Day
                "2025-01-29",  # Chinese New Year
                "2025-01-30",  # Chinese New Year
                "2025-02-01",  # Federal Territory Day
                "2025-02-11",  # Thaipusam
                "2025-03-18",  # Nuzul Al'Quran
                "2025-03-31",  # Hari Raya Puasa (Eid-ul-Fitri)
                "2025-04-01",  # Hari Raya Puasa (Eid-ul-Fitri)
                "2025-05-01",  # Workers' Day
                "2025-05-12",  # Wesak Day
                "2025-05-30",  # Harvest Festival (Labuan)
                "2025-05-31",  # Harvest Festival (Labuan)
                "2025-06-02",  # Yang Dipertuan Agong's Birthday
                "2025-06-07",  # Hari Raya Haji (Eid-ul-Adha)
                "2025-06-27",  # Awal Muharram (Maal Hijrah)
                "2025-08-31",  # National Day
                "2025-09-05",  # Birthday of Prophet Muhammad
                "2025-09-16",  # Malaysia Day
                "2025-10-20",  # Deepavali
                "2025-12-25",  # Christmas Day
            ]
        )
        self.tickers = [
            "2089.KL",
            "3301.KL",
            "1155.KL",
            "4006.KL",
            "5209.KL",
            "5347.KL",
            "5983.KL",
            "1066.KL",
            "1818.KL",
            "5228.KL",
            "5176.KL",
            "5109.KL",
            "5236.KL",
            "5606.KL",
            "6432.KL",
            "3476.KL",
            "5227.KL",
            "7100.KL",
            "2488.KL",
            "7084.KL",
            "5014.KL",
            "5212.KL",
            "7103.KL",
            "6351.KL",
            "5398.KL",
            "5008.KL",
            "5878.KL",
            "2305.KL",
            "5819.KL",
            "5185.KL",
        ]
        self.before_files = [
            "data/Min Variance.csv",
            "data/Max Sharpe.csv",
            "data/Steepest Descent.csv",
        ]
        self.after_files = [
            "data/Min Variance_updated.csv",
            "data/Max Sharpe_updated.csv",
            "data/Steepest Descent_updated.csv",
        ]


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("algo.log", mode="w")],
)

logger = logging.getLogger("algo")
