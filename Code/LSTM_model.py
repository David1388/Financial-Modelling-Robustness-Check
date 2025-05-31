import logging
import pandas as pd


class MyTransaction:
    def __init__(self):
        self.df = pd.DataFrame()
        self.scaled_data = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.DataFrame()
        self.dates = pd.DataFrame()
        self.y_rescaled = pd.DataFrame()
        self.stocks = pd.DataFrame()
        self.SEQ_LEN = 84
        self.future_preds = pd.DataFrame()
        self.updated_weights = {}
        self.portfolio_returns = {}
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
        self.test_months = [
            (2022, 7),
            (2022, 11),
            (2023, 7),
            (2023, 2),
            (2023, 12),
            (2023, 5),
            (2024, 4),
            (2024, 2),
        ]


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("algo.log", mode="w")],
)

logger = logging.getLogger("algo")
