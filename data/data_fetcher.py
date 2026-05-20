import yfinance as yf
import pandas as pd
import os

class Data_Fetching:
    def __init__(self, tickers, start, end, filename="sp500_2018_2024.pkl"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.filename = filename

    def data_download(self):
        if os.path.exists(self.filename):
            print("Loading data from local pickle file...")
            data = pd.read_pickle(self.filename)
            return data

        else:
            try:
                print("Downloading data from Yahoo Finance...")
                data = yf.download(
                    tickers=self.tickers,
                    start=self.start,
                    end=self.end,
                    auto_adjust=True)
                data.to_pickle(self.filename)
                print(f"Data saved locally as {self.filename}")
                return data

            except Exception as e:
                print(f"Error downloading data: {e}")
                return None

