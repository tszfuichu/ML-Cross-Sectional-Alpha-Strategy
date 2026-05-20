import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import pandas as pd
from data.data_fetcher import Data_Fetching

# Download the current stock table and changed stock table
def get_sp500_universe(start="2018-01-01", end="2024-12-31"):

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        tables = pd.read_html(response.text)
        print(f"Found {len(tables)} table(s) on the page.")

    if tables:
        current_table = tables[0]
        changed_table = tables[1]

    # Clean Data
    current_table["Symbol"] = current_table["Symbol"].str.replace('.', '-', regex=False)
    current_members = set(current_table["Symbol"].tolist())

    # Flatten the MultiIndex columns if they exist
    if isinstance(changed_table.columns, pd.MultiIndex):
        changed_table.columns = changed_table.columns.droplevel(0)

    # Filter changes between 2018 to 2024
    changed_table['Effective Date'] = pd.to_datetime(changed_table['Effective Date'])
    changed_table = changed_table[(changed_table['Effective Date'] >= start) & (changed_table['Effective Date'] <= end)]
    changed_table = changed_table.sort_values(by='Effective Date', ascending=False)

    # Get a master list of EVERY ticker that touched the S&P 500 between 2018-2024
    historical_all_ticker = current_members.copy()

    for _, row in changed_table.iterrows():
        ticker_added = row['Ticker'].iloc[0]
        ticker_removed = row['Ticker'].iloc[1]

        if pd.notna(ticker_added):
            clean_added = str(ticker_added).replace('.', '-')
            historical_all_ticker.add(clean_added)

        if pd.notna(ticker_removed):
            clean_removed = str(ticker_removed).replace('.', '-')
            historical_all_ticker.add(clean_removed)

    return sorted(list(historical_all_ticker))

# Download the stock data by using the ticker list
def download_market_data(tickers, start, end, filename):
    print("Downloading historical daily data from Yahoo Finance...")
    fetcher = Data_Fetching(tickers, start, end, filename=filename)
    return fetcher.data_download()


def clean_data(df):
    # Filter the useful data
    close = df.xs("Close", level=0, axis=1)
    volume = df.xs("Volume", level=0, axis=1)
    high = df.xs("High", level=0, axis=1)
    low = df.xs("Low", level=0, axis=1)

    # Remove Fully Empty Stocks
    close = close.dropna(axis=1, how="all")
    volume = volume[close.columns]

    # Remove Penny Stocks
    close = close.loc[:, close.mean() > 5]
    volume = volume[close.columns]

    # Remove Illiquid Stocks
    dollar_volume = close * volume
    avg_dollar_volume = dollar_volume.mean()
    liquid = avg_dollar_volume > 10000000
    close = close.loc[:, liquid]
    volume = volume.loc[:, liquid]

    # Forward Fill Small Gaps
    close = close.ffill(limit=3)
    high = high.ffill(limit=3)
    low = low.ffill(limit=3)

    # Remove Stocks With Too Little History
    min_days = 252
    valid_history = close.count() >= min_days
    close = close.loc[:, valid_history]
    volume = volume.loc[:, valid_history]

    high = high[close.columns]
    low = low[close.columns]

    return close, volume, high, low

