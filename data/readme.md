# 📊 S&P 500 Market Data Downloader & Cleaner

This Directory provides tools to:

- ✅ Retrieve the full S&P 500 universe (including historical constituents to avoid Survivorship Bias)
- ✅ Download historical market data from Yahoo Finance
- ✅ Cache data locally to avoid repeated downloads
- ✅ Clean and filter market data for quantitative research

## 📌 Overview

The project consists of two main modules:

### 1️⃣ `get_ticker_data.py`
**Handles:**
- Fetching S&P 500 constituents from Wikipedia  
- Building a historical ticker universe (2018–2024 by default)  
- Downloading historical price data  
- Cleaning and filtering raw market data  

### 2️⃣ `data_fetcher.py`

**Handles:**

- Downloading historical stock data from Yahoo Finance via `yfinance`  
- Saving data locally as a pickle file  
- Loading cached data if available
