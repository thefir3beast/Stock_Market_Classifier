import argparse
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm

# Predefined list of tickers for Dow Jones 30
dow_jones_30_tickers = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

def fetch_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    
    # Fetch historical (technical) data
    technical_data = stock.history(start=start_date, end=end_date)
    technical_data.reset_index(inplace=True)  # Make 'Date' a column
    technical_data['Ticker'] = ticker  # Add 'Ticker' column to technical data

    # Fetch the most recent fundamental data
    info = stock.info

    fundamental_data = {
        'EPS': info.get('trailingEps', float('nan')),
        'PE Ratio': info.get('trailingPE', float('nan')),
        'ROE': info.get('returnOnEquity', float('nan')),
        'Debt to Equity': info.get('debtToEquity', float('nan')),
        'Current Ratio': info.get('currentRatio', float('nan')),
        'EBITDA': info.get('ebitda', float('nan')),
        'Dividend Yield': info.get('dividendYield', float('nan')) * 100 if info.get('dividendYield') else float('nan')
    }

    # Repeat the fundamental data for each date in the technical data
    fundamental_data_daily = pd.DataFrame([fundamental_data] * len(technical_data), columns=fundamental_data.keys())
    fundamental_data_daily['Date'] = technical_data['Date']  # Align the dates with the technical data
    fundamental_data_daily['Ticker'] = ticker  # Ensure 'Ticker' column is added

    return technical_data, fundamental_data_daily


def main():
    parser = argparse.ArgumentParser(description="Download stock technical and fundamental data for a given index and merge them into one file.")
    parser.add_argument('--index', type=str, default='DowJones30', help='Index name (default: DowJones30)')
    parser.add_argument('--start_year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end_year', type=int, default=datetime.now().year, help='End year (default: current year)')
    parser.add_argument('--save_path', type=str, default='./data', help='Path to save the CSV file (default: ./data)')
    args = parser.parse_args()

    if args.index != 'DowJones30':
        print("Currently, only DowJones30 index is supported.")
        return

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"

    combined_all = pd.DataFrame()

    for ticker in tqdm(dow_jones_30_tickers, desc="Fetching and combining data"):
        technical_data, fundamental_data = fetch_data(ticker, start_date, end_date)

        # Merge technical and fundamental data on the 'Date' and 'Ticker' columns
        combined_data = pd.merge(technical_data, fundamental_data, on=['Date', 'Ticker'])

        combined_all = pd.concat([combined_all, combined_data], ignore_index=True)

    # Save the combined data to a single CSV file
    save_path = os.path.join(args.save_path, 'combined_data.csv')
    combined_all.to_csv(save_path, index=False)
    print(f"Combined data saved to {save_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
