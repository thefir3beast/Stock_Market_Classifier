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

    # Initialize a list to collect fundamental data for all years
    fundamental_data_list = []

    # Iterate over each year in the specified range
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        # Fetch fundamental data
        info = stock.info

        fundamental_data = {
            'Year': year,
            'Ticker': ticker,
            'EPS': info.get('trailingEps', float('nan')),
            'PE Ratio': info.get('trailingPE', float('nan')),
            'ROE': info.get('returnOnEquity', float('nan')),
            'Debt to Equity': info.get('debtToEquity', float('nan')),
            'Current Ratio': info.get('currentRatio', float('nan')),
            'EBITDA': info.get('ebitda', float('nan')),
            'Dividend Yield': info.get('dividendYield', float('nan')) * 100 if info.get('dividendYield') else float('nan')
        }

        # Append the dictionary to the list
        fundamental_data_list.append(fundamental_data)

    # Convert the list of dictionaries to a DataFrame
    fundamental_data_all_years = pd.DataFrame(fundamental_data_list)

    return technical_data, fundamental_data_all_years

def save_data(dataframe, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, filename)
    dataframe.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Download stock technical and fundamental data for a given index.")
    parser.add_argument('--index', type=str, default='DowJones30', help='Index name (default: DowJones30)')
    parser.add_argument('--start_year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end_year', type=int, default=datetime.now().year, help='End year (default: current year)')
    parser.add_argument('--save_path', type=str, default='./data', help='Path to save the CSV files (default: ./data)')
    args = parser.parse_args()

    if args.index != 'DowJones30':
        print("Currently, only DowJones30 index is supported.")
        return

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"

    technical_all = pd.DataFrame()
    fundamental_all = pd.DataFrame()

    for ticker in tqdm(dow_jones_30_tickers, desc="Fetching data"):
        technical_data, fundamental_data = fetch_data(ticker, start_date, end_date)
        technical_all = pd.concat([technical_all, technical_data])
        fundamental_all = pd.concat([fundamental_all, fundamental_data], ignore_index=True)

    save_data(technical_all, args.save_path, 'technical_data.csv')
    save_data(fundamental_all, args.save_path, 'fundamental_data.csv')

if __name__ == "__main__":
    main()
