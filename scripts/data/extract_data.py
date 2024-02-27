import argparse
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
    historical_data = stock.history(start=start_date, end=end_date)

    # Fundamental data
    info = stock.info
    eps = info.get('trailingEps', float('nan'))
    pe_ratio = info.get('trailingPE', float('nan'))
    roe = info.get('returnOnEquity', float('nan'))
    debt_to_equity = info.get('debtToEquity', float('nan'))
    current_ratio = info.get('currentRatio', float('nan'))

    # Combine all data
    combined_data = pd.DataFrame()
    for date, row in historical_data.iterrows():
        combined_row = row.to_dict()
        combined_row.update({
            'EPS': eps,
            'PE Ratio': pe_ratio,
            'ROE': roe,
            'Debt to Equity': debt_to_equity,
            'Current Ratio': current_ratio,
            'Ticker': ticker
        })

        combined_data = pd.concat([combined_data, pd.DataFrame([combined_row], index=[date])])

    # Reset the index to include the 'Date' column
    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={'index': 'Date'}, inplace=True)

    return combined_data

def main():
    parser = argparse.ArgumentParser(description="Download stock and fundamental data for a given index.")
    parser.add_argument('--index', type=str, default='DowJones30', help='Index name (default: DowJones30)')
    parser.add_argument('--start_year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end_year', type=int, default=datetime.now().year, help='End year (default: current year)')
    parser.add_argument('--save_path', type=str, default='./', help='Path to save the consolidated CSV file (default: current directory)')
    args = parser.parse_args()

    if args.index == 'DowJones30':
        tickers = dow_jones_30_tickers
    else:
        print("Currently, only DowJones30 index is supported.")
        return

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"

    all_data = pd.DataFrame()
    for ticker in tqdm(tickers, desc="Fetching data"):
        ticker_data = fetch_data(ticker, start_date, end_date)
        all_data = pd.concat([all_data, ticker_data])

    print(all_data.head())
    
    # Save the DataFrame to CSV, ensuring the 'Date' column is included
    all_data.to_csv(args.save_path + '/stock_and_fundamental_data.csv', index=False)

if __name__ == "__main__":
    main()
