import argparse
import pandas as pd
import numpy as np

def assign_labels(data, buy_threshold=5.0, sell_threshold=-5.0):
    """
    Assigns 'Buy', 'Hold', or 'Sell' labels based on predefined thresholds.
    
    :param data: DataFrame with fundamental data.
    :param buy_threshold: Threshold above which the label is 'Buy'.
    :param sell_threshold: Threshold below which the label is 'Sell'.
    :return: DataFrame with an additional column 'Label'.
    """
    conditions = [
        data['EPS'] >= buy_threshold,  # Buy condition
        data['EPS'] <= sell_threshold  # Sell condition
    ]
    choices = ['Buy', 'Sell']
    data['Label'] = np.select(conditions, choices, default='Hold')  # Default label is 'Hold'
    
    return data[['Ticker', 'Year', 'EPS', 'PE Ratio', 'ROE', 'Debt to Equity', 'Current Ratio', 'EBITDA', 'Dividend Yield', 'Label']]

def main(input_file, output_file, buy_threshold=5.0, sell_threshold=-5.0):
    # Load the data
    df = pd.read_csv(input_file)
    print("Columns in the DataFrame:", df.columns.tolist())  # Diagnostic print
    
    # Assign labels
    labeled_data = assign_labels(df, buy_threshold, sell_threshold)
    
    # Save the labeled data
    labeled_data.to_csv(output_file, index=False)
    print(f"Labeled data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label fundamental stock data based on predefined thresholds.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing fundamental data')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the labeled CSV file')
    parser.add_argument('--buy_threshold', type=float, default=5.0, help='Threshold above which the label is "Buy"')
    parser.add_argument('--sell_threshold', type=float, default=-5.0, help='Threshold below which the label is "Sell"')
    
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.buy_threshold, args.sell_threshold)
