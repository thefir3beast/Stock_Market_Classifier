import argparse
import pandas as pd

# Function to assign labels based on price movement
def assign_labels(data, threshold, look_forward_days):
    future_prices = data['Close'].shift(-look_forward_days)
    price_changes = (future_prices - data['Close']) / data['Close']
    
    labels = pd.cut(price_changes, bins=[-float('inf'), -threshold, threshold, float('inf')], labels=['Sell', 'Hold', 'Buy'])
    return labels

def main(input_file, output_file, threshold, look_forward_days):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Apply the labeling function
    df['Label'] = assign_labels(df, threshold, look_forward_days)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Save the labeled data
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label stock data for classification.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing technical data')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the labeled CSV file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold for labeling data based on price changes')
    parser.add_argument('--look_forward_days', type=int, default=5, help='Number of days to look forward for labeling')

    args = parser.parse_args()

    main(args.input_file, args.output_file, args.threshold, args.look_forward_days)
