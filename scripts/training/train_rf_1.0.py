import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Function to calculate moving averages
def add_moving_averages(df, window_sizes=[5, 10, 20]):
    for window in window_sizes:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df

# Load and preprocess the data
def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    
    # Exclude 'Close' and 'Label' from features and ensure all remaining are numeric
    X = df.drop(['Close', 'Label'], axis=1).select_dtypes(include=[np.number])
    y = df['Label']

    # Handle NaN values in X
    X.fillna(X.mean(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Main function
def main(input_file, test_size):
    X_scaled, y = preprocess_data(input_file)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)

    # Address class imbalance in the training set using SMOTE
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=30)
    rf_classifier.fit(X_train_balanced, y_train_balanced)

    # Evaluate on the test set
    y_test_pred = rf_classifier.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a stock classifier with enhanced preprocessing.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing technical data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as the test set')

    args = parser.parse_args()

    main(args.input_file, args.test_size)
