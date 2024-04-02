import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Function to calculate moving averages
def add_moving_averages(df, window_sizes=[5, 10, 20]):
    print("Adding moving averages...")
    for window in window_sizes:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    print("Moving averages added.")
    return df

# Load and preprocess the data
def preprocess_data(input_file):
    print("Loading and preprocessing data...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    
    df = add_moving_averages(df)
    X = df.drop(['Close', 'Label'], axis=1).select_dtypes(include=[np.number])
    y = df['Label']

    X.fillna(X.mean(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data preprocessing completed.")
    return X_scaled, y

# Main function
def main(input_file, test_size):
    X_scaled, y = preprocess_data(input_file)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
    print("Data split completed.")

    print("Addressing class imbalance with SMOTE...")
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("SMOTE applied to training data.")

    print("Training Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=1000, max_depth=None)
    rf_classifier.fit(X_train_balanced, y_train_balanced)
    print("Training completed.")

    print("Evaluating model on the test set...")
    y_test_pred = rf_classifier.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a stock classifier with enhanced preprocessing.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing technical data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as the test set')

    args = parser.parse_args()
    main(args.input_file, args.test_size)
