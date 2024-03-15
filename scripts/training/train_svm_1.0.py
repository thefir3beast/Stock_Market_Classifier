import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np

# Function to calculate moving averages and additional features
def add_features(df):
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df

# Load and preprocess the data
def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    
    df = add_features(df)
    X = df.drop(['Close', 'Label'], axis=1).select_dtypes(include=[np.number])
    y = df['Label']

    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Main function incorporating SVM and hyperparameter tuning
def main(input_file, test_size):
    X_scaled, y = preprocess_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)

    # Define the pipeline with SMOTE and SVC
    pipeline = Pipeline([
        ('smote', SMOTE()),
        ('svm', SVC(probability=True))
    ])

    # Parameters for Grid Search with SVM
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear'],
        'svm__gamma': ['scale', 'auto']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best model: {best_model}")

    # Evaluate on the test set
    y_test_pred = best_model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a stock classifier using SVM.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing technical data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of the data to be used as the test set')

    args = parser.parse_args()
    main(args.input_file, args.test_size)
