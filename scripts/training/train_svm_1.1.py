import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np

def preprocess_data(input_file):
    print("Loading and preprocessing data...")
    df = pd.read_csv(input_file)
    
    X = df.drop(['Label'], axis=1).select_dtypes(include=[np.number])
    y = df['Label']

    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Preprocessing completed.")
    return X_scaled, y

def main(input_file, test_size):
    print("Starting main program...")
    X_scaled, y = preprocess_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
    print("Data split into training and test sets.")

    print("Beginning Grid Search for SVM hyperparameter tuning...")
    pipeline = Pipeline([
        ('smote', SMOTE()),
        ('svm', SVC(probability=True))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['rbf', 'linear'],
        'svm__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Grid Search completed.")

    best_model = grid_search.best_estimator_
    print(f"Best model determined: {best_model}")

    print("Evaluating the best model on the test set...")
    y_test_pred = best_model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a stock classifier using SVM.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing fundamental data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of the data to be used as the test set')

    args = parser.parse_args()
    main(args.input_file, args.test_size)
