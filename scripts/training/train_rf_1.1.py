import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # Ensure GridSearchCV is imported here
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from tqdm.notebook import tqdm  # Make sure tqdm is imported for the progress bar


def add_features(df):
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()  # Example of another feature
    df.dropna(inplace=True)
    return df


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


class TqdmGridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, 
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', 
                 error_score=np.nan, return_train_score=True):
        super().__init__(estimator=estimator, param_grid=param_grid, scoring=scoring, 
                         n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose, 
                         pre_dispatch=pre_dispatch, error_score=error_score, 
                         return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        candidates = list(self._get_param_iterator())
        n_candidates = len(candidates)

        with tqdm(total=n_candidates, desc="GridSearch Progress") as pbar:
            def evaluate_candidates_with_progress_bar(candidate_params):
                out = evaluate_candidates(candidate_params)
                pbar.update(len(candidate_params))
                return out

            super()._run_search(evaluate_candidates_with_progress_bar)

# Main function update to use TqdmGridSearchCV
def main(input_file, test_size):
    X_scaled, y = preprocess_data(input_file)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)

    pipeline = Pipeline([
        ('smote', SMOTE()),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {
        'classifier': [RandomForestClassifier(), GradientBoostingClassifier()],
        'classifier__n_estimators': [100, 300],
        'classifier__max_depth': [10, 20]
    }

    # Use verbose for progress update, without custom tqdm integration
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best model: {best_model}")

    y_test_pred = best_model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a stock classifier with advanced techniques.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file containing technical data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of the data to be used as the test set')

    args = parser.parse_args()
    main(args.input_file, args.test_size)