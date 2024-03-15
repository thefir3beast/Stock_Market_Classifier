# Stock Market Classification Project

## Overview
This project aims to classify stocks based on various machine learning models. By leveraging both technical and fundamental data, the project seeks to predict stock categories or future performance indicators. The models used in this project include Support Vector Machines (SVM), Naive Bayes Classifier, Random Forest, and a simple Neural Network.

## Data Collection
The data is collected using the `yfinance` library, fetching historical stock prices and fundamental data for stocks listed in the Dow Jones 30. The fundamental data includes metrics such as Earnings Per Share (EPS), Price to Earnings (P/E) Ratio, Return on Equity (ROE), Debt to Equity Ratio, and Current Ratio.

### How to Fetch Data
To fetch the data, run the `extract_data.py` script with the desired parameters. For example:
```bash
python extract_data.py --index DowJones30 --start_year 2020 --end_year 2022 --save_path ./data/
