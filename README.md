# Importing necessary libraries for Stock prediction program

* import yfinance as yf
* import pandas as pd

# Getting S&P 500 stock data
* sp500 = yf.Ticker("^GSPC")
* sp500 = sp500.history(period ="max")

# Dropping dividends and stock splits columns
* del sp500["Dividends"]
* del sp500["Stock Splits"]

# Adding tomorrow's closing price as a new column
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Adding a target column that indicates whether the stock price increased or decreased the next day
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"] ).astype(int)

# Copying the data from 1990 onwards
sp500 = sp500.loc["1990-01-01":].copy()

# Splitting the data into training and test sets
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close","Volume","Open","High","Low"]

# Fitting the model on the training set and making predictions on the test set
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

# Calculating the precision score for the predictions
from sklearn.metrics import precision_score
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

# Combining the predictions with the true values for visualisation
combined = pd.concat([test["Target"], preds], axis =1)

# Plotting the combined data
combined.plot()

# Defining a function to fit the model, make predictions, and combine the predictions with the true values
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis =1)
    return combined

# Defining a function to perform backtesting
def backtest(data,model,predictors,start=2500, step= 250):
    all_predictions=[]
    # Iterating over the data in steps of the specified size
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Performing backtesting on the data using the model and predictors
predictions = backtest(sp500, model, predictors)

