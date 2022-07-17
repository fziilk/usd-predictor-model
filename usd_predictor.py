"""
Data source: [Yahoo Finance](https://finance.yahoo.com/quote/USDIDR%3DX/history?period1=993686400&period2=1657152000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true&guccounter=1)
Data Citation:
> None
"""

try:
    # utils
    import numpy as np
    import pandas as pd
    import os
    import pickle

    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns
    from IPython.display import display

    # model dev
    import sklearn

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor

    print(f"using Scikit Learn v{sklearn.__version__}")
except Exception:
    import sys

    exc_type, exc_obj, exc_tb = sys.exc_info()
    raise Exception(f"Error '{exc_obj}' on line {exc_tb.tb_lineno}!")
finally:
    print("Success on importing dependencies with 0 exception.")

# Data Loading

url = "https://query1.finance.yahoo.com/v7/finance/download/USDIDR=X?period1=993686400&period2=1658016000&interval=1d&events=history&includeAdjustedClose=true"
df = pd.read_csv(url, sep=",")

print(df.info())

# Data Preparation

## Data Cleaning


class data_prep():
    def detect_outliers(df):
        cwo = list()
        df_cols = df.columns.tolist()

        for column in df_cols :
            if type(df[column][0]) == str:
                df_cols.remove(column)

        for col in df_cols :
            skewness = df[col].skew()

            if skewness >= 1 or skewness <= -1:
                cwo.append(col)

        return cwo

    def clean_dataset(dataframe):
        df = dataframe
        cwo = data_prep.detect_outliers(df)

        for o_col in cwo:
            Q1 = df[o_col].quantile(0.25)
            Q3 = df[o_col].quantile(0.75)
            IQR = Q3 - Q1
            whisker_width = 1.5

            outliers = df[(df[o_col] < Q1 - whisker_width*IQR) | (df[o_col] > Q3 + whisker_width*IQR)]
            outlier_values = pd.DataFrame(outliers[o_col])
            outlier_values = outlier_values.values
            
            col_median = df[o_col].median()

            df[o_col] = df[o_col].replace(to_replace=outlier_values, value=col_median)


dp = data_prep

df.drop(labels=["Volume"], axis=1, inplace=True)

print(f"column(s) with outlier values: {dp.detect_outliers(df)}")

if len(dp.detect_outliers(df)) > 0:
    dp.clean_dataset(df)

    print(f"column(s) with outlier values: {dp.detect_outliers(df)}")

print(f"column(s) with missing values: \n{df.isnull().sum()}")

df = df.interpolate()

print(f"column(s) with missing values: \n{df.isnull().sum()}")

## Data Transformation

for date in df_dates:
    date_obj = date.split("-")
    dmy_list = [date_obj[2], date_obj[1], date_obj[0]]

    date_list.append(dmy_list)

date_df = pd.DataFrame(date_list, columns=["DD", "MM", "YY"])

date_old = df.copy()
df_new = pd.concat([date_df, df], axis=1)

df_new.drop(labels=["Date"], axis=1, inplace=True)

## Train Test Split

df_new.drop(labels=["High", "Low"], axis=1, inplace=True)
df = df_new.copy()

independent = df[df.columns[:3]]
dependent = df[df.columns[3:]]


def gss(dataset_length, test_size):
    test = round(dataset_length * test_size)
    train = dataset_length - test

    return (train, test)


for size in [0.1, 0.05, 0.2]:
    train, test = gss(len(df), size)
    
    print(f"dataset training size for test size of {size}: \n\ttrain data: {train}\n\ttest data: {test}")

X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.05, shuffle=False)

# Modelling


def get_prediction_table(model, x_data, pred_columns, y_true):
    model_pred = model.predict(x_data)
    pred_res = pd.DataFrame(model_pred, columns=pred_columns)

    y_true = y_true.reset_index()
    y_true.drop(labels=["index"], axis=1, inplace=True)

    test_result = pd.concat([pred_res, y_true], axis=1)

    return test_result

## K-Nearest Neighbor Model

KNN = KNeighborsRegressor()

knn_params = [{"n_neighbors": (range(10, 50)), 
                "algorithm": ("ball_tree", "kd_tree", "brute"), 
                "metric": ("minkowski", "euclidean")}]

knngs = GridSearchCV(KNN, 
                        knn_params, 
                        cv=15, 
                        scoring="neg_mean_squared_error")

knngs.fit(X_train, y_train)

bp = knngs.best_params_
bs = knngs.best_score_

print(f"best params: {bp}\nbest score: {bs}")

KNN = KNeighborsRegressor(algorithm=bp["algorithm"], 
                            metric=bp["metric"], 
                            n_neighbors=bp["n_neighbors"])

KNN.fit(X_train, y_train)

model_res = get_prediction_table(KNN,
                                    X_test,
                                    ["Open (prediction)", "Close (prediction)", "Adj Close (prediction)"],
                                    y_test)

print(model_res)

## Random Forest Algorithm

RF = RandomForestRegressor()

RF_params = [{"n_estimators": (range(100, 135)), "max_depth": (None, range(1, 30))}]
RFgs = GridSearchCV(RF, RF_params, cv=5, scoring="neg_mean_squared_error")

RFgs.fit(X_train, y_train)

bp = RFgs.best_params_
bs = RFgs.best_score_

print(f"best params: {bp}\nbest score: {bs}")

RF = RandomForestRegressor(n_estimators=bp["n_estimators"], max_depth=bp["max_depth"])

RF.fit(X_train, y_train)

model_res = get_prediction_table(RF, X_test, ["Open (pred)", "Close (pred)", "Adj Close (pred)"], y_test)

print(model_res)

# Evaluation

mse = pd.DataFrame(columns=["train", "test"], index=["KNN", "RF"])
model_dict = {"KNN": KNN, "RF": RF}

for name, model in model_dict.items():
    mse.loc[name, "train"] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, "test"] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

print(mse)

usd_predictor = RF

# Saving The Model

with open("usd_predictor.pkl", "wb") as f:
    pickle.dump(usd_predictor, f)

# Getting Prediction


class usd_predictor_class():
    def load_model(path):
        with open(path, "rb") as f:
            predictor = pickle.load(f)

        return predictor

    def gp(predictor, dmy):
        dd, mm, yy = dmy

        prediction = predictor.predict([[dd, mm, yy]]).tolist()
        
        open = prediction[0][0]
        close = prediction[0][1]
        adj_close = prediction[0][2]

        pred_df = pd.DataFrame([open, close, adj_close], columns=["Value"], index=["Open", "Close", "Adj Close"])

        return pred_df

usdp = usd_predictor_class

predictor = usdp.load_model("usd_predictor.pkl")
prediction = usdp.gp(predictor, (17, 7, 2022))

print(prediction)
