import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    data = pd.read_csv("Salary_Data.csv")
    X = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    regressor = LinearRegression()
    regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_predictions = regressor.predict(X_test.reshape(-1, 1))
    print(y_predictions)


if __name__ == '__main__':
    main()
