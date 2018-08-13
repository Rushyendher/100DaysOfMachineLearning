import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    data = pd.read_csv("Position_Salaries.csv")
    X = data.iloc[:, 1].values
    y = data.iloc[:, -1].values

    regressor = SVR(kernel="rbf")
    regressor.fit(X)

    predictions = regressor.predict(7.5)


if __name__ == '__main__':
    main()
