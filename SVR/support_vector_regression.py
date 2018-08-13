import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    data = pd.read_csv("Position_Salaries.csv")
    X = data.iloc[:, 1].values
    y = data.iloc[:, -1].values

    # Apply feature scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Fit the model
    regressor = SVR(kernel="rbf")
    regressor.fit(X)

    predictions = regressor.predict(7.5)


if __name__ == '__main__':
    main()
