import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def main():
    data = pd.read_csv("Position_Salaries.csv")
    X = data.iloc[:, 1].values.reshape(-1, 1)
    y = data.iloc[:, -1].values

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    linear_regressor_poly = LinearRegression()
    linear_regressor_poly.fit(X_poly, y)

    # Linear Predictions for a job level of 7.5
    linear_regressor.predict(7.5)

    # Polynomial predictions for a job level of 7.5
    linear_regressor_poly.predict(poly.fit_transform(7.5))


if __name__ == '__main__':
    main()
