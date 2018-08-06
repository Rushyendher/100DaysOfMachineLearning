import pandas as pd
import numpy as np
import statsmodels.formula.api as sm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression


def backward_elimination(X, y, p_value):
    foo = len(X[0])
    for index in range(foo):
        regressor_OLS = sm.OLS(endog=y, exog=X).fit()
        p_values = regressor_OLS.pvalues.astype(float)
        max_value_index = np.argmax(p_values)

        if p_values[max_value_index] > p_value:
            X = np.delete(X, max_value_index, 1)
        print(regressor_OLS.summary())


def main():

    data = pd.read_csv('50_Startups.csv')
    X = data.iloc[:, :-1].values

    # Label encoding for categorical values
    label_encoder = LabelEncoder()
    X[:, 3] = label_encoder.fit_transform(X[:, 3])
    one_hot_encoder = OneHotEncoder(categorical_features=[3])
    X = one_hot_encoder.fit_transform(X).toarray()

    X = X[:, 1:]
    y = data.iloc[:, -1].values

    # Split the data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

    # Fit the model and predict
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_predictions = regressor.predict(X_test)

    # Optimizing using Backward elimination
    X = np.append(np.ones((50, 1), dtype=int), X, 1)

    X_opt = X[:, [0, 1, 2, 3, 4, 5]]
    backward_elimination(X_opt, y, 0.05)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
