import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def main():
    data = pd.read_csv('50_Startups.csv')
    X = data.iloc[:, :-1].values

    label_encoder = LabelEncoder()
    X[:, 3] = label_encoder.fit_transform(X[:, 3])
    one_hot_encoder = OneHotEncoder(categorical_features=[3])
    X = one_hot_encoder.fit_transform(X).toarray()

    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


if __name__ == '__main__':
    main()
