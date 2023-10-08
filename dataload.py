import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def impute_median(var, data):
    imputed_data = data[data[var].notnull()]
    imputed_data = imputed_data[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return imputed_data


def load_diabetes_data(label_data_rate, random_seed=7):
    data = pd.read_csv(r"./data/diabetes.csv")
    data_copy = data.copy(deep=True)

    data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    missing_val_columns = ['BloodPressure', 'SkinThickness', 'Insulin', 'Glucose', 'BMI']
    for col in missing_val_columns:
        medi0, medi1 = impute_median(col, data_copy)[col]
        data_copy.loc[(data_copy['Outcome'] == 0) & (data_copy[col].isnull()), col] = medi0
        data_copy.loc[(data_copy['Outcome'] == 1) & (data_copy[col].isnull()), col] = medi1

    # Scaler
    sc_X = MinMaxScaler()
    X = pd.DataFrame(sc_X.fit_transform(data_copy.drop(["Outcome"], axis=1)),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                              'DiabetesPedigreeFunction', 'Age'])
    y = data_copy.Outcome

    X = np.asarray(X)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=9, stratify=y)

    # Divide labeled and unlabeled data
    # idx = np.random.permutation(len(y_train))
    #
    # # Label data
    # label_idx = idx[:int(len(idx) * label_data_rate)]
    # unlab_idx = idx[int(len(idx) * label_data_rate):]
    #
    # # Unlabeled data
    # x_unlab = x_train[unlab_idx, :]
    #
    # # Labeled data
    # x_label = x_train[label_idx, :]
    # y_label = y_train[label_idx]

    x_label, x_unlab, y_label, y_unlabel = train_test_split(x_train, y_train, test_size=(1 - label_data_rate),
                                                            random_state=7)

    return x_label, y_label, x_unlab, x_test, y_test, x_train, y_train
