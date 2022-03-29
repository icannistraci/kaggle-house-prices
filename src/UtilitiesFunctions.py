import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer


def check_equal_value(df, percentage):
    same_val = []
    for col in df.columns:
        df2 = df.groupby(col)['Id'].nunique().to_frame()
        for i, row in df2.iterrows():
            if row['Id'] / 1460 >= percentage:
                same_val.append(col)

    return list(dict.fromkeys(same_val))


def check_high_correlated_features(df, percentage):
    rows, cols = df.corr().shape
    flds = list(df.corr().columns)
    corr = df.corr().corr().values
    for i in range(cols):
        for j in range(i + 1, cols):
            if corr[i, j] > percentage:
                print(flds[i], flds[j], corr[i, j])


def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def scaling_data_multiple(x_train, x_test):
    scaler_b = MinMaxScaler()
    scaler_c = MinMaxScaler(feature_range=(-1, 1))
    scaler_d = MaxAbsScaler()
    scaler_e = RobustScaler(quantile_range=(25, 75))

    scaled_data = [
        ('Data after [0,1] min-max scaling',
         scaler_b.fit_transform(x_train), scaler_b.transform(x_test)),
        ('Data after [-1,+1] min-max scaling',
         scaler_c.fit_transform(x_train), scaler_c.transform(x_test)),
        ('Data after max-abs scaling',
         scaler_d.fit_transform(x_train), scaler_d.transform(x_test)),
        ('Data after robust scaling',
         scaler_e.fit_transform(x_train), scaler_e.transform(x_test)),
        ('Data after power transformation (Box-Cox)',
         PowerTransformer(method='box-cox').fit_transform(x_train),
         PowerTransformer(method='box-cox').fit_transform(x_test)),
        ('Data after quantile transformation (gaussian pdf)',
         QuantileTransformer(output_distribution='normal').fit_transform(x_train),
         QuantileTransformer(output_distribution='normal').fit_transform(x_test)),
        ('Data after quantile transformation (uniform pdf)',
         QuantileTransformer(output_distribution='uniform').fit_transform(x_train),
         QuantileTransformer(output_distribution='uniform').fit_transform(x_test))
    ]

    return scaled_data


def create_submission_file(y_pred):
    submission = pd.read_csv('../data/sample_submission.csv')
    submission['SalePrice'] = y_pred
    submission.to_csv(r'submission.csv', index=False)
