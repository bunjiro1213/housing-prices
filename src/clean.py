import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
def clean_housing_data(df):
    df['month+year'] = df['MoSold'].astype(str) + '.' + df['YrSold'].astype(str)
    df['totalbath'] = df['BsmtFullBath'] + (df['BsmtHalfBath'] * 0.6) + df['FullBath'] + (df['HalfBath'] * 0.5)
    df['Totalsqft'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + (df['PoolArea']*.2)
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['IsNewHouse'] = (df['Age'] <= 5)
    drop_cols = [
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'MoSold', 'YrSold',
        'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage',
        'MiscVal', 'PoolArea', 'ScreenPorch', '3SsnPorch', 
        'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'BedroomAbvGr',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'YearRemodAdd'
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df

train_cleaned = clean_housing_data(train.copy())
test_cleaned = clean_housing_data(test.copy())

train_cleaned.to_csv("../data/train_cleaned.csv", index=False)
test_cleaned.to_csv("../data/test_cleaned.csv", index=False)