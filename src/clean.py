import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
def clean_housing_data(df):
    df['month+year'] = df['MoSold'].astype(str) + '.' + df['YrSold'].astype(str)
    df['totalbath'] = df['BsmtFullBath'] + (df['BsmtHalfBath'] * 0.6) + df['FullBath'] + (df['HalfBath'] * 0.5)
    drop_cols = [
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'MoSold', 'YrSold',
        'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage',
        'MiscVal', 'OverallCond', 'MSSubClass', 'PoolArea', 'ScreenPorch', '3SsnPorch', 
        'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'BedroomAbvGr', 'BsmtFinSF2'
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df

train_cleaned = clean_housing_data(train.copy())
test_cleaned = clean_housing_data(test.copy())

train_cleaned.to_csv("../data/train_cleaned.csv", index=False)
test_cleaned.to_csv("../data/test_cleaned.csv", index=False)