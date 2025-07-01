import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import sqrt


def load_data():
    train = pd.read_csv("../HousePrices/data/houses_filtered.csv") 
    test = pd.read_csv("../HousePrices/data/test.csv")
    return train, test

def preprocessing(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
    ])
    return preprocessor

def main():
    train_df, test_df = load_data()
    X = train_df.drop("SalePrice", axis=1)
    y = train_df["SalePrice"]
    
    preprocessor = preprocessing(X)
    model = RandomForestRegressor(random_state=44)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10],
    }


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=44
    )

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=1
    )

    grid_search.fit(X_train, y_train)
    print('Best Parameters:', grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    print(f'RMSE: {rmse:.3f}')
    print(f'R2: {r2:.3f}')

    best_model.fit(X, y)
    test_preds = best_model.predict(test_df)
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': test_preds
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()



