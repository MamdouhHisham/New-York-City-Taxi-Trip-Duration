import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures , MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pickle


def modeling(train, val):

    numeric_features = ['distance_log', 'manhattan_distance_log', 'direction_sin', 'direction_cos']
    categorical_features = ['vendor_id', 'DayofMonth', 'DayofWeek', 'hour', 'month' ,'dayofyear' ,'passenger_count', 'rush_hour',
                            ]
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', MinMaxScaler(), numeric_features)
    ]
        , remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('poly', PolynomialFeatures(degree=2,interaction_only=True)),
        ('regression', Ridge(alpha=1, random_state=42)),
    ])

    model = pipeline.fit(train[train_features], train.trip_duration_log)
    pred_evaluation(model, train, train_features)
    pred_evaluation(model, val, train_features)

    with open('models/ridge_regression_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved as 'ridge_regression_model'.")

def pred_evaluation(model, train, train_features):
    cv = 5
    rmse_scores = cross_val_score(model, train[train_features], train.trip_duration_log, scoring='neg_root_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, train[train_features], train.trip_duration_log, scoring='r2', cv=cv)
    print(f"Cross-Validation RMSE: {-rmse_scores.mean():.4f}, R²: {r2_scores.mean():.4f}")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    train = pd.read_csv('C:\\Users\\MH\\PycharmProjects\\Taxi Trip Duration\\data\\train_df_prepared.csv')
    val = pd.read_csv('C:\\Users\\MH\\PycharmProjects\\Taxi Trip Duration\\data\\val_df_prepared.csv')

    modeling(train, val)

