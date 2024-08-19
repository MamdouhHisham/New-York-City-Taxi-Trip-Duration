import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pickle


def calculate_distance(lat1, lon1, lat2, lon2):
    start = (lat1, lon1)
    end = (lat2, lon2)
    distance = gpd.geodesic(start, end).m
    return distance


def calculate_bearing(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def is_rush_hour(hour):
    return 1 if (18 <= hour <= 22) else 0


def prepare_data(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['DayofMonth'] = df['pickup_datetime'].dt.day
    df['DayofWeek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear

    df['distance'] = df.apply(lambda row: calculate_distance(
        row['pickup_latitude'],
        row['pickup_longitude'],
        row['dropoff_latitude'],
        row['dropoff_longitude']
    ), axis=1)

    df['direction'] = df.apply(lambda row: calculate_bearing(
        row['pickup_latitude'],
        row['pickup_longitude'],
        row['dropoff_latitude'],
        row['dropoff_longitude']
    ), axis=1)

    df['rush_hour'] = df['hour'].apply(is_rush_hour)
    df['manhattan_distance'] = (abs(df['dropoff_longitude'] - df['pickup_longitude']) +
                                abs(df['dropoff_latitude'] - df['pickup_latitude']))

    df['distance_log'] = np.log1p(df['distance'])
    df['manhattan_distance_log'] = np.log1p(df['manhattan_distance'])
    df['direction_sin'] = np.sin(np.radians(df['direction']))
    df['direction_cos'] = np.cos(np.radians(df['direction']))

    return df


def pred_evaluation(model, train, train_features):
    cv = 5
    rmse_scores = cross_val_score(model, train[train_features], train.trip_duration_log,
                                  scoring='neg_root_mean_squared_error', cv=cv)
    r2_scores = cross_val_score(model, train[train_features], train.trip_duration_log, scoring='r2', cv=cv)
    print(f"RMSE: {-rmse_scores.mean():.4f}, R²: {r2_scores.mean():.4f}")


if '__main__' == __name__:

    with open('C:\\Users\\MH\\PycharmProjects\\Taxi Trip sample\\models\\ridge_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print('model loaded')

    data_path = ''
    test = pd.read_csv(path)

    test = prepare_data(test)

    numeric_features = ['distance_log', 'manhattan_distance_log', 'direction_sin', 'direction_cos']
    categorical_features = ['vendor_id', 'DayofMonth', 'DayofWeek', 'hour', 'month', 'dayofyear', 'passenger_count',
                            'rush_hour']
    train_features = categorical_features + numeric_features

    predict_eval(model, test, test_features)
