import pandas as pd
import numpy as np
import geopy.distance as gpd
from sklearn.model_selection import cross_val_score
import pickle

# calculate the distance between two coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    start = (lat1, lon1)
    end = (lat2, lon2)
    distance = gpd.geodesic(start, end).m
    return distance

# calculate the direction between two coordinates
def calculate_bearing(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

# Determine if a given hour falls within rush hour
def is_rush_hour(hour):
    return 1 if (18 <= hour <= 22) else 0 # Rush hour is between 6 PM and 10 PM

# Prepare the data by adding new features
def prepare_data(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])  # Convert pickup_datetime to a datetime object
    df['DayofMonth'] = df['pickup_datetime'].dt.day  # Extract day of the month
    df['DayofWeek'] = df['pickup_datetime'].dt.dayofweek  # Extract day of the week (0 = Monday, 6 = Sunday)
    df['month'] = df['pickup_datetime'].dt.month  # Extract month
    df['hour'] = df['pickup_datetime'].dt.hour  # Extract hour
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear  # Extract day of the year

    # Calculate the  distance between pickup and dropoff locations
    df['distance'] = df.apply(lambda row: calculate_distance(
        row['pickup_latitude'],
        row['pickup_longitude'],
        row['dropoff_latitude'],
        row['dropoff_longitude']
    ), axis=1)

    # Calculate the direction between pickup and dropoff locations
    df['direction'] = df.apply(lambda row: calculate_bearing(
        row['pickup_latitude'],
        row['pickup_longitude'],
        row['dropoff_latitude'],
        row['dropoff_longitude']
    ), axis=1)

    df['rush_hour'] = df['hour'].apply(is_rush_hour)  # Determine if the pickup time is during rush hour

    # Calculate the Manhattan distance (sum of absolute differences in latitude and longitude)
    df['manhattan_distance'] = (abs(df['dropoff_longitude'] - df['pickup_longitude']) +
                                abs(df['dropoff_latitude'] - df['pickup_latitude']))

    df['distance_log'] = np.log1p(df['distance'])  # Log-transform the distance to reduce skewness
    df['manhattan_distance_log'] = np.log1p(df['manhattan_distance'])  # Log-transform the Manhattan distance

    # Calculate sine and cosine of the direction for feature engineering
    df['direction_sin'] = np.sin(np.radians(df['direction']))
    df['direction_cos'] = np.cos(np.radians(df['direction']))

    return df


# Evaluate the model's performance using cross-validation
def pred_evaluation(model, train, train_features):
    cv = 5

    # Calculate RMSE using cross-validation
    rmse_scores = cross_val_score(model, train[train_features], train.trip_duration_log,
                                  scoring='neg_root_mean_squared_error', cv=cv)

    # Calculate R-squared using cross-validation
    r2_scores = cross_val_score(model, train[train_features], train.trip_duration_log, scoring='r2', cv=cv)

    print(f"RMSE: {-rmse_scores.mean():.4f}, R²: {r2_scores.mean():.4f}")


if '__main__' == __name__:

    # Load the pre-trained model from a pickle file
    with open('C:\\Users\\MH\\PycharmProjects\\Taxi Trip sample\\models\\ridge_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print('model loaded')

    # Load the test dataset
    data_path = ''
    test = pd.read_csv(data_path)

    # Prepare the test dataset
    test = prepare_data(test)

    # Define the features to be used for prediction
    numeric_features = ['distance_log', 'manhattan_distance_log', 'direction_sin', 'direction_cos']
    categorical_features = ['vendor_id', 'DayofMonth', 'DayofWeek', 'hour', 'month', 'dayofyear', 'passenger_count',
                            'rush_hour']
    test_features = categorical_features + numeric_features

    # Evaluate the model using the test dataset
    pred_evaluation(model, test, test_features)
