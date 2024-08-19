# New-York-City-Taxi-Trip-Duration

![Type of Project](https://img.shields.io/badge/Type%20of%20Project-Machine%20Learning-orange?style=flat)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.4.2-red.svg?style=flat&logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.22.3-lightblue.svg?style=flat&logo=numpy)

## Project Overview

This project focuses on predicting the duration of NYC taxi rides using machine learning methods. The workflow includes data preprocessing, feature engineering, model development, and performance evaluation. Ridge Regression uses as the main predictive model in this project.

![NYC Taxi](https://www.theoldie.co.uk/media/articles/57856BB1-0750-4092-96FE-9171870BB5F9.jpeg)

## Data Description

The dataset contains information about NYC taxi rides. columns include:

- `vendor_id` : A unique identifier representing the taxi service provider associated with the trip.
- `pickup_datetime` : The exact date and time when the taxi meter was started, indicating the beginning of the trip.
- `passenger_count` : The number of passengers in the taxi, as recorded by the driver.
- `pickup_longitude` : The geographic longitude of the location where the trip started.
- `pickup_latitude` : The geographic latitude of the location where the trip started.
- `dropoff_longitude` : The geographic longitude of the location where the trip ended.
- `dropoff_latitude` : The geographic latitude of the location where the trip ended.
- `trip_duration` : The total time of the trip, measured in seconds, from start to finish.

## Feature Engineering

Various features are engineered to improve the model's performance:

- **Time-Based Features**: Extracted from `pickup_datetime` (hour, dayofmonth, dayofweek, month,etc..).
- **Geographical Features**: Direction and Distances between pickup and dropoff locations.
- **Log Transformation**: Applied to trip duration, distance and manhattan distance to reduce skewness.
- **Rush Hour**: A newly created feature that identifies whether the trip occurred during a peak traffic period

## Model
