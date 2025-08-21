# trip_fare_prediction_app.py

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import streamlit as st
import pickle
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# --- 1. DATA PREPROCESSING, EDA, AND MODEL TRAINING ---
# This part of the code is for the backend. In a real-world scenario, you would
# run this part once to train and save your model. For this example, we've included
# it in a single file for completeness. You would then create a separate file
# for the Streamlit app that only loads the model.

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    """
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def train_and_save_model(data_path='train_data.csv'):
    """
    Loads data, preprocesses it, trains a regression model, and saves the best model.
    """
    try:
        # Load the dataset. Replace 'train_data.csv' with the actual path to your file.
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{data_path}' was not found.")
        return None, None

    # --- Data Cleaning and Feature Engineering ---
    # Drop rows with missing or zero values that are essential
    df.dropna(subset=['passenger_count', 'trip_distance'], inplace=True)
    df = df[df['passenger_count'] > 0]
    df = df[df['total_amount'] > 0]
    df = df[df['trip_distance'] > 0]

    # Convert datetime columns and extract features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Calculate trip duration in minutes
    df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Extract time-based features
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['is_weekend'] = df['tpep_pickup_datetime'].dt.weekday >= 5
    df['is_night_ride'] = df['pickup_hour'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0)

    # Use Haversine distance if trip_distance is not provided
    # The document suggests using it if coordinates are present
    df['haversine_distance_km'] = df.apply(
        lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'],
                              row['dropoff_latitude'], row['dropoff_longitude']),
        axis=1
    )

    # Handle outliers using IQR method for 'total_amount' and 'haversine_distance_km'
    Q1_fare = df['total_amount'].quantile(0.25)
    Q3_fare = df['total_amount'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    df = df[~((df['total_amount'] < (Q1_fare - 1.5 * IQR_fare)) | (df['total_amount'] > (Q3_fare + 1.5 * IQR_fare)))]

    Q1_dist = df['haversine_distance_km'].quantile(0.25)
    Q3_dist = df['haversine_distance_km'].quantile(0.75)
    IQR_dist = Q3_dist - Q1_dist
    df = df[~((df['haversine_distance_km'] < (Q1_dist - 1.5 * IQR_dist)) | (
                df['haversine_distance_km'] > (Q3_dist + 1.5 * IQR_dist)))]

    # Select features and target variable
    features = [
        'haversine_distance_km', 'passenger_count', 'trip_duration_minutes',
        'pickup_hour', 'is_weekend', 'is_night_ride', 'pickup_day_of_week'
    ]
    target = 'total_amount'

    X = df[features]
    y = df[target]

    # --- Exploratory Data Analysis (EDA) ---
    st.subheader("Exploratory Data Analysis (EDA)")

    # Plot 1: Distribution of Total Fare
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['total_amount'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Total Fare')
    ax.set_xlabel('Total Fare ($)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Plot 2: Fare vs. Haversine Distance
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='haversine_distance_km', y='total_amount', data=df, ax=ax)
    ax.set_title('Total Fare vs. Haversine Distance')
    ax.set_xlabel('Haversine Distance (km)')
    ax.set_ylabel('Total Fare ($)')
    st.pyplot(fig)

    # Define categorical and numerical features for the pipeline
    categorical_features = ['pickup_day_of_week']
    numerical_features = [
        'haversine_distance_km', 'passenger_count', 'trip_duration_minutes',
        'pickup_hour', 'is_weekend', 'is_night_ride'
    ]

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Building, Hyperparameter Tuning, and Evaluation ---
    # Define models
    models = {
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso()
    }

    best_model = None
    best_r2 = -np.inf

    print("Training and evaluating models...")
    for name, model in models.items():
        # Create a pipeline that first preprocesses the data then trains the model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', model)])

        # --- Hyperparameter Tuning for RandomForestRegressor ---
        if name == 'RandomForestRegressor':
            st.subheader("Hyperparameter Tuning (Random Forest)")
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [5, 10, None]
            }
            grid_search = GridSearchCV(full_pipeline, param_grid, cv=2, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            print(f"Best parameters for {name}: {grid_search.best_params_}")
            full_pipeline = grid_search.best_estimator_
            y_pred = full_pipeline.predict(X_test)
            st.write("Best Hyperparameters:", grid_search.best_params_)
        else:
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"--- {name} ---")
        print(f"R-squared: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print("-" * 20)

        # Track the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = full_pipeline

    if best_model:
        # Save the best model to a file using pickle
        model_filename = 'best_fare_predictor.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"Best model saved as '{model_filename}'")
        return best_model, model_filename
    else:
        return None, None


# --- 2. STREAMLIT UI ---

# Check if the Streamlit app is being run directly
if __name__ == "__main__":
    st.set_page_config(page_title="TripFare Predictor", layout="centered")

    # Define the path to the pickled model.
    MODEL_PATH = 'best_fare_predictor.pkl'


    # Define a helper function to create a placeholder model in case the file doesn't exist
    @st.cache_resource
    def load_or_train_model():
        """
        Loads the pre-trained model or trains a new one if it doesn't exist.
        """
        try:
            with open(MODEL_PATH, 'rb') as file:
                model = pickle.load(file)
                st.success("Pre-trained model loaded successfully!")
                return model
        except FileNotFoundError:
            st.warning("Model not found. Training a new model. This may take a moment...")

            # This is the updated, larger dummy dataset
            dummy_data = {
                'tpep_pickup_datetime': pd.to_datetime(
                    ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 12:30:00',
                     '2023-01-03 15:00:00', '2023-01-04 08:00:00', '2023-01-05 23:00:00',
                     '2023-01-06 02:00:00', '2023-01-07 18:45:00', '2023-01-08 09:15:00',
                     '2023-01-09 13:00:00']),
                'tpep_dropoff_datetime': pd.to_datetime(
                    ['2023-01-01 10:15:00', '2023-01-01 11:30:00', '2023-01-02 12:45:00',
                     '2023-01-03 15:20:00', '2023-01-04 08:10:00', '2023-01-05 23:30:00',
                     '2023-01-06 02:25:00', '2023-01-07 19:15:00', '2023-01-08 09:30:00',
                     '2023-01-09 13:40:00']),
                'passenger_count': [1, 2, 3, 1, 4, 2, 1, 3, 1, 2],
                'pickup_longitude': [-73.99, -73.98, -74.00, -73.97, -73.95, -73.99, -74.01, -73.96, -73.98, -74.00],
                'pickup_latitude': [40.76, 40.75, 40.74, 40.78, 40.76, 40.73, 40.72, 40.78, 40.75, 40.74],
                'dropoff_longitude': [-74.00, -73.99, -73.98, -74.01, -73.98, -73.97, -73.99, -73.95, -74.00, -73.98],
                'dropoff_latitude': [40.75, 40.76, 40.76, 40.75, 40.74, 40.75, 40.74, 40.77, 40.76, 40.76],
                'total_amount': [15.5, 25.0, 20.0, 30.5, 12.0, 45.0, 22.0, 38.0, 18.5, 28.0],
                'trip_distance': [2.5, 5.0, 3.2, 7.8, 1.5, 10.1, 4.5, 6.2, 2.9, 5.5]
            }
            dummy_df = pd.DataFrame(dummy_data)
            dummy_df.to_csv('dummy_data.csv', index=False)

            # Train and save the model
            model, _ = train_and_save_model(data_path='dummy_data.csv')

            if model:
                st.success("Model trained and saved. You can now use the predictor.")
                return model
            else:
                st.error("Model training failed. Please check your data and try again.")
                return None


    # Load the model at the start of the app
    model = load_or_train_model()

    # Check if the model was loaded successfully before proceeding
    if model:
        # --- UI elements ---
        st.title("🚗 TripFare: Urban Taxi Fare Predictor")
        st.markdown(
            """
            This application predicts the total taxi fare based on your trip details.

            **Instructions:**
            1.  Enter your trip information in the fields below.
            2.  Click the 'Predict Fare' button to see the estimated cost.
            """
        )

        st.divider()

        # Input fields for the user
        col1, col2 = st.columns(2)
        with col1:
            passenger_count = st.number_input("Number of Passengers", min_value=1, max_value=8, value=1)
            pickup_longitude = st.number_input("Pickup Longitude", value=-73.98)
            pickup_latitude = st.number_input("Pickup Latitude", value=40.75)
            trip_duration_minutes = st.number_input("Trip Duration (minutes)", min_value=1, value=15)

        with col2:
            dropoff_longitude = st.number_input("Dropoff Longitude", value=-74.00)
            dropoff_latitude = st.number_input("Dropoff Latitude", value=40.73)
            pickup_date = st.date_input("Pickup Date")
            pickup_time = st.time_input("Pickup Time")

        # Create a button to trigger prediction
        if st.button("Predict Fare"):
            # Combine date and time for feature engineering
            pickup_datetime_str = f"{pickup_date} {pickup_time}"
            pickup_datetime = pd.to_datetime(pickup_datetime_str)

            # Create a dictionary from user inputs to form a DataFrame
            # The structure must match the training data's features
            user_input_data = {
                'haversine_distance_km': haversine(
                    pickup_latitude, pickup_longitude,
                    dropoff_latitude, dropoff_longitude
                ),
                'passenger_count': passenger_count,
                'trip_duration_minutes': trip_duration_minutes,
                'pickup_hour': pickup_datetime.hour,
                'is_weekend': pickup_datetime.weekday() >= 5,
                'is_night_ride': 1 if (pickup_datetime.hour >= 22 or pickup_datetime.hour < 6) else 0,
                'pickup_day_of_week': pickup_datetime.day_name()
            }

            # Create a DataFrame from the user input
            input_df = pd.DataFrame([user_input_data])

            try:
                # Make a prediction
                predicted_fare = model.predict(input_df)[0]

                # Display the result
                st.success(f"Estimated Fare: **${predicted_fare:.2f}**")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
