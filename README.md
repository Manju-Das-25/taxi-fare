# 🚖 TripFare Predictor

This project is a **Streamlit-based machine learning app** that predicts taxi fares based on trip details such as pickup/dropoff locations, passenger count, trip duration, and time of travel.

## 📌 Features
- Preprocesses and cleans taxi trip data.
- Uses **Haversine distance** to calculate trip distance.
- Extracts time-based features like hour of the day, weekday/weekend, and night rides.
- Trains multiple regression models (**Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso**) and selects the best one.
- Saves the best-trained model using **pickle**.
- Provides an **interactive Streamlit UI** for fare prediction.

## ⚙️ Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
