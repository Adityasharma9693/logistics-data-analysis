# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# PHASE 1 & 2: Data Loading and Preprocessing
# ---------------------------------------------------------
def load_and_clean_data(filepath):
    # Load daily order manifest
    df = pd.read_csv(filepath)
    
    # Drop rows with missing crucial geolocation data
    df = df.dropna(subset=['latitude', 'longitude', 'package_weight'])
    
    # Filter out anomalous coordinates (e.g., outside operational bounds)
    df = df[(df['latitude'] > 20.0) & (df['latitude'] < 25.0)] 
    
    return df

orders_df = load_and_clean_data('daily_orders.csv')

# ---------------------------------------------------------
# PHASE 3: Clustering (Creating Delivery Zones)
# ---------------------------------------------------------
def create_delivery_zones(df, num_vehicles):
    # Extract coordinates for clustering
    coordinates = df[['latitude', 'longitude']]
    
    # Apply K-Means clustering to group addresses by geographical proximity
    kmeans = KMeans(n_clusters=num_vehicles, random_state=42, n_init=10)
    df['assigned_zone'] = kmeans.fit_predict(coordinates)
    
    return df

# Assume 15 vehicles available for the day
optimized_zones_df = create_delivery_zones(orders_df, num_vehicles=15)

# ---------------------------------------------------------
# PHASE 4: Predictive Modeling (ETA Prediction)
# ---------------------------------------------------------
def train_eta_model(historical_data):
    # Features: Distance from hub, historical traffic index, weather severity, weight
    features = ['distance_km', 'traffic_index', 'weather_severity', 'package_weight']
    target = 'actual_delivery_time_mins'
    
    X = historical_data[features]
    y = historical_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate Model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error: {mae:.2f} minutes")
    
    return model

# historical_df = pd.read_csv('historical_deliveries.csv')
# eta_model = train_eta_model(historical_df)