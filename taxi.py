import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import radians, sin, cos, sqrt, atan2


# Replace with the path to your dataset
df = pd.read_csv('D:/projects py/taxi  price prediction')
print(df)

# Display dataset info
print(df.info())
df.head()


# Drop rows with missing values
df = df.dropna()


print("After handling missing values:")
print(df.info())

from sklearn.preprocessing import LabelEncoder

# Label Encoding
label_encoders = {}
categorical_columns = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders if needed for deployment

print("After encoding categorical variables:")
print(df.head())

# Calculate additional features
df['Total_Cost_Calculated'] = df['Base_Fare'] + (df['Trip_Distance_km'] * df['Per_Km_Rate']) + (df['Trip_Duration_Minutes'] * df['Per_Minute_Rate'])
df['Cost_Per_KM'] = df['Trip_Price'] / df['Trip_Distance_km']

print("After feature engineering:")
print(df[['Trip_Price', 'Total_Cost_Calculated', 'Cost_Per_KM']].head())

# Features (independent variables) and target (dependent variable)
X = df[['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
        'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
        'Per_Minute_Rate', 'Trip_Duration_Minutes']]

y = df['Trip_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Forest - MAE: {mae}, RMSE: {rmse}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Display basic statistics for numerical columns
print("Numerical Column Statistics:")
print(df.describe())

# Display unique values for categorical columns
categorical_columns = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts())

numerical_columns = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate', 
                     'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Trip_Price']

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df, palette='viridis')
    plt.title(f"Count Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Trip_Distance_km', y='Trip_Price', data=df, alpha=0.6, color='green')
plt.title("Trip Distance vs. Trip Price")
plt.xlabel("Trip Distance (km)")
plt.ylabel("Trip Price")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Traffic_Conditions', y='Trip_Price', data=df, palette='Set2')
plt.title("Trip Price by Traffic Conditions")
plt.xlabel("Traffic Conditions")
plt.ylabel("Trip Price")
plt.show()


sns.pairplot(df[['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 
                 'Trip_Duration_Minutes', 'Trip_Price']])
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Time_of_Day', y='Trip_Price', data=df, palette='viridis')
plt.title("Trip Price by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Trip Price")
plt.xticks(rotation=45)
plt.show()

pivot_table = df.pivot_table(values='Trip_Price', 
                             index='Traffic_Conditions', 
                             columns='Weather', aggfunc='mean')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Average Trip Price by Traffic and Weather")
plt.ylabel("Traffic Conditions")
plt.xlabel("Weather")
plt.show()


for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], palette='coolwarm')
    plt.title(f"Box Plot for {col}")
    plt.xlabel(col)
    plt.show()

from scipy.stats import zscore

# Compute Z-scores
z_scores = np.abs(zscore(df[numerical_columns]))
outliers = (z_scores > 3).sum(axis=0)

print("Number of Outliers in Each Numerical Column:")
print(outliers)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Split data into features (X) and target (y)
X = df[['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count',
        'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 
        'Per_Minute_Rate', 'Trip_Duration_Minutes']]
y = df['Trip_Price']

# Handle missing values (drop or fill them)
X = X.dropna()
y = y.loc[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model Performance - MAE: {mae}, RMSE: {rmse}")


# New trip data
new_trip = [[10.5, 2, 4, 2, 1, 0, 50.0, 15.0, 1.5, 30]]

# Scale the input
new_trip_scaled = scaler.transform(new_trip)

# Predict the fare
predicted_price = model.predict(new_trip_scaled)
print(f"Predicted Taxi Fare: {predicted_price[0]:.2f}")


# Batch of trips
trips = [
    [10.5, 2, 4, 2, 1, 0, 50.0, 15.0, 1.5, 30],
    [5.0, 1, 6, 1, 0, 1, 40.0, 12.0, 2.0, 15],
    [20.0, 3, 2, 3, 2, 0, 100.0, 18.0, 1.8, 45]
]

# Scale the trips
trips_scaled = scaler.transform(trips)

# Predict fares
predicted_fares = model.predict(trips_scaled)
for i, fare in enumerate(predicted_fares):
    print(f"Trip {i+1}: Predicted Fare = {fare:.2f}")


import pickle

# Save the model
with open('taxi_fare_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Load the model and scalercl
with open('taxi_fare_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Predict using the loaded model
new_trip = [[10.5, 2, 4, 2, 1, 0, 50.0, 15.0, 1.5, 30]]
new_trip_scaled = loaded_scaler.transform(new_trip)
predicted_price = loaded_model.predict(new_trip_scaled)

print(f"Predicted Taxi Fare: {predicted_price[0]:.2f}")
#taxi.py