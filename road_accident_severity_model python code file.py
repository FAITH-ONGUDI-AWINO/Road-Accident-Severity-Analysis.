
# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Step 1: Load the dataset
# Assuming you have downloaded the dataset and it's in a CSV format
df = pd.read_csv('path_to_your_dataset.csv')

# Step 2: Data exploration
print(df.head())  # Inspect the first few rows
print(df.info())  # Get information about the dataset

# Step 3: Selecting features (independent variables) and the target (dependent variable)
# Example: You may want to predict 'accident_severity' based on other factors like 'speed_limit', 'weather_conditions', etc.
X = df[['speed_limit', 'weather_conditions', 'road_type']]  # Replace with relevant columns from your dataset
y = df['accident_severity']  # Replace with the appropriate dependent variable column

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the model for future use
with open('accident_severity_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Predicting accident severity for a new set of variables
# Hypothetical data: speed_limit = 30, weather_conditions = 1 (clear), road_type = 2 (urban roads)
hypothetical_data = [[30, 1, 2]]
predicted_severity = model.predict(hypothetical_data)
print(f"Predicted Accident Severity: {predicted_severity[0]}")
