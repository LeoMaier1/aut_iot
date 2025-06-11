import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Load training data (equivalent to sns.load_dataset('iris'))
data_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data.csv')
df = pd.read_csv(data_path)
print("Training data loaded:")
print(df.head())

# Prepare features and target (equivalent to iris example)
y = df['final_weight']  # Target variable
X = df.drop(['bottle', 'final_weight'], axis=1)  # Features (remove ID and target)
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nTraining features shape:", X_train.shape)
print("Test features shape:", X_test.shape)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Show model coefficients
print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Make predictions on training data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"\nModel Performance:")
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# Load prediction data (X.csv)
pred_path = os.path.join(os.path.dirname(__file__), '..', 'X.csv')
X_pred_df = pd.read_csv(pred_path)
print(f"\nPrediction data loaded. Shape: {X_pred_df.shape}")

# Prepare prediction features (same columns as training)
X_pred = X_pred_df.drop(['bottle'], axis=1)  # Remove bottle ID
X_pred = X_pred.fillna(X_pred.mean())  # Handle missing values

# Make final predictions
final_predictions = model.predict(X_pred)
print(f"\nFirst 5 predictions: {final_predictions[:5]}")

# Save predictions in required format
predictions_df = pd.DataFrame({
    'Flaschen_ID': X_pred_df['bottle'],
    'y_hat': final_predictions
})

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'reg_student1-student2-student3.csv')
predictions_df.to_csv(output_path, index=False)

print(f"\n=== RESULTS TABLE ===")
print("| Genutzte Spalten | Modell-Typ | MSE-Wert (Training) | MSE-Wert (Test) |")
print("|------------------|------------|---------------------|-----------------|")
print(f"| All features | Linear | {mse_train:.4f} | {mse_test:.4f} |")

print(f"\n=== MODEL FORMULA ===")
formula = f"final_weight = {model.intercept_:.4f}"
for feature, coef in zip(X.columns, model.coef_):
    if coef >= 0:
        formula += f" + {coef:.4f} * {feature}"
    else:
        formula += f" - {abs(coef):.4f} * {feature}"
print(formula)

print(f"\nPredictions saved to: {output_path}")
print(f"Total predictions made: {len(predictions_df)}")

# Show sample predictions in required format
print(f"\nSample predictions (Flaschen_ID, y_hat):")
for i in range(min(5, len(predictions_df))):
    print(f"{predictions_df.iloc[i]['Flaschen_ID']}, {predictions_df.iloc[i]['y_hat']:.1f}")
