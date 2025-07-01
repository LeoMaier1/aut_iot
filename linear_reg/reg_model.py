import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# 1. Daten laden
data_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data.csv')
df = pd.read_csv(data_path)
print("Training data loaded:")
print(f"Shape: {df.shape}")
print(df.head())

# 2. Features und Zielvariable definieren
y = df['final_weight']
X = df.drop(['bottle', 'final_weight'], axis=1)
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", list(X.columns))

# Check for non-numeric values in features
for col in X.columns:
    if X[col].dtype == object:
        print(f"Nicht-numerische Werte in Spalte {col}:")
        print(X[X[col].apply(lambda x: isinstance(x, str))])

# Versuche alle Spalten numerisch zu machen, entferne Zeilen mit Fehlern
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nTraining features shape:", X_train.shape)
print("Test features shape:", X_test.shape)

# 4. Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModell erfolgreich trainiert!")
print(f"Intercept (β₀): {model.intercept_:.4f}")

# 5. Modellkoeffizienten anzeigen
print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# 6. Modell evaluieren
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"\nModel Performance:")
print(f"Training MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# 7. Ergebnistabelle für Report
print("\n=== RESULTS TABLE ===")
print("| Genutzte Spalten | Modell-Typ | MSE-Wert (Training) | MSE-Wert (Test) |")
print("|------------------|------------|---------------------|-----------------|")
print(f"| All features | Linear | {mse_train:.4f} | {mse_test:.4f} |")

# 8. Modellformel (y = mx + b Form)
print(f"\n=== MODEL FORMULA ===")
formula = f"final_weight = {model.intercept_:.4f}"
for feature, coef in zip(X.columns, model.coef_):
    if coef >= 0:
        formula += f" + {coef:.4f} * {feature}"
    else:
        formula += f" - {abs(coef):.4f} * {feature}"
print(formula)

# 9. Vorhersagen für X.csv
pred_path = os.path.join(os.path.dirname(__file__), '..', 'X.csv')
X_pred_df = pd.read_csv(pred_path)
print(f"\nPrediction data loaded. Shape: {X_pred_df.shape}")
X_pred = X_pred_df.drop(['bottle'], axis=1)
X_pred = X_pred.fillna(X_pred.mean())
final_predictions = model.predict(X_pred)
print(f"First 5 predictions: {final_predictions[:5]}")
print(f"Total predictions made: {len(final_predictions)}")

# 10. Vorhersagen speichern
predictions_df = pd.DataFrame({
    'Flaschen_ID': X_pred_df['bottle'],
    'y_hat': final_predictions
})
output_path = os.path.join(os.path.dirname(__file__), 'reg_52315857-52310501.csv')
predictions_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

# Show sample predictions in required format
print("\nSample predictions (Flaschen_ID, y_hat):")
for i in range(min(5, len(predictions_df))):
    print(f"{predictions_df.iloc[i]['Flaschen_ID']}, {predictions_df.iloc[i]['y_hat']:.1f}")
