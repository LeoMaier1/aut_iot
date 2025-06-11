import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path to access database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data():
    """Load the data from CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'data.csv')
    try:
        data = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None

def explore_data(data):
    """Explore the dataset"""
    print("\n=== DATA EXPLORATION ===")
    print("Dataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nTarget Variable (final_weight) Statistics:")
    print(data['final_weight'].describe())

def prepare_features(data):
    """Prepare different feature combinations for testing"""
    # Remove bottle ID as it's not a predictive feature
    features_all = data.drop(['bottle', 'final_weight'], axis=1)
    
    # Define different feature combinations to test
    feature_combinations = {
        'vibration_only': ['vibration_index_red', 'vibration_index_blue', 'vibration_index_green'],
        'fill_level_only': ['fill_level_grams_red', 'fill_level_grams_blue', 'fill_level_grams_green'],
        'temperature_only': ['temperature_green', 'temperature_red', 'temperature_blue'],
        'vibration_fill': ['vibration_index_red', 'vibration_index_blue', 'vibration_index_green',
                          'fill_level_grams_red', 'fill_level_grams_blue', 'fill_level_grams_green'],
        'all_features': features_all.columns.tolist()
    }
    
    return feature_combinations

def train_and_evaluate_models(data, feature_combinations):
    """Train and evaluate linear regression models with different feature combinations"""
    results = []
    models = {}
    scalers = {}
    
    target = data['final_weight']
    
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    for name, features in feature_combinations.items():
        print(f"\nTraining model with features: {name}")
        print(f"Features used: {features}")
        
        # Prepare features
        X = data[features].copy()
        
        # Handle missing values by filling with mean
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate MSE
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        # Store results
        results.append({
            'model_name': name,
            'features': features,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'n_features': len(features)
        })
        
        # Store model and scaler for later use
        models[name] = model
        scalers[name] = scaler
        
        print(f"MSE Training: {mse_train:.4f}")
        print(f"MSE Test: {mse_test:.4f}")
        
        # Print model coefficients
        print("Model coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")
    
    return results, models, scalers

def find_best_model(results):
    """Find the best model based on test MSE"""
    best_model = min(results, key=lambda x: x['mse_test'])
    print(f"\n=== BEST MODEL ===")
    print(f"Best model: {best_model['model_name']}")
    print(f"Features: {best_model['features']}")
    print(f"Test MSE: {best_model['mse_test']:.4f}")
    print(f"Training MSE: {best_model['mse_train']:.4f}")
    return best_model

def create_results_table(results):
    """Create a formatted results table"""
    print("\n=== RESULTS TABLE ===")
    print("| Genutzte Spalten | Modell-Typ | MSE-Wert (Training) | MSE-Wert (Test) |")
    print("|------------------|------------|---------------------|-----------------|")
    
    for result in results:
        features_str = ', '.join(result['features'][:2]) + ('...' if len(result['features']) > 2 else '')
        print(f"| {features_str} | Linear | {result['mse_train']:.4f} | {result['mse_test']:.4f} |")

def write_model_formula(best_model, models, scalers, data):
    """Write the mathematical formula for the best model"""
    model_name = best_model['model_name']
    model = models[model_name]
    features = best_model['features']
    
    print(f"\n=== BEST MODEL FORMULA ===")
    print("Linear Regression Formula:")
    
    formula = f"final_weight = {model.intercept_:.4f}"
    for feature, coef in zip(features, model.coef_):
        if coef >= 0:
            formula += f" + {coef:.4f} * {feature}"
        else:
            formula += f" - {abs(coef):.4f} * {feature}"
    
    print(formula)
    return formula

def create_prediction_dataset():
    """Load the actual prediction dataset from X.csv"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'X.csv')
    try:
        prediction_data = pd.read_csv(csv_path)
        print(f"Prediction data loaded successfully. Shape: {prediction_data.shape}")
        print("Prediction data preview:")
        print(prediction_data.head())
        return prediction_data
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        # Fallback to sample data if X.csv not found
        sample_data = {
            'vibration_index_red': [5.0, 3.0, 7.0],
            'fill_level_grams_red': [700.0, 650.0, 750.0],
            'vibration_index_blue': [180.0, 175.0, 185.0],
            'fill_level_grams_blue': [600.0, 550.0, 650.0],
            'vibration_index_green': [-15.0, -20.0, -10.0],
            'fill_level_grams_green': [800.0, 795.0, 805.0],
            'temperature_green': [31.0, 32.0, 30.0],
            'temperature_red': [31.5, 32.5, 30.5],
            'temperature_blue': [32.0, 33.0, 31.0]
        }
        return pd.DataFrame(sample_data)

def make_predictions(best_model, models, scalers, prediction_data):
    """Make predictions using the best model"""
    model_name = best_model['model_name']
    model = models[model_name]
    scaler = scalers[model_name]
    features = best_model['features']
    
    # Prepare prediction data
    X_pred = prediction_data[features].copy()
    X_pred = X_pred.fillna(X_pred.mean())
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_pred_scaled)
    
    return predictions

def save_predictions(predictions, prediction_data):
    """Save predictions to CSV file with proper naming convention"""
    # Create predictions DataFrame using bottle IDs from X.csv
    pred_df = pd.DataFrame({
        'Flaschen_ID': prediction_data['bottle'].values,
        'y_hat': predictions
    })
    
    # Save to CSV with naming convention reg_<Matrikelnummer1-Matrikelnummer2-Matrikelnummer3>.csv
    # For now using generic name - replace with actual student numbers
    output_path = os.path.join(os.path.dirname(__file__), 'reg_student1-student2-student3.csv')
    pred_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"File follows naming convention: reg_<Matrikelnummer1-Matrikelnummer2-Matrikelnummer3>.csv")
    print("\nPrediction Results:")
    print(pred_df.head(10))
    print(f"Total predictions: {len(pred_df)}")
    
    return pred_df

def main():
    """Main function to run the regression analysis"""
    print("=== LINEAR REGRESSION MODEL FOR FINAL WEIGHT PREDICTION ===")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Explore data
    explore_data(data)
    
    # Prepare feature combinations
    feature_combinations = prepare_features(data)
    
    # Train and evaluate models
    results, models, scalers = train_and_evaluate_models(data, feature_combinations)
    
    # Find best model
    best_model = find_best_model(results)
    
    # Create results table
    create_results_table(results)
    
    # Write model formula
    formula = write_model_formula(best_model, models, scalers, data)
    
    # Load actual prediction dataset and make predictions
    prediction_data = create_prediction_dataset()
    predictions = make_predictions(best_model, models, scalers, prediction_data)
    
    # Save predictions with proper naming
    pred_df = save_predictions(predictions, prediction_data)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("The linear regression analysis has been completed successfully!")
    print("Check the results above for model performance and predictions.")
    print(f"\nIMPORTANT: Rename the output file to reg_<your-actual-student-numbers>.csv")
    print("Replace 'student1-student2-student3' with your actual Matrikelnummern")

if __name__ == "__main__":
    main()
