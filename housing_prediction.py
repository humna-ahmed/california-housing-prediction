# Import libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data():
    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['PRICE'] = california.target
    return data

# Preprocess data
def preprocess_data(data):
    # Check for missing values
    if data.isnull().sum().any():
        data = data.dropna()
    
    # Feature selection (all features are relevant in this dataset)
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']
    return X, y

# Train model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_imp = pd.Series(model.feature_importances_, 
                           index=X_test.columns).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.show()

def main():
    print("Loading California housing dataset...")
    data = load_data()
    
    print("\nFirst 5 rows:")
    print(data.head())
    
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()