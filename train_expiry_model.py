import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_expiry_model():
    # Load generated data
    df = pd.read_csv('expiry_data.csv')
    
    # Prepare features and target
    X = df[['item_type', 'storage', 'initial_quality', 'opened', 'temperature_variation']]
    y = df['days_until_expiry']
    
    # Preprocessing pipeline
    categorical_features = ['item_type', 'storage']
    numeric_features = ['initial_quality', 'opened', 'temperature_variation']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
    
    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"- MAE: {mae:.2f} days")
    print(f"- R2 Score: {r2:.2f}")
    
    # Save model
    joblib.dump(model, 'expiry_model.joblib')
    print("Model saved as expiry_model.joblib")
    
    # Save feature names for later reference
    feature_names = (list(model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features)) +
                    numeric_features)
    
    joblib.dump(feature_names, 'expiry_model_features.joblib')
    print("Feature names saved")

if __name__ == '__main__':
    train_expiry_model()