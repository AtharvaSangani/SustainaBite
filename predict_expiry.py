import joblib
import pandas as pd

def predict_expiry(item_type, storage, opened, initial_quality=0.95, temperature_variation=1.0):
    # Load model and features
    model = joblib.load('expiry_model.joblib')
    feature_names = joblib.load('expiry_model_features.joblib')
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'item_type': item_type,
        'storage': storage,
        'initial_quality': initial_quality,
        'opened': int(opened),
        'temperature_variation': temperature_variation
    }])
    
    # Predict
    days = model.predict(input_data)[0]
    return max(1, round(days))  # Ensure at least 1 day

if __name__ == '__main__':
    # Example predictions
    examples = [
        ('dairy', 'fridge', False),
        ('meat', 'fridge', True),
        ('vegetable', 'pantry', False),
        ('grain', 'pantry', False)
    ]
    
    for item, storage, opened in examples:
        days = predict_expiry(item, storage, opened)
        print(f"{item} in {storage}, opened={opened}: {days} days until expiry")