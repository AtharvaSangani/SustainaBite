import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_expiry_data(num_samples=5000):
    np.random.seed(42)
    
    # Base parameters for different item types
    type_params = {
        'dairy': {'base_life': 7, 'pantry_factor': 0.3, 'fridge_factor': 1.2},
        'vegetable': {'base_life': 5, 'pantry_factor': 0.8, 'fridge_factor': 1.5},
        'fruit': {'base_life': 7, 'pantry_factor': 1.0, 'fridge_factor': 1.3},
        'meat': {'base_life': 3, 'pantry_factor': 0.1, 'fridge_factor': 2.0},
        'grain': {'base_life': 30, 'pantry_factor': 1.2, 'fridge_factor': 1.0},
        'canned': {'base_life': 365, 'pantry_factor': 1.0, 'fridge_factor': 1.0},
        'bakery': {'base_life': 4, 'pantry_factor': 0.9, 'fridge_factor': 1.8}  # NEW
    }
    
    data = {
        'item_type': np.random.choice(list(type_params.keys()), num_samples),
        'storage': np.random.choice(['pantry', 'fridge'], num_samples),
        'initial_quality': np.random.uniform(0.7, 1.0, num_samples),
        'opened': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'temperature_variation': np.random.uniform(0.9, 1.1, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate expiry days based on item type and storage
    def calculate_expiry(row):
        params = type_params[row['item_type']]
        base_days = params['base_life']
        
        if row['storage'] == 'pantry':
            storage_factor = params['pantry_factor']
        else:
            storage_factor = params['fridge_factor']
        
        opened_factor = 0.5 if row['opened'] else 1.0
        quality_factor = row['initial_quality']
        temp_factor = row['temperature_variation']
        
        return int(base_days * storage_factor * opened_factor * quality_factor * temp_factor)
    
    df['days_until_expiry'] = df.apply(calculate_expiry, axis=1)
    
    # Add some realistic noise
    df['days_until_expiry'] = df['days_until_expiry'] * np.random.uniform(0.9, 1.1, num_samples)
    df['days_until_expiry'] = df['days_until_expiry'].round().astype(int)
    
    # Save to CSV
    df.to_csv('expiry_data.csv', index=False)
    print(f"Generated {num_samples} samples in expiry_data.csv")

if __name__ == '__main__':
    generate_expiry_data()