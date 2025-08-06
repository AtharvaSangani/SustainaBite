import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.pipeline import Pipeline

# Load the trained model and test data
def load_model_and_data():
    model = joblib.load('expiry_model.joblib')
    df = pd.read_csv('expiry_data.csv')
    
    # Prepare features and target
    X = df[['item_type', 'storage', 'initial_quality', 'opened', 'temperature_variation']]
    y = df['days_until_expiry']
    
    # Split data (use the same random_state as during training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Figure 2a: Predicted vs Actual scatter plot
def plot_predicted_vs_actual(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Days Until Expiry', fontsize=12)
    plt.ylabel('Predicted Days Until Expiry', fontsize=12)
    plt.title('Predicted vs Actual Expiry Days', fontsize=14)
    plt.grid(True)
    plt.savefig('Figure2a_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2b: Error distribution histogram
def plot_error_distribution(y_test, y_pred):
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.axvline(x=errors.mean(), color='r', linestyle='--', label=f'Mean Error: {errors.mean():.2f} days')
    plt.title('Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error (Days)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.savefig('Figure2b_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2c: Confusion matrix for expiry ranges
def plot_confusion_matrix(y_test, y_pred):
    # Create bins for days until expiry
    bins = [0, 3, 7, 14, 30, 365]
    labels = ['0-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days']
    
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Number of Predictions'})
    plt.xlabel('Predicted Expiry Range', fontsize=12)
    plt.ylabel('Actual Expiry Range', fontsize=12)
    plt.title('Confusion Matrix of Expiry Ranges', fontsize=14)
    plt.savefig('Figure2c_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2d: Feature importance chart
def plot_feature_importance(model):
    if isinstance(model, Pipeline):
        feature_importances = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    else:
        feature_importances = model.feature_importances_
        feature_names = model.feature_names_in_
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importances', fontsize=14)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.savefig('Figure2d_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load required data
    model, X_test, y_test, y_pred = load_model_and_data()
    
    # Generate all figures
    plot_predicted_vs_actual(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model)
    
    print("All figures generated successfully!")
    print("Saved files:")
    print("- Figure2a_predicted_vs_actual.png")
    print("- Figure2b_error_distribution.png")
    print("- Figure2c_confusion_matrix.png")
    print("- Figure2d_feature_importance.png")

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    main()