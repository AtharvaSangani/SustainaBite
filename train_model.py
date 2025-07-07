import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib

# Load data
df = pd.read_csv("recipe_training_data.csv")
X = df[["matched_ingredients", "total_ingredients", "avg_days_to_expiry", "match_ratio"]]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=3)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# cross-validation for better estimate
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average CV accuracy: {scores.mean():.2f}")

# Save model
joblib.dump(model, "recipe_model.joblib")
print("Model saved!")
