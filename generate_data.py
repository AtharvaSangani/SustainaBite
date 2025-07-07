import pandas as pd
import numpy as np

# Generate 300 fake examples
np.random.seed(42)
data = {
    "matched_ingredients": np.random.randint(1, 8, 300),
    "total_ingredients": np.random.randint(3, 12, 300),
    "avg_days_to_expiry": np.random.uniform(1, 14, 300),
}
df = pd.DataFrame(data)

# Calculate match ratio
df["match_ratio"] = df["matched_ingredients"] / df["total_ingredients"]

# Label data: 1=Good Match if at least 50% match AND avg expiry < 5 days
df["label"] = np.where(
    (df["match_ratio"] >= 0.5) & (df["avg_days_to_expiry"] <= 5),
    1,
    0
)

# Randomly flip 10% of labels to simulate real-world imperfection
random_flip = np.random.random(len(df)) < 0.1
df.loc[random_flip, "label"] = 1 - df.loc[random_flip, "label"]

# Save to CSV
df.to_csv("recipe_training_data.csv", index=False)
print("Generated 300 training examples!")