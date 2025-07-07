from flask import Flask, render_template, request, redirect
from datetime import datetime
import sqlite3
import joblib
import pandas as pd

app = Flask(__name__)

# Load ML model
model = joblib.load("recipe_model.joblib")


RECIPES = [
    {
        "name": "Vegetable Stir Fry",
        "ingredients": ["carrot", "broccoli", "bell pepper", "soy sauce"],
        "description": "Quick and healthy stir-fry!"
    },
    {
        "name": "Banana Smoothie",
        "ingredients": ["banana", "milk", "yogurt", "honey"],
        "description": "Creamy and delicious."
    },
    {
        "name": "Tomato Pasta",
        "ingredients": ["tomato", "pasta", "garlic", "olive oil"],
        "description": "Classic Italian dish."
    },
    {
        "name": "Scrambled Eggs",
        "ingredients": ["egg", "butter", "salt", "pepper"],
        "description": "Simple breakfast favorite."
    },
    {
        "name": "Fruit Salad",
        "ingredients": ["apple", "banana", "orange", "yogurt"],
        "description": "Refreshing and light."
    }
]

app = Flask(__name__)

# Create database connection
def get_db():
    conn = sqlite3.connect('food.db')
    return conn

# Initialize database
with get_db() as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        food TEXT,
        expiry DATE
    )
    """)

#Expiring soon
def get_expiring_soon(foods):
    today = datetime.now().date()
    expiring = []
    
    for food in foods:
        expiry_date = datetime.strptime(food[2], "%Y-%m-%d").date()
        days_left = (expiry_date - today).days
        
        if days_left <= 7:  # Show items expiring in a week
            expiring.append((food[0], food[1], food[2], days_left))  # Include days_left
    
    return expiring


@app.route("/add", methods=["POST"])
def add_food():
    food = request.form["food"]
    expiry = request.form["expiry"]
    
    with get_db() as conn:
        conn.execute("INSERT INTO inventory (food, expiry) VALUES (?, ?)", 
                    (food, expiry))
    
    return redirect("/")

@app.route("/delete/<int:food_id>")
def delete_food(food_id):
    with get_db() as conn:
        conn.execute("DELETE FROM inventory WHERE id = ?", (food_id,))
    return redirect("/")

#using model
def predict_recipe_quality(recipe, expiring_names, expiring_days):
    """Predict if recipe is good match (1) or poor match (0)"""
    matched = sum(ing.lower() in expiring_names for ing in recipe["ingredients"])
    total = len(recipe["ingredients"])
    
    if matched == 0:
        return 0  # No match
    
    # Calculate features
    match_ratio = matched / total
    avg_days = sum(expiring_days) / matched
    
    features = pd.DataFrame([[
        matched,
        total,
        avg_days,
        match_ratio
    ]], columns=["matched_ingredients", "total_ingredients", 
                "avg_days_to_expiry", "match_ratio"])
    
    return model.predict(features)[0]


@app.route("/")
def home():
    with get_db() as conn:
        foods = conn.execute("SELECT * FROM inventory").fetchall()
    
    expiring = get_expiring_soon(foods)
    expiring_names = [food[1].lower() for food in expiring]
    expiring_days = [food[3] for food in expiring]
    
    # Score all recipes
    scored_recipes = []
    for recipe in RECIPES:
        score = predict_recipe_quality(recipe, expiring_names, expiring_days)
        if score == 1:
            recipe["match_reason"] = (
                f"Uses {sum(ing.lower() in expiring_names for ing in recipe['ingredients'])} "
                f"of your expiring ingredients"
            )
            scored_recipes.append(recipe)
    
    return render_template("index.html",
                         foods=foods,
                         expiring=expiring,
                         recipes=scored_recipes)










if __name__ == "__main__":
    app.run(debug=True)



