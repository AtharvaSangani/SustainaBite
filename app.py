from flask import Flask, render_template, request, redirect
from datetime import datetime
import sqlite3
import joblib
import pandas as pd
from recipe_data import RECIPES 

app = Flask(__name__)

# Load ML model
model = joblib.load("recipe_model.joblib")


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
        
        if days_left <= 7:
            # Return as dictionary for clearer access
            expiring.append({
                "id": food[0],
                "name": food[1],
                "expiry_date": food[2],
                "days_left": days_left
            })
    
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
def predict_recipe_quality(recipe, expiring_items):
    """Predict if recipe is good match (1) or poor match (0)"""
    expiring_names = [item['name'].lower() for item in expiring_items]
    expiring_days = [item['days_left'] for item in expiring_items]
    
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
    try:
        with get_db() as conn:
            foods = conn.execute("SELECT * FROM inventory").fetchall()
            print(f"DEBUG: Retrieved {len(foods)} food items")
        
        if not foods:
            print("DEBUG: No foods found in database")
            foods = []

        expiring = get_expiring_soon(foods)
        
        # Score all recipes
        scored_recipes = []
        for recipe in RECIPES:
            score = predict_recipe_quality(recipe, expiring)
            if score == 1:
                expiring_names = [item['name'].lower() for item in expiring]
                matched = sum(ing.lower() in expiring_names for ing in recipe["ingredients"])
                recipe["match_info"] = {
                    "score": score,
                    "matched": matched,
                    "total": len(recipe["ingredients"]),
                    "percentage": int((matched / len(recipe["ingredients"])) * 100),
                    "missing": [ing for ing in recipe["ingredients"] 
                              if ing.lower() not in expiring_names]
                }
                scored_recipes.append(recipe)
        
        scored_recipes.sort(key=lambda x: x["match_info"]["percentage"], reverse=True)
        
        return render_template(
            "index.html",
            foods=foods,
            expiring=expiring,
            recipes=scored_recipes
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return render_template("error.html"), 500










if __name__ == "__main__":
    app.run(debug=True)



