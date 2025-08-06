from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime, timedelta
import sqlite3
import joblib
import pandas as pd
import numpy as np
from recipe_data import RECIPES

app = Flask(__name__)

# Load models
expiry_model = joblib.load('expiry_model.joblib')

# Database setup
def get_db():
    conn = sqlite3.connect('food.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database with proper schema
def init_db():
    with get_db() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            item_type TEXT NOT NULL,
            best_before DATE NOT NULL,
            storage TEXT NOT NULL,
            opened INTEGER DEFAULT 0,
            predicted_expiry DATE,
            confidence FLOAT,
            days_remaining INTEGER
        )
        ''')
        conn.commit()

# Call this function at startup
init_db()

# Prediction function with confidence
def predict_expiry_with_confidence(item_data):
    input_df = pd.DataFrame([{
        'item_type': item_data['item_type'],
        'storage': item_data['storage'],
        'initial_quality': 0.95,
        'opened': item_data['opened'],
        'temperature_variation': 1.0
    }])
    
    model = expiry_model.named_steps['regressor']
    preprocessor = expiry_model.named_steps['preprocessor']
    
    X = preprocessor.transform(input_df)
    days_pred = model.predict(X)[0]
    days_pred = max(1, round(days_pred))
    
    tree_preds = [tree.predict(X)[0] for tree in model.estimators_]
    confidence = 1 - (np.std(tree_preds) / days_pred if days_pred > 0 else 1.0)
    
    expiry_date = (datetime.strptime(item_data['best_before'], '%Y-%m-%d') + 
                  timedelta(days=days_pred)).strftime('%Y-%m-%d')
    
    return expiry_date, min(1.0, max(0.5, confidence))

# Routes
@app.route('/')
def home():
    with get_db() as conn:
        try:
            items = conn.execute('''
                SELECT *, 
                       julianday(predicted_expiry) - julianday('now') as days_remaining
                FROM inventory
                ORDER BY predicted_expiry
            ''').fetchall()
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            items = []
    
    today = datetime.now().date()
    return render_template('index.html', items=items, today=today)

@app.route('/add', methods=['POST'])
def add_item():
    item_data = {
        'item_name': request.form['item_name'],
        'item_type': request.form['item_type'],
        'best_before': request.form['best_before'],
        'storage': request.form['storage'],
        'opened': 1 if 'opened' in request.form else 0
    }
    
    expiry_date, confidence = predict_expiry_with_confidence(item_data)
    days_remaining = (datetime.strptime(expiry_date, '%Y-%m-%d').date() - datetime.now().date()).days
    
    with get_db() as conn:
        conn.execute('''
            INSERT INTO inventory 
            (item_name, item_type, best_before, storage, opened, 
             predicted_expiry, confidence, days_remaining)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item_data['item_name'],
            item_data['item_type'],
            item_data['best_before'],
            item_data['storage'],
            item_data['opened'],
            expiry_date,
            confidence,
            days_remaining
        ))
        conn.commit()
    
    return redirect(url_for('home'))

@app.route('/delete/<int:item_id>')
def delete_item(item_id):
    with get_db() as conn:
        conn.execute('DELETE FROM inventory WHERE id = ?', (item_id,))
        conn.commit()
    return redirect(url_for('home'))

@app.route('/recipes')
def recipes():
    with get_db() as conn:
        items = conn.execute('SELECT item_name FROM inventory').fetchall()
        item_names = [item['item_name'].lower() for item in items]
    
    matched_recipes = []
    for recipe in RECIPES:
        matched = sum(ing.lower() in item_names for ing in recipe["ingredients"])
        if matched >= 2:
            recipe['match_score'] = matched
            matched_recipes.append(recipe)
    
    matched_recipes.sort(key=lambda x: x['match_score'], reverse=True)
    return render_template('recipes.html', recipes=matched_recipes)

if __name__ == '__main__':
    app.run(debug=True)