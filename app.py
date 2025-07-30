from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify  # if needed
from flask import redirect, session
import enum
import pandas as pd
import os 
import joblib
import numpy as np

# Load the saved diest plan model and column order
model = joblib.load("ai/diet_plan_model.pkl")
model_columns = joblib.load("ai/model_columns.pkl")
# Phase 4: Load food suggestion model
food_model = joblib.load("ai/food_suggestion_model.pkl")
food_model_columns = joblib.load("ai/food_model_columns.pkl")



app = Flask(__name__)

app.secret_key = "my-secret-key"  # Needed for session to work


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./gym.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  
db = SQLAlchemy(app)
  

class GOAL(enum.Enum):
    Bulking = "Bulking"
    Cutting = "Cutting"
    Maintenance = "Maintenance"

class ACTIVITY(enum.Enum):
    Light = "🛋️ Lightly Active"
    Moderate = "🚶 Moderately Active"
    Active = "🏃 Active"
    VeryActive = "🏋️ Very Active"

class Gender(enum.Enum):
    Male = "Male"
    Female = "Female"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weight = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer,  nullable=False)
    age = db.Column(db.Integer,  nullable=False)
    gender = db.Column(db.Enum(Gender), nullable=False)
    activity_level = db.Column(db.Enum(ACTIVITY),  nullable=False) 
    goal = db.Column(db.Enum(GOAL),  nullable=False)  


with app.app_context():
    db.create_all()  

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calorie_calculator", methods=["GET", "POST"])
def calorie_calculator():
    bmr = None
    tdee = None
    goal_calorie = None
    predicted_plan = None

    if request.method == "POST":
        weight = int(request.form.get("weight"))
        height = int(request.form.get("height"))
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        activity_level = request.form.get("activity_level")
        goal = request.form.get("goal")

        # 🧮 BMR Calculation
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # 🔁 TDEE Calculation
        if activity_level == "Light":
            factor = 1.375
        elif activity_level == "Moderate":
            factor = 1.55
        elif activity_level == "Active":
            factor = 1.725
        elif activity_level == "VeryActive":
            factor = 1.9
        else:
            factor = 1.2

        tdee = bmr * factor

        # 🔁 Calorie Goal
        if goal == "Bulking":
            goal_calorie = tdee + 500
        elif goal == "Cutting":
            goal_calorie = tdee - 500
        else:
            goal_calorie = tdee

        # 🧠 Predict diet plan using ML model
        input_dict = {
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'activity_level': activity_level,
            'goal': goal
        }

        input_df = pd.DataFrame([input_dict]) #✅ Converts the raw input dictionary (from the form/session) into a DataFrame.
        input_encoded = pd.get_dummies(input_df)
        # ✅ One-hot encodes the categorical values (gender, activity_level, goal).
        # 🧠 Why? Because the ML model was trained on encoded features, not raw text
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        predicted_plan = model.predict(input_encoded)[0]

        #SO INTEGRATION FOR THE ML MODEL HAS 4 PARTS:
        # 1. user inputed dict conv to DataFrame
        # 2. one-hot encoding of categorical values 
        # 3. reindexing to match the model's expected columns
        # 4. prediction using the model

        session['user_input'] = input_dict #the entire user inpted form  as well as the next 4 system genarted values
        session['tdee'] = tdee
        session['bmr'] = bmr
        session['goal_calorie'] = goal_calorie
        session['predicted_plan'] = predicted_plan

        return redirect("/food_suggestion") # ✅ Only return this after POST
    
    return render_template("calorie_calculator.html")  # ✅ For GET request, show form
    
# User submits calorie form ➝ Predicts diet ➝ Stores in session ➝ Redirects to food_suggestion ➝ Suggests foods using 2nd model
# Exactly, bro ✅ — in Flask, to pass the same input (or any data) from one route to another, we commonly use session.
# 🔁 Why session?
# ✅ Stores data temporarily for the current user
# ✅ Data stays even after a redirect()



@app.route("/food_suggestion")
def food_suggestion():
    user_input = session.get("user_input")
    tdee = session.get("tdee")
    bmr = session.get("bmr")
    goal_calorie = session.get("goal_calorie")
    predicted_plan = session.get("predicted_plan")

    if not user_input:
        return redirect("/calorie_calculator")

    # Predict food suggestion
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=food_model_columns, fill_value=0)
    food_suggestion = food_model.predict(input_encoded)[0] #The [0] is used because .predict() returns a list/array, even for one input.

    return render_template("food_suggestion.html",
                           tdee=tdee,
                           bmr=bmr,
                           goal_calorie=goal_calorie,
                           predicted_plan=predicted_plan,
                           food_suggestion=food_suggestion,
                           user_input=user_input)



if __name__ == "__main__":
    app.run(debug=True)
