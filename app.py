from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("disease_prediction_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    sample = pd.DataFrame({
        "Pregnancies":[float(request.form["pregnancies"])],
        "Glucose":[float(request.form["glucose"])],
        "BloodPressure":[float(request.form["bp"])],
        "SkinThickness":[float(request.form["skin"])],
        "Insulin":[float(request.form["insulin"])],
        "BMI":[float(request.form["bmi"])],
        "DiabetesPedigreeFunction":[float(request.form["dpf"])],
        "Age":[float(request.form["age"])]
    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1] * 100

    if prediction == 1:
        result = f"⚠️ Diabetes Detected | Risk: {probability:.2f}%"
    else:
        result = f"✅ No Diabetes Detected | Risk: {probability:.2f}%"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)