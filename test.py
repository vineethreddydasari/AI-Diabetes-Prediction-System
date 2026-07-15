import joblib

model = joblib.load("disease_prediction_model.pkl")

print("Model Loaded Successfully")
print(type(model))