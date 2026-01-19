import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 🔥 IMPORTANT: import custom transformers BEFORE loading pickle
from feature_engineering import InvalidZeroHandler, DiabetesFeatureEngineer

# Load trained pipeline
with open("stacking_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
):
    input_df = pd.DataFrame([{
        "Pregnancies": int(Pregnancies),
        "Glucose": float(Glucose),
        "BloodPressure": float(BloodPressure),
        "SkinThickness": float(SkinThickness),
        "Insulin": float(Insulin),
        "BMI": float(BMI),
        "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
        "Age": int(Age)
    }])

    prediction = model.predict(input_df)[0]
    return "Diabetic" if prediction == 1 else "Non-Diabetic"

# Gradio inputs
inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="BloodPressure", value=70),
    gr.Number(label="SkinThickness", value=20),
    gr.Number(label="Insulin", value=79),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="DiabetesPedigreeFunction", value=0.5),
    gr.Number(label="Age", value=33)
]

# Interface
app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction",
    description="Predict whether a person is diabetic based on health parameters."
)

# Launch app
app.launch(share=True)
