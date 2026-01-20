import gradio as gr
import pandas as pd
import pickle
import numpy as np

from feature_engineering import InvalidZeroHandler, DiabetesFeatureEngineer, IQRCapper, SafeLogTransformer

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
    gr.Slider(label="Number of Pregnancies", minimum=0, maximum=20, step=1, value=0),
    gr.Number(label="Plasma Glucose Level (mg/dL)", value=120),
    gr.Number(label="Diastolic Blood Pressure (mm Hg)", value=70),
    gr.Number(label="Triceps Skinfold Thickness (mm)", value=20),
    gr.Number(label="2-Hour Serum Insulin (µU/mL)", value=80),
    gr.Number(label="Body Mass Index (BMI, kg/m²)", value=25.0),
    gr.Number(label="Diabetes Pedigree Function (genetic risk)", value=0.5),
    gr.Number(label="Age (years)", value=33)
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
