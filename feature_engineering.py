import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class InvalidZeroHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols] = X[self.cols].replace(0, np.nan)
        return X
class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bmi_bins = [0, 18.5, 25, 30, 100]
        self.bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.age_bins = [20, 30, 40, 50, 100]
        self.age_labels = ['Age_20-30', 'Age_30-40', 'Age_40-50', 'Age_50+']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # BMI Category
        X['BMI_Category'] = pd.cut(X['BMI'], bins=self.bmi_bins, labels=self.bmi_labels)

        # Age Group
        X['Age_Group'] = pd.cut(X['Age'], bins=self.age_bins, labels=self.age_labels)

        # Flags
        X['High_Glucose'] = (X['Glucose'] >= 140).astype(int)
        X['High_BP'] = (X['BloodPressure'] >= 80).astype(int)

        # Safe ratio
        X['Insulin_Glucose_Ratio'] = X['Insulin'] / X['Glucose']

        # Pregnancy risk
        X['Pregnancy_Risk'] = (X['Pregnancies'] >= 3).astype(int)

        return X
