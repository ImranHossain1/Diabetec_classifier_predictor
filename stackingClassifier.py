import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import  StackingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import pickle

df = pd. read_csv("diabetes.csv")
print(df.head())

class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.bmi_bins = [0, 18.5, 25, 30, 100]
        self.bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.age_bins = [20, 30, 40, 50, 100]
        self.age_labels = ['Age_20-30', 'Age_30-40', 'Age_40-50', 'Age_50+']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Replace invalid zeros with NaN
        X[self.invalid_zero_cols] = X[self.invalid_zero_cols].replace(0, np.nan)
        
        # BMI Category 
        X['BMI_Category'] = pd.cut(X['BMI'], bins=self.bmi_bins, labels=self.bmi_labels)
        # Add 'Missing' category for NaNs
        X['BMI_Category'] = X['BMI_Category'].cat.add_categories('Missing').fillna('Missing')
        
        # Age Group
        X['Age_Group'] = pd.cut(X['Age'], bins=self.age_bins, labels=self.age_labels)
        X['Age_Group'] = X['Age_Group'].cat.add_categories('Missing').fillna('Missing')
        
        # High Glucose & High BP flags
        X['High_Glucose'] = (X['Glucose'] >= 140).astype(int)
        X['High_BP'] = (X['BloodPressure'] >= 80).astype(int)
        
        # Insulin / Glucose ratio 
        X['Insulin_Glucose_Ratio'] = X['Insulin'] / X['Glucose']
        
        # Pregnancy risk
        X['Pregnancy_Risk'] = (X['Pregnancies'] >= 3).astype(int)
        
        return X

class DropOriginalColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.cols_to_drop, inplace=True)
        return X



# Numeric & categorical columns
numeric_features = ['SkinThickness', 'DiabetesPedigreeFunction', 'Insulin_Glucose_Ratio']

categorical_features = ['BMI_Category', 'Age_Group']

cols_to_drop = ['BMI', 'Age', 'Glucose', 'BloodPressure', 'Insulin', 'Pregnancies']


# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42, stratify=y)

logreg_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf= 1, min_samples_split= 5, random_state=42, class_weight='balanced')
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3,colsample_bytree=0.8, subsample= 1, learning_rate=0.05, random_state=42)


stacking_classifier = StackingClassifier(
    estimators= [
        ('logReg', logreg_model),
        ('rf',rf_model),
        ('xgb', xgb_model)
    ],
    final_estimator=logreg_model
)

model = Pipeline(
      [
          ('feature_engineer', DiabetesFeatureEngineer()),
          ('drop_unused', DropOriginalColumns(cols_to_drop=cols_to_drop)),
          ('preprocessor', preprocessor),
          ('model',stacking_classifier)
      ]
)

#train
model.fit(X_train,y_train)

#predict
y_pred = model.predict(X_test)

  #Evaluate
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print({
      "F1_Score": f1,
      "Accuracy": accuracy,
      "Precision": precision,
      "Recall": recall
  })



filename= 'stacking_classifier.pkl'
with open(filename, "wb") as file:
  pickle.dump(model,file)