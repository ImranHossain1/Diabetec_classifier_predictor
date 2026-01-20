import pandas as pd
from sklearn.model_selection import train_test_split
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
from feature_engineering import InvalidZeroHandler, DiabetesFeatureEngineer, IQRCapper, SafeLogTransformer

df = pd. read_csv("diabetes.csv")
print(df.head())

# Columns where zero is invalid
invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Columns to cap outliers
outlier_cols = ['SkinThickness','Insulin','DiabetesPedigreeFunction']

# Numeric & categorical columns
numeric_features = ['SkinThickness', 'DiabetesPedigreeFunction', 'Insulin_Glucose_Ratio']

categorical_features = ['BMI_Category', 'Age_Group']

# Numeric preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing pipeline
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
column_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Full preprocessing pipeline
preprocessor = Pipeline([
    ('zero_handler', InvalidZeroHandler(cols=invalid_zero_cols)),
    ('feature_engineer', DiabetesFeatureEngineer()),
    ('outlier_capper', IQRCapper(cols=outlier_cols)),
    ('log_transform', SafeLogTransformer(cols=['Insulin_Glucose_Ratio'])),
    ('preprocessor', column_preprocessor)
])

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42, stratify=y)

# Final model
logreg_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf= 1, min_samples_split= 10, random_state=42, class_weight='balanced')
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4,colsample_bytree=1, subsample= 1, learning_rate=0.05, random_state=42)

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators= [
        ('logReg', logreg_model),
        ('rf',rf_model),
        ('xgb', xgb_model)
    ],
    final_estimator=logreg_model
)

# Complete pipeline
model = Pipeline(
      [
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

# Save the trained model
filename= 'stacking_classifier.pkl'
with open(filename, "wb") as file:
  pickle.dump(model,file)