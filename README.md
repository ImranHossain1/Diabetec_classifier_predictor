# Diabetes Classifier Predictor

This repository contains a **Diabetes Prediction Classifier** built using a **Stacking Ensemble Model**. The project demonstrates a full machine learning workflow from data loading and preprocessing to model deployment, with careful evaluation for robustness.

---

## Links

- **Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1Ncpqgk0ZLDhkC_Kn_5C0E7EnT6A-osL4?usp=sharing)
- **GitHub Repository:** [View on GitHub](https://github.com/ImranHossain1/Diabetec_classifier_predictor)
- **Hugging Face App:** [Try it on Hugging Face](https://huggingface.co/spaces/ImranHossain1/Diabetes_classifier_predictor)

## Step-by-Step Workflow

1. **Dataset Loading**
   - The dataset was loaded from a public repository and inspected to understand the features and target variable.

2. **Exploratory Data Analysis (EDA)**
   - Visualizations such as histograms, correlation plots, and boxplots were used to understand feature distributions, relationships, and identify potential outliers.

3. **Feature Engineering**
   - New features were created to enhance model performance, including BMI and age categories, glucose and blood pressure flags, insulin-to-glucose ratio, and pregnancy risk indicator.
   - Invalid zeros were handled, outliers capped using the IQR method, and skewed features were log-transformed to stabilize distributions.

4. **Data Preparation**
   - Features (`X`) and target (`y`) were separated.
   - The dataset was split into training and testing sets using stratified sampling to preserve class distributions.
   - A preprocessing pipeline was built to handle numeric scaling, categorical encoding, and integrate all feature engineering steps.

5. **Model Selection**
   - Multiple classifiers were considered, including Logistic Regression, Random Forest, XGBoost, KNN, SVM, Voting Classifier, and Stacking Classifier.
   - Each model was trained using the preprocessing pipeline and evaluated on the test set using F1-score, accuracy, precision, and recall.

6. **Cross-Validation**
   - Stratified 5-fold cross-validation was applied to assess robustness and reduce variance in performance metrics.
   - The Stacking Classifier consistently achieved the **highest mean F1-score** while maintaining a good balance between precision and recall.

7. **Hyperparameter Tuning**
   - Grid Search and Randomized Search CV were applied to optimize the hyperparameters of the base models (Logistic Regression, Random Forest, XGBoost) and the meta-learner in the stacking ensemble.
   - The final selected hyperparameters improved predictive performance on unseen data.

8. **Final Model Training and Evaluation**
   - The optimized Stacking Classifier was trained on the full training set.
   - Evaluation on the test set yielded:
     - F1-Score: 0.683
     - Accuracy: 0.747
     - Precision: 0.609
     - Recall: 0.778
   - The model demonstrated robust and balanced predictive performance.

9. **Model Serialization**
   - The final trained model was saved using `pickle` for easy loading and deployment.

10. **Deployment**
    - A user-friendly interface was created using **Gradio**.
    - The model was deployed to **Hugging Face Spaces**, enabling interactive predictions for new user input.

---

## Key Decision: Why Stacking Classifier

The Stacking Classifier was chosen because it effectively combines the strengths of Logistic Regression, Random Forest, and XGBoost. Cross-validation showed it achieved the **highest F1-score**, balancing precision and recall better than any individual model. This ensemble approach captures complementary patterns in the data, leading to more robust and reliable predictions compared to single models or simpler ensembles like Voting Classifier.

---

## Results Summary

| Metric    | Value |
| --------- | ----- |
| F1 Score  | 0.683 |
| Accuracy  | 0.747 |
| Precision | 0.609 |
| Recall    | 0.778 |

---

## Conclusion

This project demonstrates a complete ML workflow: from data exploration and feature engineering to model evaluation, hyperparameter tuning, and deployment. The Stacking Classifier provides a **robust, balanced, and high-performing solution** for predicting diabetes, suitable for practical use in interactive applications.
