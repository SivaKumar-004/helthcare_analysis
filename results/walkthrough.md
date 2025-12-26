# Stroke Prediction Model Analysis

## 1. Project Overview
The goal was to create a prediction model for the `healthcare-dataset-stroke-data.csv` dataset. The target variable is `stroke` (1 for stroke, 0 for no stroke).

## 2. Data Processing
- **Imputation**: Missing `bmi` values were imputed using the median.
- **Encoding**: Categorical variables (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) were One-Hot Encoded.
- **Balancing**: **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training data to handle the severe class imbalance (strokes are rare).
- **Scaling**: Numerical features (`age`, `avg_glucose_level`, `bmi`) were standardized.

## 3. Model Evaluation
We trained and evaluated three models:
1.  **Logistic Regression**
2.  **Random Forest Classifier**
3.  **Gradient Boosting Classifier**

### Results Summary

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| **Logistic Regression** | 75.1% | **80.0%** | 0.24 | **0.85** |
| Random Forest | 92.4% | 8.0% | 0.09 | 0.75 |
| Gradient Boosting | 88.6% | 42.0% | 0.26 | 0.80 |

> [!NOTE]
> **Recall** is the most critical metric for this medical application. It measures the ability to detect actual stroke cases. A high accuracy with low recall (like the Random Forest model) is dangerous as it misses most stroke patients.

## 4. Best Model: Logistic Regression
**Logistic Regression** is selected as the main prediction model.
- **Reasoning**: It achieved the highest **Recall (0.80)**, meaning it correctly identified 80% of the stroke cases in the test set. It also had the highest **ROC-AUC (0.85)**, indicating good separability between classes.
- **Trade-off**: The precision is lower (many false positives), but in screenings, false positives are acceptable (follow-up tests can rule them out), whereas false negatives (missed strokes) are life-threatening.

## 5. Visualizations

### Model Comparison
![Model Comparison](/C:/Users/USER/.gemini/antigravity/brain/4379032b-1941-4664-a8b1-e0407c81639e/model_comparison.png)

### ROC Curve (Best Model)
The ROC curve shows the trade-off between True Positive Rate and False Positive Rate.
![ROC Curve](/C:/Users/USER/.gemini/antigravity/brain/4379032b-1941-4664-a8b1-e0407c81639e/roc_curve.png)

### Confusion Matrix (Best Model)
![Confusion Matrix](/C:/Users/USER/.gemini/antigravity/brain/4379032b-1941-4664-a8b1-e0407c81639e/confusion_matrix.png)
