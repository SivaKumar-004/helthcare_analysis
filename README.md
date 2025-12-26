# Stroke Prediction Analysis

This project performs an exploratory data analysis (EDA) and builds machine learning models to predict the likelihood of a patient having a stroke. Ideally suited for healthcare applications, the analysis prioritizes **Recall** to minimize false negatives (missed stroke cases).

## 📂 Dataset
The dataset utilized is the **Healthcare Dataset Stroke Data**. It contains various patient attributes such as gender, age, diseases, and smoking status.

- **Target Variable**: `stroke` (1: Stroke, 0: No Stroke)
- **Features**: `age`, `avg_glucose_level`, `bmi`, `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`, etc.

## 🛠️ Methodology

### 1. Data Preprocessing
- **Missing Values**: Imputed missing `bmi` values using the median.
- **Encoding**: Applied One-Hot Encoding to categorical variables (e.g., `work_type`, `smoking_status`).
- **Scaling**: Standardized numerical features (`age`, `avg_glucose_level`, `bmi`) using `StandardScaler`.
- **Imbalance Handling**: Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training dataset, as stroke cases are significantly rarer than non-stroke cases.

### 2. Models Trained
We evaluated the following classifiers:
- **Logistic Regression** (Selected as Best Model)
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

## 📊 Results
The models were evaluated based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| **Logistic Regression** | **75.1%** | **80.0%** | **0.24** | **0.85** |
| Random Forest | 92.4% | 8.0% | 0.09 | 0.75 |
| Gradient Boosting | 88.6% | 42.0% | 0.26 | 0.80 |

**Why Logistic Regression?**
In medical diagnostics, **Recall** is critical. A model that misses actual stroke cases (low recall) is dangerous, even if it has high overall accuracy. Logistic Regression identified **80%** of the stroke cases, significantly outperforming Random Forest (which only found 8%).

## 📈 Visualizations
The script generates the following plots in the `results/` directory:
- **Model Comparison**: Bar chart comparing performance metrics across models.
- **ROC Curve**: Evaluating the trade-off between sensitivity and specificity.
- **Confusion Matrix**: Visualizing true positives, false positives, etc.

## 🚀 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SivaKumar-004/helthcare_analysis.git
    cd helthcare_analysis
    ```

2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
    ```

3.  **Run the Analysis**:
    ```bash
    python stroke_analysis.py
    ```

4.  **View Results**:
    Check the console output for metrics and the generated `results/` folder for plots.
