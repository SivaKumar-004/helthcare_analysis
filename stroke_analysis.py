import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r'd:\Projects main\data science\source\healthcare-dataset-stroke-data.csv'
ARTIFACTS_DIR = r'C:\Users\USER\.gemini\antigravity\brain\4379032b-1941-4664-a8b1-e0407c81639e'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # bmi has 'N/A' which pandas might read as object if not handled, but read_csv handles 'N/A' as NaN by default often, 
    # but the preview showed 'N/A'. Let's coerce to numeric.
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop ID as it's not a feature
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from numerical_cols
    if 'stroke' in numerical_cols:
        numerical_cols.remove('stroke')
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    return df, categorical_cols, numerical_cols

def build_pipeline(model, categorical_cols, numerical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Use ImbPipeline to include SMOTE
    clf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    return clf

def evaluate_model(clf, X_train, y_train, X_test, y_test, model_name):
    print(f"\nTraining {model_name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': clf
    }

def visualize_results(results_df, best_model_result, X_test, y_test):
    print("\nGenerating visualizations...")
    
    # 1. Model Comparison Bar Plot
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Recall', 'F1', 'ROC-AUC']
    results_long = pd.melt(results_df, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    sns.barplot(data=results_long, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'model_comparison.png'))
    plt.close()
    
    # 2. Confusion Matrix for Best Model
    y_pred = best_model_result['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {best_model_result["Model"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # 3. ROC Curve for Best Model
    fpr, tpr, _ = roc_curve(y_test, best_model_result['y_prob'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{best_model_result["Model"]} (AUC = {best_model_result["ROC-AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'roc_curve.png'))
    plt.close()

def main():
    df = load_data(DATA_PATH)
    
    # Basic info
    print(f"Dataset Shape: {df.shape}")
    print(df.info())
    print("\nClass Distribution:")
    print(df['stroke'].value_counts(normalize=True))
    
    df, cat_cols, num_cols = preprocess_data(df)
    
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42))
    ]
    
    results = []
    
    for name, model in models:
        clf = build_pipeline(model, cat_cols, num_cols)
        try:
            res = evaluate_model(clf, X_train, y_train, X_test, y_test, name)
            results.append(res)
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No models trained successfully.")
        return

    results_df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(results_df[['Model', 'Accuracy', 'Recall', 'F1', 'ROC-AUC']])
    
    # Sort by Recall (usually critical for medical diagnosis to minimize false negatives) or F1
    best_model_row = results_df.sort_values(by='Recall', ascending=False).iloc[0]
    print(f"\nBest Model based on Recall: {best_model_row['Model']}")
    
    # Save results to text file
    with open(os.path.join(ARTIFACTS_DIR, 'results.txt'), 'w') as f:
        f.write("Model Performance Summary:\n")
        f.write(results_df[['Model', 'Accuracy', 'Recall', 'F1', 'ROC-AUC']].to_string())
        f.write(f"\n\nBest Model based on Recall: {best_model_row['Model']}")
    
    visualize_results(results_df, best_model_row, X_test, y_test)
    print("Analysis complete. Visualizations saved.")

if __name__ == "__main__":
    main()
