from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pandas as pd
import yaml
import os
import joblib


def load_config():
    with open("D:\GUB\Semester-7\AI Lab\Project\Identify-Best-Classifier\config\params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize_models():
    config = load_config()
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=config['models']['random_state']),
        "Random Forest": RandomForestClassifier(random_state=config['models']['random_state']),
        "Logistic Regression": LogisticRegression(random_state=config['models']['random_state']),
        "SVM": SVC(random_state=config['models']['random_state'], probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(random_state=config['models']['random_state']),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config['models']['random_state'])
    }
    return models


def train_models(X_train, y_train):
    models = initialize_models()
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def evaluate_models(models, X_test, y_test):
    config = load_config()
    metrics = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1": f1_score(y_test, y_pred, average='weighted'),
            "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        })

    metrics_df = pd.DataFrame(metrics)

    # Save metrics
    os.makedirs(os.path.dirname(config['results']['metrics_path']), exist_ok=True)
    metrics_df.to_csv(config['results']['metrics_path'], index=False)

    # Find and save best model
    best_model_name = metrics_df.sort_values('Accuracy', ascending=False).iloc[0]['Model']
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join(config['results']['models_dir'], 'best_model.pkl'))

    return metrics_df