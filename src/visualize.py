import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, auc


def load_config():
    with open("D:\GUB\Semester-7\AI Lab\Project\Identify-Best-Classifier\config\params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def plot_metrics(metrics_df):
    config = load_config()
    os.makedirs(config['results']['plots_dir'], exist_ok=True)

    plt.figure(figsize=(12, 8))
    metrics_df = metrics_df.sort_values('Accuracy')
    ax = sns.barplot(x='Accuracy', y='Model', data=metrics_df)
    plt.title('Classifier Accuracy Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(config['results']['plots_dir'], 'accuracy_comparison.png'))
    plt.close()

    # Plot all metrics
    plt.figure(figsize=(12, 8))
    melted_df = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    sns.barplot(x='Value', y='Model', hue='Metric', data=melted_df)
    plt.title('Classifier Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(config['results']['plots_dir'], 'all_metrics_comparison.png'))
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, model_name):
    config = load_config()
    os.makedirs(config['results']['plots_dir'], exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(config['results']['plots_dir'], f'confusion_matrix_{model_name}.png'))
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_name):
    config = load_config()
    if not hasattr(model, "predict_proba"):
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(config['results']['plots_dir'], f'roc_curve_{model_name}.png'))
    plt.close()