from data_preprocessing import load_and_preprocess
from train_evaluate import train_models, evaluate_models
from visualize import plot_metrics, plot_confusion_matrix, plot_roc_curve
import pandas as pd




def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    metrics_df = evaluate_models(models, X_test, y_test)
    print("\nModel Performance Metrics:")
    print(metrics_df.to_markdown())

    # Visualize results
    plot_metrics(metrics_df)

    # Plot detailed metrics for top 3 models
    top_models = metrics_df.sort_values('Accuracy', ascending=False).head(3)['Model'].values
    for model_name in top_models:
        plot_confusion_matrix(models[model_name], X_test, y_test, model_name)
        plot_roc_curve(models[model_name], X_test, y_test, model_name)

    print("\nProject execution completed. Results saved in the results/ directory.")


if __name__ == "__main__":
    main()