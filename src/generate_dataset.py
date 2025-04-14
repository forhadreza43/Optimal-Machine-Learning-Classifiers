import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def generate_synthetic_dataset(n_samples=1000, random_state=42):
    """Generate a synthetic dataset for customer churn prediction."""
    np.random.seed(random_state)

    # Generate numerical features
    age = np.random.normal(40, 10, n_samples).astype(int).clip(0, 100)  # Age between 0 and 100
    income = np.random.normal(50000, 30000, n_samples).astype(int).clip(0, 200000)  # Income between 0 and 200,000
    usage_frequency = np.random.normal(50, 15, n_samples).astype(int).clip(0, 100)  # Usage 0–100
    support_calls = np.random.poisson(lam=5, size=n_samples).clip(0, 50)  # Poisson distribution for calls

    # Generate categorical feature (education level)
    education_levels = ["High School", "Bachelor", "Master", "PhD"]
    education_level = np.random.choice(education_levels, size=n_samples)

    # Generate target variable (churn) with some correlation to features
    # Churn is more likely if age is low, income is low, usage is low, and support calls are high
    churn_probability = (
        0.2 * (1 - (age / 100)) +  # Younger people more likely to churn
        0.3 * (1 - (income / 200000)) +  # Lower income more likely to churn
        0.3 * (1 - (usage_frequency / 100)) +  # Lower usage more likely to churn
        0.2 * (support_calls / 50)  # More calls increase churn likelihood
    )
    churn = np.random.binomial(1, churn_probability, size=n_samples)
    churn = np.where(churn == 1, "Yes", "No")

    # Create DataFrame
    data = pd.DataFrame({
        "age": age,
        "income": income,
        "education_level": education_level,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "churn": churn
    })

    # Shuffle the dataset
    data = shuffle(data, random_state=random_state)

    return data

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_synthetic_dataset(n_samples=1000)

    # Save to CSV in the data folder
    output_path = "../data/raw/sample_dataset.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print("\nDataset Preview:")
    print(dataset.head())