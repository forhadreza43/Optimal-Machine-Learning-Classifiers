import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import os


def load_config():
    with open("../config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_preprocess():
    config = load_config()

    # Load data
    df = pd.read_csv(config['data']['input_path'])

    # Handle missing values
    df = df.dropna()

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode target if categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Save processed data
    os.makedirs(os.path.dirname(config['data']['output_path']), exist_ok=True)
    pd.concat([X, pd.Series(y, name='target')], axis=1).to_csv(config['data']['output_path'], index=False)

    return X_train, X_test, y_train, y_test