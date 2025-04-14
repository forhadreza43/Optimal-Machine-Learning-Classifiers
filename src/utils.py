import joblib
import yaml
import os

def load_config():
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def save_model(model, filename):
    config = load_config()
    os.makedirs(config['results']['models_dir'], exist_ok=True)
    joblib.dump(model, os.path.join(config['results']['models_dir'], filename))

def load_model(filename):
    config = load_config()
    return joblib.load(os.path.join(config['results']['models_dir'], filename))