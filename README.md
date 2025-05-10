# Machine Learning Classifier Comparison Project

This project compares the performance of various machine learning classifiers on a given dataset.


## Project Structure

```plaintext
   classifier_comparison_project/
   │
   ├── data/
   │   ├── raw/
   │   │   └── sample_dataset.csv
   │   └── processed/
   │
   ├── notebooks/
   │   └── eda.ipynb
   │
   ├── src/
   │   ├── data_preprocessing.py
   │   ├── train_evaluate.py
   │   ├── visualize.py
   │   ├── utils.py
   │   └── main.py
   │
   ├── results/
   │   ├── metrics/
   │   ├── plots/
   │   └── models/
   │
   ├── config/
   │   └── params.yaml
   │
   ├── requirements.txt
   ├── .gitignore
   └── README.md
```
## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/forhadreza43/Optimal-Machine-Learning-Classifiers.git
   cd Optimal-Machine-Learning-Classifiers
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   
3. Run the full pipeline:
   ```bash
   python src/main.py
   
## Results
   The pipeline will generate:
   
   Processed data in data/processed/
   
   Evaluation metrics in results/metrics/
   
   Visualizations in results/plots/
   
   Trained models in results/models/

## Contributing
   Fork the project

##Contact
   FORHAD REZA - forhad.bimt@gmail.com


