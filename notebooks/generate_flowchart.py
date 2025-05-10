# notebooks/generate_flowchart.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Project-specific boxes
boxes = [
    ("Raw Data\n\n(data/raw/)", 2, 6),
    ("Preprocessing\n\n(src/preprocessing.py)", 4, 6),
    ("EDA\n\n(notebooks/EDA.ipynb)", 6, 6),
    ("Model Training\n\n(src/train.py)", 8, 6),
    ("Evaluation\n\n(results/metrics/)", 10, 6),
    ("Best Model\n\n(results/models/)", 8, 4),
    ("Visualization\n\n(results/plots/)", 6, 4),
    ("Config\n\n(config.yaml)", 4, 4)
]

for text, x, y in boxes:
    ax.add_patch(Rectangle((x-1.5, y-0.7), 3, 1.4, fill=True, color='#4b8bbe', alpha=0.7))
    plt.text(x, y, text, ha='center', va='center', color='black', fontweight='light')

# Arrows showing workflow
arrows = [
    (2, 6, 4, 6), (4, 6, 6, 6), (6, 6, 8, 6), (8, 6, 10, 6),
    (8, 6, 8, 4),  # To Best Model
    (8, 4, 6, 4), (6, 4, 4, 4), (4, 4, 4, 6)  # Feedback loop
]

for x1, y1, x2, y2 in arrows:
    ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, fc='#333333', ec='#333333', lw=1.1)

plt.title('Classifier Comparison Project Workflow', pad=20, fontsize=14)
plt.savefig('../figures/churn_prediction_flowchart.pdf', bbox_inches='tight', dpi=300)