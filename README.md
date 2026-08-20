# Wine Cultivar Clustering & Anomaly Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end unsupervised machine learning pipeline for discovering chemical cultivar groupings in wine datasets. This project combines variance-preserving dimensionality reduction via Principal Component Analysis (PCA), robust anomaly detection using Isolation Forests, comparative cluster modeling across multiple algorithms, dynamic MLflow experiment tracking, and publication-ready 2D decision boundary visualization.

---

## Cluster Visualization

<p align="center">
  <img src="assets/Figure_1.jpg" alt="Wine Cultivar Clustering Analysis" width="850"/>
</p>

* **Voronoi Decision Regions:** Segmented via KD-Tree spatial indexing over the 2D PCA projection space.
* **Confidence Boundaries:** $2\sigma$ covariance ellipses capturing dispersion, orientation, and density per cultivar cluster.
* **Anomaly Pruning:** Outlier data points detected and removed via Isolation Forest ($10\%$ contamination threshold) highlighted with red **X** markers.
* **Centroids:** Gold stars mark the converged $K$-Means cluster centers in latent space.

---

## Key Highlights & Features

* **Data Transformation & Standardization:** Skew correction through logarithmic transformation followed by `StandardScaler` standard normal normalization.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) retaining $95\%$ cumulative explained variance to avoid multicollinearity.
* **Multivariate Anomaly Detection:** Outlier filtering with `IsolationForest` to ensure high cluster purity and prevent centroid drift.
* **Multi-Model Benchmark:** Quantitative comparison between **$K$-Means**, **Gaussian Mixture Models (GMM)**, and density-based **HDBSCAN**.
* **MLflow Experiment Tracking:** Automated logging of hyperparameters, algorithmic tags, and validation metrics across all pipeline runs.

---

## Benchmark & Evaluation Results

| Model / Strategy | Outlier Filtering | Silhouette Score | Status |
| :--- | :---: | :---: | :---: |
| **$K$-Means + Isolation Forest** | **Yes (10%)** | **0.359** | 🏆 **Best Performer** |
| **Gaussian Mixture + Isolation Forest** | Yes (10%) | 0.357 | Runner-Up |
| **$K$-Means (Baseline)** | No | 0.325 | Baseline |
| **Gaussian Mixture (Baseline)** | No | 0.321 | Baseline |
| **HDBSCAN** | Built-in | 0.163 | Density-Based |

---

## Project Structure

```text
wine-cultivar-clustering/
├── assets/
│   └── Figure_1.jpg          # Generated PCA decision boundary plot
├── data/
│   └── wine-clustering.csv   # Chemical cultivar dataset
├── mlruns/                   # MLflow experiment tracking metadata & artifacts
├── src/
│   └── wine_clustering.py    # Main training and visualization script
├── requirements.txt          # Environment dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/wine-cultivar-clustering.git
cd wine-cultivar-clustering

python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Contents of `requirements.txt`)*:
```text
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
hdbscan>=0.8.33
mlflow>=2.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### 3. Run Pipeline & Generate Visuals

Execute the main pipeline to run transformations, train models, log experiments, and display the visualization:

```bash
python src/wine_clustering.py
```

### 4. Launch MLflow Tracking Dashboard

To explore logged metrics, parameters, and compare runs side-by-side:

```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your web browser.

---

## Methodology Overview

1. **Feature Transformation & Scaling**:
   $$X_{\text{norm}} = \text{StandardScaler}(\ln(X))$$
2. **Variance-Preserving PCA**:
   Retains the minimum number of principal components such that:
   $$\sum_{i=1}^{k} \lambda_i \ge 0.95 \cdot \sum_{j=1}^{p} \lambda_j$$
3. **Outlier Filtering**:
   Multivariate isolation using tree ensembles:
   $$X_{\text{clean}} = X_{\text{PCA}}[\text{IsolationForest}(c=0.10) == 1]$$
4. **Cluster Evaluation**:
   Clustering cohesion and separation measured via the Silhouette Coefficient:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/wine-cultivar-clustering/issues).

---

## Author

**Alkiviadis Agrogiannhs**  
Data Scientist & Machine Learning Engineer 
[LinkedIn](https://www.linkedin.com/in/alkiviadis-agrogiannhs/)
[Email](mailto:alkiviadisagrogiannhs@gmail.com)
[GitHub](https://github.com/alkiviadisss)

---
