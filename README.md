# 📚 Book Rating Prediction — End-to-End Machine Learning Project

> An end-to-end machine learning system that predicts whether a book is likely to receive a **high or low rating**, from data collection and preprocessing to model development, explainability, FastAPI deployment, frontend integration, testing, and production deployment.

## 🚀 Live Demo

### 🌐 Deployed Application

**[Book Rating Prediction — Live Demo](https://book-rating-prediction.onrender.com/)**

The application provides an interactive frontend where users can enter book information and receive a predicted rating category along with the model's estimated probability.

### 🔗 Application Endpoints

| Endpoint   | Description                           |
| ---------- | ------------------------------------- |
| `/`        | Interactive prediction frontend       |
| `/docs`    | Interactive Swagger API documentation |
| `/health`  | Application and model health check    |
| `/predict` | Book rating prediction API            |

**Live URLs:**

* **Web Application:** https://book-rating-prediction.onrender.com/
* **Swagger API:** https://book-rating-prediction.onrender.com/docs
* **Health Check:** https://book-rating-prediction.onrender.com/health

---

## 📌 Project Overview

This is a production-style, end-to-end machine learning project designed to demonstrate the complete ML lifecycle.

The system predicts whether a book is likely to receive a **high rating** or **low rating** using structured book information such as:

* Book title
* Book category
* Price
* Availability status

The project covers:

* Ethical data collection
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Class imbalance handling
* Baseline modeling
* Model development
* Hyperparameter tuning
* Cross-validation
* Model interpretability
* Final model evaluation
* Model persistence
* FastAPI REST API
* Custom frontend UI
* Automated API testing
* Render deployment
* Documentation

---

# 🎯 Problem Statement

The objective is to build a machine learning classification system that predicts whether a book is likely to be **high-rated** or **low-rated** based on information available before making the prediction.

### Target Variable

```text
is_high_rating
```

Target definition:

```text
1 → Rating ≥ 4
0 → Rating < 4
```

The original `rating` feature is deliberately excluded from model inputs because it directly determines the target.

Including it would result in **target leakage**, producing artificially strong model performance.

---

# 📊 Dataset

## Source

Data was collected from:

**Books to Scrape**

https://books.toscrape.com/

Books to Scrape is a publicly available website specifically designed as a web scraping practice environment.

### Dataset Size

```text
1,000 records
```

### Collected Attributes

* Book title
* Category
* Price
* Rating
* Availability status
* Product URL

### ⚠️ Dataset Limitation

The website contains fictional/demo book data.

Therefore:

* Prices are not real commercial prices.
* Ratings are not representative of real customer behavior.
* Business conclusions should not be interpreted as real-world market insights.

This project is intended for **educational, internship, and portfolio purposes**.

---

# 🏗️ Machine Learning Pipeline

```text
                    ┌─────────────────────┐
                    │  Books to Scrape    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Ethical Scraping   │
                    │ Requests + BS4      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Cleaning + Target   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │ Statistics + Plots │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Encoding + Scaling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Class Balancing     │
                    │       SMOTE         │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌──────────────────────────────┐
              │       Model Development     │
              │                              │
              │ Logistic Regression          │
              │ Random Forest                │
              │ Support Vector Machine       │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ F1 + ROC-AUC etc.   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Interpretability    │
                    │ Permutation FI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Persistence   │
                    │ Inference Bundle    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Frontend UI         │
                    │ HTML/CSS/JavaScript │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Render Deployment   │
                    └─────────────────────┘
```

---

# 🔬 Machine Learning Workflow

## 1. Ethical Data Collection

The scraping pipeline uses:

* Python Requests
* BeautifulSoup
* Custom request headers
* Rate limiting
* Error handling
* Logging
* `robots.txt` verification
* CSV storage
* Source metadata documentation

### Generated Files

```text
data/raw/books.csv
data/raw/source_metadata.json
```

---

## 2. Data Preprocessing

The preprocessing pipeline handles:

* Missing values
* Duplicate records
* Invalid values
* Data type corrections
* Column renaming
* Outlier detection
* Target creation

### Generated Files

```text
data/processed/books_cleaned.csv
data/processed/cleaning_metadata.json
```

---

## 3. Exploratory Data Analysis

The EDA stage analyzes:

* Rating distribution
* Price distribution
* Target distribution
* Price by rating
* Correlations
* Summary statistics
* Dataset characteristics
* Business-oriented observations

### Generated Reports

```text
reports/docs/eda_summary.json
```

### Generated Visualizations

```text
reports/figures/price_distribution.png
reports/figures/rating_distribution.png
reports/figures/target_distribution.png
reports/figures/price_by_rating.png
reports/figures/correlation_heatmap.png
```

---

## 4. Feature Engineering

The feature engineering pipeline creates additional predictive features.

### Numerical Features

* Price transformation
* Price band
* Title character count
* Title word count
* Title digit count
* Title punctuation count
* Title uppercase ratio
* Availability quantity

### Categorical Features

Categorical variables are encoded for machine learning models.

### Numerical Processing

Numeric variables are appropriately scaled where required.

### Excluded Features

```text
rating
product_url
```

### Why?

`rating` directly determines the target and would create **data leakage**.

`product_url` is an identifier-like feature and does not provide meaningful generalizable predictive information.

### Generated Artifacts

```text
data/processed/books_features.csv
data/processed/feature_metadata.json
models/preprocessing_pipeline.joblib
```

---

# ⚖️ 5. Class Imbalance Handling

Class distribution was evaluated before model training.

The following approaches were considered:

* SMOTE
* ADASYN
* Random Oversampling

### Selected Strategy

```text
SMOTE
```

SMOTE was selected because it generates synthetic minority-class observations rather than simply duplicating existing observations.

### Generated Artifacts

```text
data/processed/train_test_split.joblib
data/processed/balanced_training_sets.joblib
data/processed/balancing_metadata.json
models/fitted_preprocessing_pipeline.joblib
```

---

# 🧪 6. Baseline Model

A `DummyClassifier` using the most-frequent-class strategy was trained as a baseline.

The purpose of the baseline is to establish a minimum performance benchmark that the actual machine learning models must outperform.

### Generated Artifacts

```text
models/baseline_model.joblib
data/processed/baseline_metrics.json
```

---

# 🤖 7. Model Development

Three classification algorithms were evaluated:

### Logistic Regression

Provides a strong linear classification baseline and interpretable coefficients.

### Random Forest

Captures nonlinear relationships and feature interactions.

### Support Vector Machine

Provides a powerful classification approach, particularly after feature scaling.

### Training Strategy

* GridSearchCV
* RandomizedSearchCV
* 5-fold cross-validation
* F1-based model selection

### Generated Artifacts

```text
models/logistic_regression_model.joblib
models/random_forest_model.joblib
models/svm_model.joblib
models/final_model.joblib
```

### Training Results

```text
data/processed/model_comparison.csv
data/processed/model_training_results.json
```

---

# 🔎 8. Model Interpretability

Permutation Feature Importance was used to understand how individual features influence model performance.

This approach is model-agnostic and can therefore be applied consistently across different model families.

### Generated Artifacts

```text
data/processed/model_interpretation.json
reports/figures/feature_importance.png
```

---

# 📈 9. Final Model Evaluation

The final model is evaluated using multiple classification metrics.

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* Classification Report

### Generated Artifacts

```text
data/processed/final_evaluation_metrics.json
reports/docs/classification_report.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

---

# 💾 10. Model Persistence

The trained model and preprocessing pipeline are packaged together into an inference bundle.

This ensures that the same preprocessing logic used during training can be reproduced during inference.

### Artifacts

```text
models/inference_bundle.joblib
models/inference_bundle.pkl
data/processed/persistence_metadata.json
```

### Preferred Inference Artifact

```text
models/inference_bundle.joblib
```

---

# 🌐 Application

The machine learning model is exposed through a **FastAPI REST API** and integrated with a custom frontend.

### Technology Stack

```text
Python
Scikit-learn
Pandas
NumPy
Imbalanced-learn
BeautifulSoup
Requests
FastAPI
Uvicorn
HTML
CSS
JavaScript
Pytest
Render
```

---

# 🖥️ Frontend Features

The custom frontend provides:

* Book prediction form
* Example input functionality
* Loading state
* Success state
* Error handling
* High-rating probability display
* Probability progress bar
* API integration
* Responsive interface

---

# 🔌 REST API

## `GET /`

Returns the custom frontend interface.

**Live:**

https://book-rating-prediction.onrender.com/

---

## `GET /health`

Checks whether the API and model are operational.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "bundle_path": "models/inference_bundle.joblib"
}
```

**Live:**

https://book-rating-prediction.onrender.com/health

---

## `POST /predict`

Predicts whether a book is likely to receive a high or low rating.

### Example Request

```json
{
  "book_title": "A Light in the Attic",
  "book_category": "Books",
  "price_gbp": 51.77,
  "availability_status": "In stock"
}
```

### Example Response

```json
{
  "prediction": 0,
  "prediction_label": "low_rating",
  "high_rating_probability": 0.42,
  "model_version": "1.0.0"
}
```

---

# 📖 Swagger API Documentation

FastAPI automatically provides interactive API documentation.

**Swagger UI:**

https://book-rating-prediction.onrender.com/docs

The Swagger interface can be used to test the `/predict` endpoint directly from the browser.

---

# 📁 Project Structure

```text
Book-Rating-Prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│
├── reports/
│   ├── docs/
│   └── figures/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── config/
│   │
│   ├── data_collection/
│   │   └── scrape_books.py
│   │
│   ├── eda/
│   │   └── analyze_books.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── interpret_model.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── modeling/
│   │   ├── handle_imbalance.py
│   │   ├── persist_model.py
│   │   ├── train_baseline.py
│   │   └── train_models.py
│   │
│   └── preprocessing/
│       └── clean_books.py
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/
│   └── test_api.py
│
├── .python-version
├── Procfile
├── render.yaml
├── requirements.txt
└── README.md
```

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/PriyanshuRaj0019/Book-Rating-Prediction.git
cd Book-Rating-Prediction
```

## 2. Create Virtual Environment

```bash
py -3.12 -m venv .venv
```

## 3. Activate Virtual Environment

### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

### Windows Git Bash

```bash
source .venv/Scripts/activate
```

## 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Verify Environment

```bash
python -c "import pandas, sklearn, imblearn, fastapi, bs4; print('Environment ready')"
```

Expected output:

```text
Environment ready
```

---

# 🔄 Run the Complete ML Pipeline

Execute the following modules in order.

### 1. Data Collection

```bash
python -m src.data_collection.scrape_books
```

### 2. Data Preprocessing

```bash
python -m src.preprocessing.clean_books
```

### 3. Exploratory Data Analysis

```bash
python -m src.eda.analyze_books
```

### 4. Feature Engineering

```bash
python -m src.features.build_features
```

### 5. Data Balancing

```bash
python -m src.modeling.handle_imbalance
```

### 6. Baseline Training

```bash
python -m src.modeling.train_baseline
```

### 7. Model Training

```bash
python -m src.modeling.train_models
```

### 8. Model Interpretability

```bash
python -m src.evaluation.interpret_model
```

### 9. Model Evaluation

```bash
python -m src.evaluation.evaluate_model
```

### 10. Model Persistence

```bash
python -m src.modeling.persist_model
```

---

# 🚀 Run the Application Locally

From the project root:

```bash
python -m uvicorn src.api.app:app --reload
```

Then open:

### Frontend

```text
http://127.0.0.1:8000/
```

### Swagger API

```text
http://127.0.0.1:8000/docs
```

### Health Check

```text
http://127.0.0.1:8000/health
```

---

# 🧪 Testing

API tests are implemented using `pytest`.

Run:

```bash
python -m pytest tests/test_api.py -q
```

Expected result:

```text
11 passed
```

---

# ☁️ Deployment

The application is deployed using **Render**.

### Production Start Command

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

### Build Command

```bash
pip install -r requirements.txt
```

### Health Check

```text
/health
```

### Production Application

**https://book-rating-prediction.onrender.com/**

---

# 🔐 Deployment Artifacts

The deployment requires:

```text
.python-version
Procfile
render.yaml
requirements.txt
models/inference_bundle.joblib
src/api/app.py
static/index.html
static/styles.css
static/app.js
```

The virtual environment is intentionally excluded from version control.

```text
.venv/
```

Dependencies are reproduced using:

```text
requirements.txt
```

---

# 📌 Key Engineering Practices

This project emphasizes several practical machine learning engineering principles:

### Data Leakage Prevention

The target-defining `rating` feature is excluded from model inputs.

### Reproducible Preprocessing

The preprocessing pipeline is persisted alongside the model.

### Class Imbalance Handling

SMOTE is applied to address minority-class representation during training.

### Baseline Comparison

A `DummyClassifier` establishes a minimum performance benchmark.

### Cross-Validation

5-fold cross-validation is used during model selection.

### Hyperparameter Optimization

Grid search and randomized search are used to identify stronger model configurations.

### Model Interpretability

Permutation feature importance provides model-agnostic explanations.

### API Separation

The trained model is separated from the presentation layer and exposed through FastAPI.

### Automated Testing

The API includes automated tests using Pytest.

### Deployment

The trained inference application is deployed and publicly accessible through Render.

---

# ⚠️ Limitations

This project has several important limitations.

1. **Demo Dataset**
   Books to Scrape contains fictional/demo data.

2. **Small Dataset**
   The dataset contains approximately 1,000 records.

3. **Limited Feature Space**
   The model uses structured book information rather than richer signals such as reviews, authors, publishers, or textual descriptions.

4. **Synthetic Class Balancing**
   SMOTE generates synthetic observations and therefore does not add genuinely new real-world observations.

5. **Educational Objective**
   Model predictions should not be interpreted as commercially validated book-rating forecasts.

---

# 🎓 Project Purpose

This project was developed as a **major machine learning project for internship and portfolio purposes**.

It demonstrates the transition from:

```text
Raw Data
   ↓
Data Collection
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Interpretation
   ↓
Model Persistence
   ↓
REST API
   ↓
Frontend
   ↓
Cloud Deployment
```

The primary objective is to demonstrate practical **Machine Learning + Data Science + ML Engineering + API Deployment** skills rather than simply training a classification model.

---

# 👨‍💻 Author

**Priyanshu Raj**

---

# 📄 License

This project is created for **educational purposes**.
