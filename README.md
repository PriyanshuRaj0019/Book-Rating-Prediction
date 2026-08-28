# Book Rating Prediction — End-to-End Machine Learning Project

## Project Overview

This is a production-style end-to-end machine learning project that predicts whether a book is likely to receive a high or low rating.

The project covers the complete machine learning lifecycle:

- Ethical data collection
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Data balancing
- Baseline modeling
- Model development
- Model interpretability
- Final evaluation
- Model persistence
- FastAPI application development
- Frontend UI integration
- Deployment preparation
- Automated testing
- Documentation

The final application provides both:

- A custom frontend prediction UI
- A FastAPI backend with Swagger API documentation

## Problem Statement

The objective of this project is to build a machine learning system that predicts whether a book is likely to be high-rated based on structured book information.

The prediction target is:

```text
is_high_rating
```

Target definition:

```text
1 = rating greater than or equal to 4
0 = rating below 4
```

The raw `rating` column is not used as an input feature because it directly defines the target and would cause data leakage.

## Dataset Source

Data was collected from:

```text
Books to Scrape
https://books.toscrape.com/
```

The website is a public scraping sandbox created for web scraping practice.

Collected data includes:

- Book title
- Category
- Price
- Rating
- Availability status
- Product URL

Dataset size:

```text
1000 records
```

Important limitation:

The source website contains fictional/demo data. Prices and ratings are not real market data. Therefore, business conclusions are educational and should not be treated as real commercial insights.

## Machine Learning Workflow

### 1. Ethical Data Collection

The scraper uses:

- BeautifulSoup
- Requests
- Custom request headers
- Rate limiting
- Error handling
- Logging
- robots.txt verification
- CSV storage
- Source metadata documentation

Output files:

```text
data/raw/books.csv
data/raw/source_metadata.json
```

### 2. Data Preprocessing

The preprocessing step handles:

- Missing values
- Duplicate records
- Invalid values
- Incorrect data types
- Column renaming
- Outlier detection
- Target column creation

Output files:

```text
data/processed/books_cleaned.csv
data/processed/cleaning_metadata.json
```

### 3. Exploratory Data Analysis

EDA includes:

- Summary statistics
- Rating distribution
- Price distribution
- Target distribution
- Price by rating analysis
- Correlation analysis
- Business observations

Output files:

```text
reports/docs/eda_summary.json
reports/figures/price_distribution.png
reports/figures/rating_distribution.png
reports/figures/target_distribution.png
reports/figures/price_by_rating.png
reports/figures/correlation_heatmap.png
```

### 4. Feature Engineering

Created features include:

- Price transformation
- Price band
- Title character count
- Title word count
- Title digit count
- Title punctuation count
- Title uppercase ratio
- Availability quantity
- Encoded categorical variables
- Scaled numeric variables

Excluded columns:

```text
rating
product_url
```

Reason:

- `rating` causes target leakage.
- `product_url` is identifier-like and not useful for generalizable prediction.

Output files:

```text
data/processed/books_features.csv
data/processed/feature_metadata.json
models/preprocessing_pipeline.joblib
```

### 5. Data Balancing

Class imbalance was checked before modeling.

Balancing techniques applied:

- SMOTE
- ADASYN
- Random oversampling

Chosen strategy:

```text
SMOTE
```

Reason:

SMOTE creates synthetic minority-class samples instead of only duplicating existing rows.

Output files:

```text
data/processed/train_test_split.joblib
data/processed/balanced_training_sets.joblib
data/processed/balancing_metadata.json
models/fitted_preprocessing_pipeline.joblib
```

### 6. Baseline Model

A `DummyClassifier` baseline was trained using the most frequent class strategy.

Purpose:

To create a minimum benchmark that real models must outperform.

Output files:

```text
models/baseline_model.joblib
data/processed/baseline_metrics.json
```

### 7. Model Development

Models trained:

- Logistic Regression
- Random Forest
- Support Vector Machine

Training methods used:

- GridSearchCV
- RandomizedSearchCV
- 5-fold cross-validation
- F1-based model selection

Output files:

```text
models/logistic_regression_model.joblib
models/random_forest_model.joblib
models/svm_model.joblib
models/final_model.joblib
data/processed/model_comparison.csv
data/processed/model_training_results.json
```

### 8. Model Interpretability

Permutation feature importance was used because it works across different model families.

Output files:

```text
data/processed/model_interpretation.json
reports/figures/feature_importance.png
```

### 9. Final Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- ROC curve
- Classification report

Output files:

```text
data/processed/final_evaluation_metrics.json
reports/docs/classification_report.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

### 10. Model Persistence

The final model and fitted preprocessing pipeline are saved together as an inference bundle.

Output files:

```text
models/inference_bundle.joblib
models/inference_bundle.pkl
data/processed/persistence_metadata.json
```

Preferred inference artifact:

```text
models/inference_bundle.joblib
```

## Application

The application is built with:

```text
FastAPI
HTML
CSS
JavaScript
```

The app provides:

- Custom frontend UI
- Prediction form
- Example input button
- Loading state
- Success state
- Error state
- Probability progress bar
- API documentation
- Health check endpoint

## API Endpoints

### Frontend UI

```text
GET /
```

Displays the custom prediction interface.

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "bundle_path": "models/inference_bundle.joblib"
}
```

### Prediction

```text
POST /predict
```

Example request:

```json
{
  "book_title": "A Light in the Attic",
  "book_category": "Books",
  "price_gbp": 51.77,
  "availability_status": "In stock"
}
```

Example response:

```json
{
  "prediction": 0,
  "prediction_label": "low_rating",
  "high_rating_probability": 0.42,
  "model_version": "1.0.0"
}
```

### Swagger Documentation

```text
GET /docs
```

Opens the interactive FastAPI Swagger UI.

## Project Structure

```text
Major Project/
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
│   ├── config/
│   ├── data_collection/
│   │   └── scrape_books.py
│   ├── eda/
│   │   └── analyze_books.py
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── interpret_model.py
│   ├── features/
│   │   └── build_features.py
│   ├── modeling/
│   │   ├── handle_imbalance.py
│   │   ├── persist_model.py
│   │   ├── train_baseline.py
│   │   └── train_models.py
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

## Setup Instructions

Create virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate virtual environment:

```bash
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify environment:

```bash
python -c "import pandas, sklearn, imblearn, fastapi, bs4; print('Environment ready')"
```

Expected output:

```text
Environment ready
```

## Run Complete Pipeline

Run each phase in order.

### Data Collection

```bash
python -m src.data_collection.scrape_books
```

### Data Preprocessing

```bash
python -m src.preprocessing.clean_books
```

### EDA

```bash
python -m src.eda.analyze_books
```

### Feature Engineering

```bash
python -m src.features.build_features
```

### Data Balancing

```bash
python -m src.modeling.handle_imbalance
```

### Baseline Model

```bash
python -m src.modeling.train_baseline
```

### Model Training

```bash
python -m src.modeling.train_models
```

### Model Interpretability

```bash
python -m src.evaluation.interpret_model
```

### Model Evaluation

```bash
python -m src.evaluation.evaluate_model
```

### Model Persistence

```bash
python -m src.modeling.persist_model
```

## Run Application Locally

From the `Major Project` folder:

```bash
python -m uvicorn src.api.app:app --reload
```

Open frontend UI:

```text
http://127.0.0.1:8000/
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Open health check:

```text
http://127.0.0.1:8000/health
```

## Run Tests

```bash
python -m pytest tests/test_api.py -q
```

Expected output:

```text
11 passed
```

## Deployment

The project is prepared for Render deployment.

Required deployment files:

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

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/health
```

## Author

```text
Priyanshu Raj
Registration Number: INBT021231
Internship: iNeuBytes Data Science Internship
```

## License

This project is created for educational, internship, and portfolio purposes.
