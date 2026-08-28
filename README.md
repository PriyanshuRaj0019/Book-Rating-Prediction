# 📚 Book Rating Prediction — End-to-End Machine Learning

An end-to-end Machine Learning application that predicts whether a book is likely to receive a **high or low rating** based on book metadata such as title, category, price, and availability.

The project covers the complete ML lifecycle — from **data collection and preprocessing to model training, evaluation, interpretability, API development, frontend integration, testing, and deployment**.

## 🚀 Live Demo

🌐 **Live Application:**  
https://book-rating-prediction-heu5.onrender.com/

📖 **Swagger API Documentation:**  
https://book-rating-prediction-heu5.onrender.com/docs

🏥 **Health Check:**  
https://book-rating-prediction-heu5.onrender.com/health

💻 **GitHub Repository:**  
https://github.com/PriyanshuRaj0019/Book-Rating-Prediction

---

## 🎯 Project Objective

The objective is to predict whether a book is likely to receive a **high rating** or **low rating** using structured book information.

### Target Variable

```text
is_high_rating
1 → Rating >= 4
0 → Rating < 4

The original rating feature is excluded from model inputs because it directly defines the target and would cause target leakage.

📊 Dataset

Data was collected from Books to Scrape, a public website designed for web scraping practice.

Source: https://books.toscrape.com/

Dataset Size
1,000 book records
Collected Attributes
Book title
Book category
Price
Rating
Availability status
Product URL

Note: Books to Scrape contains fictional/demo data. Prices and ratings are not real-world commercial data, so all insights and predictions should be considered educational.

🔄 Machine Learning Pipeline
Data Collection
      ↓
Data Validation & Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Class Imbalance Handling
      ↓
Baseline Model
      ↓
Model Training & Hyperparameter Tuning
      ↓
Model Comparison
      ↓
Model Interpretation
      ↓
Final Evaluation
      ↓
Model Persistence
      ↓
FastAPI Backend
      ↓
Frontend UI
      ↓
Automated Testing
      ↓
Render Deployment
1. 🕷️ Data Collection

The scraping pipeline uses:

Python
Requests
BeautifulSoup
Custom request headers
Rate limiting
Error handling
Logging
robots.txt verification
CSV storage
Source metadata tracking

Outputs:

data/raw/books.csv
data/raw/source_metadata.json
2. 🧹 Data Preprocessing

The preprocessing pipeline handles:

Missing values
Duplicate records
Invalid values
Data type conversion
Column standardization
Outlier detection
Target creation
Data validation

Outputs:

data/processed/books_cleaned.csv
data/processed/cleaning_metadata.json
3. 📈 Exploratory Data Analysis

EDA includes:

Rating distribution
Price distribution
Target distribution
Price by rating
Correlation analysis
Summary statistics
Dataset-level observations

Generated visualizations:

reports/figures/
├── price_distribution.png
├── rating_distribution.png
├── target_distribution.png
├── price_by_rating.png
└── correlation_heatmap.png
4. ⚙️ Feature Engineering

Engineered features include:

Numerical Features
Price transformation
Price band
Title character count
Title word count
Title digit count
Title punctuation count
Title uppercase ratio
Availability quantity
Categorical Features
Book category
Availability status

Categorical variables are encoded and numerical variables are scaled using a preprocessing pipeline.

Excluded Features
rating
product_url

rating is excluded to prevent target leakage, while product_url is excluded because it is identifier-like and does not provide a meaningful generalizable signal.

5. ⚖️ Class Imbalance

Class imbalance was evaluated before model training.

The following techniques were considered:

SMOTE
ADASYN
Random Oversampling
Selected Strategy
SMOTE

SMOTE generates synthetic minority-class observations rather than simply duplicating existing observations.

6. 🤖 Baseline Model

A DummyClassifier using the most-frequent-class strategy was trained as a baseline.

The baseline establishes a minimum benchmark that the actual Machine Learning models should outperform.

7. 🧠 Machine Learning Models

The following models were trained and compared:

Logistic Regression
Random Forest
Support Vector Machine
Optimization
GridSearchCV
RandomizedSearchCV
5-fold cross-validation
F1-based model selection

The final model was selected based on comparative validation performance.

8. 🔍 Model Interpretability

Permutation Feature Importance was used to understand which features contributed most to model predictions.

Outputs:

data/processed/model_interpretation.json
reports/figures/feature_importance.png
9. 📊 Model Evaluation

The final model is evaluated using:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
ROC Curve
Classification Report

Outputs:

data/processed/final_evaluation_metrics.json
reports/docs/classification_report.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
10. 💾 Model Persistence

The trained model and preprocessing pipeline are stored as an inference bundle.

Preferred artifact:

models/inference_bundle.joblib

This ensures that the same preprocessing and trained model configuration is used during inference.

🌐 Web Application

The application is built using:

FastAPI
HTML
CSS
JavaScript
Scikit-learn
Joblib
Features
Custom prediction UI
Prediction form
Example input
Loading state
Success state
Error handling
High-rating probability
Probability progress bar
REST API integration
Swagger documentation
Health check endpoint
🔌 API
GET /

Returns the custom prediction interface.

GET /health

Checks API and model availability.

Example:

{
  "status": "ok",
  "model_loaded": true,
  "bundle_path": "models/inference_bundle.joblib"
}
POST /predict

Predicts whether a book is likely to receive a high rating.

Request
{
  "book_title": "A Light in the Attic",
  "book_category": "Books",
  "price_gbp": 51.77,
  "availability_status": "In stock"
}
Response
{
  "prediction": 0,
  "prediction_label": "low_rating",
  "high_rating_probability": 0.42,
  "model_version": "1.0.0"
}
Swagger

Interactive API documentation is available at:

https://book-rating-prediction-heu5.onrender.com/docs

🧪 Testing

API functionality is tested using Pytest.

python -m pytest tests/test_api.py -q

Tests cover:

API availability
Health endpoint
Prediction endpoint
Valid inputs
Response structure
Error handling
📁 Project Structure
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
⚙️ Local Setup
Clone Repository
git clone https://github.com/PriyanshuRaj0019/Book-Rating-Prediction.git
cd Book-Rating-Prediction
Create Virtual Environment
py -3.12 -m venv .venv
Activate

Windows PowerShell:

.venv\Scripts\Activate.ps1

Git Bash:

source .venv/Scripts/activate
Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Verify Environment
python -c "import pandas, sklearn, imblearn, fastapi, bs4; print('Environment ready')"

Expected:

Environment ready
▶️ Run the Pipeline
python -m src.data_collection.scrape_books
python -m src.preprocessing.clean_books
python -m src.eda.analyze_books
python -m src.features.build_features
python -m src.modeling.handle_imbalance
python -m src.modeling.train_baseline
python -m src.modeling.train_models
python -m src.evaluation.interpret_model
python -m src.evaluation.evaluate_model
python -m src.modeling.persist_model
🚀 Run Locally
python -m uvicorn src.api.app:app --reload

Application:

http://127.0.0.1:8000/

Swagger:

http://127.0.0.1:8000/docs

Health Check:

http://127.0.0.1:8000/health
☁️ Deployment

The application is deployed using Render.

Build Command
pip install -r requirements.txt
Start Command
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
Health Check
/health
Live Application

https://book-rating-prediction-heu5.onrender.com/

🛠️ Tech Stack

Languages & Frameworks

Python
FastAPI
HTML
CSS
JavaScript

Data Science & ML

Pandas
NumPy
Matplotlib
Scikit-learn
Imbalanced-learn

Data Collection

Requests
BeautifulSoup

Machine Learning

Logistic Regression
Random Forest
Support Vector Machine
SMOTE
GridSearchCV
RandomizedSearchCV
Cross-validation

Testing & Deployment

Pytest
Uvicorn
Render

Model Persistence

Joblib
🎓 Key Learning Outcomes

This project demonstrates practical experience with:

End-to-end ML pipeline development
Web data collection
Data preprocessing
Exploratory data analysis
Feature engineering
Target leakage prevention
Imbalanced classification
Hyperparameter optimization
Cross-validation
Model comparison
Model interpretability
Model persistence
REST API development
Frontend-backend integration
Automated testing
Cloud deployment
⚠️ Disclaimer

This project is intended for educational and portfolio purposes.

The dataset comes from a fictional/demo web scraping sandbox. Predictions should not be interpreted as real-world book-market recommendations or commercial decisions.

👤 Author

Priyanshu Raj

GitHub:
https://github.com/PriyanshuRaj0019

📄 License

This project is available for educational and portfolio purposes.
