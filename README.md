# 💼 Salary Prediction & Insight Platform

## 📌 Project Overview

This project is an end-to-end machine learning system that predicts salaries based on job-related features and provides human-readable insights.

It combines:

* Machine Learning (salary prediction)
* FastAPI (model serving)
* Streamlit (interactive dashboard)
* Supabase (data storage)
* Ollama (local LLM for explanations)

The goal is to make salary insights **accessible and understandable for non-technical users**.

---

## 🚀 Live Demo

* 🌐 **Streamlit App: https://salary-prediction-app-georgeselhajj-uvjvurvz46ejukfxnxmdep.streamlit.app/
* 🔗 **API Endpoint: https://salary-prediction-app-georgeselhajj.onrender.com/predict

---

## 🧠 Features

### 🔮 Live Salary Prediction

* Users input job details (role, experience, location, etc.)
* The system predicts salary using a trained ML model
* A short explanation is generated for easy understanding

### 📊 Saved Estimates Dashboard

* All predictions are stored in Supabase
* Users can explore previous estimates
* Filter by experience, employment type, and company size

### 📈 Visual Insights

* Salary trends by experience level
* Top job roles by predicted salary
* Clean, simple visualizations for non-technical users

### 🤖 AI-Powered Explanations

* Uses **Ollama (local LLM)** to generate human-friendly insights
* Falls back to rule-based explanation when deployed

---

## 🏗️ System Architecture

```text
User (Streamlit)
      ↓
FastAPI (Model Inference)
      ↓
Ollama (Local LLM - optional in deployment)
      ↓
ML Model (Decision Tree)
      ↓
Supabase (Storage)
      ↓
Streamlit Dashboard (Visualization + Insights)      
```

---

## 📂 Project Structure

```text
salary-prediction-app/
│
├── data/
│   └── raw/                  # Original dataset
│
├── artifacts/
│   └── model.joblib          # Trained model
│
├── src/
│   ├── data/                 # Loading & preprocessing
│   ├── models/               # Training logic
│   ├── api/                  # FastAPI app
│   ├── dashboard/            # Streamlit app
│   ├── db/                   # Supabase client
│   └── llm/                  # Ollama integration
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Machine Learning

### Model Used

* **DecisionTreeRegressor**

### Features Used

* work_year
* experience_level
* employment_type
* job_title
* employee_residence
* remote_ratio
* company_location
* company_size

### Target

* salary_in_usd

### Performance 

* MAE: 28177.00
* RMSE: 49611.31
* R²: 0.49

---

## 🔌 API (FastAPI)

### Endpoint

```http
GET /predict
```

### Example Request

```text
/predict?work_year=2022&experience_level=SE&employment_type=FT&job_title=Data Scientist&employee_residence=US&remote_ratio=100&company_location=US&company_size=M
```

### Example Response

```json
{
  "predicted_salary_usd": 172425.51,
  "model_name": "DecisionTreeRegressor"
}
```

---

## 🗄️ Database (Supabase)

### Tables

#### prediction_runs

* id
* run_name
* model_name
* created_at

#### predictions

* run_id
* job features
* predicted_salary_usd
* status

#### analyses

* run_id
* summary stats
* analysis_text

---

## 🤖 LLM Integration (Ollama)

* Model: `llama3.2:3b`
* Generates short explanations for predictions
* Used locally
* Disabled in deployed version (fallback logic used)

---

## 🌐 Deployment

### FastAPI

* Hosted on **Render**
* Public API endpoint available

### Streamlit

* Hosted on **Streamlit Community Cloud**
* Connected to Supabase and FastAPI

### Important Note

Ollama runs locally, so:

* Live AI explanations work locally
* Deployed version uses fallback explanation

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/GeorgeElHajj/salary-prediction-app-GeorgesELHAJJ.git
cd salary-prediction-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create `.env`:

```env
SUPABASE_URL=url
SUPABASE_KEY=key
MODEL_PATH=artifacts/model.joblib
FASTAPI_PREDICT_URL=http://127.0.0.1:8000/predict
OLLAMA_URL=http://localhost:11434
ENABLE_LIVE_OLLAMA=true
```

### 4. Run FastAPI

```bash
uvicorn src.api.main:app --reload
```

### 5. Run Streamlit

```bash
streamlit run src/dashboard/app.py
```

---

## 📊 Data Source

Dataset: **ds_salaries.csv**

Contains global data science job salaries with features like:

* experience level
* location
* job role
* company size

---

## 🧪 Future Improvements

* Better ML model (Random Forest / XGBoost)
* More accurate feature engineering
* Deploy LLM in cloud (instead of local Ollama)
* User authentication
* Salary comparison across countries

---

## 👨‍💻 Author

* Georges EL HAJJ

---

