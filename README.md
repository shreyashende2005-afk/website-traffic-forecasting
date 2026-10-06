# 📈 Website Traffic Forecasting

### Daily Website Traffic Prediction using Time Series Analysis & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Project-Completed-success)]()

---

## 🌐 Live Demo

🚀 **Streamlit Application:**  
https://website-traffic-forecasting-9x2amx8uyrzytea4379gdj.streamlit.app/

---

## 📌 Project Overview

Website Traffic Forecasting is a machine learning and time-series forecasting project developed to analyze historical daily website traffic and predict future traffic.

The project combines traditional time-series forecasting techniques with machine-learning regression models. Historical traffic patterns are transformed into meaningful lag, rolling and calendar-based features, which are then used to train and evaluate multiple forecasting models.

An interactive **Streamlit dashboard** is also developed to visualize traffic patterns, compare model performance and generate future traffic forecasts.

---

## 🎯 Objectives

- Analyze historical website traffic data.
- Perform data cleaning and preprocessing.
- Identify daily, weekly and monthly traffic patterns.
- Create time-series based features.
- Apply machine-learning forecasting techniques.
- Compare multiple forecasting models.
- Evaluate models using RMSE, MAE and MAPE.
- Generate future website traffic predictions.
- Develop an interactive Streamlit forecasting dashboard.

---

## 📊 Dataset

The project uses a daily website visitors dataset containing:

- **2,167 daily observations**
- Date range: **September 2014 to August 2020**
- Target variable: **Page.Loads**

### Dataset Features

| Feature | Description |
|---|---|
| Date | Date of website activity |
| Page.Loads | Total page loads |
| Unique.Visits | Number of unique visitors |
| First.Time.Visits | First-time visitors |
| Returning.Visits | Returning visitors |
| New.Sessions | New user sessions |
| Returning.Sessions | Returning user sessions |
| New.Visits | New visitors |

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit web application for website traffic analysis and forecasting.

### Dashboard Features

- 📊 Traffic overview
- 📈 Daily traffic visualization
- 📅 Weekday traffic analysis
- 📆 Monthly traffic trends
- 🤖 Machine-learning model comparison
- 🎯 Actual vs predicted traffic
- 🔮 Future traffic forecasting
- 📌 Feature importance
- 📥 Forecast CSV download

---

## 📸 Application Screenshots

### 🏠 Traffic Overview

![Traffic Overview](screenshots/dashboard.png)

---

### 📈 Traffic Analysis

![Traffic Analysis](screenshots/traffic_analysis.png)

---

### 🤖 Model Performance

![Model Performance](screenshots/comparison.png)

---

### 🔮 Future Traffic Forecast

![Future Forecast](screenshots/futurescope.png)

---

## 🔬 Exploratory Data Analysis

The historical traffic data was analyzed to understand:

- Overall traffic trends
- Weekly traffic patterns
- Monthly traffic variations
- High and low traffic periods
- Recurring traffic behavior

These patterns were used to design useful forecasting features.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Converted the `Date` column into datetime format.
2. Sorted observations chronologically.
3. Removed duplicate dates where required.
4. Converted comma-separated traffic values into numeric format.
5. Checked for missing values.
6. Verified the daily time-series structure.
7. Set the date as the time-series index.
8. Interpolated missing values where necessary.

---

## 🛠️ Feature Engineering

Several time-series and calendar-based features were created.

### Lag Features

- Lag 1
- Lag 2
- Lag 3
- Lag 7
- Lag 14
- Lag 21
- Lag 28

### Rolling Features

- 7-day rolling mean
- 7-day rolling standard deviation
- 14-day rolling mean
- 14-day rolling standard deviation
- 28-day rolling mean
- 28-day rolling standard deviation

### Calendar Features

- Day of week
- Day of month
- Month
- Week of year
- Trend

---

## 🤖 Forecasting Models

Multiple approaches were evaluated.

### Time-Series Models

- Weekly Seasonal Naive
- Holt-Winters Exponential Smoothing

### Machine Learning Models

- Random Forest Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor

The data was divided chronologically into:

- **80% Training Data**
- **20% Testing Data**

A chronological split was used to prevent future information from leaking into the training data.

---

## 📏 Model Evaluation

The models were evaluated using:

### RMSE
Root Mean Squared Error measures the magnitude of prediction errors.

### MAE
Mean Absolute Error measures the average absolute difference between actual and predicted values.

### MAPE
Mean Absolute Percentage Error measures prediction error as a percentage.

### Benchmark Results

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Weekly Naive | 615.07 | 407.84 | 11.28% |
| Holt-Winters | 1413.52 | 1161.29 | 28.84% |
| Random Forest | 372.31 | 278.77 | 7.71% |
| HistGradientBoosting | 349.42 | 255.97 | 6.86% |
| **Extra Trees** | **347.28** | **254.97** | **6.86%** |

### 🏆 Best Benchmark Model

**Extra Trees Regressor**

It achieved the lowest benchmark RMSE and MAE among the evaluated models, with a MAPE of approximately **6.86%**.

---

## 🔄 Project Workflow

```text
Raw Website Traffic Data
          ↓
Data Cleaning & Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Train/Test Time-Series Split
          ↓
Multiple Forecasting Models
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
Future Traffic Forecast
          ↓
Streamlit Dashboard
