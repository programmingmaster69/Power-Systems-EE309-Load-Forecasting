Weather Based Load Forecasting credit to - https://github.com/KomalGoel18

# ⚡ Short-Term Load Forecasting using Time-Series Models

This repository presents an implementation of multiple short-term load forecasting techniques, inspired by IEEE research on electricity demand prediction. The work focuses on modeling intraday and intraweek patterns using statistical and machine learning approaches.

---

## 📌 Overview

Accurate short-term load forecasting is essential for efficient power system operation. This project implements and compares multiple forecasting approaches, including:

- Singular Value Decomposition (SVD)-based models  
- Exponential smoothing techniques  
- Neural network-based models (LSTM)  
- Statistical baselines  

The models are evaluated using log-transformed load data and standard error metrics such as MAPE.

---

## 📊 Features

- 📈 Actual vs Forecast visualization (log scale)  
- 📉 MAPE vs Forecast Horizon (up to 48 half-hours)  
- 📦 Export of results (CSV, Excel, PNG figures)  
- 🔍 Rolling forecast evaluation  
- ⚙️ Parameter tuning (grid search)  
- 📊 IEEE-style plots for publication  

---


---

## ⚙️ Data Pre-processing

The following preprocessing steps are applied:

- Removal of missing and invalid values  
- Interpolation to ensure continuity  
- Conversion to uniform half-hour intervals  
- Logarithmic transformation for variance stabilization  
- Chronological splitting into training, validation, and testing sets  
- Preservation of seasonal (intraday and intraweek) structures  

---

## 🚀 Methodology

### 1. Time-Series Modeling
- Models capture both **short-term dependencies** and **seasonal patterns**  
- Feature extraction is performed where necessary (e.g., decomposition-based methods)

### 2. Forecasting
- Recursive multi-step forecasting up to **48 half-hours (24 hours)**  
- Rolling-origin evaluation is used for realistic performance estimation  

### 3. Error Metric

Mean Absolute Percentage Error (MAPE):

```math
MAPE = \frac{100}{N} \sum_{t=1}^{N} \left| \frac{y_t - \hat{y}_t}{y_t} \right|

