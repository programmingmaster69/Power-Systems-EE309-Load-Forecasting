#  Short-Term Load Forecasting using Exponentially Weighted Methods

This repository presents MATLAB implementations and simulations of short-term electrical load forecasting methods based on the IEEE research paper:

> **James W. Taylor, “Short-Term Load Forecasting With Exponentially Weighted Methods,” IEEE Transactions on Power Systems, Vol. 27, No. 1, Feb. 2012.**

The project focuses on forecasting half-hourly electrical demand using univariate time-series techniques without relying on weather inputs. Multiple statistical and decomposition-based forecasting approaches are implemented and evaluated using MATLAB simulations.

---

#  Objective

The objective of this work is to study and compare different short-term load forecasting models capable of capturing:

- Intraday seasonal patterns  
- Intraweek seasonal variations  
- Temporal dependencies in load demand  

The models are evaluated using rolling multi-step forecasting up to 48 half-hour intervals (24 hours ahead).

---

#  Forecasting Models Implemented

The following forecasting methods were studied and implemented:

###  Intraday Cycle (IC) Exponential Smoothing
- Models recurring daily load patterns
- Different seasonal structures for weekdays and weekends

###  Holt-Winters-Taylor (HWT) Method
- Double seasonal exponential smoothing
- Captures intraday and intraweek cycles

###  Singular Value Decomposition (SVD)-Based Forecasting
- Uses low-rank decomposition of weekly load matrices
- Extracts dominant temporal structures

###  Dynamic Weighted Regression (DWR)
- Uses adaptive weighted regression
- Includes trigonometric and spline-based formulations

###  Artificial Neural Network (ANN)
- Feedforward neural network for nonlinear forecasting

###  AutoRegressive Moving Average (ARMA)
- Statistical baseline model for time-series prediction

---

#  Data Pre-processing

The following preprocessing steps were performed prior to model implementation:

- Removal of missing and invalid observations  
- Interpolation to preserve continuity of the time series  
- Conversion to uniform half-hour intervals  
- Logarithmic transformation for variance stabilization  
- Chronological division into training and testing datasets  
- Preservation of intraday and intraweek seasonal structures  

---

#  MATLAB Simulations

The repository includes MATLAB implementations for forecasting and performance evaluation.

## Implemented Features

- Weekly matrix formation for SVD decomposition  
- Recursive exponential smoothing updates  
- Multi-step rolling forecasts  
- Actual vs Forecast visualization  
- Zoomed forecasting comparison  
- MAPE vs Forecast Horizon analysis  
- Export of plots and result tables  

---

#  Simulation Outputs

The MATLAB simulations generate the following outputs:

###  Actual vs Forecast Plot
Compares the predicted load profile with the actual load demand.

###  Zoomed Forecast Plot
Provides a detailed comparison over a smaller interval.

###  MAPE vs Forecast Horizon
Evaluates forecasting accuracy for horizons up to 48 half-hours ahead.

###  Error Tables
Exports horizon-wise MAPE values in CSV and Excel formats.

---

#  Performance Metric

Forecast accuracy is evaluated using Mean Absolute Percentage Error (MAPE):

```math
MAPE = \frac{100}{N}\sum_{t=1}^{N}\left|\frac{y_t-\hat{y}_t}{y_t}\right|
