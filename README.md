# Tesla Stock Forecasting Project - Detailed Documentation

## Project Overview
This project is focused on forecasting Tesla’s stock prices from **2023-07-01** to **2024-07-15**. It encompasses both **univariate** and **multivariate** forecasting approaches. The main objective is to predict the stock prices for the next **30 days** and **90 days** using a variety of statistical and machine learning models. The project is organized into four Jupyter notebooks, each leveraging a different model or set of models.

### Common Workflow Across Notebooks
Each notebook follows a consistent workflow:
1. **Data Loading and Preprocessing**
   - Import necessary libraries.
   - Load the Tesla stock price dataset.
   - Filter the dataset for the date range **2023-07-01** to **2024-07-15**.
   - Handle missing data and normalize/standardize the values if needed.
   
2. **Exploratory Data Analysis (EDA)**
   - Plot historical stock prices.
   - Check for trends, seasonality, and volatility in the data.

3. **Data Splitting**
   - Split the data into **training** and **test** sets.
   - The training set is typically used to train the models, while the test set is used to evaluate performance.
   
4. **Model Implementation**
   - Build and train the respective model(s).
   - Tune hyperparameters using grid search or manual tuning methods.
   
5. **Forecasting**
   - Use the trained model to predict stock prices for the next **30 days** and **90 days**.
   - Compare the forecasts against actual stock prices (if available) or validate based on historical patterns.

6. **Performance Evaluation**
   - Calculate evaluation metrics such as:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Square Error (RMSE)**
     - **Mean Absolute Percentage Error (MAPE)**
   - Plot actual vs. predicted values to visually assess model performance.

---

### 1. `Tesla_Stock_forecast_(arima__and_sarima) (1).ipynb`
**Description**:  
This notebook implements **univariate time series forecasting** using the **ARIMA** and **SARIMA** models on the **'Close'** column of Tesla stock data.

#### Workflow:
1. **ARIMA Model**:
   - **Data Preprocessing**: The 'Close' column is checked for stationarity using the Augmented Dickey-Fuller (ADF) test. If the data is non-stationary, differencing is applied.
   - **Model Selection**: Use the **auto_arima** function to automatically select the best ARIMA parameters (p, d, q) by minimizing AIC.
   - **Model Training**: Fit the ARIMA model on the training data.
   - **Forecasting**: Predict Tesla stock prices for the next **30 days** and **90 days**.
   - **Evaluation**: Compare forecasts with actual values using MAE and RMSE.
  
   - Test Data Prediction
  
     
   - ![download (6)](https://github.com/user-attachments/assets/4ac3c804-55c1-4c3b-889f-e9e2017e5d8f)

  
   - Next 30 Days Forecast
  
   - ![download (21)](https://github.com/user-attachments/assets/55fa2e22-a20c-4102-8844-ea8d9d87af9c)
     
   - Next 90 Days Forecast
  
   - ![download (47)](https://github.com/user-attachments/assets/22234d1b-bad2-4dad-8941-43969d637516)


2. **SARIMA Model**:
   - **Seasonality Analysis**: Seasonal components in the data are identified using seasonal decomposition and autocorrelation plots.
   - **Model Selection**: Use **auto_arima** to select the best SARIMA parameters (p, d, q) x (P, D, Q, s) that capture both short-term and seasonal patterns.
   - **Model Training**: Fit the SARIMA model to the training data.
   - **Forecasting**: Predict the next **30 days** and **90 days**.
   - **Evaluation**: Compare SARIMA performance with ARIMA using the same evaluation metrics.
  
   - Test Data Prediction
  
   - ![download (7)](https://github.com/user-attachments/assets/225a443b-a6ce-453c-b38d-021771a43d37)
  
   - Next 30 Days Forecast
  
   - ![download (22)](https://github.com/user-attachments/assets/a3dfbdc2-e454-494d-be5f-986b413d3fb0)
  
   - Next 90 Days Forecast
  
   - ![download (48)](https://github.com/user-attachments/assets/9d9f3c39-045a-4b04-8af9-a39ab7708e77)




---

### 2. `Tesla_Stock_forecastin_with_ML_Models.ipynb`
**Description**:  
This notebook applies **machine learning regression models** for **univariate forecasting** using the **'Close'** column of Tesla stock data. It compares several popular regression models to evaluate their performance in predicting stock prices.

#### Workflow:
1. **Feature Engineering**:
   - Use rolling windows (e.g., 5-day, 10-day averages) as features for predicting future prices.
   - Create lag features representing past stock prices to inform future predictions.
   
2. **Model Implementation**:
   - **Linear Regression**:
     - Build a simple linear regression model using past stock prices as the independent variable(s).
     - Train the model on the training data and forecast for the next **30 days** and **90 days**.
     - Evaluate performance using MAE and RMSE.
    
     - Actual Data VS Prediction
    
     - ![download (50)](https://github.com/user-attachments/assets/b39c4d85-f3b9-487c-8a5d-7428fc436437)
    
     - Next 30 Days Forecast
    
     - ![download (51)](https://github.com/user-attachments/assets/0683efc7-17d0-4277-9fec-8d5c57b82ac2)
    
     - Next 90 Days Forecast
    
     - ![download (52)](https://github.com/user-attachments/assets/baef0970-c2d1-48af-970b-41f6312119d1)



   
   - **Random Forest Regressor**:
     - Build a random forest model using ensemble learning.
     - Train the model and use it to predict stock prices.
     - Random Forest provides robust performance for nonlinear relationships between features and targets.
     - Evaluate the forecast accuracy for both **30 days** and **90 days**.
    
     - Actual Data Vs Prediction
    
     - ![download (53)](https://github.com/user-attachments/assets/71c47f46-f632-439e-923f-72427694d8be)
    
     - Next 30 Days Forecast
    
     - ![download (54)](https://github.com/user-attachments/assets/5b6ebf75-851b-47ae-9288-f1ab6c5acbd5)

    
     - Next 90 Days Forecast
    
     - ![download (55)](https://github.com/user-attachments/assets/613bde77-1525-412c-9c99-feecc8e41b75)



   - **Support Vector Regression (SVR)**:
     - Use SVR to map data into higher dimensions for better separation and prediction.
     - Train and forecast stock prices for **30 days** and **90 days** using SVR.
     - Evaluate the model’s performance based on actual vs. predicted values.
    
     - Actual Data Vs Prediction
    
     - ![download (56)](https://github.com/user-attachments/assets/0a2894f9-219f-4f5c-9c08-c774da14ad98)

    
     - Next 30 Days Forecast
    
     - ![download (57)](https://github.com/user-attachments/assets/3dd04905-97f0-4492-9624-548c2efb0f64)

    
     - Next 90 Days Forecast
     
     -   ![download (58)](https://github.com/user-attachments/assets/ed2c1222-50e2-4f96-a909-b60f31106fa4)

     - 
     - 
   
   - **XGBoost Regressor**:
     - Build and train an XGBoost model, which is known for its effectiveness in handling complex and non-linear data.
     - Forecast for **30 days** and **90 days**.
     - Use feature importance to understand which features contribute most to predictions.
    
     - Actual Data Vs Prediction
     
     -   ![download (59)](https://github.com/user-attachments/assets/3d9a25c6-15e3-43b7-8608-5f34c4f761d7)
    
     - Next 30 Days Forecast
    
     - ![download (60)](https://github.com/user-attachments/assets/5da4cb06-8a7c-495c-ac1c-4ea56d652b23)

    
     - Next 30 Days Forecast
    
     - ![download (61)](https://github.com/user-attachments/assets/c1d30b7a-ca13-4574-9ca9-5def43e913ad)

---

### 3. `Tesla_Univariate_Stock_forecast_with_LSTM.ipynb`
**Description**:  
This notebook uses a **Long Short-Term Memory (LSTM)** network to perform **univariate time series forecasting** based solely on the **'Close'** column.

#### Workflow:
1. **Data Preparation**:
   - Normalize the 'Close' column using MinMaxScaler to scale the values between 0 and 1, which helps the LSTM model converge faster.
   - Split the data into sequences of input (X) and output (y) windows to train the model (e.g., use the last 60 days of data to predict the next day).

2. **Model Architecture**:
   - Build an LSTM model with:
     - Input layer corresponding to the sequence length (e.g., 10 days).
     - Hidden layers consisting of LSTM cells.
     - Dense output layer predicting the next day's stock price.
   
3. **Model Training**:
   - Train the LSTM model on the training data, typically over 50-100 epochs to minimize loss (e.g., Mean Squared Error).
   
4. **Forecasting**:
   - Predict Tesla’s stock prices for the next **30 days** and **90 days** by feeding in the latest 60-day sequences.
   - Rescale the predicted values back to the original scale using the inverse of MinMaxScaler.

5. **Evaluation**:
   - Plot the actual vs. predicted values.
   - Calculate and analyze MAE, RMSE, and MAPE.
  
   - Next 30 Days Forecast
  
   - ![download (62)](https://github.com/user-attachments/assets/4c92ece2-d398-42fb-866d-12004d994de8)
  
   - Next 90 Days Forecast
  
   - ![download (63)](https://github.com/user-attachments/assets/9067fda4-8e44-462f-86a1-86e6e9b013df)



---

### 4. `Tesla_Multivariate_Stock_forecast_with_LSTM.ipynb`
**Description**:  
This notebook uses a **multivariate LSTM** model to forecast Tesla stock prices using **multiple columns** of data (e.g., 'Open', 'Close', 'High', 'Low', 'Volume').

#### Workflow:
1. **Feature Selection**:
   - Include multiple features such as 'Open', 'High', 'Low', 'Volume' in addition to the 'Close' column.
   - Correlation analysis is done to understand the relationship between features.

2. **Data Preparation**:
   - Scale all features using MinMaxScaler.
   - Create sequences of input data using rolling windows across all selected features.
   - Split the data into training and test sets, ensuring all features are aligned with the output (future 'Close' values).

3. **Model Architecture**:
   - Build a more complex LSTM model that takes into account multiple input features.
   - The architecture includes input layers for each feature, followed by LSTM layers and a dense output layer that predicts future 'Close' prices.

4. **Model Training**:
   - Train the multivariate LSTM model using multiple epochs.
   - Tune the model's hyperparameters such as learning rate, batch size, and the number of LSTM units.

5. **Forecasting**:
   - Predict stock prices for the next **30 days** and **90 days** based on the input features.
   - Convert the scaled predictions back to their original values.

6. **Evaluation**:
   - Plot actual vs. predicted prices and compare the multivariate model's performance with the univariate LSTM model.
   - Evaluate MAE, RMSE, and MAPE.
  
   - Next 30 Days Forecast
   
   - ![download (64)](https://github.com/user-attachments/assets/ca44cee0-017c-4628-89bc-cca47d664719)

   - Next 90 Days Forecast
     
   - ![download (65)](https://github.com/user-attachments/assets/821ac209-21fd-487b-9dc8-81eeff8e4fba)
 

---

## Conclusion
This project uses a variety of methods to predict Tesla’s stock prices over short-term (30 days) and medium-term (90 days) horizons. The models range from traditional statistical techniques like ARIMA and SARIMA to more advanced machine learning and deep learning approaches such as LSTM. The goal is to provide insight into which model performs best for forecasting Tesla's stock under different conditions and datasets.
