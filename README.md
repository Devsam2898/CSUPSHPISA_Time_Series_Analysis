# Time Series Forecasting Project

This project focuses on time series forecasting using the CSUSHPISA dataset. It explores two different modeling approaches: AutoARIMA and ARIMA models for traditional time series forecasting, and LSTM (Long Short-Term Memory) neural network for more complex sequence prediction.

## Dataset

- **Dataset Name**: CSUSHPISA
- **Source**: [(http://fred.stlouisfed.org/series/CSUSHPISA)]
- **Description**: The CSUSHPISA dataset contains two three columns mainly, Index, Date, and CSUSHPISA. It is commonly used in Financial Research.
- **S&P/Case-Shiller U.S. National Home Price Index (CSUSHPISA). 

## Project Structure

- `data/`: Folder containing the dataset.
- `notebooks/`: Jupyter notebooks for data exploration, modeling, and results.
- `src/`: Python source code files.
- `README.md`: This file, providing an overview of the project.

## AutoARIMA and ARIMA Models

AutoARIMA is used to automatically select the optimal hyperparameters for the ARIMA model. The ARIMA model is then trained on the dataset to make predictions.

```python
from pmdarima import auto_arima
model = auto_arima(train_set.dropna())
arima_model = ARIMA(train_set.dropna(), order=model.order)
arima_model_fit = arima_model.fit()
```

## LSTM Model

An LSTM model is implemented using TensorFlow and Keras for deep learning-based time series forecasting. It's trained using a sequence length of 10.

```python
model = Sequential()
model.add(LSTM(110, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

## Results

The LSTM model achieved a Root Mean Squared Error (RMSE) of approximately 5.

## Usage

1. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn pmdarima tensorflow
   ```

2. Run the Jupyter notebooks in the `notebooks/` directory to see the data exploration and modeling process.

3. Execute the source code in the `src/` directory for a more automated workflow.

## Contributing

Feel free to contribute to this project by opening issues or pull requests. Your feedback and enhancements are highly appreciated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

