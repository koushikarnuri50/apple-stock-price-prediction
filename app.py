

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# --- 1. Define Model and Data Paths ---
arima_filename = 'arima_model.joblib'
sarima_filename = 'sarima_model.joblib'
rf_filename = 'random_forest_model.joblib'
xgb_filename = 'xgboost_model.joblib'
lstm_model_filename = 'lstm_model.h5'
AAPL_DATA_FILE = 'AAPL.csv'

# --- 2. Load Models ---
@st.cache_resource
def load_all_models():
    """Loads all trained models."""
    try:
        arima_model = joblib.load(arima_filename)
        sarima_model = joblib.load(sarima_filename)
        rf_model = joblib.load(rf_filename)
        xgb_model = joblib.load(xgb_filename)
        lstm_model = tf.keras.models.load_model(lstm_model_filename)

        return {
            'ARIMA': arima_model,
            'SARIMA': sarima_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model,
            'LSTM': lstm_model,
        }
    except Exception as e:
        st.error(f"Error loading models. Please ensure all model files ({arima_filename}, {sarima_filename}, {rf_filename}, {xgb_filename}, {lstm_model_filename}) and {AAPL_DATA_FILE} are in the same directory. Error: {e}")
        st.stop()

models = load_all_models()

# --- 3. Load Original Data and Perform Feature Engineering ---
@st.cache_data
def load_and_preprocess_data():
    df_raw = pd.read_csv(AAPL_DATA_FILE)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)
    df_raw = df_raw.sort_values("Date")
    df_raw.set_index("Date", inplace=True)

    # Feature Engineering (as in the notebook)
    df_raw["MA07"] = df_raw["Close"].rolling(7).mean()
    df_raw["MA30"] = df_raw["Close"].rolling(30).mean()
    df_raw["Volatility"] = df_raw["Close"].rolling(10).std()
    df_raw["Daily_Returns"] = df_raw["Close"].pct_change()
    df_raw.dropna(inplace=True) # Drop NaNs introduced by rolling features

    return df_raw

df = load_and_preprocess_data()

# Constants from notebook
LOOKBACK = 60 # Lookback for LSTM model

# Re-initialize and fit scaler for LSTM within the app
# This is necessary because the scaler is no longer saved separately.
@st.cache_resource
def get_lstm_scaler(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[['Close']].values)
    return scaler

lstm_scaler = get_lstm_scaler(df)

# --- Model Performance Data ---
model_performance_df = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Random Forest', 'XGBoost', 'LSTM'],
    'RMSE': [33.386831, 26.726207, 32.313498, 32.540103, 8.712613],
    'MAPE': [0.115383, 0.105267, 0.090483, 0.091569, 0.034375]
})

# Sort for better visualization and identifying the best model
model_performance_df = model_performance_df.sort_values(by='RMSE', ascending=True)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Apple Stock Price Prediction Dashboard")

# --- Model Performance Visualization ---
st.subheader("Model Performance Comparison")

# Combined RMSE and MAPE Bar Chart
fig = go.Figure()

# Add RMSE bars
fig.add_trace(go.Bar(
    x=model_performance_df['Model'],
    y=model_performance_df['RMSE'],
    name='RMSE',
    marker_color='blue' # All RMSE bars are blue
))

# Add MAPE bars
fig.add_trace(go.Bar(
    x=model_performance_df['Model'],
    y=model_performance_df['MAPE'] * 100, # Multiply by 100 for percentage
    name='MAPE (%)',
    marker_color='darkblue' # All MAPE bars are darkblue
))

fig.update_layout(
    barmode='group',
    title_text='Model Performance: RMSE and MAPE',
    xaxis_title='Model',
    yaxis_title='Metric Value',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.write("### All Model Metrics:")
st.dataframe(model_performance_df.round(4))

best_model = model_performance_df.iloc[0]
st.write(
    f"##### The best performing model (based on lowest RMSE and MAPE) is **{best_model['Model']}** "
    f"with RMSE: {best_model['RMSE']:.2f} and MAPE: {best_model['MAPE']:.4f}"
)

st.sidebar.header("Prediction Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model for Forecast:",
    ('ARIMA', 'SARIMA', 'Random Forest', 'XGBoost', 'LSTM')
)
prediction_horizon = st.sidebar.slider("Prediction Horizon (days):", 1, 90, 30)

# Removed slider for ML model confidence range

st.write(f"### Using {selected_model_name} to predict the next {prediction_horizon} days of Apple Stock Prices")

# --- 5. Prediction Logic ---
forecast_prices = None
conf_int = None

# Create future dates based on the last date in the preprocessed DataFrame
future_dates = pd.date_range(
    start=df.index[-1] + timedelta(days=1),
    periods=prediction_horizon,
    freq="B" # Business days
)

# Get the last 100 historical prices to display with the forecast
historical_prices_for_plot = df["Close"].tail(100)

if st.button("Generate Forecast"):
    with st.spinner(f"Generating forecast using {selected_model_name}..."):
        if selected_model_name == 'ARIMA':
            # ARIMA model is already trained on the full 'Close' series
            forecast_res = models['ARIMA'].get_forecast(steps=prediction_horizon)
            forecast_prices = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

        elif selected_model_name == 'SARIMA':
            # SARIMA model is already trained on the full 'Close' series
            forecast_res = models['SARIMA'].get_forecast(steps=prediction_horizon)
            forecast_prices = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

        elif selected_model_name in ['Random Forest', 'XGBoost']:
            ml_features = ['Open', 'High', 'Low', 'Volume', 'MA07', 'MA30', 'Volatility', 'Daily_Returns']

            # Create a DataFrame for features up to the last known point
            current_ml_df = df[ml_features].copy()

            # To forecast future steps, we need to iteratively generate future features.
            # This is a simplification and assumes features like Open, High, Low, Volume,
            # MA07, MA30, Volatility, Daily_Returns can be approximated or extrapolated.
            # For this example, we'll use the last available row's features as a proxy
            # for the prediction horizon, or a more sophisticated approach if context allows.

            # For a more robust solution, these features would need to be forecast themselves.
            # For now, we'll repeat the last known feature set for simplicity in this Streamlit app.
            last_known_features = current_ml_df.iloc[-1].to_dict()
            future_ml_features_df = pd.DataFrame([last_known_features] * prediction_horizon, index=future_dates)

            if selected_model_name == 'Random Forest':
                forecast_prices = models['Random Forest'].predict(future_ml_features_df)
            else: # XGBoost
                forecast_prices = models['XGBoost'].predict(future_ml_features_df)

            forecast_prices = pd.Series(forecast_prices, index=future_dates)
            conf_int = None # Tree-based models don't directly provide confidence intervals this way

        elif selected_model_name == 'LSTM':
            # Get the scaled 'Close' prices from the entire dataset
            scaled_close_data = lstm_scaler.transform(df[['Close']].values)

            # Use the last LOOKBACK days from the scaled data for the initial sequence
            last_sequence = scaled_close_data[-LOOKBACK:]

            future_predictions_scaled = []
            current_sequence = last_sequence.copy()

            for _ in range(prediction_horizon):
                current_sequence_reshaped = current_sequence.reshape((1, LOOKBACK, 1))
                next_pred_scaled = models['LSTM'].predict(current_sequence_reshaped, verbose=0)
                future_predictions_scaled.append(next_pred_scaled[0, 0])

                # Slide window: append the new prediction and remove the oldest value
                current_sequence = np.append(current_sequence[1:], next_pred_scaled, axis=0)

            # Inverse transform to get actual prices
            forecast_prices = lstm_scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()
            forecast_prices = pd.Series(forecast_prices, index=future_dates)
            conf_int = None # LSTM typically doesn't directly provide confidence intervals this way


    # --- 6. Visualization with Plotly ---
    if forecast_prices is not None:
        fig = go.Figure()

        # Plot historical prices
        fig.add_trace(go.Scatter(
            x=historical_prices_for_plot.index,
            y=historical_prices_for_plot,
            mode='lines',
            name='Historical Prices',
            line=dict(color='yellow')
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_prices,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

        # Plot confidence interval if available (only for ARIMA/SARIMA in this setup)
        if conf_int is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int.iloc[:, 0], # Lower bound
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int.iloc[:, 1], # Upper bound
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                name='Confidence Interval'
            ))

        fig.update_layout(
            title=f"Apple Stock Price - {prediction_horizon} Day Forecast ({selected_model_name})",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecasted Prices:")

        # Create a detailed table for forecasted prices
        if conf_int is not None:
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Price': forecast_prices.round(2),
                'Lower Bound': conf_int.iloc[:, 0].round(2),
                'Upper Bound': conf_int.iloc[:, 1].round(2)
            })
        else:
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Price': forecast_prices.round(2)
            })

        forecast_df = forecast_df.set_index('Date')
        st.dataframe(forecast_df)

        # --- Forecast Summary Block ---
        st.subheader("Forecast Summary:")
        current_price = df["Close"].iloc[-1]
        predicted_end_price = forecast_prices.iloc[-1]
        expected_change = predicted_end_price - current_price
        expected_change_percent = (expected_change / current_price) * 100
        avg_predicted_price = forecast_prices.mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Current Price", value=f"${current_price:.2f}")
        with col2:
            st.metric(label=f"Predicted ({prediction_horizon} days)", value=f"${predicted_end_price:.2f}")
        with col3:
            st.metric(
                label="Expected Change",
                value=f"${expected_change:.2f}",
                delta=f"{expected_change_percent:.2f}%"
            )
        with col4:
            st.metric(label="Avg Predicted Price", value=f"${avg_predicted_price:.2f}")

    else:
        st.warning("Please select a model and click 'Generate Forecast'.")
