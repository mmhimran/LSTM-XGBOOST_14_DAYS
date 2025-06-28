import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import io
import plotly.express as px

# Page Config
st.set_page_config(page_title="14-Day Offshore Temperature Forecast", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hybrid_lstm_all.h5")
model = load_model()

# Forecasting Function
def predict_temperature(df_input, lookback=1008, forecast_horizon=336):
    df_input = df_input.fillna(method='ffill')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_input[['Layer 1', 'Layer 2', 'Layer 3']])
    X_input = np.array([scaled[-lookback:]])
    predicted_scaled = model.predict(X_input).reshape(forecast_horizon, 3)
    predicted = scaler.inverse_transform(predicted_scaled)
    forecast_dates = pd.date_range(start=df_input['Date'].iloc[-1] + timedelta(hours=1), periods=forecast_horizon, freq='H')
    return pd.DataFrame({'Date': forecast_dates, 'Pred_Layer 1': predicted[:, 0], 'Pred_Layer 2': predicted[:, 1], 'Pred_Layer 3': predicted[:, 2]})

# Compare Actual vs Predicted
def compare_with_actual(df_actual, df_predicted):
    merged = pd.merge(df_actual, df_predicted, on='Date')
    for i in range(1, 4):
        merged[f'Error_Layer {i}'] = merged[f'Layer {i}'] - merged[f'Pred_Layer {i}']
        merged[f'Accuracy_Layer {i} (%)'] = 100 - (np.abs(merged[f'Error_Layer {i}']) / merged[f'Layer {i}']) * 100
    return merged

# Plot Function
def plot_layer(df, layer):
    fig = px.line(df, x='Date', y=[f'Layer {layer}', f'Pred_Layer {layer}'],
                  labels={'value': 'Temperature (°C)', 'variable': 'Legend'},
                  title=f"Actual vs Predicted for Layer {layer}",
                  color_discrete_sequence=["blue", "red"])
    fig.update_traces(line=dict(width=4), hovertemplate='<b>%{y:.2f}</b>')
    fig.update_layout(font=dict(family="Times New Roman", size=22, color="black"),
                      plot_bgcolor='white', paper_bgcolor='white')
    return fig

# Export to Excel
def export_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparison')
    return output.getvalue()

# ----------------- UI -------------------
st.title("📊 14-Day Offshore Temperature Forecast Dashboard")

file = st.file_uploader("Upload your .xlsx file (must contain 1008 + 336 rows for forecast)", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    df['Date'] = pd.to_datetime(df['Date'])

    if len(df) < 1344:
        st.error("File must contain at least 1344 rows (1008 for input + 336 for comparison)")
    else:
        input_df = df.iloc[:1008].copy()
        actual_df = df.iloc[1008:1344].copy()

        st.success("✅ Forecasting started...")
        predicted_df = predict_temperature(input_df)
        comparison_df = compare_with_actual(actual_df, predicted_df)

        # Show plots
        for i in range(1, 4):
            st.plotly_chart(plot_layer(comparison_df, i), use_container_width=True)

        # Show RMSE and MAE
        st.subheader("📈 Evaluation Metrics")
        for i in range(1, 4):
            rmse = mean_squared_error(comparison_df[f'Layer {i}'], comparison_df[f'Pred_Layer {i}'], squared=False)
            mae = mean_absolute_error(comparison_df[f'Layer {i}'], comparison_df[f'Pred_Layer {i}'])
            st.write(f"**Layer {i}:** RMSE = {rmse:.4f}, MAE = {mae:.4f}")

        # Download Excel
        st.download_button("📥 Download Result Excel", export_excel(comparison_df),
                           file_name="Comparison_Actual_vs_Predicted.xlsx")
