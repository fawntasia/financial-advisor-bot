"""
Financial Advisor Bot - LSTM Stock Price Prediction Prototype
A Streamlit application demonstrating ML-based stock price prediction.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# New Architecture Imports
import sys
sys.path.insert(0, '.')
from src.data.stock_data import StockDataProcessor
from src.models.lstm_wrapper import LSTMStockPredictor

MODEL_PATH = "lstm_model.pth"

# Page configuration
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def calculate_metrics(y_true, y_pred):
    """Calculate error metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae}


def create_price_chart(df, predictions_df=None, future_df=None):
    """Create an interactive stock price chart with predictions."""
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#667eea', width=2)
    ))
    
    # Predicted prices on test set (if available)
    if predictions_df is not None:
        fig.add_trace(go.Scatter(
            x=predictions_df.index,
            y=predictions_df['Predicted'],
            mode='lines',
            name='Model Fit (Test Set)',
            line=dict(color='#f093fb', width=2, dash='dash')
        ))
    
    # Future forecast (if available)
    if future_df is not None:
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['Forecast'],
            mode='lines+markers',
            name='Future Forecast',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='AAPL Stock Price - Actual vs Predicted vs Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 LSTM Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header(" Settings")
        ticker = st.text_input("Stock Ticker", value="AAPL", disabled=True)
        st.caption("Prototype is configured for AAPL")
        
        st.markdown("---")
        st.header(" Model Parameters")
        model_type = st.selectbox("Model Type", ["LSTM (PyTorch)"]) # Extensible!
        sequence_length = st.slider("Sequence Length (days)", 30, 90, 60)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        forecast_days = st.slider("Forecast Days Ahead", 5, 60, 30)
        
        st.markdown("---")
        train_button = st.button(" Train Model", use_container_width=True)
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    # Initialize Processor
    processor = StockDataProcessor(ticker)
    
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        try:
            df = processor.fetch_data(years=5)
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
            price_change_pct = (price_change / df['Close'].iloc[-2]) * 100
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return
    
    # Display current metrics
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Data Points",
            value=f"{len(df):,}",
            delta="5 years"
        )
    
    with col3:
        model_status = "Trained" if (st.session_state.model_trained or os.path.exists(MODEL_PATH)) else "⏳ Not Trained"
        st.metric(label="Model Status", value=model_status)
    
    st.markdown("---")
    
    # Train model if button clicked
    if train_button:
        with st.spinner("Preparing data..."):
            # Use Processor
            X_train, y_train, X_test, y_test, scaler, df = processor.prepare_for_lstm(df, sequence_length)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training LSTM model...")
        progress_bar.progress(10)
        
        # Initialize and Train Model
        model = LSTMStockPredictor()
        
        with st.spinner("Training model (this may take a few minutes)..."):
            history = model.train(X_train, y_train, X_test, y_test, epochs=epochs)
            model.save(MODEL_PATH)
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Invert scaling (handling externally to model)
        predictions = scaler.inverse_transform(predictions)
        
        # Get actual values for comparison
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        metrics = calculate_metrics(actual, predictions)
        
        # Store in session state
        st.session_state.model_trained = True
        st.session_state.predictions = predictions
        st.session_state.metrics = metrics
        st.session_state.actual = actual
        st.session_state.scaler = scaler
        st.session_state.df = df
        st.session_state.X_test = X_test
        st.session_state.sequence_length = sequence_length
        st.session_state.forecast_days = forecast_days
        
        # Create predictions dataframe for plotting
        test_dates = df.index[-len(predictions):]
        predictions_df = pd.DataFrame({
            'Predicted': predictions.flatten()
        }, index=test_dates)
        st.session_state.predictions_df = predictions_df
        
        # Generate future forecast
        latest_sequence = processor.get_latest_sequence(df, sequence_length)
        last_price = df['Close'].iloc[-1]
        recent_prices = df['Close'].values[-20:]
        
        # Use Model's forecast method
        # NOTE: We must pass the FEATURE scaler (processor.scaler), not the passed 'scaler' 
        # (which is likely the target_scaler returned by prepare_for_lstm)
        future_prices = model.forecast(latest_sequence, processor.scaler, forecast_days, 
                                       last_actual_price=last_price, recent_prices=recent_prices)
        
        # Create future dates
        from datetime import timedelta
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_df = pd.DataFrame({
            'Forecast': future_prices
        }, index=future_dates)
        st.session_state.future_df = future_df
        
        st.success("Model trained successfully!")
        st.rerun()
    
    # Display chart
    st.subheader("📊 Price Chart")
    
    predictions_df = st.session_state.get('predictions_df', None)
    future_df = st.session_state.get('future_df', None)
    fig = create_price_chart(df, predictions_df, future_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics if model is trained
    if st.session_state.model_trained and st.session_state.metrics:
        st.markdown("---")
        st.subheader("📈 Model Performance")
        
        metrics = st.session_state.metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="RMSE (Root Mean Square Error)",
                value=f"${metrics['rmse']:.2f}",
                help="Lower is better. Average prediction error in dollars."
            )
        
        with col2:
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"${metrics['mae']:.2f}",
                help="Lower is better. Average absolute prediction error."
            )
        
        with col3:
            # Calculate MAPE
            actual = st.session_state.actual
            predictions = st.session_state.predictions
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            st.metric(
                label="MAPE",
                value=f"{mape:.2f}%",
                help="Mean Absolute Percentage Error"
            )
        
        # Next day prediction
        st.markdown("---")
        st.subheader("🔮 Next Day Prediction")
        
        # Load model for inference
        model = LSTMStockPredictor()
        try:
            model.load(MODEL_PATH)
            
            scaler = st.session_state.scaler
            latest_sequence = processor.get_latest_sequence(df, st.session_state.sequence_length)
            
            # Make prediction
            next_day_pred_scaled = model.predict(latest_sequence)
            next_day_price = scaler.inverse_transform(next_day_pred_scaled)[0][0]
            
            predicted_change = next_day_price - current_price
            predicted_change_pct = (predicted_change / current_price) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Predicted Next Day Price",
                    value=f"${next_day_price:.2f}",
                    delta=f"{predicted_change_pct:.2f}%"
                )
            
            with col2:
                direction = "📈 Bullish" if predicted_change > 0 else "📉 Bearish"
                st.metric(label="Predicted Direction", value=direction)
                
        except Exception:
             st.info("Train the model to see next day predictions.")
             
    # Load existing model if available but not in session
    elif os.path.exists(MODEL_PATH) and not st.session_state.model_trained:
        st.info("Click 'Train Model' in the sidebar to train the LSTM model and see predictions.")
    else:
        st.info("Click 'Train Model' in the sidebar to train the LSTM model and see predictions.")
    
    # Footer
    st.markdown("---")
    st.caption("**Disclaimer**: This is a prototype for educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()
