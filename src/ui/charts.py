import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartGenerator:
    """Generator for interactive financial charts using Plotly."""
    
    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    def create_ohlcv_chart(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Creates a candlestick chart with OHLCV data and technical indicators.
        Expected columns in df: ['Open', 'High', 'Low', 'Close', 'Volume']
        Technical indicators (if present): ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'BB_Middle']
        """
        # Create subplots: 1. Candlestick + BB, 2. Volume, 3. MACD, 4. RSI
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f'{ticker} Price & BB', 'Volume', 'MACD', 'RSI'),
            row_width=[0.2, 0.2, 0.1, 0.5]
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Bollinger Bands (Overlays)
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='BB Lower', fill='tonexty'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], line=dict(color='orange', width=1), name='BB Middle'), row=1, col=1)

        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='gray'),
            row=2, col=1
        )

        # MACD
        if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='red', width=1.5), name='Signal'), row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Hist', marker_color='silver'), row=3, col=1)

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=4, col=1)
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1) # type: ignore
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1) # type: ignore

        # Layout updates
        fig.update_layout(
            template=self.theme,
            title=f'Technical Analysis: {ticker}',
            xaxis_rangeslider_visible=False,
            height=900,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        return fig

    def create_sentiment_chart(self, sentiment_df: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Creates a sentiment timeline chart.
        Expected columns: ['Date', 'SentimentScore']
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['Date'], 
                y=sentiment_df['SentimentScore'],
                mode='lines+markers',
                line=dict(color='cyan', width=2),
                name='Sentiment'
            )
        )
        
        fig.update_layout(
            template=self.theme,
            title=f'Sentiment Timeline: {ticker}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score (-1 to 1)',
            height=400,
            yaxis=dict(range=[-1.1, 1.1])
        )
        
        return fig
