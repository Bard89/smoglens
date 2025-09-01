import pandas as pd
import plotly.graph_objects as go
from typing import Optional

def create_prediction_chart(
    predictions_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    
    fig = go.Figure()
    
    if not predictions_df.empty:
        fig.add_trace(go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        if 'actual' in predictions_df.columns:
            valid_actuals = predictions_df.dropna(subset=['actual'])
            if not valid_actuals.empty:
                fig.add_trace(go.Scatter(
                    x=valid_actuals['timestamp'],
                    y=valid_actuals['actual'],
                    mode='markers',
                    name='Actual',
                    marker=dict(color='red', size=10, symbol='circle')
                ))
    
    if historical_df is not None and not historical_df.empty:
        fig.add_trace(go.Scatter(
            x=historical_df['timestamp'],
            y=historical_df['pm25_ugm3_mean'],
            mode='lines',
            name='Historical',
            line=dict(color='gray', width=1, dash='dash')
        ))
    
    fig.add_hline(y=12, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Good")
    fig.add_hline(y=35, line_dash="dot", line_color="yellow", opacity=0.5, annotation_text="Moderate")
    fig.add_hline(y=55, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="Unhealthy for Sensitive")
    
    fig.update_layout(
        title='PM2.5 Predictions vs Actual Values',
        xaxis_title='Time',
        yaxis_title='PM2.5 (μg/m³)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig