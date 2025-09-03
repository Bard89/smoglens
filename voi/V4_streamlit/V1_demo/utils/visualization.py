import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_activity_prediction_chart(predictions, activity_threshold, activity_name):
    fig = go.Figure()
    
    safe_color = '#10B981'
    unsafe_color = '#EF4444'
    colors = [safe_color if val < activity_threshold else unsafe_color 
              for val in predictions['pm25']]
    
    fig.add_trace(go.Scatter(
        x=predictions['time'],
        y=predictions['pm25'],
        mode='markers+lines',
        name='<b>PM2.5 Level</b>',
        marker=dict(
            size=10,
            color=colors,
            line=dict(color='white', width=1.5)
        ),
        line=dict(color='#334155', width=2.5),
        hovertemplate='<b>%{x|%H:%M}</b><br>PM2.5: %{y:.1f} μg/m³<extra></extra>'
    ))
    
    if 'std_error' in predictions.columns and predictions['std_error'].iloc[1] > 0:
        upper_bound = predictions['pm25'] + 1.96 * predictions['std_error']
        lower_bound = (predictions['pm25'] - 1.96 * predictions['std_error']).clip(lower=0)
        
        fig.add_trace(go.Scatter(
            x=predictions['time'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['time'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='<b>95% Confidence</b>',
            fillcolor='rgba(100, 116, 139, 0.15)',
            hoverinfo='skip'
        ))
    
    fig.add_hline(
        y=activity_threshold,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"  {activity_name} Limit: {activity_threshold} μg/m³",
        annotation_position="right",
        annotation=dict(
            font=dict(size=14, color='#DC2626', family='system-ui, -apple-system, sans-serif'),
            bgcolor='rgba(255, 255, 255, 0.95)',
            borderpad=6
        )
    )
    
    fig.add_hrect(
        y0=0,
        y1=activity_threshold,
        fillcolor=safe_color,
        opacity=0.08,
        layer="below",
        line_width=0
    )
    
    y_max = max(predictions['pm25'].max() * 1.3, activity_threshold * 1.5, 50)
    
    fig.add_hrect(
        y0=activity_threshold,
        y1=y_max,
        fillcolor=unsafe_color,
        opacity=0.08,
        layer="below",
        line_width=0
    )
    
    fig.add_annotation(
        x=predictions['time'].iloc[len(predictions)//3],
        y=activity_threshold * 0.5,
        text="SAFE",
        showarrow=False,
        font=dict(size=16, color=safe_color, family='system-ui, -apple-system, sans-serif', weight=600),
        opacity=0.5
    )
    
    fig.add_annotation(
        x=predictions['time'].iloc[len(predictions)//3],
        y=min(activity_threshold + (y_max - activity_threshold) * 0.3, y_max - 5),
        text="UNSAFE",
        showarrow=False,
        font=dict(size=16, color=unsafe_color, family='system-ui, -apple-system, sans-serif', weight=600),
        opacity=0.5
    )
    
    fig.update_layout(
        title=None,
        xaxis=dict(
            title=dict(
                text='Time (Tokyo)',
                font=dict(size=16, family='system-ui, -apple-system, sans-serif', color='#111827', weight=500)
            ),
            tickformat='%H:%M',
            dtick=3600000,
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.08)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=14, family='system-ui, -apple-system, sans-serif', color='#1F2937'),
            linecolor='#9CA3AF',
            linewidth=2
        ),
        yaxis=dict(
            title=dict(
                text='PM2.5 Concentration (μg/m³)',
                font=dict(size=16, family='system-ui, -apple-system, sans-serif', color='#111827', weight=500)
            ),
            range=[0, y_max],
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.08)',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#9CA3AF',
            zerolinewidth=2,
            tickfont=dict(size=14, family='system-ui, -apple-system, sans-serif', color='#1F2937'),
            linecolor='#9CA3AF',
            linewidth=2
        ),
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor='#9CA3AF',
            borderwidth=1,
            font=dict(size=14, family='system-ui, -apple-system, sans-serif', color='#1F2937')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(family='system-ui, -apple-system, sans-serif')
    )
    
    fig.update_traces(hoverlabel=dict(
        bgcolor='white',
        font_size=14,
        font_family='system-ui, -apple-system, sans-serif',
        bordercolor='#E5E7EB'
    ))
    
    return fig

