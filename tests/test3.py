# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:22:07 2023

@author: aidan
"""

import streamlit as st
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(layout="wide")
stock_data = pd.read_csv('NFLX.csv',index_col='Date')

trace1 = go.Scatter(y=df['ax'], x=df['time_stamp'], mode='lines', name='AccX')
trace2 = go.Scatter(y=df['az'], x=df['time_stamp'], mode='lines', name='AccZ')
data = [trace1, trace2]
fig = go.FigureWidget(data=data)
scatter = fig.data[0]
fig.update_layout(
        width=1000,
        height=500,
        xaxis_title='Time_Stamp',
        yaxis_title='Values',
        dragmode="select",
        selectdirection="h",
    )
st.plotly_chart(fig)
