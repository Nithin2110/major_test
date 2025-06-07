import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from prepdata import *
from getmodel import *

com={
    'rice': [
        {'lstm_units': 195, 'dense_units': 73, 'learning_rate': 0.001},
        {'lstm_units': 195, 'dense_units': 88, 'learning_rate': 0.001},
        {'lstm_units': 176, 'dense_units': 46, 'num_lstm_layers': 3, 'learning_rate': 0.001},
        {'gru_units': 64, 'dense_units': 32, 'learning_rate': 0.001},
        {'num_filters': 78, 'kernel_size': 4, 'dense_units': 40, 'learning_rate': 0.001}
    ],
    'wheat': [
        {'lstm_units': 153, 'dense_units': 115, 'learning_rate': 0.001},
        {'lstm_units': 219, 'dense_units': 110, 'learning_rate': 0.001},
        {'lstm_units': 132, 'dense_units': 81, 'num_lstm_layers': 2, 'learning_rate': 0.001},
        {'gru_units': 256, 'dense_units': 51, 'learning_rate': 0.001},
        {'num_filters': 25, 'kernel_size': 6, 'dense_units': 91, 'learning_rate': 0.001} 
    ],
    'maize': [
        {'lstm_units': 209, 'dense_units': 107, 'learning_rate': 0.001},
        {'lstm_units': 221, 'dense_units': 77, 'learning_rate': 0.001},
        {'lstm_units': 167, 'dense_units': 123, 'num_lstm_layers': 2, 'learning_rate': 0.001},
        {'gru_units': 160, 'dense_units': 54, 'learning_rate': 0.001},
        {'num_filters': 112, 'kernel_size': 5, 'dense_units': 124, 'learning_rate': 0.001}
    ],
    'potato': [
        {'lstm_units': 32, 'dense_units': 96, 'learning_rate': 0.001},
        {'lstm_units': 32, 'dense_units': 77, 'learning_rate': 0.001},
        {'lstm_units': 178, 'dense_units': 38, 'num_lstm_layers': 3, 'learning_rate': 0.001},
        {'gru_units': 135, 'dense_units': 17, 'learning_rate': 0.001},
        {'num_filters': 64, 'kernel_size': 3, 'dense_units': 46, 'learning_rate': 0.001}
    ],
    'onion': [
        {'lstm_units': 102, 'dense_units': 99, 'learning_rate': 0.001},
        {'lstm_units': 183, 'dense_units': 70, 'learning_rate': 0.001},
        {'lstm_units': 101, 'dense_units': 87, 'num_lstm_layers': 2, 'learning_rate': 0.001},
        {'gru_units': 145, 'dense_units': 51, 'learning_rate': 0.001},
        {'num_filters': 42, 'kernel_size': 6, 'dense_units': 107, 'learning_rate': 0.001}
    ]


}
index={'LSTM':1,'Bi-LSTM':2,'Stacked-LSTM':3,'GRU':4,'1D-CNN':5}
def load(commodity,model,days):
    bestperms=com[commodity][index[model]-1]
    st.write(bestperms)
    dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scalar,dates,X,y=preparedata(commodity,model)
    m1=generateModel(model,bestperms,X_train,y_train,X_val,y_val,X_test,y_test,shift,scalar)
    predict(m1,bestperms,dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scalar)
    predictfuture(days,m1,bestperms,dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scalar,dates,X,y)



