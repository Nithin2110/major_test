def load_data(op):
    if op:
        option=str(op)
        df=pd.read_csv("DATA/Backup/"+option+".csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
    shift=1
    for i in range(shift,0,-1):
        df['target-'+str(i)] = df['price'].shift(i)
    d=df[shift:]
    dates, X, y = windowed_df_to_date_X_y(d)
    dates.shape, X.shape, y.shape
    return dates, X, y, df, scaler
    
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 2:]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, 1:2]
    return dates, X.astype(np.float32), Y.astype(np.float32)

    
def load(option):
    l=['rice','wheat','maize','potato','onion']
    s=l.index(option)
    bestperms=[{'lstm_units': 176, 'dense_units': 46, 'num_lstm_layers': 3, 'learning_rate': 0.001},{'lstm_units': 132, 'dense_units': 81, 'num_lstm_layers': 2, 'learning_rate': 0.001}, {'lstm_units': 167, 'dense_units': 123, 'num_lstm_layers': 2, 'learning_rate': 0.001},{'lstm_units': 178, 'dense_units': 38, 'num_lstm_layers': 3, 'learning_rate': 0.001},{'lstm_units': 101, 'dense_units': 87, 'num_lstm_layers': 2, 'learning_rate': 0.001}]
    lstm_units=bestperms[s][lstm_units]
    dense_units=bestperms[s][dense_units]
    num_lstm_layers=bestperms[s][num_lstm_layers]
    learning_rate=bestperms[s][learning_rate]
    model= Sequential()
    model.add(layers.Input((1, 1)))
    for _ in range(num_lstm_layers):
        model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(loss='mse', 
                optimizer=Adam(learning_rate=best_learning_rate),
                metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)
    return model

def predict(X,y,m1,scaler,number_input):
    n = number_input
    forecast = []
    last_data = X[-1]  
    for i in range(n):
        next_y = model.predict(last_data.reshape(1, X.shape[1], X.shape[2]))  
        forecast.append(next_y)
        last_data = np.append(last_data[1:], next_y)  
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    y_act=scaler.inverse_transform(np.array(y).reshape(-1, 1))
    return y_act,forecast
        




dates,X,y,df,scaler=load_data(option)
'''q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
    '''
    
    '''m1 = load(option,X_val,y_val)
    y_act,forecast=predict(X,y,m1,scaler,number_input)

    # User input for forecasting
    st.subheader('Forecasted Prices')
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_act, label='Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Original Data and Forecast')
    plt.xticks(rotation=45)
    plt.legend()
    forecast_dates = np.arange(len(y_act), len(y_act) + n)
    plt.plot(forecast_dates, forecast, label='Forecast', color='red')
    plt.legend()
    st.pyplot()'''
    #n_days = st.slider('Select number of days to forecast', min_value=1, max_value=30, value=7)

    # Forecast next n days
    #forecast = forecast_next_n_days(lstm_model, X_train[-1], n_days)

    # Display forecast
    
    #st.write(forecast)
    #fig, ax = plt.subplots()
    #ax.plot(forecast)
    #ax.set_xlabel('Historical')
    #ax.set_ylabel('forecast')
    #st.pyplot(fig)
