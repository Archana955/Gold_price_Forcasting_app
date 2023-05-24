# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:24:11 2023

@author: Adith
"""

import pandas as pd 

import streamlit as st 
import pickle
import plotly.graph_objs as go
import plotly.express as px
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore', message='X does not have valid feature names')
  
loaded_model = pickle.load(open('final_model1.sav'))

df_gold1 = pd.read_csv("gold_ext.csv")
df_stock = pd.read_csv("stock_price_gold.csv")


page = st.sidebar.radio("Navigate",("Home","Forecast","custom"))
   
if page == 'Home':
    
    
    
    st.header("Gold Forecasting App")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.write("Overview")
    st.write("This App constitutes Data of Gold prices from the year 2016 - 2021,Based on the historical data this app forecast the prices for future,This app will also help investors and companies associated with investment of gold")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("Gold Data(2016 - 2021) :")
    st.dataframe(df_gold1)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("Data from (2016 - 2021)")
    fig = px.line(df_gold1,x = 'date',y = 'price',labels={'x':'date','y':'price'})
    st.plotly_chart(fig)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.header("Monthwise price of gold")
    
    fig = go.Figure(data=[go.Candlestick(x=df_stock['date'],
                open=df_stock['open'],
                high=df_stock['high'],
                low=df_stock['low'],
                close=df_stock['close'])])
    fig.update_layout(xaxis_rangeslider_visible=False)

    
    st.plotly_chart(fig)
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.header("Quick Future Forecasts")
    
    tab1,tab2,tab3,tab4 = st.tabs(["1M","3M","6M","1Y"])
    with tab1:
        st.header("1Month")
        m_Days = loaded_model.forecast(30)
        m_Days = pd.DataFrame({'Price':m_Days})
       
        st.write(m_Days)
        st.write(m_Days.describe().T)
        st.line_chart(m_Days)
    with tab2:
        st.header("3Months")
        nin_day = loaded_model.forecast(90)
        nin_day = pd.DataFrame({'Price':nin_day})
       
        st.write(nin_day)
        st.write(nin_day.describe().T)
        st.line_chart(nin_day)
    with tab3:
        st.header("6Months")
        six_mon = loaded_model.forecast(180)
        six_mon = pd.DataFrame({'Price':six_mon})
       


        st.write(six_mon)
        st.write(six_mon.describe().T)
        st.line_chart(six_mon)

    with tab4:
        st.header("1Y")
        year_one = loaded_model.forecast(365)
        year_one = pd.DataFrame({'Price':year_one})

        st.write(year_one)
        st.write(year_one.describe().T)

        st.line_chart(year_one)
  
        
if page == 'Forecast':
   
    
    days = st.sidebar.number_input("Add number of periods", min_value=(1),max_value=(10000))
    
    forecast = st.sidebar.button("Forecast")
    
    if forecast:
        forecast = loaded_model.forecast(days)
        forecast_df = pd.DataFrame({'Price':forecast})
       
       
        
        
        
        st.write(forecast_df)
        st.write(forecast_df.describe().T)
        #plotting for next 30 days 
        plt.figure(figsize=(10,4))
        plt.plot(forecast_df)
        plt.xlabel("date",fontsize = 12)
        plt.ylabel("price",fontsize = 12)
       
       
        plt.xticks(rotation =45)
        plt.tight_layout()
        plt.grid()
        
        plt.show()
        st.pyplot()
        
elif page == "custom":
   

       
    st.sidebar.title("upload custom data")
    uploaded_file = st.sidebar.file_uploader("Add csv file",type="csv")
    st.sidebar.error("Input takes only two dimensions (ie date,price)")
    data = pd.DataFrame()
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.lower()
        formats = ["%m/%d/%Y","%Y-%m-%d"]
        for fmt in formats:
            try:
                
                data['date'] = pd.to_datetime(data['date'], format=fmt)
                break
            except ValueError:
                pass
        data = data.set_index('date')
        data = data.resample('D').mean()
        data = data.interpolate(method = 'linear')
       
       
    #fitting our model on user data
    if not data.empty:
        
        texp_mul_mul_model_fin = ExponentialSmoothing(data,trend = 'mul',seasonal='mul',seasonal_periods=30).fit(optimized=True)
        periods = st.sidebar.number_input("add number of periods",min_value=1,max_value=10000)
        Prediction = st.sidebar.button("Predict")
        if Prediction:
        
            custom_forecast = texp_mul_mul_model_fin.forecast(periods)
            custom_forecast = pd.DataFrame(custom_forecast,columns=data.columns)
            st.dataframe(custom_forecast)
            st.dataframe(custom_forecast.describe().T)
            plt.figure(figsize=(10,4))
            plt.plot(custom_forecast)
           
            plt.ylabel("price",fontsize = 12)
           
           
            plt.xticks(rotation =45)
            plt.tight_layout()
            plt.grid()
            
            plt.show()
            st.pyplot()
    else:
            
        st.sidebar.error("No data uploaded")
    

    
    
    
    
    
       
   

   
   
   
   


    
    

