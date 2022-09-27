#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# ## Group members: 
#            Minh Ta & Varun Ramakrishnan
# 
# ## Central Question:
#           This report delves into two potential cryptocurrencies and seeks to identify an optimal investment in terms of the stock market. 
#     
# ## Purpose: 
#            To see the fluctuations of two cryptocurrencies for investment opportunity
# ## Data: 
#            This report enlists a stock market website which stores accurate market data pertaining to the cryptocurrencies.
#         
# 
# ##### coin_Cardano, coin_Ethereum from SRK Kaggle
# ##### https://coinmarketcap.com/currencies/ethereum/
# ##### https://coinmarketcap.com/currencies/cardano/

# ### 1. Data extraction and Cleaning
# 
# In this part, we want to collect all the data needed for our analysis and prediction. The first method we use is to have two csv file from Kaggle that documented Cardano and Ethereum cryptocurrencies price in the past and transform it to database format so that we could perfrom SQL queries on those. The second source we use is from an online site call coinmarketcap to see live data on the same two coin that we are doing analysis on.

# In[70]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sqlalchemy import create_engine

Cardano_data = pd.read_csv('coin_Cardano.csv')   
Cardano_dataset = pd.DataFrame(Cardano_data)
Cardano_dataset = Cardano_dataset.dropna()
Cardano_dataset = Cardano_dataset.drop(['Name', 'SNo'], axis = 1)
Cardano_dataset['Date'] = pd.to_datetime(Cardano_dataset['Date']).dt.date
engine = create_engine('sqlite://', echo = False)
Cardano_dataset.to_sql(name = 'Cardano_Data', con = engine)
Cardano_sql = engine.execute("SELECT High, Low, High - Low AS Differences FROM Cardano_Data ORDER BY Differences DESC").fetchall()


CDN_sql_pd = pd.read_sql("""
SELECT High, Low, High - Low AS Differences FROM Cardano_Data ORDER BY Differences DESC
;""", con = engine)


CDN_sql_pd['Differences'] = CDN_sql_pd.astype({'Differences': int})
CDN_sql_pd


# In[81]:


Ethereum_data = pd.read_csv('coin_Ethereum.csv')   
Ethereum_dataset = pd.DataFrame(Ethereum_data)
Ethereum_dataset = Ethereum_dataset .dropna()
Ethereum_dataset = Ethereum_dataset .drop(['Name', 'SNo'], axis = 1)
Ethereum_dataset ['Date'] = pd.to_datetime(Ethereum_dataset['Date']).dt.date
engine = create_engine('sqlite://', echo = False)
Ethereum_dataset.to_sql(name = 'Ethereum_Data', con = engine)
Ethereum_sql = engine.execute("SELECT High, Low, High - Low AS Differences FROM Ethereum_Data ORDER BY Differences DESC").fetchall()

ETH_sql_pd = pd.read_sql("""
SELECT High, Low, High - Low AS Differences FROM Ethereum_Data ORDER BY Differences DESC
;""", con = engine)

ETH_sql_pd['Differences'] = ETH_sql_pd.astype({'Differences': int})
ETH_sql_pd


# The first two block just transform the data to a SQL format and then we want to find the difference within a day of transaction which we later use for our prediction model.

# In[95]:


web_Ethereum = pd.read_html('https://coinmarketcap.com/currencies/ethereum/')
web_Cardano = pd.read_html('https://coinmarketcap.com/currencies/cardano/')
#Purpose of this part is to read in the data from the online site
web_Ethereum
web_Cardano


# ### 2. Visualization
# 

# In[83]:


print(web_Ethereum[0].head(15))
web_Ethereum[0].to_csv('lsr.csv')
web_Ethereum_df = pd.read_csv('lsr.csv',header =None)
web_Ethereum_df.plot()
plt.show


print(web_Cardano[0].head(15))
web_Cardano[0].to_csv('lsrr.csv')
web_Cardano_df = pd.read_csv('lsrr.csv')
web_Cardano_df.plot()
plt.show()


# In[84]:


Cardano_data.plot(figsize=(10,4))
plt.axhline(0,color="black", lw=1)
plt.ylabel("Daily Changes CDN")


# In[85]:


Ethereum_data.plot(figsize=(10,4))
plt.axhline(0, color="black", lw=1)
plt.ylabel("Daily Changes ETH")


# 

# In[86]:


def scatter (df, title, separate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    "Generate a scatter plot of the entire dataframe"
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))
    #Designing the layout 
    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels= not separate_y_axis,
            type=scale
        )
    )
    
    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale )
    
    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'
        
    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index, 
            y=series, 
            name=label_arr[index],
            visible=visibility
        )
        
        # Add separate axis for the series
        if separate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config    
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.iplot(fig)


# ### 3. Prediction

# In this part, we tried to implement a linear regression model. By getting the differences from the two cryptocurrencies, we want to see the effect of the fluctuation of the coin in a day. The differences were continuous so we have to import utils to transfer it to multiclass for out linear regression model.

# In[121]:


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import utils


# In[122]:


X = CDN_sql_pd.drop(columns=['Differences'])
y = CDN_sql_pd['Differences']

CDN_enc = preprocessing.LabelEncoder()
CDN_encoded = lab_enc.fit_transform(y)
CDN_encoded


# In[123]:


utils.multiclass.type_of_target(y)
utils.multiclass.type_of_target(y.astype('int'))
y =utils.multiclass.type_of_target(CDN_encoded)


# In[124]:


CDN_model = LinearRegression()
CDN_model.fit(X, CDN_encoded)
CDN_predictions = CDN_model.predict([[5,1]])
CDN_predictions


# In[125]:


A = ETH_sql_pd.drop(columns=['Differences'])
b = ETH_sql_pd['Differences']

ETH_enc = preprocessing.LabelEncoder()
ETH_encoded = lab_enc.fit_transform(b)
ETH_encoded


# In[126]:


utils.multiclass.type_of_target(b)
utils.multiclass.type_of_target(b.astype('int'))
b =utils.multiclass.type_of_target(ETH_encoded)


# In[127]:


ETH_model = LinearRegression()
ETH_model.fit(A, ETH_encoded)
ETH_predictions = ETH_model.predict([[1636,523]])
ETH_predictions


# #### Conclusion
# The more popular a cryptocurrency, the more fluctuation it will have. Thus the range is suited for those who have large capital funding to yield beneficial from the variance it occurs.

# In[ ]:




