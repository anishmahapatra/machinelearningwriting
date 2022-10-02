# -*- coding: utf-8 -*-
"""Copy of Detection and Prediction of Climate Change with Temperature - Adnan Thesis v3.ipynb

# Detection and Prediction of Climate Change with Temperature
* Author: Adnan Ahmed 
* Year: 2022

This is a dataset to analyze climate change that has been sourced from the Univerity of Dayton (https://academic.udayton.edu/kissock/http/Weather/source.htm). We will analyse this dataset, perform analysis, model and interpret the results.
"""

!pip install pystan~=2.14
!pip install fbprophet
!pip install -q pmdarima
# ! pip install --upgrade Cython
# ! pip install --upgrade git+https://github.com/statsmodels/statsmodels
# import statsmodels.api as sm

!pip install pmdarima scipy==1.2 -Uqq

# Importing the required libraries

from keras.layers import Activation, Dense, LSTM, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from dateutil.relativedelta import relativedelta
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm_notebook as tqdm
import matplotlib.gridspec as gridspec
from pmdarima.arima import auto_arima
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime
import plotly.express as px
from scipy import stats

import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import warnings
import plotly
import random

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings("ignore")
sns.set_style('darkgrid')

# Mounting the data
from google.colab import drive
drive.mount('/content/drive')

# Reading the csv file and viewing a sample to ensure the data has been correctly read
temperature = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AdnanThesisTemperature/city_temperature.csv',low_memory=False)
temperature.head(5)

# Creating a copy of the dataset
df = temperature.copy(deep=True)

# Typecasting of features to ensure that the data type is standardized 
# We also set the right number of bytes to ensure that memory is optimaaly used
df[['Month', 'Day']] = df[['Month', 'Day']].astype('int8')
df[['Year']] = df[['Year']].astype('int16')
df['AvgTemperature'] = df['AvgTemperature'].astype('float16')

# Resetting the index of the pandas dataframe
df.reset_index(drop=True, inplace=True)

"""## Preprocessing and data cleaning"""

df.head()

# Removing ambiguous values from Year column in the data based on preliminary analysis 
df = df[df.Year != 200]
df = df[df.Year != 201]
df = df[df.Year != 2020]

# Dropping the State column as it only applies to the US and we will not be analyzing it in detail
df = df.drop('State', axis = 1)

# Creating a new column, Date in datetime format
df['Date'] = pd.to_datetime(df.Year.astype(str) + '/' + df.Month.astype(str))

# Let us look at the percentage of missing values
missing = pd.DataFrame(df.loc[df.AvgTemperature == -99, 'Country'].value_counts())

# We group by Country and look at the percentage of missing values
missing['TotalNumberOfMissingPoints'] = df.groupby('Country').AvgTemperature.count()

# Making the dataframe for percentge of missing values and sorting it by descending order of percentage of missing values
missing['PercentageOfValuesMissing'] = missing.apply(lambda row: (row.Country/row.TotalNumberOfMissingPoints)*100, axis = 1).round(2)
missing['PercentageOfValuesMissing'] = missing['PercentageOfValuesMissing'].astype(str) + '%'
missing.sort_values(by=['PercentageOfValuesMissing'], inplace=True, ascending = False)
missing.head(20)

# Missing values in AvgTemperature column as indicated as -99, so we replace it by NumPy's NaN function
df.loc[df.AvgTemperature == -99, 'AvgTemperature'] = np.nan

# Let us have a look at the total number of missing data points
df.AvgTemperature.isna().sum()

# Filling in the missing values with the mean temperature of the same city and same date
df['AvgTemperature'] = df['AvgTemperature'].fillna(df.groupby(['City']).AvgTemperature.transform('mean'))

# Looking at the rest of the missing values after we have performed missing value imputation
df.AvgTemperature.isna().sum()

df.loc[df.AvgTemperature.isna(), 'City'].value_counts()

# Missing values should be removed now; the result of this query is 0
df.AvgTemperature.isna().sum()

# Let us now convert this dataset into Celsius
# °F to °C: (°F − 32) × 5/9 = °C
df['AvgTempCelsius'] = (df.AvgTemperature -32)*(5/9)
df  = df.drop(['AvgTemperature'], axis = 1)

# Temperatures beyond two decimal points are not as useful, so let limit the float values to one and two decimal points
df['AvgTempCelsius_rounded'] = df.AvgTempCelsius.apply(lambda x: "{0:0.2f}".format(x))
df['AvgTempCelsius_rounded2'] = df.AvgTempCelsius.apply(lambda x: "{0:0.1f}".format(x))

# Converting the new rounded values to numeric
df['AvgTempCelsius_rounded'] = pd.to_numeric(df['AvgTempCelsius_rounded'])
df['AvgTempCelsius_rounded2'] = pd.to_numeric(df['AvgTempCelsius_rounded2'])

# Sanity check of a sample of the data
df.sample(5)

data = df[['Country','Year','AvgTempCelsius']].groupby(['Country','Year']).mean().reset_index()
px.choropleth(data_frame=data,locations="Country",locationmode='country names',animation_frame="Year",color='AvgTempCelsius',color_continuous_scale = 'Turbo',title="Average temperature of countries over the years 1995 to 2019")

# Plotting the Yearly Global Average Temperature
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'AvgTempCelsius', data = df , palette='tab10')

# Setting the plot parameters
plt.title('Average Global Temperatures')
plt.ylabel('Average Temperature in degrees centigrade (°C)')
plt.xlabel('')
plt.xticks(range(1995,2020))
plt.show();

# Making a new dataframe to plot the average temperature by month
df_mean_month = df.groupby(['Month', 'Year']).AvgTempCelsius_rounded2.mean()
df_mean_month = df_mean_month.reset_index()
df_mean_month = df_mean_month.sort_values(by = ['Year'])

# Making a pivoted dataframe with Month, Average Temperature and Year
df_pivoted = pd.pivot_table(data= df_mean_month,
                    index='Month',
                    values='AvgTempCelsius_rounded2',
                    columns='Year')

# Plotting the average temperatures by year and month
plt.figure(figsize=(20, 8))
sns.heatmap(data = df_pivoted, cmap='coolwarm', annot = True, fmt=".1f", annot_kws={'size':11})
plt.xlabel('Year')
plt.ylabel('Month (Number)')
plt.title('Average Global Temperatures in degrees centigrade (°C)')
plt.show();

# Plotting the average temperature of the Regions
s = df.groupby(['Region'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=False)
s.style.background_gradient(cmap="coolwarm")

# Plotting the average temperature of different regions over time
f = plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'AvgTempCelsius', hue = 'Region', data = df , palette='muted')

# Filling the required parameters of the plot
plt.title('Average Temperature in Different Regions')
plt.ylabel('Average Temperature in degree celsius (°C)')
plt.xlabel('Year')
plt.xticks(range(1995,2020))
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5),ncol=1)
plt.tight_layout()
plt.show();

# Distribution of temperatures across various regions
region_sorted = df.groupby('Region')['AvgTempCelsius'].median().sort_values().index

with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data = df.sort_values("AvgTempCelsius"), x = 'Region', y = 'AvgTempCelsius', order = region_sorted)
    plt.xticks(rotation = 70)
    plt.title('Distribution of Temperatures (1995-2019)')
    plt.xlabel('')
    plt.ylabel('Average Temperature (°C)')

# Making a violin plot of the distribution of average temperatures by Country
with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(data = df.sort_values("AvgTempCelsius"), x = 'Region', y = 'AvgTempCelsius_rounded', order = region_sorted)
    plt.xticks(rotation = 45)
    plt.title('Distribution of Average Temperatures (1995-2019)')
    plt.xlabel('')
    plt.ylabel('Average Temperature (°C)')
    plt.show;

# Plotting the monthly average temperature by month in various Regions

regions = df.Region.unique().tolist()

number_plot = [0, 0, 0, 1, 1, 1, 2]
position_a = [0, 2, 4, 0, 2, 4, 2]
position_b = [2, 4, 6, 2, 4, 6, 4]

fig = plt.figure(figsize = (25,15))
plt.suptitle('Global Monthly Temperatures (1995-2019)', y = 1.05, fontsize=15)

gs = gridspec.GridSpec(3, 6)

for i in range(7): 
    #ax = plt.subplot(3, 3, i+1)
    ax = plt.subplot(gs[number_plot[i], position_a[i]:position_b[i]])
    sns.barplot(x = 'Month', y = 'AvgTempCelsius_rounded2', data = df[df.Region == regions[i]])
    ax.title.set_text(regions[i])
    ax.set_ylim((0,35))
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.subplots_adjust(wspace = 0.5)
plt.rcParams.update({'font.size': 10})
plt.savefig('demographics.png')
plt.tight_layout()
plt.show();

# Top 5 hottest countries

s = df.groupby(['Country'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=False)[:5]
s.style.background_gradient(cmap="Reds")

# Top 5 coldest countries

s = df.groupby(['Country'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=True)[:5]
s.style.background_gradient(cmap="Blues")

# Plotting the Kerner Density Estimation (KDE) for AvgTempCelsius
sns.set(rc={'figure.figsize':(15,8)})
sns.kdeplot(data=df, x="AvgTempCelsius")

# Selecting a sub-section of the data and filtering for Europe
df_europe = df[df.Region == 'Europe'].copy()

# Sanity check via the sample function on the Europe dataset
df_europe.sample(5)

# Plotting the distribution of temperatures in Europe
f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(df_europe.AvgTempCelsius_rounded, bins = 20)
plt.title('Distribution of Temperatures in Europe (1995-2019)')
plt.xlabel('Temperature (°C)')
#ax.axes.yaxis.set_visible(False)
ax.axes.yaxis.set_ticklabels(['']);

# Distribution of temperatures across different Europen Countries

countries_sorted = df_europe.groupby('Country')['AvgTempCelsius_rounded2'].median().sort_values().index

with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(20, 7))
    sns.boxplot(data = df_europe, x = 'Country', y = 'AvgTempCelsius_rounded', order = countries_sorted)
    plt.xticks(rotation = 90)
    plt.title('Distribution of Temperatures in Europe (1995-2019)')
    plt.ylabel('Temperature (°C)')
    plt.xlabel('');

# Average temperature of the countries in Europe, sorted by temperature in ascending order

countries_mean_sorted = df_europe.groupby('Country').AvgTempCelsius_rounded2.mean().sort_values().index

plt.figure(figsize = (15,8))
sns.barplot(x = 'Country', y = 'AvgTempCelsius_rounded2', data = df_europe, 
            order = countries_mean_sorted)
plt.xticks(rotation = 90)
plt.xlabel('')
plt.title('Average Temperatures in Europe (1995-2019)')
plt.ylabel('Average Temperature (°C)');

# Grouping the European data by month and year

europe_mean_month = df_europe.groupby(['Month', 'Year']).AvgTempCelsius_rounded2.mean()
europe_mean_month = europe_mean_month.reset_index()
europe_mean_month = europe_mean_month.sort_values(by = ['Year'])

# Making a pivot on the data

europe_pivoted = pd.pivot_table(data= europe_mean_month,
                    index='Month',
                    values='AvgTempCelsius_rounded2',
                    columns='Year')

# Plotting a grid to understand the month-wise trends across 1995 to 2019

plt.figure(figsize=(20, 8))
sns.heatmap(data = europe_pivoted, cmap='coolwarm', annot = True, fmt=".1f", annot_kws={'size':11})
plt.ylabel('Month')
plt.xlabel('')
plt.title('Average Temperatures in Europe (°C)')
plt.show();

"""## Prophet"""

# First, we will take backup of the code
dataBackup = df.copy(deep=True)

# df = dataBackup.copy(deep = True)

df.head()

cols=["Year","Month","Day"]
df['Date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")

df['AvgTemperatureCelsius'] = df['AvgTempCelsius_rounded']

df = df[df['City'] == 'London']

df = df.drop(['Region', 'Country', 'Month', 'Day', 'Year', 'City', 'AvgTempCelsius', 'AvgTempCelsius_rounded', 'AvgTempCelsius_rounded2'], axis = 1)

df = df[df['AvgTemperatureCelsius'] < 50]
df = df[df['AvgTemperatureCelsius'] > -20]

df['ds'] = pd.to_datetime(df['Date'])

df = df.drop('Date', axis = 1)

df = df.rename(columns={'AvgTemperatureCelsius': 'y'})

df.tail()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=730, freq = 'd')
future.tail()

forecast_rainfall = m.predict(future)
forecast_rainfall[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast_rainfall)
plt.axvline(x= datetime.date(2020,5,13), color='red');

"""## Pre-processing for ARIMA & LSTM"""

df = dataBackup.copy(deep = True)

# Viewing sample of the data - sanity check
df.sample(5)

# Dropping any duplicates and ensuring that there are no null values
df = df.drop_duplicates()
df.isnull().sum()

# Typecasting AvgTempCelsius
df['AvgTempCelsius'] = df['AvgTempCelsius'].astype(np.int16)

# Average temperature Global from 1995 to 2019
data = df[['Year','AvgTempCelsius']].groupby('Year').mean()
linfit = np.polyfit(data.index,data['AvgTempCelsius'],deg=1)
linfit = linfit[0]*data.index + linfit[1]

fig = px.line(data,title='Average Temperature of the World from 1995 to 2019')
fig.add_trace(go.Scatter(x=data.index,y=linfit,name='Linear Fit'))

#Preparing data
data = df[['Region','Month','AvgTempCelsius']]
data = data.groupby(['Region','Month']).mean()

months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:"Dec"}
data = data.reset_index(level=1)

#Changing the month label from integer to name
data['Month'] = data.loc[:,'Month'].map(months)
data.head(3)

# Viewing the highest and lowest recorded tempratures in this dataset
data1=df.sort_values(by=['AvgTempCelsius'],ascending=False).head(1)
data2=df.sort_values(by=['AvgTempCelsius'],ascending=True).head(1)
data = pd.concat([data1,data2],)
data.index = ['Highest','Lowest']
data

# Selecting data points from London
data = df[df['Country'] == 'United Kingdom']
bom = data.loc[data['City'] == 'London',['Month','Day','Year','AvgTempCelsius']].reset_index(drop=True)
bom.head(3)

# Sanity check to ensure that the temperature is being recorded every day 
size = bom.groupby(['Year','Month']).size().reset_index()
size_max = size[0].max()
size_min = size[0].min()
n = size_max - size_min +1
cmap = sns.color_palette("deep",n)
size = size.pivot(index='Year',columns='Month',values=0)
size.head(3)

# We create the datetime column from month, day and year
bom['Date'] = bom[['Year','Month','Day']].apply(lambda row:'-'.join([str(row['Year']),str(row['Month']),str(row['Day'])]),axis=1)
bom['Date'] = pd.to_datetime(bom['Date'])
bom = bom.drop(columns=['Month','Day','Year']).set_index('Date')
bom.head(3)

# Plotting hte average temperature of the city of London
px.line(data_frame=bom,color_discrete_sequence=['blue'],title="Daily Average Temperature - London (1995-2019)")

# Plotting the rolling statistics
roll_mean = bom.rolling(window=31).mean()
roll_mean2 = bom.rolling(window=365).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=bom.index,y=bom['AvgTempCelsius'],marker=dict(color='grey'),name='Daily'))
fig.add_trace(go.Scatter(x=roll_mean.index,y=roll_mean['AvgTempCelsius'],marker=dict(color='red'),name='31DaysRolling'))
fig.add_trace(go.Scatter(x=roll_mean2.index,y=roll_mean2['AvgTempCelsius'],marker=dict(color='green'),name='365DaysRolling'))
fig.update_layout(dict(title='Rolling Mean'))

# Plotting the rolling standard deviation
roll_mean = bom.rolling(window=31).std()
roll_mean2 = bom.rolling(window=365).std()
fig = go.Figure()
fig.add_trace(go.Scatter(x=roll_mean.index,y=roll_mean['AvgTempCelsius'],marker=dict(color='red'),name='31DaysRolling'))
fig.add_trace(go.Scatter(x=roll_mean2.index,y=roll_mean2['AvgTempCelsius'],marker=dict(color='green'),name='365DaysRolling'))
fig.update_layout(dict(title='Rolling Std'))

# Performing seasonal decompose on the dataframe
decompose = seasonal_decompose(bom,period=365)
decompose.plot();

# Performing the Dickey-Fuller-test to verify the stationarity
# Here, the the NULL hypothesis is that the time series data is non-stationary 
adf = adfuller(x=bom['AvgTempCelsius'])
print('pvalue:',adf[1])
print('adf:',adf[0])
print('usedlag:',adf[2])
print('nobs:',adf[3])
print('critical_values:',adf[4])
print('icbest:',adf[5])

"""Here, the p-value is very small, and close to 0, so we can reject the NULL Hypothesis.

## LSTM Again!
"""

# Train-test-split
test = bom[bom.index>'2019']
train = bom[bom.index<'2019']

# Performing Min-Max scaling
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

time_steps = 20
features = 1

train_gen = TimeseriesGenerator(train,train,time_steps,batch_size=32)
test_gen = TimeseriesGenerator(test,test,time_steps,batch_size=32)

# Defining the sequential model and using ReLU as the activation function
model = Sequential()
model.add(LSTM(64,activation='relu',input_shape=(time_steps,features),return_sequences=True))
model.add(LSTM(32,activation='relu'))
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='mse')

model.summary()

# Defining the callbacks
early_stop = EarlyStopping(patience=5)

# Fitting the model and setting the parameters
model.fit(x=train_gen,epochs=40,callbacks=[early_stop],validation_data=test_gen)

#Save the model
model.save('LSTMAvgTemp.h5')

# Input the test genertor values to the model and see the predictions
predict = model.predict(test_gen)

# Collect the original target values from the test generator
test_targets = test_gen.targets[test_gen.start_index:test_gen.end_index+1]

# Performing inverse transform on the scaled results
predict = scaler.inverse_transform(predict).ravel()
test_targets = scaler.inverse_transform(test_targets).ravel()

# Evaluation of the model

_,ax=plt.subplots(1,1,figsize=(12,8))
sns.lineplot(x=range(len(predict)),y=predict,ax=ax,label='Predictions')
sns.lineplot(x=range(len(test_targets)),y=test_targets,ax=ax,label='Actual')
plt.legend()
_=plt.title('Comparison - Predictions vs Actual Target Temperatures - London (2019)')

# Printing the output of the model
print('The RMSE Score is:',format(np.sqrt(mean_squared_error(predict,test_targets)),'.2f'))

# Now, we will forecast beyond 2019

data = bom.iloc[-time_steps:].to_numpy() #2D Array
data = scaler.transform(data)

#expand to include batch dimension
data = np.expand_dims(data,0)

#record the last date of observartion from the data
date = bom.index[-1]

date_store = bom.iloc[-time_steps:].index.to_list()

#forecasting
forecasts=10
for i in range(forecasts):
    predicted = model.predict(data[:,-20:,:])
    date = date+datetime.timedelta(days=1)
    data = np.append(data,[predicted],axis=1)
    date_store.append(date)
data = scaler.inverse_transform(data.reshape(1,-1))
forecast_df = pd.DataFrame(index=date_store[time_steps-1:],data={'AvgTempCelsius':data.ravel()[time_steps-1:]})

# Let us look at the table of the forecast (Sanity check)
forecast_df

# Plotting the output
_,ax=plt.subplots(1,1,figsize=(12,8))
sns.lineplot(data=bom.iloc[-100:,:],y='AvgTempCelsius',x=bom.iloc[-100:,:].index,color='blue',ax=ax,label='AvgTemp - 2019 end')
sns.lineplot(data=forecast_df,y='AvgTempCelsius',x=forecast_df.index,color='red',ax=ax,label= 'AvgTemp - 2020 forecasted')
_=plt.title(f'Temperature Forecasting - {forecasts} days (2020)')

"""## Seasonal ARIMA Model"""

# We are downsampling to monthly frequency
bom_monthly = bom.resample('M').mean()
bom_monthly.head(5)

# Plotting the output - the seasonal element is 12 months
data =[go.Scatter(x=bom.index,y=bom['AvgTempCelsius'],name='AvTemperature-Daily',marker=dict(color='grey')),go.Scatter(x= bom_monthly.index,y=bom_monthly['AvgTempCelsius'],name='AvgTempCelsius-Monthly',marker=dict(color='red'))]
fig = go.Figure(data)

buttons = [dict(label='Both',method='restyle',args=[{'visible':[True,True]}]),
           dict(label='Daily',method='restyle',args=[{'visible':[True,False]}]),
           dict(label='Monthly',method='restyle',args=[{'visible':[False,True]}])]

updatemenus=[dict(type="buttons",direction='down',buttons=buttons)]
fig.update_layout(updatemenus=updatemenus,title='Average Temperature - Daily and Monthly - London')

# Re-sampling the data
bom_monthly = bom.resample('M').mean()
bom_monthly.head(5)

# Plotting the average temperature to compare monthly and daily
data =[go.Scatter(x=bom.index,y=bom['AvgTempCelsius'],name='AvTemperature-Daily',marker=dict(color='grey')),go.Scatter(x= bom_monthly.index,y=bom_monthly['AvgTempCelsius'],name='AvgTempCelsius-Monthly',marker=dict(color='red'))]
fig = go.Figure(data)

buttons = [dict(label='Both',method='restyle',args=[{'visible':[True,True]}]),
           dict(label='Daily',method='restyle',args=[{'visible':[True,False]}]),
           dict(label='Monthly',method='restyle',args=[{'visible':[False,True]}])]

updatemenus=[dict(type="buttons",direction='down',buttons=buttons)]
fig.update_layout(updatemenus=updatemenus,title='Average Temperature - Daily and Monthly - London')

"""## Auto-ARIMA"""

import statsmodels.api as sm

#Using default auto_arima arguments
model = auto_arima(bom_monthly,seasonal=True,m=12)

# Storing date and temperature 
date = bom_monthly.index[-1]
last_val = bom_monthly.iloc[-1].to_numpy()

# Forecasting the first 5 months
forecasts=5
date_store = [(date + relativedelta(months=i)) for i in range(0,forecasts+1)]

predict = model.predict(forecasts)
predict =  np.append(last_val,predict)
forecast_df = pd.DataFrame(data=predict,index=date_store,columns=['AvgTempCelsius'])
forecast_df.head(5)

# Plotting the results
_,ax=plt.subplots(1,1,figsize=(10,10))
sns.lineplot(data=forecast_df['AvgTempCelsius'],ax=ax,color='red')
sns.lineplot(data=bom_monthly['AvgTempCelsius'][-4*12:],ax=ax,color='blue')
_=plt.title(f'Forecast - Monthly Temperature Average (2020) - First {forecasts} months')

























































































































































































































































































































































































































