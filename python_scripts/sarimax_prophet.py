from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import warnings
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


data = pd.read_csv("data/INDPRO.csv")
# Check the last 5 elements of the dataframe
data.tail()


data.dtypes

data['DATE'] = pd.to_datetime(data['DATE'])
#Visualize the dataframe
plt.figure(figsize=(10,5))
sns.lineplot(data=data, x="DATE", y="INDPRO")
plt.title("U.S. Industrial Production Index (INDPRO)")
plt.grid(True)
plt.show()

data.columns = ["ds","y"]
model = Prophet(growth="linear", seasonality_mode="multiplicative", changepoint_prior_scale=30, seasonality_prior_scale=35,
               daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
               ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=30)

model.fit(data)


future = model.make_future_dataframe(periods= 120, freq='m')

future.tail()


forecast = model.predict(future)
forecast.tail()


forecast[["ds","yhat","yhat_lower","yhat_upper"]].head()


model.plot(forecast);
plt.title("U.S. Industrial Production Index (INDPRO)")
plt.show()


# calculate MAE between expected and predicted values
y_true = data['y'].values
y_pred = forecast['yhat'][:1225].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
r = r2_score(y_true, y_pred)
print('R-squared Score: %.3f' % r)
rms = mean_squared_error(y_true, y_pred, squared=False)
print('RMSE: %.3f' % rms)


plt.figure(figsize=(10,5))
# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title("United States Industrial Production Index (INDPRO)")
plt.grid(True)
plt.legend()
plt.show()
