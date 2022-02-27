import pandas as pd
import matplotlib.pyplot as plt
from auto_ts import auto_timeseries


df = pd.read_csv("data/AMZN_2006-01-01_to_2018-01-01.csv", usecols=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')


train_df = df.iloc[:2800]
test_df = df.iloc[2800:]



train_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Train')
test_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Test')
plt.legend()
plt.grid()
plt.show()

model = auto_timeseries(forecast_period=219, score_type='rmse', time_interval='D', model_type='best')
model.fit(traindata= train_df, ts_column="Date", target="Close")


model.get_leaderboard()

model.plot_cv_scores()

plt.show()


future_predictions = model.predict(testdata=219)