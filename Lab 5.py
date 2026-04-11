# Bài 1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_5_Supermarket.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df['revenue'] = df['revenue'].ffill().bfill()
df['revenue'] = df['revenue'].interpolate(method='linear')

df['Year'] = df.index.year
df['Month'] = df.index.month
df['Quarter'] = df.index.quarter
df['Day_of_Week'] = df.index.dayofweek
df['Day'] = df.index.day
df['Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
df['Month_Name'] = df.index.month_name()
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Autumn')

print(df.head())
print(df.isnull().sum())

df_month = df.groupby('Month')['revenue'].sum().reset_index()
df_week = df.groupby('Day_of_Week')['revenue'].sum().reset_index()

df_month.plot(x='Month', y='revenue', kind='bar', title='Total Revenue by Month')
plt.tight_layout()
plt.show()

df_week.plot(x='Day_of_Week', y='revenue', kind='bar', title='Total Revenue by Day of Week')
plt.tight_layout()
plt.show()

df_month['Rolling_Mean'] = df_month['revenue'].rolling(window=3).mean()
df_month.plot(x='Month', y=['revenue', 'Rolling_Mean'], kind='line', title='Revenue & Rolling Mean')
plt.tight_layout()
plt.show()

# Bài 2:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_5_Web_traffic.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

df = df.asfreq('h')
df['visits'] = df['visits'].interpolate(method='linear')

df['Hour'] = df.index.hour
df['Day'] = df.index.day
df['Day_of_Week'] = df.index.dayofweek
df['Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
df['Month'] = df.index.month

print(df.head())
print(df.isnull().sum())

df_hour = df.groupby('Hour')['visits'].mean().reset_index()
df_dow = df.groupby('Day_of_Week')['visits'].mean().reset_index()

df_hour.plot(x='Hour', y='visits', kind='line', title='Avg Traffic by Hour')
plt.tight_layout()
plt.show()

df_dow.plot(x='Day_of_Week', y='visits', kind='line', title='Avg Traffic by Day of Week')
plt.tight_layout()
plt.show()

df_daily = df['visits'].resample('D').sum()
df_daily_rolling = df_daily.rolling(window=7).mean()
pd.DataFrame({'Daily': df_daily, 'Rolling_7d': df_daily_rolling}).plot(title='Daily Traffic & 7-day Rolling Mean')
plt.tight_layout()
plt.show()

# Bài 3:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ITA105_Lab_5_Stock.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df['close_price'] = df['close_price'].ffill()

df['Rolling_Mean_7'] = df['close_price'].rolling(window=7).mean()
df['Rolling_Mean_30'] = df['close_price'].rolling(window=30).mean()

df[['close_price', 'Rolling_Mean_7', 'Rolling_Mean_30']].plot(title='Stock Close Price & Rolling Means')
plt.tight_layout()
plt.show()

df['Month'] = df.index.month
df_month = df.groupby('Month')['close_price'].mean().reset_index()
df_month.plot(x='Month', y='close_price', kind='line', title='Avg Close Price by Month')
plt.tight_layout()
plt.show()

# Bài 4:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('ITA105_Lab_5_Production.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['week_start'] = pd.to_datetime(df['week_start'])
df.set_index('week_start', inplace=True)

df['production'] = df['production'].ffill().bfill()
df['production'] = df['production'].interpolate(method='linear')

df['Week'] = df.index.isocalendar().week.astype(int)
df['Month'] = df.index.month
df['Quarter'] = df.index.quarter
df['Year'] = df.index.year

df['Rolling_Mean_4'] = df['production'].rolling(window=4).mean()

df[['production', 'Rolling_Mean_4']].plot(title='Production & 4-week Rolling Mean')
plt.tight_layout()
plt.show()

df_quarter = df.groupby('Quarter')['production'].mean().reset_index()
df_quarter.plot(x='Quarter', y='production', kind='bar', title='Avg Production by Quarter')
plt.tight_layout()
plt.show()

result = seasonal_decompose(df['production'], model='additive', period=52)
result.plot()
plt.tight_layout()
plt.show()