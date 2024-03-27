import pandas as pd 
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_day_trend(data,weather_data,date,path=None,type_='value'):
    plt.figure(figsize=(12,6))
    day_data = data[data['date'] == date]
    plt.plot(range(len(day_data)), day_data[type_].values)
    weather_day_data = weather_data[weather_data['date']==date]
    weather = weather_day_data['weather'].iloc[0]
    wind = weather_day_data['wind'].iloc[0]
    plt.title(f'Weather: {weather} Wind: {wind} Date: {date}')
    if path is None:
        plt.show()
    else:
        plt.savefig(path+f'/{date}.png')

data = pd.read_csv('./merged_data.csv')
weather_data = pd.read_csv('../Cache/dunhuang_weather.csv')


# 1 月份的数据
plot_day_trend(data,weather_data,'2021-01-15',type_='value')


# 6 月份的数据

plot_day_trend(data,weather_data,'2021-06-15',type_='value')