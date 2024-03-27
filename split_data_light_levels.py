import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from math import radians, tan, cos, sin, acos, degrees
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pd.read_csv('./merged_data.csv')


alpha = 94.5061111  # 经度
beta = 40.0663889  # 纬度

# 修正时区为中国统一时区 UTC+8
tz = 8
# 使用敦煌的经纬度重新计算日出时刻
def compute_Rt_corrected(alpha, beta, t, tz):
    # 将赤纬角的计算修正为基于天数t的函数，保持其余参数不变
    beta_radians = radians(beta)
    Rt = 24 * (180 + 15*tz - alpha - degrees(acos(-tan(radians(-23.4 * cos(radians(360 * (t + 9) / 365)))) * tan(beta_radians)))) / 360
    Ss = 24 * (1 + (15*tz - alpha) / 180) - Rt

    return Rt, Ss

# 重新调用函数进行计算，使用敦煌的经纬度，日期为年中的第一天，时区为8
# 由于计算的是日出时刻，所以取Rt


Rt_list = []
Ss_list = []
St_list = []
for t in range(1, 366):
    Rt, Ss = compute_Rt_corrected(alpha, beta, t, tz)
    Rt_list.append(Rt)
    Ss_list.append(Ss)
    St_list.append(Ss - Rt)
    
    
daylight_hours = np.array(St_list)  # Daylight hours array
quartiles = np.percentile(daylight_hours, [25, 50, 75])  # Quartiles for classification
print(quartiles)
def classify_daylight(hours, quartiles):
    """Classify daylight hours based on quartiles."""
    if hours <= quartiles[0]:
        return '低'
    elif hours <= quartiles[1]:
        return '中'
    elif hours <= quartiles[2]:
        return '高'
    else:
        return '非常高'

daylight_levels = [classify_daylight(hour, quartiles) for hour in daylight_hours]  # Classify each day



# Create a dictionary to hold dates for each daylight level
dates_by_level = {'低': [], '中': [], '高': [], '非常高': []}


# Populate the dictionary with dates for each level
for i, level in enumerate(daylight_levels):
    # Convert day of year to date. Year is a placeholder, we assume a non-leap year for simplicity.
    date = pd.Timestamp(year=2021, month=1, day=1) + pd.Timedelta(days=i)
    dates_by_level[level].append(date.strftime('%Y-%m-%d'))
    
print('低:',dates_by_level['中'])    
print('中:',dates_by_level['中'])
print('高:',dates_by_level['高'])
print('非常高:',dates_by_level['非常高'])

def get_daylight_level(date, dates_by_level):
    """Returns the daylight level for a given date."""
    for level, dates in dates_by_level.items():
        if date in dates:
            return level
    return None  # Return None if the date doesn't match any level

# Convert 'date' column to datetime to extract the date in YYYY-MM-DD format
data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

# Apply the function to add a new 'daylight_level' column to the dataset
data['daylight_level'] = data['date'].apply(lambda date: get_daylight_level(date, dates_by_level))
# data.to_csv('temp1.csv',index=False)




# Split the dataset based on the daylight level
data_low = data[data['daylight_level'] == '低']
data_medium = data[data['daylight_level'] == '中']
data_high = data[data['daylight_level'] == '高']
data_very_high = data[data['daylight_level'] == '非常高']



def filter_time_range(data, start_time_str, end_time_str):
    """
    Filter the data to only include rows between the specified start and end times.

    :param data: Pandas DataFrame containing a 'time_of_day' column with times.
    :param start_time_str: String representing the start time (e.g., '09:00:00').
    :param end_time_str: String representing the end time (e.g., '20:00:00').
    :return: Pandas DataFrame containing only the rows within the specified time range.
    """
    # Convert 'time_of_day' to datetime time type if not already
    if data['time_of_day'].dtype != 'datetime64[ns]':
        data['time_of_day'] = pd.to_datetime(data['time_of_day'], format='%H:%M:%S').dt.time

    # Define the start and end times
    start_time = pd.to_datetime(start_time_str).time()
    end_time = pd.to_datetime(end_time_str).time()

    # Filter the data between start_time and end_time
    filtered_data = data[(data['time_of_day'] >= start_time) & (data['time_of_day'] <= end_time)]
    
    return filtered_data

data_low = filter_time_range(data_low, '09:30:00', '18:30:00')
data_medium = filter_time_range(data_medium,'09:00:00', '19:00:00')
data_high = filter_time_range(data_high,'07:45:00', '19:45:00')
data_very_high = filter_time_range(data_very_high,'07:00:00', '21:00:00')

data_low.to_csv('./levels/low.csv',index=False)
data_medium.to_csv('./levels/medium.csv',index=False)
data_high.to_csv('./levels/high.csv',index=False)
data_very_high.to_csv('./levels/very_high.csv',index=False)




