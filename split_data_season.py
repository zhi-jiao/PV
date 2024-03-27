import numpy as np 
import pandas as pd 


data = pd.read_csv('./merged_data.csv')

# Convert 'date' column to datetime format and sort the data
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# 划分


# Define the seasons based on month
def assign_season(month):
    if 0 <= month <= 3:
        return 'Spring'
    elif 4 <= month <= 6:
        return 'Summer'
    elif 7 <= month <= 9:
        return 'Autumn'
    else:  # December to February
        return 'Winter'

# Assign season to each row
data['season'] = data['date'].dt.month.apply(assign_season)

# Separate data into four datasets based on seasons
spring_data = data[data['season'] == 'Spring']
summer_data = data[data['season'] == 'Summer']
autumn_data = data[data['season'] == 'Autumn']
winter_data = data[data['season'] == 'Winter']

spring_data = spring_data.sort_values(by=['date', 'time_of_day'])
summer_data = summer_data.sort_values(by=['date', 'time_of_day'])
autumn_data = autumn_data.sort_values(by=['date', 'time_of_day'])
winter_data = winter_data.sort_values(by=['date', 'time_of_day'])

# Show the first few rows of each season to verify
print('Spring:',len(spring_data))
print('Summer:',len(summer_data))
print('Autumen:',len(autumn_data))
print('Winter:',len(winter_data))

# save data
spring_data.to_csv('./seasons/spring.csv')
summer_data.to_csv('./seasons/summer.csv')
autumn_data.to_csv('./seasons/autumn.csv')
winter_data.to_csv('./seasons/winter.csv')



