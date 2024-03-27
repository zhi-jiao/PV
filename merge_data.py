import pandas as pd
from datetime import time

def preprocess_and_adjust_data(pv_file_path, output_file_path,qxz_file_path,start_time='05:00',end_time='20:00',description='样板法'):
    # 读取并预处理数据
    data = pd.read_csv(pv_file_path)
    data = data[data['description'] == description]
    data['time'] = pd.to_datetime(data['time'])
    data['date'] = data['time'].dt.date
    data['time_of_day'] = data['time'].dt.time

    # 筛选数据：剔除开始时间晚于9:00和结束时间早于18:00的日期
    filtered_data = data.groupby('date').filter(lambda x: x['time_of_day'].min() <= time(9, 0) and x['time_of_day'].max() >= time(18, 0))

    # 为筛选后的数据创建新的时间段，每5分钟一个时间间隔，从早上5点到晚上8点
    time_slots = pd.date_range(start=start_time, end=end_time, freq='5T').time
    time_template = pd.DataFrame({'time_of_day': time_slots})

    # 初始化最终数据集
    adjusted_data = pd.DataFrame()

    # 获取日期范围
    date_range = pd.date_range(start=filtered_data['date'].min(), end=filtered_data['date'].max())

    # 处理每一天的数据
    for single_date in date_range:
        day_data = filtered_data[filtered_data['date'] == single_date.date()].copy()
        if not day_data.empty:
            day_data['time_of_day'] = day_data['time'].dt.time
            day_time_frame = pd.merge(time_template, day_data[['time_of_day', 'value']], how='left', on='time_of_day')
            day_time_frame.fillna(0, inplace=True)  # 缺失值用0填充
            day_time_frame['date'] = single_date
            adjusted_data = pd.concat([adjusted_data, day_time_frame], ignore_index=True)

    # 
    # Combine 'date' and 'time_of_day' into a new column 'time'
    print(adjusted_data.head())
    adjusted_data['time'] = pd.to_datetime(adjusted_data['date'].apply(lambda x: str(x)) + ' ' + adjusted_data['time_of_day'].apply(lambda x : str(x)))
    # adjusted_data.drop(columns=['date', 'time_of_day'], inplace=True)
    # 保存处理后的数据
    adjusted_data.to_csv(output_file_path, index=False)

        
    # 对来自qxz的数据进行拼接
    qxz_data = pd.read_csv(qxz_file_path)
    GlobalR = qxz_data[qxz_data['property'] == 'GlobalR']['value'].values
    # print(GlobalR.shape)
    DirectR = qxz_data[qxz_data['property'] == 'DirectR']['value'].values
    DiffuseR = qxz_data[qxz_data['property'] == 'DiffuseR']['value'].values
    AirT = qxz_data[qxz_data['property'] == 'AirT']['value'].values
    CellT = qxz_data[qxz_data['property'] == 'CellT']['value'].values
    WS = qxz_data[qxz_data['property'] == 'WS']['value'].values
    WD = qxz_data[qxz_data['property'] == 'WD']['value'].values
    P = qxz_data[qxz_data['property'] == 'P']['value'].values
    RH = qxz_data[qxz_data['property'] == 'RH']['value'].values
    
    new_qxz_data = pd.DataFrame([])
    new_qxz_data['time'] = qxz_data['time'].drop_duplicates()
    new_qxz_data['GlobalR'] = GlobalR
    new_qxz_data['DirectR'] = DirectR
    new_qxz_data['DiffuseR'] = DiffuseR
    new_qxz_data['AirT'] = AirT
    new_qxz_data['CellT'] = CellT
    new_qxz_data['WS'] = WS
    new_qxz_data['WD'] = WD
    new_qxz_data['P'] = P
    new_qxz_data['RH'] = RH
    
    new_qxz_data.to_csv('new_qxz_data.csv', index=False)
    
    # 拼接一下数据
    new_qxz_data['time'] = pd.to_datetime(new_qxz_data['time'])
    adjusted_data['time'] = pd.to_datetime(adjusted_data['time'])
    
    merged_data = pd.merge(left=adjusted_data, right=new_qxz_data, how='left', on='time')
    merged_data = merged_data.fillna(0)
    merged_data.to_csv('merged_data.csv', index=False)
    
input_file_path = '.\dhgdgf_theory\dhgdgf_theory.csv'  # 替换为您的输入文件路径

qxz_file_path = './dhgdgf_qxz/dhgdgf_qxz.csv'  # 替换为您的输入文件路径

output_file_path = './sorted_pv_data.csv'  # 替换为您的输出文件路径

preprocess_and_adjust_data(input_file_path, output_file_path,qxz_file_path,start_time='05:30',end_time='21:30')

