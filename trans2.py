import pandas as pd
import re

path = r'D:\Pythonproject\TelescopeScheduling\test_result\2022-10-01'
file_name = r'\20221101'
date = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


def time_seconds(st_day, day, hour, min, sec):
    if day == st_day:
        return hour * 3600 + min * 60 + sec
    else:
        return (hour + 24) * 3600 + min * 60 + sec


for _ in date:
    # data = pd.read_csv(path+file_name+'.csv', header=None, sep='\s+')
    path = r'D:\Pythonproject\TelescopeScheduling\test_result\2022-10-{}'.format(_)
    data = pd.read_csv(path + file_name + '.log', header=None, sep='\s+', dtype=str)

    # print(data)
    # print(data.shape)
    # print(data.info)
    # print(data.dtypes)

    # 时间转换
    # 记录开始日期
    pattern = re.compile(r'\d+(?:\.\d+)?')  # ?: find_all启用“不捕捉模式”
    # pattern = re.compile(r'\d+')
    st_day = 0
    temp = 0
    for i in range(data.shape[0]):
        time_str = data.loc[i, 4]
        # print(time_str)
        year, month, day, hour, min, sec = \
            int(time_str[0: 4]), int(time_str[4: 6]), int(time_str[6: 8]), \
            int(time_str[8: 10]), int(time_str[10: 12]), int(time_str[12: 14])
        # print(year, month, day, hour, min, sec)
        temp2 = year * 365 + month * 30 + day
        if i == 0:
            st_day = day
            temp = temp
        else:
            if temp2 < temp:
                st_day = day
                break
            elif temp2 > temp:
                break
    # print('开始日期为', st_day)
    # exit()

    # 修改时间格式
    for i in range(data.shape[0]):
        # 开始时间
        time_str = data.loc[i, 4]
        # print(time_str)
        year, month, day, hour, min, sec = \
            int(time_str[0: 4]), int(time_str[4: 6]), int(time_str[6: 8]), \
            int(time_str[8: 10]), int(time_str[10: 12]), int(time_str[12: 14])

        # print(day, hour, min, sec)
        data.loc[i, 4] = time_seconds(st_day, day, hour, min, sec)

    # 保存
    data.to_csv(path+file_name+'_preprocessing.log', header=None, index=False, sep=' ')
