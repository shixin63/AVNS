import pandas as pd
import re

path = r'D:\Pythonproject\TelescopeScheduling\test_result\2022-10-01'
file_name = r'\20221101_sta'
date = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


def time_seconds(st_day, day, hour, min, sec):
    if day == st_day:
        return hour * 3600 + min * 60 + sec
    else:
        return (hour + 24) * 3600 + min * 60 + sec


for _ in date:
    # data = pd.read_csv(path+file_name+'.csv', header=None, sep='\s+')
    path = r'D:\Pythonproject\TelescopeScheduling\test_result\2022-10-{}'.format(_)
    data = pd.read_csv(path + file_name + '.csv', header=None, sep='\s+')

    # print(data)
    # print(data.shape)
    # print(data.info)
    data.columns = ['序号', '站编号', '目标编号', '开始日期', '开始时间', '结束日期',
                    '结束时间', '弧段长度', '最近距离', '最大角速度', '亮度']
    # print(data.dtypes)

    data['开始时间'] = data['开始日期'] + ' ' + data['开始时间']
    data.drop(labels='开始日期', axis=1, inplace=True)
    data['结束时间'] = data['结束日期'] + ' ' + data['结束时间']
    data.drop(labels='结束日期', axis=1, inplace=True)
    # '序号', '站编号', '目标编号', '开始时间', '结束时间', '弧段长度', '最近距离', '最大角速度', '亮度'

    data.insert(loc=3, column='等级', value=0)
    # '序号', '站编号', '目标编号', '等级', '开始时间', '结束时间', '弧段长度', '最近距离', '最大角速度', '亮度'
    # print(data)

    # 读入等级表
    df_level = pd.read_csv(r'D:\Pythonproject\TelescopeScheduling\test_result\目标等级.csv', header=None,
                           sep=',')
    df_level.columns = ['目标编号', '等级']
    # print(df_level)

    # 时间转换
    # 记录开始日期
    pattern = re.compile(r'\d+(?:\.\d+)?')  # ?: find_all启用“不捕捉模式”
    # pattern = re.compile(r'\d+')
    st_day = 0
    temp = 0
    for i in range(data.shape[0]):
        # print(data.loc[i, '开始时间'])
        time_info = pattern.findall(data.loc[i, '开始时间'])
        # print(time_info)
        if i == 0:
            st_day = int(time_info[2])
            temp = int(time_info[0]) * 365 + int(time_info[1]) * 30 + st_day
        else:
            day = int(time_info[2])
            temp2 = int(time_info[0]) * 365 + int(time_info[1]) * 30 + day
            if temp2 < temp:
                st_day = day
                break
            elif temp2 > temp:
                break
    # print('开始日期为', st_day)
    # 修改时间格式
    for i in range(data.shape[0]):
        # 开始时间
        time_info_st = pattern.findall(data.loc[i, '开始时间'])
        day, hour, min, sec = int(time_info_st[2]), int(time_info_st[3]), int(time_info_st[4]), float(time_info_st[5])
        # print(day, hour, min, sec)
        data.loc[i, '开始时间'] = time_seconds(st_day, day, hour, min, sec)

        # 结束时间
        time_info_ed = pattern.findall(data.loc[i, '结束时间'])
        day, hour, min, sec = int(time_info_ed[2]), int(time_info_ed[3]), int(time_info_ed[4]), float(time_info_ed[5])
        # print(day, hour, min, sec)
        data.loc[i, '结束时间'] = time_seconds(st_day, day, hour, min, sec)
        # print(data.loc[i, '结束时间'] - data.loc[i, '开始时间'], data.loc[i, '弧段长度'])

    # 写入等级
    for i in range(data.shape[0]):
        bianhao = data.loc[i, '目标编号']
        # print(bianhao, end=' ')
        temp = df_level.loc[df_level['目标编号'] == bianhao].reset_index()
        # print(temp)

        if temp.shape[0] > 0:
            data.loc[i, '等级'] = temp.loc[0, '等级']
            # print(data.loc[i, '等级'])
        else:
            print('未找到编号为', bianhao, '的等级')
            exit(-1)

    # print(data.head())
    # data['序号'].map(lambda x: ('%4d' % x))
    # data['目标编号'].map(lambda x: ('%5d' % x))
    # data['等级'].map(lambda x: ('%2d' % x))
    # data['开始时间'].map(lambda x: ('%.3f' % x))
    # data['结束时间'].map(lambda x: ('%.3f' % x))
    # data['弧段长度'].map(lambda x: ('%.3f' % x))
    # data['最近距离'].map(lambda x: ('%.3f' % x))
    # data['最大角速度'].map(lambda x: ('%.3f' % x))
    # data['亮度'].map(lambda x: ('%.3f' % x))
    # print(data.head())

    # 保存
    data.to_csv(path+file_name+'_preprocessing.csv', header=None, index=False, sep=' ', float_format='%.3f')
    data.to_csv(path+file_name+'_preprocessing(comma).csv', header=None, index=False, float_format='%.3f')
