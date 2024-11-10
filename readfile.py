import csv
from copy import deepcopy


# 观测数据存储位置
# file_path = open(r'D:\Pythonproject\TelescopeScheduling\test_result\2022-10-01\20221101_sta_preprocessing(comma).csv',
#                  'rb')


# 读取并处理观测数据
def readfile():
    # 列表记录处理后的观测数据
    observation_target_data = []
    # 读取文件
    with open(
            'D:/Pythonproject/TelescopeScheduling/test_result/2022-10-10/20221101_sta_preprocessing.csv',
            encoding='utf-8') as f:
        for row in csv.reader(f):
            r_temp = row[0]
            r_ = r_temp.split()
            # 保存需要的数据至列表
            temp = []
            for _ in range(2, 6):
                temp.append(int(float(r_[_])))
            observation_target_data.append(temp)
            # print(temp)
    # 根据观测目标的可观测时间排序
    # observation_target_data = sorted(observation_target_data, key=lambda x: x[2])
    # for _ in observation_target_data:
    #     print(_)
    # 去除不满足条件的目标数据（可观测时间段不足90s）
    for i_ in range(len(observation_target_data) - 1, -1, -1):
        if observation_target_data[i_][3] - observation_target_data[i_][2] < 90:
            observation_target_data.pop(i_)
    print('数据读取完成')
    return observation_target_data


if __name__ == '__main__':
    list0 = readfile()


