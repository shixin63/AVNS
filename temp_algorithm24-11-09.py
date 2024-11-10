"""
2023.2.24 贪婪插入（未发现bug）
下一步的工作是将未观测的目标插入到望远镜的空闲时间
2023.02.25
将未观测的目标根据目标等级降序排列  符合约束条件的目标插入到望远镜空闲时间观测
插空实验结果：无效果望远镜排的太满了
2023.02.25-2
将未观测目标中的高等级目标替换望远镜中符合替换条件的低等级目标
替换实验效果：有提升
考虑代码的模块化问题（未解决）
2023.02.27
插入实验有明显提升  乱序贪婪插入 插空 替换 实验效果高于目标开始观测时间升序贪婪插入替换
2023.02.28
可视化函数
增加观测调度序列最前列最后列插入
2023.03.01
修复队头插入错误
实验发现当前贪心方式    插入队尾无法插入
2023.03.02
封装调度函数 构建生成解的函数 测试生成编码方式
首先 测试只对望远镜编号的方式
2023.03.03
增加调度函数返回调度后解的功能
调度函数在插入过程存在优化空间：考虑每个望远镜都可以插入，而不是只尝试插入解编码中的望远镜，但是这样做会增加调度的时间复杂度，考虑只对较优解进行此种调度
在完成上述工作后 初步计划采用自适应变邻域搜索算法
开始构建自适应邻域搜索算法
2023.03.04
实验结果不理想，重新补充允许目标所有望远镜，但尽量插入解中对应的望远镜
修复bug 下一步的工作是引入变异算子
2023.03.06
引入变异算子  实验效果不明显
下一步实验计划设计基于目标观测优先级的编码方式进行实验
2023.03.07
设计基于目标观测优先级的编码方式，并在次基础上进行主动调度
实验效果好于望远镜编码 同时缩小了解空间的大小    存在解增加长度的bug（初步认为修复03.08）
下一步的工作是实验两种编码方式的组合
2023.03.08-2
文件为ga3-10
这份文件进行两种编码方式的组合实验
计划使用遗传算法求解
需要修改的部分如下：解生成函数、调度方式函数、返回调度后的解函数
需要新的函数：POX交叉算子 、适应值函数
2023.03.09
文件为ga3-10
实现新函数：POX交叉算子 、适应值函数
2023.03.10
文件为ga3-10
实现新函数：POX交叉算子 、种群加入策略
双重编码实验性能不逊与优先级编码方式计划进行下一步实验
交叉算子存在局部最优的问题，下一步计划使用变邻域搜索求解双重编码，
计划初步实现可能存在bug
2023.03.12
文件为algo03-12
更新算法种群加入策略  算法性能获得明显提升
2023.03.15
文件为algo03-15
替换后被替换的低等级目标没有加入未观测列表，也未从已观测列表pop(无关紧要的处理，不影响算法执行和性能)
2023.03.29
修改价值目标价值
处理掉不足90秒的数据
问题分解 将数据根据时间分段 每段分别求解
2023.03.29-2
尝试问题分段求解，每次先取200个目标数据，共400个数据分两段求解（面临拼接问题）
直接把数据根据中间时间切成两部分（或者多个部分），直接拼接。拼接再用调度方式对未观测目标进行插入替换
2023.04.03
分段求解成功
2023.04.06
拼接所有子问题的解，将拼接后的解中未观测目标进行插入替换
2023.10.04
增加绘制收敛曲线
2024.11.09
修正拼接函数的错误，增加成本函数终止条件
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import pickle
from time import time
import datetime
from Telescope import Telescope
from readfile import readfile

# 望远镜数目
telescope_number = 10
# 暂时获取数据个数
temp_get_data_num = 1000
# 种群大小
POP_SIZE = 10
# 算子个数
operator_num = 3
EPISILON = 0.00000001  # 一个极小的正数
SEED = 1
# random.seed(SEED)  # 随机种子
TODAY = datetime.date.today()
# 最大迭代次数
Maximum_Iterations = 10
# 问题要分段的次数
segment_num = 3

# 保存种群信息文件地址
algorithm_data_path = r'D:\Pythonproject\TelescopeScheduling\Algorithmdata\{}-{}-seed{}-time{}'.format(telescope_number,
                                                                                                       temp_get_data_num,
                                                                                                       SEED, TODAY)
# algorithm_data_path = r'D:\ZKY\TelescopeScheduling\Algorithmdata\{}-{}-seed{}-time{}'.format(telescope_number,
#                                                                                                        temp_get_data_num,
#                                                                                                        SEED, TODAY)
# 新加入种群的解与种群中其他个体的最小相同望远镜队列差异数
Min_telescope_process_difference = 0
# 初始化望远镜信息
telescope_list = []
for _ in range(telescope_number):
    telescope_list.append(Telescope())
# 读取数据集信息
observation_target_data = readfile()
list1 = []
for z_ in range(temp_get_data_num):
    list1.append(observation_target_data[z_])
# 计算目标总数量
count = []
for _ in list1:
    # 获取每个观测目标的编号
    if _[0] not in count:
        count.append(_[0])
total_target_num = len(count)
print('目标总数据量:', total_target_num)
# 数据排序为分段做准备
list2 = sorted(list1, key=lambda x: x[2])
# 存放分段后的数据
block_list = []


# 分段函数
def block_problem(list_temp, n):
    # 递归次数
    n += 1
    # 获取分段点
    waypoint = int((list_temp[0][2] + list_temp[-1][3]) / 2)
    list1_temp0 = []
    list1_temp1 = []
    # 分段操作
    for i in range(len(list_temp)):
        if list_temp[i][3] <= waypoint:
            list1_temp0.append(list_temp[i])
        elif list_temp[i][2] >= waypoint:
            list1_temp1.append(list_temp[i])
        else:
            if waypoint - list_temp[i][2] >= 90:
                temp0 = deepcopy(list_temp[i])
                temp0[3] = waypoint
                list1_temp0.append(temp0)
            if list_temp[i][3] - waypoint >= 90:
                temp1 = deepcopy(list_temp[i])
                temp1[2] = waypoint
                list1_temp1.append(temp1)
    if n < segment_num - 1:
        block_problem(list1_temp0, n)
        block_problem(list1_temp1, n)
    else:
        block_list.append(list1_temp0)
        block_list.append(list1_temp1)


block_problem(list2, 0)
# 分段块数
block_num = len(block_list)


class Individual:
    def __init__(self, solution, value, temp_telescope_list, temp_unobserved_list):
        self.solution = solution
        self.value = value
        self.temp_telescope_list = temp_telescope_list
        self.temp_unobserved_list = temp_unobserved_list
        self.traveled_insert = [1.0 for _ in range(temp_target_num)]
        self.traveled_exchange = [1.0 for _ in range(temp_target_num)]
        self.traveled_two_opt = [1.0 for _ in range(temp_target_num)]
        # self.traveled_mutation = [1.0 for _ in range(temp_get_data_num)]
        self.__telescope_divide_process = None

    def update(self):
        pass

    # 获取每个望远镜的队列信息
    def get_telescope_divide_process(self):
        if self.__telescope_divide_process is None:
            telescope_divide_process = [[] for _ in range(telescope_number)]
            for i_ in range(telescope_number):
                for _ in self.temp_telescope_list[i_].record_target_observation_sequence:
                    telescope_divide_process[i_].append(_[0])
            self.__telescope_divide_process = telescope_divide_process
            return self.__telescope_divide_process
        else:
            return self.__telescope_divide_process


class Algorithm:
    def __init__(self):
        # 迭代次数
        self.gen = 0
        # 种群
        self.popu = []
        self.convergence_curve_record = []
        # 邻域搜索权重记录列表
        self.operator_weight_list = [1.0 for _ in range(operator_num)]
        # 初始化种群
        self.init_pop()

        self.show_popu()
        self.best_indi = self.popu[-1]
        self.best_indi_gen = 0

    def reset(self):
        # 迭代次数
        self.gen = 0
        # 种群
        self.popu = []
        # 邻域搜索权重记录列表
        self.operator_weight_list = [1.0 for _ in range(operator_num)]
        # 初始化种群
        self.init_pop()
        self.show_popu()
        self.best_indi = self.popu[0]
        self.best_indi_gen = 0

    # 初始化种群
    def init_pop(self):
        while len(self.popu) < POP_SIZE:
            solution = self.generating_solution()

            # print(len(solution))
            solution, value, temp_telescope_list, temp_unobserved_list = self.insert_replacement_schedule(solution)

            # print(len(solution))
            temp_indi = Individual(solution, value, temp_telescope_list, temp_unobserved_list)
            self.select_individual_jion([temp_indi])
            # self.popu.append(temp_indi)

        self.popu = self.popu_sort(self.popu)
        self.popu.pop(-1)
        self.popu.pop(-1)

    # 生成单个解 初步拟定目标优先级的编码方式
    def generating_solution(self):
        # 存储不完整解的临时列表
        temp_solution = target_count
        # for _ in temp_target_data:
        #     # 获取每个观测目标的编号
        #     if _[0] not in temp_solution:
        #         temp_solution.append(_[0])
        # print(temp_soultion)
        # 使用洗牌算法打乱顺序
        # Fisher - Yates shuffle
        for _ in range(len(temp_solution)):
            r = random.randint(0, len(temp_solution) - 1)
            tmp = temp_solution[r]
            temp_solution[r] = temp_solution[_]
            temp_solution[_] = tmp
        # print(temp_solution)
        return temp_solution

    # 根据solution改变数据顺序
    def data_sort(self, solution):
        temp_target_data1 = deepcopy(temp_target_data)
        target_data_sort = []
        for i_ in range(len(solution)):
            for n__ in temp_target_data1:
                if solution[i_] == n__[0]:
                    target_data_sort.append(n__)
        return target_data_sort

    # 输出种群中的解
    def show_popu(self):
        for _ in self.popu:
            print(_.value, _.solution)

    # 种群个体排序
    def popu_sort(self, popu):
        popu = sorted(popu, key=lambda x: x.value)
        return popu

    # 返回调度后的解
    def dispatched_solution(self, telescope_list, solution):
        temp_solution = []
        temp_list = []
        for k_ in range(telescope_number):
            # 遍历每个望远镜观测队列
            for i_ in telescope_list[k_].record_target_observation_sequence:
                if i_ not in temp_list:
                    temp_list.append(i_)
        temp_list = sorted(temp_list, key=lambda x: x[2])
        for x_ in range(len(temp_list)):
            if temp_list[x_][0] not in temp_solution:
                temp_solution.append(temp_list[x_][0])
        for y_ in solution:
            if y_ not in temp_solution:
                temp_solution.append(y_)
        return temp_solution

    # 解码函数 插入替换调度
    def insert_replacement_schedule(self, solution):
        # print('开始分配观测任务给望远镜')
        temp_telescope_list = deepcopy(telescope_list)
        # 目标价值
        target_value = 0
        # 已观测目标队列
        observed_target_list = []
        # 数据根据解排序
        # print(len(solution))
        target_data_sort = self.data_sort(solution)

        # print(len(solution))
        for o_ in target_data_sort:
            if o_[0] in observed_target_list:
                continue
            for t in range(telescope_number):
                # 若开始观测时间大于等于望远镜工作结束时间加入望远镜观测   尾插
                if temp_telescope_list[t].time + 90 <= o_[3]:
                    temp = o_
                    # 获取望远镜观测时间
                    temp[2] = max(temp_telescope_list[t].time, o_[2])
                    temp[3] = temp[2] + 90
                    # 更新望远镜时间
                    temp_telescope_list[t].time = temp[3]
                    # 添加望远镜观测记录
                    temp_telescope_list[t].record_target_observation_sequence.append(temp)
                    temp_telescope_list[t].record_target_observation_sequence = sorted(
                        temp_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                    # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                    # 更新望远镜观测目标总价值
                    target_value += (10 - o_[1])
                    # 更新以观察目标记录
                    observed_target_list.append(o_[0])
                    # is_observed = True
                    break
                # 头插
                elif temp_telescope_list[t].record_target_observation_sequence[0][2] - o_[2] >= 90:
                    temp = o_
                    temp[3] = temp[2] + 90
                    # 添加望远镜观测记录
                    temp_telescope_list[t].record_target_observation_sequence.append(temp)
                    temp_telescope_list[t].record_target_observation_sequence = sorted(
                        temp_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                    # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                    # 更新望远镜观测目标总价值
                    target_value += (10 - o_[1])
                    # 更新以观察目标记录
                    observed_target_list.append(o_[0])
                    # is_observed = True
                    break
                # 队中插入
                else:
                    t_ = 0
                    # 指示变量表示目标是否插入望远镜空闲时间被观测
                    is_observed = False
                    while t_ < (len(temp_telescope_list[t].record_target_observation_sequence) - 1):
                        # 如果望远镜两次观测之间没有空闲时间跳过
                        if temp_telescope_list[t].record_target_observation_sequence[t_][3] == \
                                temp_telescope_list[t].record_target_observation_sequence[t_ + 1][2]:
                            t_ += 1
                            continue
                        # 如果目标可观测时间小于望远镜空闲时间结束循环
                        elif o_[3] < temp_telescope_list[t].record_target_observation_sequence[t_][3]:
                            break
                        # 如果望远镜空闲时间段小于90s跳过
                        elif temp_telescope_list[t].record_target_observation_sequence[t_ + 1][2] - \
                                temp_telescope_list[t].record_target_observation_sequence[t_][3] < 90:
                            t_ += 1
                            continue
                        # 如果望远镜空闲时间段开始时间大于目标可观测时间跳过
                        elif o_[3] - temp_telescope_list[t].record_target_observation_sequence[t_][3] < 90:
                            t_ += 1
                            continue
                        # 如果望远镜空闲时间段小于目标可观测时间跳过
                        elif temp_telescope_list[t].record_target_observation_sequence[t_ + 1][2] - o_[2] < 90:
                            t_ += 1
                            continue
                        # 满足加入条件加入望远镜空闲时间观测
                        else:
                            temp = o_
                            temp[2] = max(temp_telescope_list[t].record_target_observation_sequence[t_][3], o_[2])
                            temp[3] = temp[2] + 90
                            # 加入望远镜观测
                            temp_telescope_list[t].record_target_observation_sequence.append(temp)
                            temp_telescope_list[t].record_target_observation_sequence = sorted(
                                temp_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                            # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                            # print('未观测目标加入望远镜')
                            # 更新以观察目标记录
                            observed_target_list.append(o_[0])
                            target_value += (10 - o_[1])
                            is_observed = True
                            break
                    if is_observed:
                        break

        # 未观测目标列表
        unobserved_list = []
        # 查找未观测的目标加入列表
        for _ in target_data_sort:
            # 如果目标未被观测
            if _[0] not in observed_target_list:
                unobserved_list.append(_)
        # 将未观测的目标根据目标等级降序排列
        unobserved_list = sorted(unobserved_list, key=lambda x: x[1])
        unobserved_list.reverse()
        # 符合约束条件的高等级目标替换低等级目标
        # temp_telescope_list = deepcopy(telescope_list)

        for _ in unobserved_list:
            # 指示变量表示目标是否替换低等级目标
            is_observed = False
            # 获取望远镜观测记录
            for i in range(telescope_number - 1, -1, -1):
                t_ = 0
                while t_ < (len(temp_telescope_list[i].record_target_observation_sequence)):
                    # 若未观测目标等级更高
                    if _[1] < temp_telescope_list[i].record_target_observation_sequence[t_][1]:
                        # 且符合替换队列中低等级目标的时间条件
                        """这里除了一个目标的观测时间  还可以改成一个目标的观测时间加空闲时间  或者其他形式 
                        主体思想用观测价值更大的目标替换观测价值不大的目标"""
                        if temp_telescope_list[i].record_target_observation_sequence[t_][2] >= _[2] and \
                                temp_telescope_list[i].record_target_observation_sequence[t_][3] <= _[3]:
                            target_value += temp_telescope_list[i].record_target_observation_sequence[t_][1] - _[1]
                            # print('高等级目标', _[0], '替换低等级目标',
                            #       temp_telescope_list[i].record_target_observation_sequence[t_][0],
                            #       '超出等级', _[1] - temp_telescope_list[i].record_target_observation_sequence[t_][1])
                            temp_telescope_list[i].record_target_observation_sequence[t_][0] = _[0]
                            temp_telescope_list[i].record_target_observation_sequence[t_][1] = _[1]
                            is_observed = True
                            break
                    t_ += 1
                if is_observed:
                    break

        # print(temp_target_data)
        # print('未观测目标数量:', len(unobserved_list))
        # print('替换结束观测价值为：', target_value)
        temp_solution = self.dispatched_solution(temp_telescope_list, solution)
        # print(len(temp_solution))
        # print(temp_target_data)
        return temp_solution, target_value, temp_telescope_list, unobserved_list

    # 插入算子
    def insertion_operator(self, indi_temp):
        indi_backup = deepcopy(indi_temp)
        # 选择插入的元素
        choosed_list = self.chose_n_by_weight(weight=indi_temp.traveled_insert, n=1,
                                              index=[_ for _ in range(temp_target_num)])
        if choosed_list:
            r = choosed_list[0]
            indi_temp.traveled_insert[r] = 0
        else:
            return
        temp = indi_backup.solution.pop(r)
        for insert_postion in range(len(indi_backup.solution)):
            indi = deepcopy(indi_backup)
            if insert_postion != r:
                # 选择插入位置
                indi.solution.insert(insert_postion, temp)
                indi.solution, indi.value, indi.temp_telescope_list, indi.temp_unobserved_list = self.insert_replacement_schedule(
                    indi.solution)
                # 找到邻域更优才停，否则返回NONE
                if indi.value > indi_temp.value:
                    self.operator_weight_list[0] += 1
                    return indi
                else:
                    if random.random() < 0.005:
                        self.operator_weight_list[1] += 0.5
                        self.operator_weight_list[2] += 0.5
                        # self.operator_weight_list[3] += 0.5
                        return indi
        self.operator_weight_list[1] += 0.5
        self.operator_weight_list[2] += 0.5
        # self.operator_weight_list[3] += 0.5

    # 交换算子
    def commutative_operator(self, indi_temp):
        # temp_commutative = []
        choosed_list = self.chose_n_by_weight(weight=indi_temp.traveled_exchange, n=1,
                                              index=[_ for _ in range(temp_target_num)])
        if choosed_list:
            r1 = choosed_list[0]
            indi_temp.traveled_exchange[r1] = 0
        else:
            return
        for r2 in range(temp_target_num):
            # 已进行交换过不在交换
            if indi_temp.traveled_exchange[r2] == 0:
                continue
            # 两个相同的工件不需要交换
            # print('r1', r1, 'r2', r2)
            if indi_temp.solution[r1] != indi_temp.solution[r2]:
                # 将单个解中两个位置的元素交换
                _temp_solution = deepcopy(indi_temp.solution)
                _temp = _temp_solution[r2]
                _temp_solution[r2] = _temp_solution[r1]
                _temp_solution[r1] = _temp
                solution, value, temp_telescope_list, temp_unobserved_list = self.insert_replacement_schedule(
                    _temp_solution)
                indi = Individual(solution, value, temp_telescope_list, temp_unobserved_list)
                if indi.value > indi_temp.value:
                    self.operator_weight_list[1] += 1
                    return indi
                else:
                    if random.random() < 0.005:
                        self.operator_weight_list[0] += 0.5
                        self.operator_weight_list[2] += 0.5
                        # self.operator_weight_list[3] += 0.5
                        return indi
        self.operator_weight_list[0] += 0.5
        self.operator_weight_list[2] += 0.5
        # self.operator_weight_list[3] += 0.5

    # 2opt算子
    def two_opt(self, indi):
        indi_temp = deepcopy(indi)
        # 随机选取对解翻转的位置
        choosed_list = self.chose_n_by_weight(weight=indi_temp.traveled_two_opt, n=1,
                                              index=[_ for _ in range(temp_target_num)])
        if choosed_list:
            point1 = choosed_list[0]
            indi.traveled_two_opt[point1] = 0
        else:
            return
        # point1 = np.random.randint(0, solution_len - 2)
        for point2 in range(len(indi_temp.solution) - 1):
            if point1 != point2:
                (st, ed) = (point1, point2) if point1 < point2 else (point2, point1)
                indi_temp.solution = self.list_reverse(li=indi_temp.solution, start=st, end=ed)
                indi_temp.solution, indi_temp.value, indi_temp.temp_telescope_list, indi_temp.temp_unobserved_list = self.insert_replacement_schedule(
                    indi_temp.solution)
                # 找到邻域更优就停止，否则返回NONE
                if indi_temp.value > indi.value:
                    self.operator_weight_list[2] += 1
                    return indi_temp
                else:
                    if random.random() < 0.005:
                        self.operator_weight_list[0] += 0.5
                        self.operator_weight_list[1] += 0.5
                        # self.operator_weight_list[3] += 0.5
                        return indi
        self.operator_weight_list[0] += 0.5
        self.operator_weight_list[1] += 0.5
        # self.operator_weight_list[3] += 0.5

    # 变异算子
    def mutation_operator(self, indi_temp):
        indi_backup = deepcopy(indi_temp)
        # 选择变异的元素
        choosed_list = self.chose_n_by_weight(weight=indi_temp.traveled_mutation, n=1,
                                              index=[_ for _ in range(temp_get_data_num)])
        if choosed_list:
            r = choosed_list[0]
            indi_temp.traveled_mutation[r] = 0
        else:
            return
        # 保存当前变异位置的初始值
        temp_t = indi_backup.solution[r]
        for _ in range(telescope_number):
            if _ == temp_t:
                continue
            else:
                temp_solution = deepcopy(indi_backup.solution)
                temp_solution[r] = _
                solution, value, temp_telescope_list, temp_unobserved_list = self.insert_replacement_schedule(
                    temp_solution)
                new_indi = Individual(solution, value, temp_telescope_list, temp_unobserved_list)
                if new_indi.value > indi_backup.value:
                    self.operator_weight_list[3] += 1
                    return new_indi
                else:
                    if random.random() < 0.005:
                        self.operator_weight_list[0] += 0.5
                        self.operator_weight_list[1] += 0.5
                        self.operator_weight_list[2] += 0.5
                        return new_indi
        self.operator_weight_list[0] += 0.5
        self.operator_weight_list[1] += 0.5
        self.operator_weight_list[2] += 0.5

    # 邻域搜索
    def neighborhood_search(self, indi):
        # neighb_search_operators = [self.insertion_operator, self.commutative_operator, self.two_opt,
        #                            self.mutation_operator]
        neighb_search_operators = [self.insertion_operator, self.commutative_operator, self.two_opt]
        random_operator_pointer = self.chose_n_by_weight(weight=self.operator_weight_list, n=1,
                                                         index=[_ for _ in range(len(neighb_search_operators))])[0]
        # print('random_operator_pointer', random_operator_pointer[0], '************')
        # indi_temp = deepcopy(indi)
        new_indi = neighb_search_operators[random_operator_pointer](indi)
        if new_indi:
            if new_indi.value > indi.value:
                return new_indi

    # 选择个体加入种群
    def select_individual_jion(self, temp_popu):
        # 若种群为空直接将个体加入种群
        if len(self.popu) == 0:
            print('新个体加入')
            self.popu.append(temp_popu.pop(0))
        for _ in temp_popu:
            # print(_.time, '**')
            # 若新个体质量比种群中最差的还差且popu满了直接跳过
            if _.value <= self.popu[0].value:
                # print('质量比种群中最差的还差')
                # 如果popu满了直接跳过
                if len(self.popu) == POP_SIZE:
                    continue
            # 可能能加入
            # 判断是否存在同类解
            # 保存更差的同类解位置
            similar_postion = []
            # 同类解标记 若为TRUE存在同类解
            similar_flag = False
            # 有相似解但是新解的完工时间更大被淘汰 为TRUE表示存在同类解切新解性能更差
            unqualified_solution = False
            for i_ in range(len(self.popu)):
                if self.is_similar(indi1=_, indi2=self.popu[i_]):
                    # 对更新后种群中出现的更差相同解进行替换
                    if similar_flag:
                        similar_postion.append(i_)
                        continue
                    # popu里有同类解选择完工时间最小的加入种群
                    if _.value > self.popu[i_].value:
                        similar_flag = True
                        similar_postion.append(i_)
                        continue
                    else:
                        # 有相似解但是新解的完工时间更大被淘汰
                        unqualified_solution = True
                        break
            if unqualified_solution:
                # 有相似解但是新解的完工时间更大被淘汰 为TRUE表示存在同类解切新解性能更差
                # 跳出循环
                break
            if not similar_flag:  # popu里无同类解 插入
                # print('无同类解')
                # print('_.solution:', _.solution)
                if len(self.popu) == POP_SIZE:
                    print("没有找到同类解，且比种群中最差的个体优秀，替换最差的解")
                    self.popu[0] = _
                elif len(self.popu) < POP_SIZE:
                    print('新个体加入')
                    self.popu.append(_)
                else:
                    exit('种群溢出bug')

                # self.showpopu()
                self.popu = self.popu_sort(self.popu)
            else:
                for pop_pos in similar_postion[::-1]:
                    print('移除相似解', self.popu[pop_pos].value)
                    self.popu.pop(pop_pos)
                self.popu.append(_)
                print('新个体加入')
                self.popu = self.popu_sort(self.popu)

    # 判断解不同类  False为不相似
    def is_similar(self, indi1, indi2):
        same_count = self.telescope_divide_same_count(indi1, indi2)
        if same_count > telescope_number - Min_telescope_process_difference:
            return True
        return False

    # 将两个解按望远镜划分成数条队列，若在相同望远镜中的队列顺序相同，计数加一
    def telescope_divide_same_count(self, indi1, indi2):
        indi1_telescope_divide = deepcopy(indi1.get_telescope_divide_process())
        indi2_telescope_divide = deepcopy(indi2.get_telescope_divide_process())
        count = 0
        for _ in indi1_telescope_divide:
            for j in range(len(indi2_telescope_divide)):
                if self.telescope_process_compare(_, indi2_telescope_divide[j]):
                    count += 1
                    indi2_telescope_divide.pop(j)
                    break
        return count

    # 绘制收敛曲线
    def draw_convergence_curve(self):
        temp_value = []
        temp_iters = []
        for _ in self.convergence_curve_record:
            temp_value.append(_[0])
            temp_iters.append(_[1])
        plt.figure(dpi=500)
        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('value')  # y轴标签
        # 反转y轴
        plt.ylim(max(temp_value), min(temp_value))

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(temp_iters, temp_value, linewidth=1, linestyle="solid", label="Observation value")
        plt.legend()
        plt.title('Convergence curve')
        plt.show()
        # plt.savefig('{}-{}-{}.png'.format(temp_get_data_num, telescope_number, block_num), dpi=500)

    def test(self):
        pass

    def solve(self):
        # 记录时间
        st_time = time()
        # 记录未更新代数
        update_num = 0
        # # 记录收敛曲线
        self.convergence_curve_record.append([self.best_indi.value, self.best_indi_gen])
        # self.reset()
        while True:
            temp_popu = []
            for i_ in range(len(self.popu)):
                # 保存第_个解邻域搜索的得到的解并排序
                new_indi = self.neighborhood_search(indi=self.popu[i_])
                # print(self.popu[i_].traveled_two_opt)
                # print(self.popu[i_].traveled_insert)
                # print(self.popu[i_].traveled_exchange)
                if new_indi:
                    print('产生新个体')
                    temp_popu.append(new_indi)
            # 更新种群
            self.select_individual_jion(temp_popu)
            temp_popu = []
            # 保存最优解
            if self.popu[-1].value > self.best_indi.value:
                self.best_indi = self.popu[-1]
                self.best_indi_gen = self.gen + 1
                update_num = 0
                print('best_value:', self.best_indi.value, self.best_indi.solution)
                # current_run_time = time() - st_time
                # # 记录收敛曲线
                self.convergence_curve_record.append([self.best_indi.value, self.best_indi_gen])
            self.gen += 1
            update_num += 1
            self.show_popu()
            print('gen:', self.gen, 'current_best_value:', self.popu[-1].value, 'cur_worst_time', self.popu[0].value)
            print('插入算子：', self.operator_weight_list[0], '交换算子：', self.operator_weight_list[1], '2-opt:',
                  self.operator_weight_list[2])
            # 淘汰迭代一定次数还没有更新的个体
            for z_ in range(len(self.popu) - 1, -1, -1):
                # 二阶段搜索结束
                if sum(self.popu[z_].traveled_exchange) + sum(self.popu[z_].traveled_insert) + sum(
                        self.popu[z_].traveled_two_opt) <= (temp_target_num * 3 * 0.5):
                    temp = self.popu.pop(z_)
                    print('搜索结束淘汰个体:', z_)
            # 生成新个体填充种群
            while len(self.popu) < POP_SIZE:
                solution = self.generating_solution()
                solution, value, temp_telescope_list, temp_unobserved_list = self.insert_replacement_schedule(solution)
                temp_indi = Individual(solution, value, temp_telescope_list, temp_unobserved_list)
                self.select_individual_jion([temp_indi])
                # self.popu.append(temp_indi)
            self.popu = self.popu_sort(self.popu)
            # if time() - st_time > run_time:
            #     return
            if update_num >= 5:
                return
            if self.gen > Maximum_Iterations:
                # self.draw_convergence_curve()
                return


    # 根据权重赌盘选择n个点（下标）
    def chose_n_by_weight(self, weight, n, index, key=lambda x: x):
        # print(weight)
        weight = [key(_) for _ in weight]
        if n > 0:
            # 根据权重以赌盘选择方式选择一个节点
            total = sum(weight)
            if total < EPISILON:
                return []
            prob = 0.
            r = random.random()
            for i in range(len(weight)):
                prob += (weight[i] / total)
                if prob > r:  # 选中第i个节点
                    choosed = [index[i]]
                    weight.pop(i)
                    index.pop(i)
                    return self.chose_n_by_weight(weight, n - 1, index) + choosed
        else:
            return []

    # 比较机器中的工序是否相等
    @staticmethod
    def machine_process_compare(list1, list2):
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                return False
        return True

    # 比较机器中的工序是否相等
    @staticmethod
    def telescope_process_compare(list1, list2):
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if list1[i] != list2[i]:
                return False
        return True

    @staticmethod
    def list_reverse(li, start, end):  # 翻转列表的一部分
        # 包括start和end
        li[start: end + 1] = [li[_] for _ in range(end, start - 1, -1)]
        return li

    @staticmethod
    def split_list(origin_list, n):  # 把列表分为n份，最后一份长度不一定和前面一样
        cnt = len(origin_list) // n

        for i in range(0, n):
            if i < n - 1:
                yield origin_list[i * cnt: (i + 1) * cnt]
            else:
                yield origin_list[i * cnt:]

    @staticmethod
    def list_count_tar_num(list, tar):  # 计算列表中某个值出现的次数
        count = 0
        for _ in list:
            if _ == tar:
                count += 1
        return count

    @staticmethod
    def print_2dlist(list, format=False, enum=False):  # 打印二维列表
        if enum:
            if format:
                for e, i in enumerate(list):
                    for j in i:
                        print(e + 1, '%.2f' % j, end=' ')
                    print()
            else:
                for e, _ in enumerate(list):
                    print(e + 1, _)
        else:
            if format:
                for i in list:
                    for j in i:
                        print('%.2f' % j, end=' ')
                    print()
            else:
                for _ in list:
                    print(_)


# 返回队列中的目标和未观测目标
def target_state_in_solution(indi):
    indi_temp = deepcopy(indi)
    observed_list = []
    unobserved_list = []
    for w_ in range(telescope_number):
        for i__ in indi_temp.temp_telescope_list[w_].record_target_observation_sequence:
            observed_list.append(i__[0])
    for j__ in indi_temp.solution:
        if j__ not in observed_list:
            unobserved_list.append(j__)
    return observed_list, unobserved_list


# 可视化函数
def telescope_observation_visualization(indi_list):
    # 数据可视化
    # 输出中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 设置画布大小和分辨率
    plt.figure(figsize=(16, 9), dpi=500)
    # 列行下标
    plt.xlabel(
        "观测时间\n" + str('总目标数：' + str(total_target_num) + '已观测目标数：' + str(len(ob_list)) + '总价值' + str(value_temp)))
    plt.ylabel("望远镜编号")
    # 坐标轴刻度
    y_ticks = np.arange(0, telescope_number, 1)
    plt.yticks(y_ticks)
    for t__ in indi_list:
        telescope_list_temp = deepcopy(t__.temp_telescope_list)
        for _ in range(telescope_number):
            for i_ in telescope_list_temp[_].record_target_observation_sequence:
                plt.barh(y=_, width=90, left=i_[2], edgecolor='black', linewidth=1)
                # plt.text(i_[2], _, i_[0], fontsize=5)
    plt.show()


# 拼接所有子问题的解，将拼接后的解中未观测目标进行插入替换
def montage(indi_list, value_temp):
    total_telescope_list = deepcopy(telescope_list)
    temp_ob_list = deepcopy(ob_list)
    # 遍历所有子问题观测队列，获得一个原问题的一个观测队列
    for t__ in indi_list:
        telescope_list_temp = deepcopy(t__.temp_telescope_list)
        for _ in range(telescope_number):
            # 将子问题的观测队列拼接
            total_telescope_list[_].record_target_observation_sequence.extend(
                telescope_list_temp[_].record_target_observation_sequence)
    # 更新每个望远镜的观测结束时间
    for t_ in range(telescope_number):
        if len(total_telescope_list[t_].record_target_observation_sequence) != 0:
            # print(total_telescope_list[t_].record_target_observation_sequence)
            total_telescope_list[t_].time = total_telescope_list[t_].record_target_observation_sequence[-1][3]
    # 未观测目标列表
    total_unobserved_list = []
    # 查找未观测的目标加入列表
    for _ in list1:
        # 如果目标未被观测
        if _[0] not in temp_ob_list:
            total_unobserved_list.append(_)
    for o_ in total_unobserved_list:
        if o_[0] in temp_ob_list:
            continue
        for t in range(telescope_number):
            # 若开始观测时间大于等于望远镜工作结束时间加入望远镜观测   尾插
            if total_telescope_list[t].time + 90 <= o_[3]:
                temp = o_
                # 获取望远镜观测时间
                temp[2] = max(total_telescope_list[t].time, o_[2])
                temp[3] = temp[2] + 90
                # 更新望远镜时间
                total_telescope_list[t].time = temp[3]
                # 添加望远镜观测记录
                total_telescope_list[t].record_target_observation_sequence.append(temp)
                total_telescope_list[t].record_target_observation_sequence = sorted(
                    total_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                # 更新望远镜观测目标总价值
                value_temp += (10 - o_[1])
                # 更新以观察目标记录
                temp_ob_list.append(o_[0])
                # is_observed = True
                break
            # 头插
            elif total_telescope_list[t].record_target_observation_sequence[0][2] - o_[2] >= 90:
                temp = o_
                temp[3] = temp[2] + 90
                # 添加望远镜观测记录
                total_telescope_list[t].record_target_observation_sequence.append(temp)
                total_telescope_list[t].record_target_observation_sequence = sorted(
                    total_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                # 更新望远镜观测目标总价值
                value_temp += (10 - o_[1])
                # 更新以观察目标记录
                temp_ob_list.append(o_[0])
                # is_observed = True
                break
            # 队中插入
            else:
                t_ = 0
                # 指示变量表示目标是否插入望远镜空闲时间被观测
                is_observed = False
                while t_ < (len(total_telescope_list[t].record_target_observation_sequence) - 1):
                    # 如果望远镜两次观测之间没有空闲时间跳过
                    if total_telescope_list[t].record_target_observation_sequence[t_][3] == \
                            total_telescope_list[t].record_target_observation_sequence[t_ + 1][2]:
                        t_ += 1
                        continue
                    # 如果目标可观测时间小于望远镜空闲时间结束循环
                    elif o_[3] < total_telescope_list[t].record_target_observation_sequence[t_][3]:
                        break
                    # 如果望远镜空闲时间段小于90s跳过
                    elif total_telescope_list[t].record_target_observation_sequence[t_ + 1][2] - \
                            total_telescope_list[t].record_target_observation_sequence[t_][3] < 90:
                        t_ += 1
                        continue
                    # 如果望远镜空闲时间段开始时间大于目标可观测时间跳过
                    elif o_[3] - total_telescope_list[t].record_target_observation_sequence[t_][3] < 90:
                        t_ += 1
                        continue
                    # 如果望远镜空闲时间段小于目标可观测时间跳过
                    elif total_telescope_list[t].record_target_observation_sequence[t_ + 1][2] - o_[2] < 90:
                        t_ += 1
                        continue
                    # 满足加入条件加入望远镜空闲时间观测
                    else:
                        temp = o_
                        temp[2] = max(total_telescope_list[t].record_target_observation_sequence[t_][3], o_[2])
                        temp[3] = temp[2] + 90
                        # 加入望远镜观测
                        total_telescope_list[t].record_target_observation_sequence.append(temp)
                        total_telescope_list[t].record_target_observation_sequence = sorted(
                            total_telescope_list[t].record_target_observation_sequence, key=lambda x: x[2])
                        # print('目标编号：', temp[0], '目标等级：', temp[1], '观测开始时间：', temp[2], '观测结束：', temp[3])
                        # print('未观测目标加入望远镜')
                        # 更新以观察目标记录
                        temp_ob_list.append(o_[0])
                        value_temp += (10 - o_[1])
                        is_observed = True
                        break
                if is_observed:
                    break

    # 未观测目标列表
    unobserved_list = []
    # 查找未观测的目标加入列表
    for _ in list1:
        # 如果目标未被观测
        if _[0] not in temp_ob_list:
            unobserved_list.append(_)
    # 将未观测的目标根据目标等级降序排列
    unobserved_list = sorted(unobserved_list, key=lambda x: x[1])
    unobserved_list.reverse()
    # 符合约束条件的高等级目标替换低等级目标
    # temp_telescope_list = deepcopy(telescope_list)
    for _ in unobserved_list:
        # 指示变量表示目标是否替换低等级目标
        is_observed = False
        # 获取望远镜观测记录
        for i in range(telescope_number - 1, -1, -1):
            t_ = 0
            while t_ < (len(total_telescope_list[i].record_target_observation_sequence)):
                # 若未观测目标等级更高
                if _[1] < total_telescope_list[i].record_target_observation_sequence[t_][1]:
                    # 且符合替换队列中低等级目标的时间条件
                    """这里除了一个目标的观测时间  还可以改成一个目标的观测时间加空闲时间  或者其他形式 
                    主体思想用观测价值更大的目标替换观测价值不大的目标"""
                    if total_telescope_list[i].record_target_observation_sequence[t_][2] >= _[2] and \
                            total_telescope_list[i].record_target_observation_sequence[t_][3] <= _[3]:
                        value_temp += total_telescope_list[i].record_target_observation_sequence[t_][1] - _[1]
                        # print('高等级目标', _[0], '替换低等级目标',
                        #       temp_telescope_list[i].record_target_observation_sequence[t_][0],
                        #       '超出等级', _[1] - temp_telescope_list[i].record_target_observation_sequence[t_][1])
                        total_telescope_list[i].record_target_observation_sequence[t_][0] = _[0]
                        total_telescope_list[i].record_target_observation_sequence[t_][1] = _[1]
                        is_observed = True
                        break
                t_ += 1
            if is_observed:
                break

    # print(temp_target_data)
    # print('未观测目标数量:', len(unobserved_list))
    # print('替换结束观测价值为：', target_value)
    return total_telescope_list, value_temp, temp_ob_list


if __name__ == '__main__':
    st_time = time()
    # 保存每个分段得到的观测队列信息
    indi_list = []
    # 记录收敛曲线数据
    temp_convergence_curve = []
    # 保存已观测目标
    ob_list = []
    for q__ in range(block_num):
        if q__ != 0:
            # 暂时获取前XX个数据
            temp_target_data = deepcopy(block_list[q__])
            for k_ in range(len(temp_target_data) - 1, -1, -1):
                if temp_target_data[k_][0] in ob_list:
                    temp_target_data.pop(k_)
        else:
            temp_target_data = deepcopy(block_list[q__])
        # 计算目标数量
        target_count = []
        for _ in temp_target_data:
            # 获取每个观测目标的编号
            if _[0] not in target_count:
                target_count.append(_[0])
        temp_target_num = len(target_count)

        algo = Algorithm()

        algo.solve()

        indi0 = deepcopy(algo.best_indi)
        # 保存收敛曲线数据
        temp_convergence_curve.append(algo.convergence_curve_record)
        indi_list.append(indi0)
        # 保存已观测目标
        ob_list_temp, unob_list_temp = target_state_in_solution(indi0)
        ob_list.extend(ob_list_temp)
        print('第', q__, '部分搜索结束')
    value_temp = 0
    for v__ in indi_list:
        value_temp += v__.value

    print('问题观测价值：', value_temp)
    total_telescope_list, value_temp, ob_list = montage(indi_list, value_temp)
    print(str('总目标数：' + str(total_target_num) + '已观测目标数：' + str(len(ob_list)) + '总价值' + str(value_temp)))
    end_time = time() - st_time
    print("总运行时间：", end_time)
    # telescope_observation_visualization(indi_list)
    # file = open(algorithm_data_path, 'wb')
    # print('save successfully')
    # pickle.dump(indi_list, file)
    # file.close()
