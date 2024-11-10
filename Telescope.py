class Telescope:
    def __init__(self):
        # 望远镜初始时间为0
        self.time = 0
        # 望远镜观测目标记录
        self.record_target_observation_sequence = []

    # 显示望远镜观测目标记录
    def show_telescope(self):
        for _ in self.record_target_observation_sequence:
            print(_)
        return self.record_target_observation_sequence
