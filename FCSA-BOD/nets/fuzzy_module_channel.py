import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
# import tensorflow as tf
# import matplotlib.pyplot as plt
import time


def fuzzy_channel_inference(data_avg, data_max):

    # 定义range
    avg = np.arange(0, 1, 0.01, np.float32)
    max = np.arange(0, 1, 0.01, np.float32)
    merge = np.arange(0, 1, 0.01, np.float32)

    # 创建模糊控制变量
    avg_v = ctrl.Antecedent(avg, 'avg_attention')
    max_v = ctrl.Antecedent(max, 'max_attention')
    merge_v = ctrl.Consequent(merge, 'merge_attention')

    # 定义模糊集和其隶属度函数
    avg_v['S'] = fuzz.gaussmf(avg_v.universe,  0, 0.2)
    avg_v['M'] = fuzz.gaussmf(avg_v.universe,  0.5, 0.2)
    avg_v['L'] = fuzz.gaussmf(avg_v.universe,  1, 0.2)


    max_v['S'] = fuzz.gaussmf(max_v.universe, 0, 0.2)
    max_v['M'] = fuzz.gaussmf(max_v.universe, 0.5, 0.2)
    max_v['L'] = fuzz.gaussmf(max_v.universe, 1, 0.2)

    merge_v['S'] = fuzz.gaussmf(merge_v.universe, 0, 0.2)
    merge_v['M'] = fuzz.gaussmf(merge_v.universe, 0.5, 0.2)
    merge_v['L'] = fuzz.gaussmf(merge_v.universe, 1, 0.2)

    # 设置解模糊方法
    merge_v.defuzzify_method = 'centroid'

    # 输出为S规则
    rule1 = ctrl.Rule(antecedent=((avg_v['S'] & max_v['S'])|
                                  (avg_v['M'] & max_v['S'])|
                                  (avg_v['S'] & max_v['M'])),
                      consequent=merge_v['S'], label='rule S')

    # 规则为M的规则
    rule2 = ctrl.Rule(antecedent=((avg_v['L'] & max_v['S'])|
                                  (avg_v['M'] & max_v['M'])|
                                  (avg_v['L'] & max_v['M'])|
                                  (avg_v['S'] & max_v['L'])|
                                  (avg_v['M'] & max_v['L'])),
                      consequent=merge_v['M'], label='rule M')

    # 规则为L的规则
    rule3 = ctrl.Rule(antecedent=((avg_v['L'] & max_v['L'])),
                      consequent=merge_v['L'], label='rule L')


    # 系统和运行环境初始化
    system = ctrl.ControlSystem(rules=[rule1, rule2, rule3])
    sim = ctrl.ControlSystemSimulation(system)

    # 传入输入，
    sim.input['avg_attention'] = data_avg
    sim.input['max_attention'] = data_max

    # 运行系统
    sim.compute()
    output_attention = sim.output['merge_attention']

    return output_attention


#     print(output_attention)
#
#     # 查看模糊子集图片
#     avg_v.view(sim=sim)
#     max_v.view(sim=sim)
#     merge_v.view(sim=sim)
#
#
# fuzzy_channel_inference(0.2, 0.3)

