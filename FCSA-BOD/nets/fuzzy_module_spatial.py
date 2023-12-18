import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
# import tensorflow as tf
# import matplotlib.pyplot as plt
import time



def fuzzy_spatial_inference(data_3x3, data_5x5, data_7x7):

    # 定义range
    s3 = np.arange(0, 1, 0.01, np.float32)
    s5 = np.arange(0, 1, 0.01, np.float32)
    s7 = np.arange(0, 1, 0.01, np.float32)
    merge = np.arange(0, 1, 0.01, np.float32)

    # 创建模糊控制变量
    s3_v = ctrl.Antecedent(s3, 's3_attention')
    s5_v = ctrl.Antecedent(s5, 's5_attention')
    s7_v = ctrl.Antecedent(s7, 's7_attention')
    merge_v = ctrl.Consequent(merge, 'merge_attention')

    # 定义模糊集和其隶属度函数
    s3_v['S'] = fuzz.gaussmf(s3_v.universe,  0, 0.2)
    s3_v['M'] = fuzz.gaussmf(s3_v.universe,  0.5, 0.2)
    s3_v['L'] = fuzz.gaussmf(s3_v.universe,  1, 0.2)

    s5_v['S'] = fuzz.gaussmf(s5_v.universe, 0, 0.2)
    s5_v['M'] = fuzz.gaussmf(s5_v.universe, 0.5, 0.2)
    s5_v['L'] = fuzz.gaussmf(s5_v.universe, 1, 0.2)

    s7_v['S'] = fuzz.gaussmf(s7_v.universe, 0, 0.2)
    s7_v['M'] = fuzz.gaussmf(s7_v.universe, 0.5, 0.2)
    s7_v['L'] = fuzz.gaussmf(s7_v.universe, 1, 0.2)

    merge_v['S'] = fuzz.gaussmf(merge_v.universe, 0, 0.2)
    merge_v['M'] = fuzz.gaussmf(merge_v.universe, 0.5, 0.2)
    merge_v['L'] = fuzz.gaussmf(merge_v.universe, 1, 0.2)

    # 设置解模糊方法
    merge_v.defuzzify_method = 'centroid'

    # 规则1-9
    rule1 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['S'] & s7_v['S'])),consequent=merge_v['S'], label='rule 1')
    rule2 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['S'] & s7_v['M'])),consequent=merge_v['S'], label='rule 2')
    rule3 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['S'] & s7_v['L'])),consequent=merge_v['M'], label='rule 3')
    rule4 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['M'] & s7_v['S'])),consequent=merge_v['S'], label='rule 4')
    rule5 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['M'] & s7_v['M'])),consequent=merge_v['M'], label='rule 5')
    rule6 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['M'] & s7_v['L'])),consequent=merge_v['M'], label='rule 6')
    rule7 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['L'] & s7_v['S'])),consequent=merge_v['M'], label='rule 7')
    rule8 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['L'] & s7_v['M'])),consequent=merge_v['M'], label='rule 8')
    rule9 = ctrl.Rule(antecedent=((s3_v['S'] & s5_v['L'] & s7_v['L'])),consequent=merge_v['M'], label='rule 9')

    # 规则10-18
    rule10 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['S'] & s7_v['S'])),consequent=merge_v['S'], label='rule 10')
    rule11 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['S'] & s7_v['M'])),consequent=merge_v['M'], label='rule 11')
    rule12 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['S'] & s7_v['L'])),consequent=merge_v['M'], label='rule 12')
    rule13 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['M'] & s7_v['S'])),consequent=merge_v['M'], label='rule 13')
    rule14 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['M'] & s7_v['M'])),consequent=merge_v['M'], label='rule 14')
    rule15 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['M'] & s7_v['L'])),consequent=merge_v['M'], label='rule 15')
    rule16 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['L'] & s7_v['S'])),consequent=merge_v['M'], label='rule 16')
    rule17 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['L'] & s7_v['M'])),consequent=merge_v['M'], label='rule 17')
    rule18 = ctrl.Rule(antecedent=((s3_v['M'] & s5_v['L'] & s7_v['L'])),consequent=merge_v['M'], label='rule 18')

    # 规则19-27
    rule19 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['S'] & s7_v['S'])),consequent=merge_v['M'], label='rule 19')
    rule20 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['S'] & s7_v['M'])),consequent=merge_v['M'], label='rule 20')
    rule21 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['S'] & s7_v['L'])),consequent=merge_v['M'], label='rule 21')
    rule22 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['M'] & s7_v['S'])),consequent=merge_v['M'], label='rule 22')
    rule23 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['M'] & s7_v['M'])),consequent=merge_v['M'], label='rule 23')
    rule24 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['M'] & s7_v['L'])),consequent=merge_v['M'], label='rule 24')
    rule25 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['L'] & s7_v['S'])),consequent=merge_v['M'], label='rule 25')
    rule26 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['L'] & s7_v['M'])),consequent=merge_v['M'], label='rule 26')
    rule27 = ctrl.Rule(antecedent=((s3_v['L'] & s5_v['L'] & s7_v['L'])),consequent=merge_v['L'], label='rule 27')


    # 系统和运行环境初始化
    system = ctrl.ControlSystem(rules=[rule1,  rule2,  rule3,  rule4,  rule5,  rule6,  rule7,  rule8,  rule9,
                                       rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
                                       rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])

    sim = ctrl.ControlSystemSimulation(system)

    # 传入输入，
    sim.input['s3_attention'] = data_3x3
    sim.input['s5_attention'] = data_5x5
    sim.input['s7_attention'] = data_7x7

    # 运行系统
    sim.compute()
    output_attention = sim.output['merge_attention']

    return output_attention


    # print(output_attention)

#     # 查看模糊子集图片
#     s3_v.view(sim=sim)
#     s5_v.view(sim=sim)
#     s7_v.view(sim=sim)
#     merge_v.view(sim=sim)
#
#
# fuzzy_channel_inference(0.2, 0.3, 0.7)

