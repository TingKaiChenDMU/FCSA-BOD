import os # 导入os模块，模块的概念后面讲到
file_path = '/home/tingkai/1workspace/laptop/4_01_yolov3_pytorch/logs'

# 循环遍历所有的文件
pth_file_name = [d for d in os.listdir(file_path)] # os.listdir可以列出文件和目录

# 只保存带有.pth文件,防止有readme文件
pth_file_name_update = [data for data in pth_file_name if data.split('.')[-1]=='pth']

# # 写成字典
dict_pre = {pth_file_name_update[i].split('-')[0].split('Epoch')[-1]: pth_file_name_update[i] for i in range(len(pth_file_name_update))}

# # 以epoch顺序，按照key：value方式排序
dict_aft = {int(i+1):dict_pre[str(i+1)] for i in range(len(dict_pre)) }

# 按照顺序，只取value
output = [dict_aft[i+1] for i in range(len(dict_aft))]
print(output)


