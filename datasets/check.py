
import os

# 打印当前工作目录
current_dir = os.getcwd()
print("当前工作目录:", current_dir)

# 假设你的目标目录是这样的
#target_dir = './datasets/Synapse/train_npz'datasets/Synapse/test_vol_h5/case0001.npy.h5//datasets/Synapse/train_npz/case0024_slice005.npz
#target_dir = './Synapse/train_npz\\case0024_slice005.npz/'
target_dir = 'E:\Medicalsegmentationcode\MISSFormermain\\test_log\\test_log_\epoch_399.pth.txt'
# 获取目标目录的绝对路径
absolute_target_dir = os.path.abspath(target_dir)
print("目标目录的绝对路径:", absolute_target_dir)

# 检查目标目录是否存在
if os.path.exists(absolute_target_dir):
    print("目录存在。")
    print("目录内容:", os.listdir(absolute_target_dir))
else:
    print("目录不存在。请检查路径是否正确。")

# 如果存在疑问，可以列出上一级目录的内容来帮助确定问题
parent_dir = os.path.dirname(absolute_target_dir)
print("上级目录:", parent_dir)
print("上级目录的内容:", os.listdir(parent_dir))