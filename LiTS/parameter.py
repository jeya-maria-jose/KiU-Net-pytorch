
train_ct_path = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/LITS17/train/CT/'  

train_seg_path = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/LITS17/train/seg/'  

test_ct_path = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/LITS17/test/CT/'  

test_seg_path = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/LITS17/test/seg/'  

training_set_path = './train/'  
pred_path = './liver_pred'  

crf_path = './crf' 

module_path = './module/kiunet2d/net150-0.183-0.172.pth'  

size = 48  # 使用48张连续切片作为网络的输入

down_scale = 0.5  # 横断面降采样因子

expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本

slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

upper, lower = 200, -200  # CT数据灰度截断窗口

# ---------------------训练数据获取相关参数-----------------------------------


# -----------------------网络结构相关参数------------------------------------

drop_rate = 0.3  # dropout随机丢弃概率

# -----------------------网络结构相关参数------------------------------------


# ---------------------网络训练相关参数--------------------------------------

gpu = '0'  # 使用的显卡序号

Epoch = 200

learning_rate = 1e-4

learning_rate_decay = [500, 750]

alpha = 0.33  # 深度监督衰减系数

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

# ---------------------网络训练相关参数--------------------------------------


# ----------------------模型测试相关参数-------------------------------------

threshold = 0.5  # 阈值度阈值

stride = 12  # 滑动取样步长

maximum_hole = 5e4  # 最大的空洞面积

# ----------------------模型测试相关参数-------------------------------------


# ---------------------CRF后处理优化相关参数----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # 根据预测结果在三个方向上的扩展数量

max_iter = 20  # CRF迭代次数

s1, s2, s3 = 1, 10, 10  # CRF高斯核参数

# ---------------------CRF后处理优化相关参数----------------------------------