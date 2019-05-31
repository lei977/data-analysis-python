# coding:utf-8
import tensorflow as tf

row_dim = 1
col_dim = 1

# 1、固定张量
# 创建指定维度的零张量
zero_tsr = tf.zeros([row_dim, col_dim])
# 创建指定维度的单位张量
ones_tsr = tf.ones([row_dim, col_dim])
# 创建指定维度的常数填充的张量
filled_str = tf.fill([row_dim, col_dim], 42)
# 用已知常数张量创建一个张量
constant_str = tf.constant([1, 2, 3])

# 2、相似形状的张量
# 新建一个与给定的tensor类型大小一致的tensor，其所有元素为0或1，使用方式如下：
zeros_similar = tf.zeros_like(constant_str)
ones_similar = tf.ones_like(constant_str)

# 3、序列张量
# TensorFlow可以创建指定间隔的张量。
linear_str = tf.linspace(start=0, stop=1, num=3)

# 4、随机张量
# 生成均匀分布的随机数：
randunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)
# 生成正态分布的随机数
randnorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
# 生成带有指定边界的正态分布的随机数，其正态分布的随机数位于指定均值（期望）到两个标准差之间的区间：
rancnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
# 张量/数组的随机化
input_tensor = 1
shuffled_output = tf.random_shuffle(input_tensor)
crop_size = 1
corpped_output = tf.random_crop(input_tensor, crop_size)
# 张量的随机剪裁
# cropped_image = tf.random_crop(my_image, [height / 2, width / 2, 3])

# 封装张量来作为变量
my_var = tf.Variable(tf.zeros([row_dim, col_dim]))
