# coding:utf-8
# 两层简单神经网络实现（全连接）
# 基于TensorFlow的神经网络
# 用张量表示数据，用计算图搭建神经网络，用会话执行。

import tensorflow as tf

# 定义输入和参数
# stddev:标准差, seed:随机种子
x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
# y is: [[3.0904665]]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y is:", sess.run(y))
