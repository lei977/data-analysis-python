# coding = utf-8

import tensorflow as tf

# 张量（tensor）：多维数组（列表）
# 阶：张量的维数
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b

# Tensor("add:0", shape=(2,), dtype=float32)
print(result)

# 计算图（Graph）：搭建神经网络的计算过程，只搭建，不运算
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])

y = tf.matmul(x, w)

# Tensor("MatMul:0", shape=(1, 1), dtype=float32)
print(y)

# 会话（session）：执行计算图中的节点运算
with tf.Session() as sess:
    print(sess.run(y))
