# coding:utf-8

# GAN：生成器采用8层BP神经网络
# 0导入模块，生成模拟数据集。
import tensorflow as tf
import numpy as np

# 设置数据格式
# np.set_printoptions(precision=3)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

index_data = [[17, 375, 3950, 25.5, 90, 0.272, 62, 2.7],
              [20, 420, 3750, 27.5, 92, 0.28, 53, 2.6],
              [21.5, 435, 3500, 28, 91.5, 0.285, 54, 2.65],
              [21, 450, 3450, 28, 90.5, 0.287, 50, 2.9],
              [22, 470, 3050, 29, 86, 0.29, 49, 3.25],
              [24, 485, 3000, 36, 81, 0.282, 48, 3.45],
              [27.5, 495, 2800, 40.5, 75, 0.285, 47, 3.6],
              [29.5, 510, 2600, 41, 66, 0.28, 46, 3.7],
              [17, 365, 4100, 28.5, 92, 0.27, 61, 2.9],
              [20, 400, 4050, 29, 93, 0.275, 60, 2.85],
              [22, 410, 3950, 30, 91, 0.28, 60, 2.9],
              [24, 440, 3650, 36.5, 91.5, 0.283, 56, 2.8],
              [25, 455, 3450, 38, 86, 0.29, 55, 2.95],
              [25.5, 490, 3300, 39.5, 84, 0.285, 55, 3.2],
              [28, 500, 3150, 43, 81, 0.286, 53, 3.3],
              [29, 510, 2250, 47, 74, 0.295, 52, 3.4],
              [16.5, 390, 3900, 31, 89, 0.275, 26, 3.2],
              [20.5, 480, 2950, 41.5, 94, 0.272, 25, 3.25],
              [22.5, 500, 2900, 44, 91, 0.275, 24, 3.3],
              [25.5, 520, 2850, 48, 90.5, 0.28, 23, 3.45],
              [26, 545, 2800, 50.5, 92, 0.283, 22, 3.35],
              [27, 570, 2700, 52.5, 88, 0.28, 23.5, 3.55],
              [28, 580, 2600, 54.5, 82, 0.278, 22, 3.6],
              [29.5, 590, 2400, 58, 79, 0.275, 21, 3.7]]

index_data = np.array(index_data)
# print(index_data.shape)
print('data:')
print(index_data)
# 数据预处理及标准化
for iii in index_data:
    # print(iii)
    # 脂肪酸值
    iii[0] = iii[0] / 50
    # 降落数值
    iii[1] = iii[1] / 500 - 0.5
    # 过氧化物酶
    iii[2] = iii[2] / 5000
    # 电导率
    iii[3] = iii[3] / 100
    # 发芽率
    iii[4] = iii[4] / 100
    # 还原糖
    iii[5] = iii[5] * 2
    # 沉降值
    iii[6] = iii[6] / 100
    # 丙二醛
    iii[7] = iii[7] / 5
    # 小麦品种->强筋：1；中筋：0.5；弱筋：0
    # iii[8]-=2

print('new data:')
print(index_data)

# 判定指标（筋力，储藏时间）：强筋1，中筋2，弱筋3。储藏时间1-8月内。
Y_ = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
      [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
      [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8]]

# # X = np.array(X1)
# X -= np.mean(X, axis=0)
# X /= np.std(X, axis=0)

BATCH_SIZE = 8

# 1定义神经网络的输入、参数和输出,定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 8))
y_ = tf.placeholder(tf.float32, shape=(None, 2))

w1 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w3 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w4 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w5 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w6 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w7 = tf.Variable(tf.random_normal([8, 8], stddev=1, seed=1))
w8 = tf.Variable(tf.random_normal([8, 2], stddev=1, seed=1))

a = tf.matmul(x, w1)
b = tf.matmul(a, w2)
c = tf.matmul(b, w3)
d = tf.matmul(c, w4)
e = tf.matmul(d, w5)
f = tf.matmul(e, w6)
g = tf.matmul(f, w7)
y = tf.matmul(g, w8)

# 2定义损失函数及反向传播方法。
loss_mse = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

    # 训练模型。
    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 8
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: index_data[start:end], y_: Y_[start:end]})
        if i % 100 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: index_data, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))

    # 输出训练后的参数取值。
    # print("w1:\n", sess.run(w1))
    # print("w2:\n", sess.run(w2))

    test_data = [[0.34, 0.25, 0.79, 0.255, 0.9, 0.544, 0.62, 0.54],
                 [0.4, 0.34, 0.75, 0.275, 0.92, 0.56, 0.53, 0.52],
                 [0.43, 0.37, 0.7, 0.28, 0.915, 0.57, 0.54, 0.53],
                 [0.42, 0.4, 0.69, 0.28, 0.905, 0.574, 0.5, 0.58],
                 [0.44, 0.44, 0.61, 0.29, 0.86, 0.58, 0.49, 0.65],
                 [0.48, 0.47, 0.6, 0.36, 0.81, 0.564, 0.48, 0.69],
                 [0.55, 0.49, 0.56, 0.405, 0.75, 0.57, 0.47, 0.72],
                 [0.59, 0.52, 0.52, 0.41, 0.66, 0.56, 0.46, 0.74]]

    aa = tf.matmul(test_data, w1)
    bb = tf.matmul(aa, w2)
    cc = tf.matmul(bb, w3)
    dd = tf.matmul(cc, w4)
    ee = tf.matmul(dd, w5)
    ff = tf.matmul(ee, w6)
    gg = tf.matmul(ff, w7)
    yy = tf.matmul(gg, w8)
    print('判定值：')
    print(np.round(yy.eval()))

    # 反向生成数据
    input_index = [[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4],
                   [2, 5],
                   [2, 6],
                   [3, 7],
                   [3, 8]]

    input_index = np.array(input_index)
    # w1 = np.array(w1)
    # w2 = np.array(w2)

    ww1 = np.array(sess.run(w1))
    ww2 = np.array(sess.run(w2))
    ww3 = np.array(sess.run(w3))
    ww4 = np.array(sess.run(w4))
    ww5 = np.array(sess.run(w5))
    ww6 = np.array(sess.run(w6))
    ww7 = np.array(sess.run(w7))
    ww8 = np.array(sess.run(w8))

    temp1 = np.dot(input_index, np.linalg.pinv(ww8))
    temp2 = np.dot(temp1, np.linalg.pinv(ww7))
    temp3 = np.dot(temp2, np.linalg.pinv(ww6))
    temp4 = np.dot(temp3, np.linalg.pinv(ww5))
    temp5 = np.dot(temp4, np.linalg.pinv(ww4))
    temp6 = np.dot(temp5, np.linalg.pinv(ww3))
    temp7 = np.dot(temp6, np.linalg.pinv(ww2))
    output_index = np.dot(temp7, np.linalg.pinv(ww1))

    # print('temp:', temp)
    # print('type(temp):', type(temp))
    # print('type(np.linalg.pinv(ww1)):', type(np.linalg.pinv(ww1)))
    # print('np.linalg.pinv(ww2):', np.linalg.pinv(ww2))
    print('output_index:')
    print(output_index)

    # 还原生成数据
    for iii in output_index:
        # print(iii)
        # 脂肪酸值
        # iii[0] = iii[0] / 50
        iii[0] = iii[0] * 50
        # 降落数值
        # iii[1] = iii[1] / 500 - 0.5
        iii[1] = (iii[1] + 0.5) * 500
        # 过氧化物酶
        # iii[2] = iii[2] / 5000
        iii[2] = iii[2] * 5000
        # 电导率
        # iii[3] = iii[3] / 100
        iii[3] = iii[3] * 100
        # 发芽率
        # iii[4] = iii[4] / 100
        iii[4] = iii[4] * 100
        # 还原糖
        # iii[5] = iii[5] * 2
        iii[5] = iii[5] / 2
        # 沉降值
        # iii[6] = iii[6] / 100
        iii[6] = iii[6] * 100
        # 丙二醛
        # iii[7] = iii[7] / 5
        iii[7] = iii[7] * 5

        # 小麦品种->强筋：1；中筋：0.5；弱筋：0
        # iii[8]-=2

    print('new data:')
    print(output_index)
    # np.set_printoptions(precision=3)
    # print('new data:', output_index)
    # print('new data(int):', output_index.astype(int))

    # print('生成数据:')
    # for i in output_index:
    #     print('index1:', i[0], 'index2:', i[1], 'index3:', i[2], 'index4:', i[3],
    #           'index5:', i[4], 'index6:', i[5], 'index7:', i[6], 'index8:', i[7])
