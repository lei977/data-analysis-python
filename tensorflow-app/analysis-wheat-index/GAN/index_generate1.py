# coding:utf-8

# 参考源码
# https://blog.csdn.net/qq_38826019/article/details/80620922
# GAN生成数据

from __future__ import division, print_function, absolute_import
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import xrange

np.set_printoptions(threshold=np.inf)  # 输出全部矩阵不带省略号

# data = np.load('data/final37.npy')
# data = data[:, :, 0:60]
data = [[17, 375, 3950, 25.5, 90, 0.272, 62, 2.7, 1],
        [20, 420, 3750, 27.5, 92, 0.28, 53, 2.6, 1],
        [21.5, 435, 3500, 28, 91.5, 0.285, 54, 2.65, 1],
        [21, 450, 3450, 28, 90.5, 0.287, 50, 2.9, 1],
        [22, 470, 3050, 29, 86, 0.29, 49, 3.25, 1],
        [24, 485, 3000, 36, 81, 0.282, 48, 3.45, 1],
        [27.5, 495, 2800, 40.5, 75, 0.285, 47, 3.6, 1],
        [29.5, 510, 2600, 41, 66, 0.28, 46, 3.7, 1],
        [17, 365, 4100, 28.5, 92, 0.27, 61, 2.9, 2],
        [20, 400, 4050, 29, 93, 0.275, 60, 2.85, 2],
        [22, 410, 3950, 30, 91, 0.28, 60, 2.9, 2],
        [24, 440, 3650, 36.5, 91.5, 0.283, 56, 2.8, 2],
        [25, 455, 3450, 38, 86, 0.29, 55, 2.95, 2],
        [25.5, 490, 3300, 39.5, 84, 0.285, 55, 3.2, 2],
        [28, 500, 3150, 43, 81, 0.286, 53, 3.3, 2],
        [29, 510, 2250, 47, 74, 0.295, 52, 3.4, 2],
        [16.5, 390, 3900, 31, 89, 0.275, 26, 3.2, 3],
        [20.5, 480, 2950, 41.5, 94, 0.272, 25, 3.25, 3],
        [22.5, 500, 2900, 44, 91, 0.275, 24, 3.3, 3],
        [25.5, 520, 2850, 48, 90.5, 0.28, 23, 3.45, 3],
        [26, 545, 2800, 50.5, 92, 0.283, 22, 3.35, 3],
        [27, 570, 2700, 52.5, 88, 0.28, 23.5, 3.55, 3],
        [28, 580, 2600, 54.5, 82, 0.278, 22, 3.6, 3],
        [29.5, 590, 2400, 58, 79, 0.275, 21, 3.7, 3]]

data = np.array(data)
print(data.shape)


################################
# 显示图片
# index = 0
# for n in range(len(data)):
#     gen = data[index:index+100]
#     gen = gen.reshape(100,3,60,1)
#     r, c = 10, 10
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             xy = gen[cnt]#第n个分叉图，有三个分支，每个分支60个数
#             for k in range(len(xy)):
#                 x = xy[k][0:30]
#                 y = xy[k][30:60]
#                 if k == 0 :
#                     axs[i,j].plot(x,y,color='blue')
#                 if k == 1 :
#                     axs[i,j].plot(x,y,color='red')
#                 if k == 2 :
#                     axs[i,j].plot(x,y,color='green')
#                     axs[i,j].axis('off')
#             cnt += 1
#     index += 100
#     plt.show()
###################################
def get_inputs(real_size, noise_size):
    """
    真实图像tensor与噪声图像tensor
    """
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

    return real_img, noise_img


def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """
    生成器

    noise_img: 生成器的输入
    n_units: 隐层单元个数
    out_dim: 生成器输出tensor的size，这里应该为360*3=180
    alpha: leaky ReLU系数
    """
    with tf.variable_scope("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise_img, 128)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.batch_normalization(hidden1, momentum=0.8, training=True)
        hidden1 = tf.layers.dropout(hidden1, rate=0.25)

        hidden2 = tf.layers.dense(hidden1, 512)
        hidden2 = tf.maximum(alpha * hidden2, hidden2)
        hidden2 = tf.layers.batch_normalization(hidden2, momentum=0.8, training=True)
        hidden2 = tf.layers.dropout(hidden2, rate=0.25)

        # logits & outputs
        logits = tf.layers.dense(hidden2, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs


def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    """
    判别器

    n_units: 隐层结点数量
    alpha: Leaky ReLU系数
    """

    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden layer1
        hidden1 = tf.layers.dense(img, 512)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        # hidden layer2
        hidden2 = tf.layers.dense(hidden1, g_units)
        hidden2 = tf.maximum(alpha * hidden2, hidden2)

        # logits & outputs
        logits = tf.layers.dense(hidden2, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


# 定义参数
# batch_size
# batch_size = 256
batch_size = 8
# 训练迭代轮数
epochs = 20
# 抽取样本数
# n_sample = 100
n_sample = 10
# 真实图像的size
# img_size = 180
img_size = 9
# 传入给generator的噪声size
noise_size = 100
# 生成器隐层参数
g_units = 128
# 判别器隐层参数
d_units = 128
# leaky ReLU的参数
alpha = 0.01
# learning_rate
learning_rate = 0.0002  # 0.00002
# label smoothing
smooth = 0.1
tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)

# generator
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)

# discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)
# discriminator的loss
# 识别真实图片
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))
# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real, d_loss_fake)

# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
train_vars = tf.trainable_variables()

# generator中的tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator中的tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
saver = tf.train.Saver()


# 开始训练
def train():
    samples = []
    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for i in xrange(len(data) // batch_size):
                batch_images = data[i * batch_size:(i + 1) * batch_size]
                print("batch_images1",batch_images)
                batch_images = batch_images.reshape(batch_size, 9)
                print("batch_images2", batch_images)
                # batch_images = batch_images.reshape(batch_size, 180)
                # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
                batch_images = batch_images * 2 - 1
                print("batch_images3", batch_images)
                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                print("batch_noise", batch_noise)

                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
                _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})
            # 每一轮结束计算loss
            train_loss_d = sess.run(d_loss,
                                    feed_dict={real_img: batch_images,
                                               noise_img: batch_noise})
            print(train_loss_d)
            # real img loss
            train_loss_d_real = sess.run(d_loss_real,
                                         feed_dict={real_img: batch_images,
                                                    noise_img: batch_noise})
            print(train_loss_d_real)
            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake,
                                         feed_dict={real_img: batch_images,
                                                    noise_img: batch_noise})
            print(train_loss_d_fake)
            # generator loss
            train_loss_g = sess.run(g_loss,
                                    feed_dict={noise_img: batch_noise})
            print(train_loss_g)

            if e % 1 == 0:
                print('[' + datetime.now().strftime('%c') + ']', "...Epoch {}/{}...".format(e + 1, epochs),
                      "[ Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f}) ]...".format(train_loss_d,
                                                                                              train_loss_d_real,
                                                                                              train_loss_d_fake),
                      "[ Generator Loss: {:.4f} ] ".format(train_loss_g))
                # 记录各类loss值
                losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
            with open('loss.txt', 'a') as f:
                f.write('[' + datetime.now().strftime('%c') + ']' + "...Epoch {}/{}...".format(e + 1, epochs) +
                        "[ Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f}) ]...".format(train_loss_d,
                                                                                                train_loss_d_real,
                                                                                                train_loss_d_fake) +
                        "[ Generator Loss: {:.4f} ] ".format(train_loss_g) + '\n')
            ###########################################################################################################
            # 抽取样本后期进行观察
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            print(sample_noise)
            gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                                   feed_dict={noise_img: sample_noise})
            print(gen_samples)
            samples.append(gen_samples)
            saver.save(sess, 'checkpoints/b.ckpt')
            if e % 2 == 0:
                # gen = gen_samples[1].reshape(100, 3, 60, 1)
                gen = gen_samples[1].reshape(100, 3, 60, 1)
                print(gen.shape)
                r, c = 10, 10
                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        xy = gen[cnt]  # 第n个分叉图，有三个分支，每个分支21个数
                        for k in range(len(xy)):
                            x = xy[k][0:30]
                            y = xy[k][30:60]
                            if k == 0:
                                axs[i, j].plot(x, y, color='blue')
                            if k == 1:
                                axs[i, j].plot(x, y, color='red')
                            if k == 2:
                                axs[i, j].plot(x, y, color='green')
                                axs[i, j].axis('off')
                        cnt += 1
                if not os.path.exists('images2'):
                    os.makedirs('images2')
                fig.savefig("images2/%d.png" % e)
                plt.close()
    ############################################################################################################
    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    fig, ax = plt.subplots(figsize=(20, 7))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator Total Loss')
    plt.plot(losses.T[1], label='Discriminator Real Loss')
    plt.plot(losses.T[2], label='Discriminator Fake Loss')
    plt.plot(losses.T[3], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('a.png')
    plt.show()


############################################################################################################
# (-1,10)
# 加载我们的生成器变量
# def test():
#     saver = tf.train.Saver(var_list=g_vars)
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#         saver.restore(sess, 'checkpoints/b.ckpt')
#         sample_noise = np.random.uniform(-1, 1, size=(10000, noise_size))
#         gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
#                                feed_dict={noise_img: sample_noise})
#         gen_images = gen_samples[1]
#         gen_images = (gen_images + 1) / 2
#         print(np.max(gen_images))
#         print(np.min(gen_images))
#         print(gen_images.shape)
#         index = 0
#         for n in range(len(gen_images) // 100):
#             gen_image = gen_images[index:index + 100]
#             gen_image = gen_image.reshape(100, 3, 60, 1)
#             r, c = 10, 10
#             fig, axs = plt.subplots(r, c)
#             cnt = 0
#             for i in range(r):
#                 for j in range(c):
#                     xy = gen_image[cnt]  # 第n个分叉图，有三个分支，每个分支21个数
#                     for k in range(len(xy)):
#                         x = xy[k][0:30]
#                         y = xy[k][30:60]
#                         if k == 0:
#                             axs[i, j].plot(x, y, color='blue')
#                         if k == 1:
#                             axs[i, j].plot(x, y, color='red')
#                         if k == 2:
#                             axs[i, j].plot(x, y, color='green')
#                             axs[i, j].axis('off')
#                     cnt += 1
#             index += 100
#             plt.show()


##############################################
train()
# test()
