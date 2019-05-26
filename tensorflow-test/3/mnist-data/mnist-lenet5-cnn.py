# coding:utf-8

import time
import tensorflow as tf
import os
import numpy as np
import struct
import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# import mnist_lenet5_forward
# import mnist_lenet5_backward

'''
lennet5卷积神经网络实现MNIST数据集手写数字识别
修改内容：
    forward、backward、test：三个文件整合进一个文件
    使用本地数据集：
        路径：./mnist-data/
        文件：t10k-images.idx3-bbyte, t10k-labels.idx3-bbyte, train-images.idx3-bbyte, train-labels.idx3-bbyte
    实现数据集文件读取操作

'''

# forward文件


IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y


# backwaed文件


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./lenet5/model/"
MODEL_NAME = "mnist_model"


def backward(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.IMAGE_SIZE,
        mnist_lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    y = mnist_lenet5_forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.IMAGE_SIZE,
                mnist_lenet5_forward.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# def main():
#     mnist = input_data.read_data_sets("./data/", one_hot=True)
#     backward(mnist)
#
#
# if __name__ == '__main__':
#     main()


# test文件


# TEST_INTERVAL_SECS = 5
#
#
# def test1(mnist):
#     with tf.Graph().as_default() as g:
#         x = tf.placeholder(tf.float32, [
#             mnist.test.num_examples,
#             mnist_lenet5_forward.IMAGE_SIZE,
#             mnist_lenet5_forward.IMAGE_SIZE,
#             mnist_lenet5_forward.NUM_CHANNELS])
#         y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
#         y = mnist_lenet5_forward.forward(x, False, None)
#
#         ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
#         ema_restore = ema.variables_to_restore()
#         saver = tf.train.Saver(ema_restore)
#
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#         while True:
#             with tf.Session() as sess:
#                 ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
#                 if ckpt and ckpt.model_checkpoint_path:
#                     saver.restore(sess, ckpt.model_checkpoint_path)
#
#                     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                     reshaped_x = np.reshape(mnist.test.images, (
#                         mnist.test.num_examples,
#                         mnist_lenet5_forward.IMAGE_SIZE,
#                         mnist_lenet5_forward.IMAGE_SIZE,
#                         mnist_lenet5_forward.NUM_CHANNELS))
#                     accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
#                     print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
#                 else:
#                     print('No checkpoint file found')
#                     return
#             time.sleep(TEST_INTERVAL_SECS)


# read文件：MNIST数据集读取


# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# fig, ax = plt.subplots(
#     nrows=2,
#     ncols=5,
#     sharex=True,
#     sharey=True, )
#
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。
    # 我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # 获得数据在缓存中的指针位置，
    # 从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。
    # 这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
    #        plt.imshow(images[i],'gray')
    #        plt.pause(0.00001)
    #        plt.show()
    # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# param: idx_ubyte_file: idx文件路径
# return: n*row*col维np.array对象，n为图片数量
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


# param: idx_ubyte_file: idx文件路径
# return: n*1维np.array对象，n为图片数量
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


# param: idx_ubyte_file: idx文件路径
# return: n*row*col维np.array对象，n为图片数量
def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


# param: idx_ubyte_file: idx文件路径
# return: n*1维np.array对象，n为图片数量
def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


# 主函数
def main():
    # mnist = input_data.read_data_sets("./data/", one_hot=True)

    path = './mnist-data/'

    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.pause(0.000001)
        plt.show()
    print('done')

    # mnist=1
    # test(mnist)


if __name__ == '__main__':
    main()
