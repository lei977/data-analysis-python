import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

# index_data = [17, 375, 3950, 25.5, 90, 0.272, 62, 2.7, 1]
# index_data = [17, 375, 3950, 25.5, 90, 0.272, 62, 2.7, 1]

index_data = [[17, 375, 3950, 25.5, 90, 0.272, 62, 2.7, 1],
              [20, 420, 3750, 27.5, 92, 0.28, 53, 2.6, 1],
              [21.5, 435, 3500, 28, 91.5, 0.285, 54, 2.65, 1],
              [21, 450, 3450, 28, 90.5, 0.287, 50, 2.9, 1],
              [22, 470, 3050, 29, 86, 0.29, 49, 3.25, 1],
              [24, 485, 3000, 36, 81, 0.282, 48, 3.45, 1],
              [27.5, 495, 2800, 40.5, 75, 0.285, 47, 3.6, 1],
              [29.5, 510, 2600, 41, 66, 0.28, 46, 3.7, 1],
              [17, 365, 4100, 28.5, 92, 0.27, 61, 2.9, 0.5],
              [20, 400, 4050, 29, 93, 0.275, 60, 2.85, 0.5],
              [22, 410, 3950, 30, 91, 0.28, 60, 2.9, 0.5],
              [24, 440, 3650, 36.5, 91.5, 0.283, 56, 2.8, 0.5],
              [25, 455, 3450, 38, 86, 0.29, 55, 2.95, 0.5],
              [25.5, 490, 3300, 39.5, 84, 0.285, 55, 3.2, 0.5],
              [28, 500, 3150, 43, 81, 0.286, 53, 3.3, 0.5],
              [29, 510, 2250, 47, 74, 0.295, 52, 3.4, 0.5],
              [16.5, 390, 3900, 31, 89, 0.275, 26, 3.2, 0],
              [20.5, 480, 2950, 41.5, 94, 0.272, 25, 3.25, 0],
              [22.5, 500, 2900, 44, 91, 0.275, 24, 3.3, 0],
              [25.5, 520, 2850, 48, 90.5, 0.28, 23, 3.45, 0],
              [26, 545, 2800, 50.5, 92, 0.283, 22, 3.35, 0],
              [27, 570, 2700, 52.5, 88, 0.28, 23.5, 3.55, 0],
              [28, 580, 2600, 54.5, 82, 0.278, 22, 3.6, 0],
              [29.5, 590, 2400, 58, 79, 0.275, 21, 3.7, 0]]

index_data = np.array(index_data)
# print(index_data.shape)
print('data:', index_data)
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

print('new data:', index_data)

# new data: [[0.34  0.25  0.79  0.255 0.9   0.544 0.62  0.54  1.   ]
#  [0.4   0.34  0.75  0.275 0.92  0.56  0.53  0.52  1.   ]
#  [0.43  0.37  0.7   0.28  0.915 0.57  0.54  0.53  1.   ]
#  [0.42  0.4   0.69  0.28  0.905 0.574 0.5   0.58  1.   ]
#  [0.44  0.44  0.61  0.29  0.86  0.58  0.49  0.65  1.   ]
#  [0.48  0.47  0.6   0.36  0.81  0.564 0.48  0.69  1.   ]
#  [0.55  0.49  0.56  0.405 0.75  0.57  0.47  0.72  1.   ]
#  [0.59  0.52  0.52  0.41  0.66  0.56  0.46  0.74  1.   ]
#  [0.34  0.23  0.82  0.285 0.92  0.54  0.61  0.58  0.5  ]
#  [0.4   0.3   0.81  0.29  0.93  0.55  0.6   0.57  0.5  ]
#  [0.44  0.32  0.79  0.3   0.91  0.56  0.6   0.58  0.5  ]
#  [0.48  0.38  0.73  0.365 0.915 0.566 0.56  0.56  0.5  ]
#  [0.5   0.41  0.69  0.38  0.86  0.58  0.55  0.59  0.5  ]
#  [0.51  0.48  0.66  0.395 0.84  0.57  0.55  0.64  0.5  ]
#  [0.56  0.5   0.63  0.43  0.81  0.572 0.53  0.66  0.5  ]
#  [0.58  0.52  0.45  0.47  0.74  0.59  0.52  0.68  0.5  ]
#  [0.33  0.28  0.78  0.31  0.89  0.55  0.26  0.64  0.   ]
#  [0.41  0.46  0.59  0.415 0.94  0.544 0.25  0.65  0.   ]
#  [0.45  0.5   0.58  0.44  0.91  0.55  0.24  0.66  0.   ]
#  [0.51  0.54  0.57  0.48  0.905 0.56  0.23  0.69  0.   ]
#  [0.52  0.59  0.56  0.505 0.92  0.566 0.22  0.67  0.   ]
#  [0.54  0.64  0.54  0.525 0.88  0.56  0.235 0.71  0.   ]
#  [0.56  0.66  0.52  0.545 0.82  0.556 0.22  0.72  0.   ]
#  [0.59  0.68  0.48  0.58  0.79  0.55  0.21  0.74  0.   ]]

# 原数据调用
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        return index_data
        # samples = np.random.normal(self.mu, self.sigma, N)
        # samples.sort()
        # print('samples:', samples)
        # return samples


# 生成数据调用
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        # print('GeneratorDistribution-sample:', np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01)
        # return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01
        return (np.random.random(N) - 0.5) * 2
        # return index_data


def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    # 如果没有小批处理层，鉴别器需要一个额外的层来有足够的能力正确地分离两个分布
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    有时由于数值四舍五入，描述器输出可以达到接近(甚至略小于)零的值。这只是确保我们排除了那些值，这样我们就不会在优化过程中得到NaNs。
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        # 这定义了生成器网络——它从噪声中提取样本分布作为输入，并通过MLP传递它们。
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        # 鉴别器试图区分真实数据分布(self.x)和生成的样本(self.z)之间的差异。
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        # 这里我们创建了共享参数的判别器网络的两个副本，因为您不能在TensorFlow中使用具有不同输入的同一个网络。
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        # 定义鉴别器和生成器网络的损失(详见原文)，并为两者创建优化器
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # update discriminator 更新分类器
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            # update generator 更新生成器
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(
                    samples(model, session, data, gen.range, params.batch_size)
                )

        if params.anim_path:
            save_animation(anim_frames, params.anim_path, gen.range)
        else:
            samps = samples(model, session, data, gen.range, params.batch_size)
            plot_distributions(samps, gen.range)


def samples(
        model,
        session,
        data,
        sample_range,
        batch_size,
        num_points=10000,
        num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    返回一个元组(db, pd, pg)，其中db为当前决策边界，pd为数据分布中样本的直方图，pg为生成样本的直方图。
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary 决定边界
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )

    # data distribution 数据分布
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples 生成的样本
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=1), args)


def parse_args():
    parser = argparse.ArgumentParser()
    # 循环迭代轮数
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='the number of training steps to take')
    # 多层感知机隐含层的大小
    parser.add_argument('--hidden-size', type=int, default=9,
                        help='MLP hidden size')
    # 批次大小
    parser.add_argument('--batch-size', type=int, default=9,
                        help='the batch size')
    # 最小批次
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    # 每轮之后记录损失函数
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    # 输出动态图片文件的位置
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    # 保存每N帧动画
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
