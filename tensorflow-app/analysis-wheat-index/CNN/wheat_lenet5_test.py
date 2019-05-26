# coding:utf-8
import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import wheat_lenet5_forward
import wheat_lenet5_backward
import numpy as np

TEST_INTERVAL_SECS = 5


def test(wheat):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            wheat.test.num_examples,
            wheat_lenet5_forward.IMAGE_SIZE,
            wheat_lenet5_forward.IMAGE_SIZE,
            wheat_lenet5_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, wheat_lenet5_forward.OUTPUT_NODE])
        y = wheat_lenet5_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(wheat_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(wheat_lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(wheat.test.images, (
                        wheat.test.num_examples,
                        wheat_lenet5_forward.IMAGE_SIZE,
                        wheat_lenet5_forward.IMAGE_SIZE,
                        wheat_lenet5_forward.NUM_CHANNELS))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: wheat.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    # wheat = input_data.read_data_sets("./data/", one_hot=True)
    wheat = 1
    test(wheat)


if __name__ == '__main__':
    main()
