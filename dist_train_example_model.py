# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from cluster_helper import  get_cluster_device_info, get_sess

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
tf.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.flags.DEFINE_string("data_dir", "./train_data",
                    "Directory for storing training data")


IMAGE_PIXELS = 28
mnist = None
batch_size = 100


def get_input(data_dir):
    """
    输入数据的加载到内存对象
    :param data_dir: 训练数据目录
    :return: 加载的数据对象
    """
    global mnist
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    return mnist


def next_batch(data_dir):
    """
    输入数据的加载到内存对象
    :param data_dir: 训练数据目录
    :return: 要训练的批次数据
    """
    global mnist
    if not mnist:
        get_input(data_dir)
    return mnist.train.next_batch(batch_size)


def get_validate_data():
    """
    验证数据集
    :return: X, y
    """
    return mnist.validation.images, mnist.validation.labels


def model_graph(device_info):
    """
    build the graph
    :return: x, y_, loss, train_step, global_step, opt
    """
    with tf.device(device_info):
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])
        hidden_units = 100
        learning_rate = 0.01
        global_step = tf.Variable(0, name="global_step", trainable=False)
        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name="hid_w")
        hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
        sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                                               stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)
        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        opt = tf.train.AdamOptimizer(learning_rate)
        train_step = opt.minimize(cross_entropy, global_step=global_step)
        loss = cross_entropy
    return x, y_, loss, train_step, global_step, opt


def main(args):
    # get the cluster info
    device_info = get_cluster_device_info()
    x, y_, loss, train_step, global_step, opt = model_graph(device_info)

    # get needed session and special treat for opt
    sess, train_step = get_sess(opt, loss, global_step, FLAGS.log_dir)

    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    local_step = 0
    while True:
        batch_xs, batch_ys = next_batch(FLAGS.data_dir)
        train_feed = {x: batch_xs, y_: batch_ys}
        _, step = sess.run([train_step, global_step], feed_dict=train_feed)
        local_step += 1
        now = time.time()
        print("%f: Worker %d: training step %d done (global step: %d)" %
              (now, FLAGS.task_index, local_step, step))
        if step >= FLAGS.train_steps:
            break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # Validation feed
    validate_x, validate_y = get_validate_data()
    val_feed = {x: validate_x, y_: validate_y}
    val_xent = sess.run(loss, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))

if __name__ == '__main__':
    tf.app.run()
