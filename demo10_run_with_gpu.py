# coding=utf-8
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        add = tf.add(a, b)
        sum = sess.run(add, feed_dict={a: 3, b: 4})
        print(sum)