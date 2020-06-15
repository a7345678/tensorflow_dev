# coding=utf-8
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

g = tf.Graph()

with g.as_default():
    c1 = tf.constant(2.5, name='c1_constant')
    c2 = tf.Variable(1.5, dtype=tf.float32, name='c2_constant')
    add = tf.multiply(c1, c2, name='op_add')

    c_1 = g.get_tensor_by_name(name='c1_constant:0')
    c_2 = g.get_tensor_by_name(name='c2_constant:0')
    c_3 = g.get_tensor_by_name(name='op_add:0')


    print(c_1)
    print(c_2)
    print(c_3)