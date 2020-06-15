# coding=utf-8
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor_1 = tf.matmul(a, b, name='matmul_1')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1 = tf.get_default_graph().get_operation_by_name(name='matmul_1')
    t2 = tf.get_default_graph().get_tensor_by_name(name='matmul_1:0')
    print(t1)
    print('t1: ', sess.run(t1))
    print('t2: ', sess.run(t2))