# coding=utf-8
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

c = tf.constant(1.5)
g = tf.Graph()

with g.as_default():

    c1 = tf.constant(2.0)
    print(c1.graph)
    print(g)
    print(c.graph)

g2 = tf.get_default_graph()
print(g2)

tf.reset_default_graph()
g3 = tf.get_default_graph()
print(g3)