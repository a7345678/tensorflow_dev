# coding=utf-8
# fix https://www.cnblogs.com/123456www/p/12584427.html
import tensorflow as tf
# fix with tensorflow2.0 by ausar
tf.compat.v1.disable_eager_execution()
# 创建两个常量节点
node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(adder))