# coding=utf-8
# demo by https://www.jianshu.com/p/0ad3761e3614
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# 声明两个变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer()  # 初始化全部变量
saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    sess.run(init_op)
    print("v1:", sess.run(v1))  # 打印v1、v2的值一会读取之后对比
    print("v2:", sess.run(v2))
    # 定义保存路径，一定要是绝对路径，且用‘/ '分隔父目录与子目录
    saver_path = saver.save(sess, '/tmp/tensorflow/ckpt/demo08_save_ckpt_a/model.ckpt')
    print("Model saved in file:", saver_path)

