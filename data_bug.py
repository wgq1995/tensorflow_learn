# coding:utf-8
"""
一个维度的tensor t1（3，）和两个维度的tensor t2（3， 1）做加减法时：
    返回的是 （3， 3）的tensor
    相当于 t2 的每一个元素依次和 t1 的每个元素相加减
"""

import tensorflow as tf


if __name__ == "__main__":

    a = tf.Variable([
        [0.5],
        [0.2],
        [0.3]
    ])  # 二维tensor

    b = tf.Variable([
        0.6,
        0.1,
        1.4
    ])  # 一维tensor

    c = tf.abs(a - b)
    d = tf.abs(a - tf.reshape(b, shape=(-1, 1)))
    loss1 = tf.losses.absolute_difference(tf.reshape(b, (-1, 1)), a)
    loss2 = tf.reduce_mean(loss1)

    loss3 = tf.reduce_mean(c)
    loss4 = tf.reduce_mean(d)

    with tf.Session() as sess:
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            tf.tables_initializer(),
        ])

        aa, bb, cc, dd, l1, l2, l3, l4 = sess.run([
            a, b, c, d, loss1, loss2, loss3, loss4
        ])

        print("aa shape is: ", aa.shape)
        print(aa)

        print("bb shape is: ", bb.shape)
        print(bb)

        print("cc shape is: ", cc.shape)
        print(cc)

        print("dd shape is: ", dd.shape)
        print(dd)

        print("l1 = {}\n l2 = {}\n l3 = {}\n l4 = {}".format(
            l1, l2, l3, l4
        ))
        
