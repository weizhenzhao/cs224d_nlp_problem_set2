'''
Created on 2017年9月27日

@author: weizhen
'''
import tensorflow as tf
import numpy as np

A = [[1, 2, 3], [4, 5, 6]]
x = tf.split(1, 3, tf.convert_to_tensor(A))

with tf.Session() as sess:
    c = sess.run(x)
    for ele in c:
        print(ele)
