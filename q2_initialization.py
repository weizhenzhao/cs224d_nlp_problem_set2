'''
Created on 2017年9月22日

@author: weizhen
'''
import numpy as np
import tensorflow as tf

def xavier_weight_init():
    """
    Returns funtion that creates random tensor.
         这个特定的函数会接收一个数组是一维d列的
         一定要返回一个随机的有特定大小的
         这些过程都是在Xavier 分布的初始化 函数里边完成的
    """
    def _xavier_initializer(shape, **kwargs):
        """
                    定义一个初始化函数对于 Xavier分布
                    这个函数会被用作变量作用域的初始化
        Args:
            shape:Tuple or 1-d array that species dimensions of requested tensor
        Returns:
            out:tf.Tensor of specified shape sampled from Xavier distribution.
        """
        m = shape[0]
        n = shape[1] if len(shape) > 1 else shape[0]
        
        bound = np.sqrt(6) / np.sqrt(m + n)
        out = tf.random_uniform(shape, minval=-bound, maxval=bound)
        
        return out
    return _xavier_initializer

def test_initialization_basic():
    """
    Some simple tests for the initialization
    """
    print("Running basic tests...")
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    
    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    print("Basic (non-exhaustive) Xavier initialization tests pass")

def test_initialization():
    """
    Use this space to test your Xavier initialization code by running:
        python q1_initialization.py
    This function will not be called by the autograder, nor will
    your tests be graded
    """
    print("Running your tests...")
    raise NotImplementedError

if __name__ == "__main__":
    test_initialization_basic()
