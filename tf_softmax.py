'''
Created on 2017年9月21日

@author: weizhen
'''
import tensorflow as tf
import numpy as np
def softmax(x):
    """
    tensorflow 版本的softmax函数
    compute the softmax function in tensorflow
    interal functions may be used:
    tf.exp,tf.reduce_max,tf.reduce_sum,tf.expend_dims
    Args:
        x:tf.Tensor with shape (n_samples,n_features)
        feature vectors are represented by row-vectors (no need to handle 1-d
        input as in the previous homework)
    Returns:
        out:tf.Tensor with shape (n_sample,n_features). You need to construct
        this tensor in this problem
    """
    
    # tf.reduce_max沿着tensorflow的某一维度计算元素的最大值
    # tf.reduce_sum沿着tensorflow的某一维度计算元素的和
    # tf.expand_dims在tensorflow的某一维度插入一个tensor
    maxes = tf.expand_dims(tf.reduce_max(x, reduction_indices=[1]), 1)
    x_red = x - maxes
    x_exp = tf.exp(x_red)
    sums = tf.expand_dims(tf.reduce_sum(x_exp, reduction_indices=[1]), 1)
    out = x_exp / sums
    
    return out

def cross_entropy_loss(y, yhat):
    """
                  计算交叉熵在tensorflow中
       y是一个one-hot tensor  大小是(n_samples,n_classes)这么大，类型是tf.int32
       yhat是一个tensor 大小是(n_samples,n_classes)  类型是 tf.float32
       function:
           tf.to_float,tf.reduce_sum,tf.log可能会用到
                  参数:
           y:tf.Tensor with shape(n_samples,n_classes) One-hot encoded
           yhat: tf.Tensorwith shape (n_samples,n_classes) Each row encodes a
                probability distribution and should sum to 1
                  返回:
           out: tf.Tensor with shape(1,) (Scalar output).You need to construct
              this tensor in the problem.
    """
    y = tf.to_float(y)
    out = -tf.reduce_sum(y * tf.log(yhat))
    return out

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive
    """
    print("Running basic tests...")
    test1 = softmax(tf.convert_to_tensor(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6
    test2 = softmax(tf.convert_to_tensor(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session():
        test2 = test2.eval()
    assert np.amax(np.fabs(test2 - np.array([0.73105858, 0.26894142]))) <= 1e-6
    print("Basic (non-exhaustive) softmax tests pass\n")
    
def test_cross_entropy_loss_basic():
    """
    Some simple tests to get you started
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])
    
    test1 = cross_entropy_loss(tf.convert_to_tensor(y, dtype=tf.int32),
                               tf.convert_to_tensor(yhat, dtype=tf.float32))
    with tf.Session():
        test1 = test1.eval()
    result = -3 * np.log(.5)
    assert np.amax(np.fabs(test1 - result)) <= 1e-6
    print("Basic (non-exhaustive) cross-entropy tests pass\n")
    
if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
