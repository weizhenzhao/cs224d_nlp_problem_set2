'''
Created on 2017年9月21日

@author: weizhen
'''
import time

from model import Model
import numpy as np
import tensorflow as tf
from tf_softmax import cross_entropy_loss
from tf_softmax import softmax
from utils import data_iterator


class Config(object):
    """
    Holds model hyperparams and data information
    The config class is used to store various hyperparameters and dataset
    information parameters.
    Model objects are passed a Config() object at instantiation.
    """
    batch_size = 64
    n_samples = 1024
    n_features = 100
    n_classes = 5
    #You may adjust the max_epochs to ensure Convergence
    max_epochs = 50
    #You may adjust this learning rate to ensure convergence.
    lr = 1e-4

class SoftmaxModel(Model):
    """实现一个softmax 交叉熵分类器"""
    def load_data(self):
        """创建一个预测数据集，并且在内存中储存它"""
        np.random.seed(1234)
        self.input_data = np.random.rand(self.config.n_samples,self.config.n_features)
        self.input_labels = np.ones((self.config.n_samples,), dtype=np.int32)
    
    def add_placeholders(self):
        """生成    placeholder 变量来呈现输入的 tensor 
                            这些    placeholders 是被用做输入，
                            在建立别的模型的时候还可以用到，
                            在训练的过程中可以对其填入数据
        input_placeholder:Input placeholder :tensor of shape (batch_size,n_features),type tf.float32
                          labels_placeholder:Labels placeholder tensor of shape (batch_size,n_classes),type tf.int32
        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        (Don't change the variable names)
    """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size,self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32,shape=(self.config.batch_size,self.config.n_classes))
    
    def create_feed_dict(self,input_batch,label_batch):
        """为softmax classifier创建feed_dict"""
        feed_dict={
            self.input_placeholder:input_batch,
            self.labels_placeholder:label_batch,
        }
        return feed_dict
    
    def add_training_op(self,loss):
        """设置训练目标，创建一个优化器，应用梯度下降到所有的训练变量上面
        Args:
            loss:Loss tensor,from cross_entropy_loss
        Returns:
            train_op:The Op for training
        """
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
        return train_op
    
    def add_model(self,input_data):
        """添加一个线性层，增加一个softmax变换
        y = softmax(xW+b)
        Hint:make sure to create tf.Variables as needed
             also make sure to use tf.name_scope to ensure that your name
             spaces are clean
        Hint:For this simple use-case, it's sufficient to initialize both
            weights W and biases b with zeros
        Args:
            input_data:A tensor of shape (batch_size,n_features)
        Returns:
            out:       A tensor of shape (batch_size,n_classes)
        """
        n_features,n_classes = self.config.n_features,self.config.n_classes
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(tf.zeros([n_features,n_classes]),name='weights')
            biases = tf.Variable(tf.zeros([n_classes]),name='biases')
            #矩阵乘法，不是内积
            logits = tf.matmul(input_data,weights)+biases
            out = softmax(logits)
        return out
    
    def add_loss_op(self,pred):
        """将交叉熵损失添加到目标的损失函数上
        Hint: use the cross_entropy_loss function we defined. This should be a very
              short function.
        Args:
            pred: A tensor of shape (batch_size,n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        return loss
    
    def run_epoch(self,sess,input_data,input_labels):
        """运行一段epoch大小数据的训练
            Trains the model for one-epoch
        Args:
            sess:tf.Session() object
            input_data : np.ndarray of shape (n_samples,n_features)
            input_labels : np.ndarray of shape (n_samples,n_classes)
        Returns:
            average_loss : scalar . Average minibatch loss of model on epoch
        """
        #And then after everything is built , start the training loop.
        average_loss = 0
        for step,(input_batch,label_batch) in enumerate(
            data_iterator(input_data,input_labels,
                          batch_size=self.config.batch_size,
                          label_size=self.config.n_classes)):
            #Fill a feed dictionary with the actual set of images and labels
            #for this particular training step
            feed_dict = self.create_feed_dict(input_batch, label_batch)
            
            #Run one step of the model. The return values are the activations
            #from the 'self.train_op' (which is discard) and the 'loss' Op.
            #To inspect the values of your Ops or variables, you may include then
            #in the list passed to sess.run() and the value tensors will be
            #returned in the tuple from the call.
            _,loss_value = sess.run([self.train_op,self.loss],feed_dict=feed_dict)
            average_loss += loss_value
        average_loss = average_loss / step
        return average_loss
    
    def fit(self,sess,input_data,input_labels):
        """Fit model on provided data
        Args:
            sess:tf.Session()
            input_data : np.ndarray of shape (n_samples,n_features)
            input_labels : np.ndarray of shape (n_samples,n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, input_data, input_labels)
            duration = time.time() - start_time
            #Print status to stdout
            print('Epoch %d: loss = %.2f (%3.f sec)'%(epoch,average_loss,duration))
            losses.append(average_loss)
        return losses
    
    def __init__(self,config):
        """Initializes the model.
           Args:
               config:A model configuration object of type Config
        """
        self.config = config
        #Generate placeholders for the images and labels
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

def test_SoftmaxModel():
    """Train softmax model for a number of steps."""
    config = Config()
    with tf.Graph().as_default():
        model = SoftmaxModel(config)
        
        #create a session for running Ops on the Graph
        sess = tf.Session()
        
        #Run the Op to initialize the variables
        init = tf.initialize_all_variables()
        sess.run(init)
        
        losses = model.fit(sess, model.input_data, model.input_labels)
    #If ops are implemented correctly, the average loss should fall close to zero
    #repidly
    assert losses[-1]<.5
    print("Basic (non-exhaustive) classifier tests pass\n")

if __name__=="__main__":
    test_SoftmaxModel()
        
            