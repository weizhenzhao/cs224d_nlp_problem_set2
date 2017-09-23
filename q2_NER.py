'''
Created on 2017年9月22日

@author: weizhen
'''
import os
import getpass
import sys
import time
import struct

import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

class Config(object):
    """
                    配置模型的超参数和数据信息
             这个配置类是用来存储超参数和数据信息，模型对象被传进Config() 实例对象在初始化的时候
    """
    embed_size = 50
    batch_size = 64
    label_size = 5
    hidden_size = 100
    max_epochs = 50
    early_stopping = 2
    dropout = 0.9
    lr = 0.001
    l2 = 0.001
    window_size = 3

class NERModel(LanguageModel):
    """
    Implements a NER (Named Entity Recognition) model.
          实现命名实体识别的模型
          这个类实现了一个深度的神经网络用来进行命名实体识别
        它继承自LanguageModel 一个有着add_embedding 方法，除了标准的模型方法
    """
    def load_data(self, debug=False):
        """
                    加载开始的word-vectors 并且开始训练 train/dev/test data
        """
        # Load the starter word vectors
        self.wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt', 'data/ner/wordVectors.txt')
        tagnames = ['O', 'LOC', 'MISC', 'ORG', 'PER']
        self.num_to_tag = dict(enumerate(tagnames))
        tag_to_num = {v:k for k, v in self.num_to_tag.items()}
        
        # Load the training set
        docs = du.load_dataset("data/ner/train")
        self.X_train, self.y_train = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_train = self.X_train[:1024]
            self.y_train = self.y_train[:1024]
        
        # Load the dev set (for tuning hyperparameters)
        docs = du.load_dataset('data/ner/dev')
        self.X_dev, self.y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        if debug:
            self.X_dev = self.X_dev[:1024]
            self.y_dev = self.y_dev[:1024]
        
        # Load the test set (dummy labels only)
        docs = du.load_dataset("data/ner/test.masked")
        self.X_test, self.y_test = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=self.config.window_size)
        
    def add_placeholders(self):
        """
                    生成placeholder 变量去接收输入的 tensors
                    这些placeholder 被用作输入在模型的其他地方调用，并且会在训练的时候被填充数据
                    当"None"在placeholder的大小当中的时候 ,是非常灵活的
                    在计算图中填充如下节点：
                    input_placeholder: tensor(None,window_size) . type:tf.int32
                    labels_placeholder: tensor(None,label_size) . type:tf.float32
                    dropout_placeholder: Dropout value placeholder (scalar), type: tf.float32
                    把这些placeholders 添加到 类对象自己作为    常量
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.window_size], name='Input')
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.label_size], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
    
    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        """
                    为softmax分类器创建一个feed字典
                    feed_dict={
                        <placeholder>:<tensor of values to be passed for placeholder>,
                    }
                    
                    Hint:The keys for the feed_dict should be a subset of the placeholder
                         tensors created in add_placeholders.
                    Hint:When label_batch is None,don't add a labels entry to the feed_dict
                    
                    Args:
                        input_batch:A batch of input data
                        label_batch:A batch of label data
                    Returns:
                        feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
                self.input_placeholder:input_batch,
            }
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict
    
    def add_embedding(self):
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding', [len(self.wv), self.config.embed_size])
            window = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            window = tf.reshape(
                window, [-1, self.config.window_size * self.config.embed_size])
            # ## END YOUR CODE
            return window
    
    def add_model(self, window):
        """Adds the 1-hidden-layer NN
        Hint:使用一个variable_scope ("layer") 对于第一个隐藏层
                                另一个("Softmax")用于线性变换在最后一个softmax层之前
                                确保使用xavier_weight_init 方法，你之前定义好的
        Hint:确保添加了正则化和dropout在这个网络中
                                正则化应该被添加到损失函数上，
             dropout应该被添加到每一个变量的梯度上面
        Hint:可以考虑使用tensorflow Graph 集合 例如(total_loss)来收集正则化
                                 和损失项，你之后会在损失函数中添加的
        Hint:这里会需要创建不同维度的变量，如下所示:
            W:(window_size*embed_size,hidden_size)
            b1:(hidden_size,)
            U:(hidden_size,label_size)
            b2:(label_size)
        Args:
            window: tf.Tensor of shape(-1,window_size*embed_size)
        Returns:
            output: tf.Tensor of shape(batch_size,label_size)
        """
        with tf.variable_scope('layer1', initializer=xavier_weight_init()) as scope:
            W = tf.get_variable('w', [self.config.window_size * self.config.embed_size, self.config.hidden_size])
            b1 = tf.get_variable('b1', [self.config.hidden_size])
            h = tf.nn.tanh(tf.matmul(window, W) + b1)
            if self.config.l2:
                tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(W))
        
        with tf.variable_scope('layer2', initializer=xavier_weight_init()) as scope:
            U = tf.get_variable('U', [self.config.hidden_size, self.config.label_size])
            b2 = tf.get_variable('b2', [self.config.label_size])
            y = tf.matmul(h, U) + b2
            if self.config.l2:
                tf.add_to_collection('total_loss', 0.5 * self.config.l2 * tf.nn.l2_loss(U))
        output = tf.nn.dropout(y, self.dropout_placeholder)
        return output
    
    def add_loss_op(self, y):
        """将交叉熵损失添加到计算图上
        Hint:你或许可以使用tf.nn.softmax_cross_entropy_with_logits 方法来简化你的
                                实现，
                                或许可以使用tf.reduce_mean
                                参数：
               pred:A tensor shape:(batch_size,n_classes)
                                返回值:
                loss:A 0-d tensor (数值类型)
        """
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.labels_placeholder))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    def add_training_op(self, loss):
        """设置训练目标
                            创建一个优化器并且将梯度下降应用到所有变量的更新上面
           Hint:对于这个模型使用tf.train.AdamOptimizer优化方法
                                       调用optimizer.minimize()会返回一个train_op的对象
            Args:
                loss:Loss tensor,from cross entropy_loss
            Returns:
                train_op:The Op for training
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    def __init__(self, config):
        """使用上面定义好的函数来构造神经网络"""
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        window = self.add_embedding()
        y = self.add_model(window)
        
        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        one_hot_prediction = tf.arg_max(self.predictions, 1)
        correct_prediction = tf.equal(tf.arg_max(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
        self.train_op = self.add_training_op(self.loss)
    
    def run_epoch(self, session, input_data, input_labels, shuffle=True, verbose=True):
        orig_X, orig_y = input_data, input_labels
        dp = self.config.dropout
        # We 're interested in keeping track of the loss and accuracy during training 
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        for step, (x, y) in enumerate(data_iterator(orig_X, orig_y, batch_size=self.config.batch_size, label_size=self.config.label_size, shuffle=shuffle)):
            feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
            loss, total_correct, _ = session.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)
            
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{}/{} : loss = {}'.format(step, total_steps, np.mean(total_loss)))
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()
            return np.mean(total_loss), total_correct_examples / float(total_processed_examples)
    
    def predict(self, session, X, y=None):
        """从提供的模型中进行预测"""
        # 如果y已经给定，loss也已经计算出来了
        # 我们对dropout求导数通过把他设置为1
        dp = 1
        losses = []
        results = []
        if np.any(y):
            data = data_iterator(X, y, batch_size=self.config.batch_size,
                                label_size=self.config.label_size, shuffle=False)
        else:
            data = data_iterator(X, batch_size=self.config.batch_size,
                                 label_size=self.config.label_size, shuffle=False)
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run([self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
        return np.mean(losses), results
            
def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix"""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print("")
    print(confusion)
    for i, tag in sorted(num_to_tag.items()):
        print(i, "-----", tag)
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print("Tag: {} - P {:2.4f} / R {:2.4f}".format(tag, prec, recall))
    
def calculate_confusion(config, predicted_indices, y_indices):
    """帮助方法计算混淆矩阵"""
    confusion = np.zeros((config.label_size, config.label_size), dtype=np.int32)
    for i in range(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion

def save_predictions(predictions, filename):
    """保存predictions 到 提供的文件中"""
    with open(filename, "w") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")

def test_NER():
    """测试NER模型的实现
            你可以使用这个函数来测试你实现了的命名实体识别的神经网络
            当调试的时候，设置最大的max_epochs 在 Config 对象里边为1
            这样便可以快速地进行迭代
    """
    config = Config()
    with tf.Graph().as_default():
        model = NERModel(config)
        
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        
        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0
            
            session.run(init)
            for epoch in range(config.max_epochs):
                print('Epoch {}'.format(epoch))
                start = time.time()
                # ##
                train_loss, train_acc = model.run_epoch(session, model.X_train, model.y_train)
                val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
                print('Training loss : {}'.format(train_loss))
                print('Training acc : {}'.format(train_acc))
                print('Validation loss : {}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    saver.save(session, './weights/ner.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
                confusion = calculate_confusion(config, predictions, model.y_dev)
                print_confusion(confusion, model.num_to_tag)
                print('Total time: {}'.format(time.time() - start))

            #saver.restore(session, './weights/ner.weights')
            #print('Test')
            #print('=-=-=')
            #print('Writing predictions t o q2_test.predicted')
            #_, predictions = model.predict(session, model.X_test, model.y_test)
            #save_predictions(predictions, "q2_test.predicted")

if __name__ == "__main__":
    test_NER()    
    
