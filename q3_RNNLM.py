'''
Created on 2017年9月26日

@author: weizhen
'''
import getpass
import sys
import time
import numpy as np
from copy import deepcopy
from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample
import tensorflow as tf
from model import LanguageModel


class Config(object):
    """储存超参数和数据信息"""
    batch_size = 64
    embed_size = 50
    hidden_size = 100
    num_steps = 10
    max_epochs = 16
    early_stopping = 2
    dropout = 0.9
    lr = 0.001


class RNNLM_Model(LanguageModel):
    def load_data(self, debug=False):
        """加载词向量并且训练   train/dev/test 数据"""
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array([self.vocab.encode(word) for word in get_ptb_dataset('train')], dtype=np.int32)
        self.encoded_valid = np.array([self.vocab.encode(word) for word in get_ptb_dataset('valid')], dtype=np.int32)
        self.encoded_test = np.array([self.vocab.encode(word) for word in get_ptb_dataset('test')])
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        """生成placeholder 变量来表示输入的 tensors
            这些placeholder 被用来在模型的其他地方被填充
                            并且在训练的过程中会被填充
            input_placeholder:Input placeholder shape (None,num_steps),type  tf.int32
            labels_placeholder:label placeholder shape (None,num_steps) type tf.float32
            dropout_placeholder:dropput value placeholder (scalar), type tf.float32
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        """添加词嵌入层
        Hint : 这一层应该用input_placeholder 来索引词嵌入
        Hint : 你或许能发现tf.nn.embedding_lookup 是有用的
        Hint : 你或许能发现tf.split , tf.squeeze 是有用的在构造tensor 的输入的时候
        Hint : 下面是你需要创建的变量的维度
                L:(len(self.vocab),embed_size)
        Returns:
            inputs:一个训练次数的列表,每一个元素应该是
                    一个张量 大小是 (batch_size,embed_size)
        """
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding',[len(self.vocab),self.config.embed_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]

            return inputs

    def add_projection(self, rnn_outputs):
        """添加一个投影层
            投影层将隐藏层的表示变换到整个词向量上的分布式表示
            Hint:下面是你需要去创建的维度
                U(hidden_size,len(vocab))
                b_2:(len(vocab),)
            参数:
                rnn_outputs:一个训练次数的列表，每一个元素应该是一个张量
                            大小是(batch_size,embed_size)
            Returns:
                outputs:一个长度的列表，每一个元素是一个张量(batch_size,len(vocab))
        """
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix', [self.config.hidden_size, len(self.vocab)])
            proj_b = tf.get_variable('Bias', [len(self.vocab)])
            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
        return outputs

    def add_loss_op(self, output):
        """将目标损失添加到计算图上
            创建一个优化器并且应用梯度下降到所有的训练变量上面
            Hint:使用tf.train.AdamOptimizer 对于这个模型
                使用optimizer.minimize() 会返回一个train_op的对象
            参数:
                loss: 损失张量，来自于cross_entropy_loss 交叉熵损失
            返回:
                train_op:训练的目标
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.calculate_loss)

        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)

        # 我们想去检验下一个词预测得多好
        # 我们把o转变成float64 位 因为如果不这样就会有数值问题
        # sum(output of softmax) = 1.00000298179 并且不是 1
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        # 将输出值转变成 len(vocab) 的大小
        output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_model(self, input_data):
        """创建RNN LM 模型
        在下面的实现里面你需要去实现RNN LM 模型的等式
        Hint: 使用一个零向量 大小是 (batch_size,hidden_size) 作为初始的RNN的状态
        Hint: 将最后RNN输出 作为实例变量
            self.final_state
        Hint : 确保将dropout应用到 输入和输出的 变量上面
        Hint : 使用变量域 RNN 来定义 RNN变量
        Hint : 表现一个明显的 for-loop 在输入上面
                你可以使用scope.reuse_variable() 来确定权重
                在每一次迭代都是相同的
                确保不会在第一次循环的时候调用这个，因为没有变量会被初始化
        Hint : 下面变量的不同的维度 ， 你需要去创建的

            H: (hidden_size,hidden_size)
            I: (embed_size,hidden_size)
            b_1:(hidden_size,)
        Args:
            inputs:一个记录num_steps的列表，里边的每一个元素应该是一个张量
                    大小是(batch_size,embed_size)的大小
        Returns:返回
            outputs:一个记录num_steps的列表，里面每一个元素应该是一个张量
                    大小是(batch_size,hidden_size)
        """
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                if tstep > 0:
                    scope.reuse_variables()
                RNN_H = tf.get_variable('HMatrix', [self.config.hidden_size, self.config.hidden_size])
                RNN_I = tf.get_variable('IMatrix', [self.config.embed_size, self.config.hidden_size])
                RNN_b = tf.get_variable('B', [self.config.hidden_size])
                state = tf.nn.sigmoid(tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)
                rnn_outputs.append(state)
            self.final_state = rnn_outputs[-1]

        with tf.variable_scope('RNNDropout'):
            rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]
        return rnn_outputs

    def rnn_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(ptb_iterator(data, config.batch_size, config.num_steps)):
            # 我们需要通过初始状态，并且从最终状态中抽取数据来进行填充
            # RNN 合适的 历史
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp
                    }
            loss, state, _ = session.run([self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {} '.format(step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>', stop_length=100, stop_tokens=None, temp=1.0):
    """从模型自动生成文字
        Hint:创建一个feed-dictionary 并且使用sess.run()方法去执行这个模型
                你会需要使用model.initial_state 作为一个键传递给feed_dict
        Hint:得到model.final_state 和 model.predictions[-1].
             在add_model()方法中设置model.final_state  。
             model.predictions 是在 __init__方法中设置的
        Hint:在模型的训练中存储输出的参数值，和预测的y_pred的值
        参数:
        Args:
            session : tf.Session() object
            model : Object of type RNNLM Model
            config : A Config() object
            starting_text:Initial text passed to model
        Returns:
            output : List of word idxs
    """
    state = model.initial_state.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in range(stop_length):
        feed = {model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.dropout_placeholder: 1}
        state, y_pred = session.run([model.final_state, model.predictions[-1]], feed_dict=feed)
        next_word_idx = sample(y_pred[0], temperature=temp)
        tokens.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output

def generate_sentence(session, model, config, *args, **kwargs):
    """方便从模型来生成句子"""
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
    config = Config()
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1

    # 创建训练模型，并且生成模型
    with tf.variable_scope('RNNLM') as scope:
        model = RNNLM_Model(config)
        # 这个指示gen_model来重新使用相同的变量作为以上的模型
        scope.reuse_variables()
        gen_model = RNNLM_Model(gen_config)

    init = tf.initialize_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
        session.run(init)
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_pp = model.run_epoch(session,
                                       model.encoded_train,
                                       train_op=model.train_step)
            valid_pp = model.run_epoch(session, model.encoded_valid)
            print('Training perplexity: { } '.format(train_pp))
            print('Validation perplexity: { } '.format(valid_pp))
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time : { }'.format(time.time() - start))

        saver.restore(session, 'ptb_rnnlm.weights')
        test_pp = model.run_epoch(session, model.encoded_test)
        print('=-=' * 5)
        print('Test perplexity: {} '.format(test_pp))
        print('=-=' * 5)
        starting_text = 'in palo alto'
        while starting_text:
            print(
                ' '.join(generate_sentence(session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))
            starting_text = raw_input('>')


if __name__ == "__main__":
    test_RNNLM()
