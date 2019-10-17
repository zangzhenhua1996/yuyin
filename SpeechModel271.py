#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import keras as kr  #导入keras的包
import numpy as np  #导入numpy
import random
# 从keras模型中导入序贯型以及函数式的模型
from keras.models import Sequential, Model
# 从神经网络层中导入 全连接层,随机失活,输入层,矩阵维度变换,BN归一化
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization # , Flatten
from keras.layers import LSTM,Bidirectional
# 从神经网络层中导入 ,Lambda层(如果你只是想对流经该层的数据做个变换，而这个变换本身没有什么需要学习的参数，那么直接用Lambda Layer是最合适的了。)
# imeDistributed这个封装器将一个层应用于输入的每个时间片。
# Activation 激活函数
# 二维的卷积层
# 二维的池化层
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
# 导入keras的底层框架 这里是TensorFlow
from keras import backend as K   
# 导入优化器 SGD,Adam
# Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。即： 
from keras.optimizers import SGD, Adadelta, Adam
# 导入数据处理的类
from readdata24 import DataSpeech 

abspath = ''  #绝对路径
ModelName='271'  #模型名
#NUM_GPU = 2

# 声学模型
class ModelSpeech(): # 语音模型类
    def __init__(self, datapath):
        '''
        初始化
        默认输出的拼音的表示大小是1424，即1423个拼音+1个空白块
        '''
        MS_OUTPUT_SIZE = 1424   # 拼音符号的个数 1423个拼音还有一个空白符
        self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE  # 神经网络最终输出的每一个字符向量维度的大小   (因为有1424个拼音因此,因此字符向量维度是1424)
        #self.BATCH_SIZE = BATCH_SIZE  # 一次训练的batch
        self.label_max_string_length = 64 #拼音串的最大长度,也就是说一句话不能超过这些个数的字符

        self.AUDIO_LENGTH = 1600   # 序列长度(转换成特征语谱图后的序列的长度)

        self.AUDIO_FEATURE_LENGTH = 200  #特征矩阵的长度(200维)

        self._model, self.base_model = self.CreateModel()   #初始化的时候调用模型创建函数进行模型的创建
        
        self.datapath = datapath  # 数据存储的的路径
        self.slash = '' #根据平台的不同进行反斜杠正斜杠的添加
        system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
        if(system_type == 'Windows'):
            self.slash='\\' # 反斜杠
        elif(system_type == 'Linux'):
            self.slash='/' # 正斜杠
        else:
            print('*[Message] Unknown System\n')
            self.slash='/' # 正斜杠
        if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
            self.datapath = self.datapath + self.slash
    
        
    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
        
        '''
        # 输入层  name: the_input 形状: ,batch_size:none(后面会根据给定的batch_zize的),序列长度1600,特征维度200,通道个数1
        input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        # 第一个卷积池化模块

        # 卷积层: 32 通道.3*3的卷积核,不使用偏置项,激活函数使用relu,padding使用的是same_padding(用来填充不能满足一次卷积的部分,保证卷积完尺寸不变) ,
        # kernel_initializer 权重初始化(he_normal(seed=None),He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，
        # 其中fan_in权重张量的扇入    seed：随机数种子)   
        #采用函数式模型输入是 the_input
        #参数的个数 3*3*32 = 288
        layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层

        #随机失活（dropout）是对具有深度结构的人工神经网络进行优化的方法，在学习过程中通过将隐含层的部分权重或输出随机归零，
        # 降低节点间的相互依赖性（co-dependence ）从而实现神经网络的正则化（regularization），降低其结构风险（structural risk)
        layer_h1 = Dropout(0.05)(layer_h1)  #输入的是第一个卷积层

        # 卷积层 参数个数 ((3*3*32) +1)*32 = 9248 这里使用了偏置项,因此呢在乘上卷积核个数之前需要加 1
        #   use_bias=True 使用偏置项
        #输入是经过随机失活 的卷积层
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层

        # 池化层 池化大小是2 ,步长设为None(为None则默认值为pool_size) ,padding不增补0, 这样经过池化以后长宽各缩小一倍变成800*100 
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
        #layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合

        # dropout层
        layer_h3 = Dropout(0.05)(layer_h3)



        #第二个卷积池化模块
        # 第一层卷积的参数 ((3*3*32)+1)*64 = 9248 这里使用了bias
        layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        # 第二层卷积的参数 ((3*3*64)+1)*64 = 36928
        layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
        #  经过第二次的池化后 变成 400*50
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        
        layer_h6 = Dropout(0.1)(layer_h6)


        # 第三个卷积池化模块
        # 第一层卷积后的参数 ((3*3*64)+1)*128 = 73856
        layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6) # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        # 第二层卷积后的参数 ((3*3*128)+1)*128 = 147584
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7) # 卷积层
        #  经过第三次池化以后 变成 200 * 25
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
        
        layer_h9 = Dropout(0.15)(layer_h9)

        # 第四个卷积模块
        # 第一层卷积后的参数 ((3*3*128)+1)*128 = 147584
        layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        # 第二层卷积后的参数 ((3*3*128)+1)*128 = 147584
        layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10) # 卷积层
        # 1*1的核做不做没有啥太大区别
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层
        
        
        layer_h12 = Dropout(0.2)(layer_h12)

        # 第五个卷积模块
        # 第一层卷积后的参数 ((3*3*128)+1)*128 = 147584
        layer_h13 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12) # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        # 第二层卷积后的参数 ((3*3*128)+1)*128 = 147584
        layer_h14 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13) # 卷积层
        # 又是流于形式的池化层,可要可不要但是为了完整性
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14) # 池化层
        
        #test=Model(inputs = input_data, outputs = layer_h12)
        #test.summary()
        
        # Reshape层将 200*25*128 的图进行变换 成 200*3200 (如果前面改动记得这里也是需要进行改动的)
        layer_h16 = Reshape((200, 3200))(layer_h15) #Reshape层

        #layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        #layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        
        #LSTM(zangz)
#       layer_h51 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h16)
        blstm = Bidirectional(LSTM(256, activation='relu', use_bias=True, return_sequences=True), merge_mode='sum')(layer_h16)  #注意使用双向LSTM书写的格式
        
        

        #随机中断部分神经网络连接，防止过拟合
        layer_h16 = Dropout(0.3)(blstm)
        # layer_h16 = Dropout(0.3)(layer_h16)

        # 全连接层 输入 200*3200 (200是时间序列也可以看做是200个样本,因为最后是每一行对应的1424个概率值)
        # 参数个数 (3200+1) * 128 = 409728
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16) # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)

        # 全连接层 输入是200*128 这里的神经元的个数就是拼音的个数 
        # 参数个数 (128+1)*1424 = 183696
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17) # 全连接层
        
        # 激活层 进行概率值的求解
        y_pred = Activation('softmax', name='Activation0')(layer_h18)

        # 最后将输入输出全部传递给Model() 函数式模型
        model_data = Model(inputs = input_data, outputs = y_pred)
        #model_data.summary()
        
        # 标签输入层
        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')  # 输入是64个拼音因此shape是64,当然默认的batch是None后续进行处理
        # 输入序列长度
        input_length = Input(name='input_length', shape=[1], dtype='int64') #shape是batch跟输入长度大小(这里是1只是说明占了1维而不是长度是1注意区别)
        #标签的长度
        label_length = Input(name='label_length', shape=[1], dtype='int64') # 标签的长度同上
        # Keras doesn't currently support loss funcs with extra parameters Keras当前不支持具有额外参数的损失函数
        # so CTC loss is implemented in a lambda layer   所以CTC损失是在lambda层实现的
        
        #layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
        
        # 损失函数的定义(使用Lambda进行定义)
        # 调用了Tensorflow 封装的底层CTC函数,不过也要符合这个Lambda层的书写.传递的参数方式跟函数式模型一样的 
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        
        
        # 建立最后的模型 输入是 [input_data, labels, input_length, label_length] , 输出是 loss_out
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        
        model.summary()  #显示整个的函数式模型的参数网络结构
        
        # clipnorm seems to speeds up convergence
        #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        #opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)

#Adam配置参数
#alpha也被称为学习速率或步长。权重比例被校正(例如0.001)。更大的值(例如0.3)在速率校正之前会加快初始学习速度。较小的值(例如1.0e-5)在训练期间降低学习速度
#beta1。第一次估计的指数衰减率(如0.9)。
#beta2。第二次估计的指数衰次减率(例如0.999)。在稀疏梯度问题(例如NLP和计算机视觉问题)上，这个值应该接近1.0。
#epsilon是一个非常小的数字，可以防止任何在实施中被0划分(例如，10e-8)

#最后，这里推荐一些比较受欢迎的使用默认参数的深度学习库：
#TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
#Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
#Blocks: learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
#Lasagne: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
#Caffe: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
#MxNet: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
#Torch: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8


        # 定义优化器,这里的优化器使用的是Adam  
        #lr 是学习率 
        
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        
        #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        # 本函数编译模型以供训练,这里的loss使用ctc损失函数的时候都是这么用的不用纠结以后再说
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)  
        
        
        # captures output of softmax so we can decode the output during visualization
        # 捕获softmax的输出，这样我们就可以在可视化过程中解码输出。
        test_func = K.function([input_data], [y_pred])
        
        #print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        # 返回一个不带CTC的一个带CTC的模型
        return model, model_data


    # CTC的损失函数  
    def ctc_lambda_func(self, args):

        '''
        添加CTC损失函数，由backend引入
        注意：CTC_batch_cost输入为：
        labels 标签：[batch_size, l]    y_true：包含真值标签的张量(samples, max_string_length) 。
        y_pred cnn网络的输出：[batch_size, t, vocab_size]    y_pred：包含预测值，或 softmax 输出的张量(samples, time_steps, num_categories) 。
        input_length 网络输出的长度：[batch_size]  张量 (samples, 1)，包含y_pred中每个批处理项的序列长度。
        label_length 标签的长度：[batch_size]   张量(samples, 1)，包含y_true中每个批处理项的序列长度。
        '''
        y_pred, labels, input_length, label_length = args  #通过元组接收输入
        y_pred = y_pred[:, :, :]  #切片
        #y_pred = y_pred[:, 2:, :]
        # 返回：shape为 (samples,1) 的张量，包含每一个元素的 CTC 损失。
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)  #返回CTC损失,这里的labels其实就是 y_true
    
    
    # 模型的训练
    def TrainModel(self, datapath, epoch = 2, save_step = 1000, batch_size = 32, filename = abspath + 'model_speech/m' + ModelName + '/speech_model'+ModelName):
        '''
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        '''
        data=DataSpeech(datapath, 'train')  #首先获取的是 train数据集
        
        num_data = data.GetDataNum() # 获取数据的数量  
        
        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH) #将所有的数据使用生成器进行batch_size的封装,封装成一个个的对象
        
        for epoch in range(epoch): # 迭代轮数
            print('[running] train epoch %d .' % epoch)
            n_step = 0 # 迭代数据数
            while True:
                try:
                    print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
                    # data_genetator是一个生成器函数
                    
                    #self._model.fit_generator(yielddatas, save_step, nb_worker=2)
                    # 利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
                    self._model.fit_generator(yielddatas, save_step)    # self._model这个是初始化调用creatmodel返回的模型
                    # samples_per_epoch：整数，当模型处理的样本达到此数目时计一个epoch结束，执行下一个epoch
                    n_step += 1
                except StopIteration:
                    print('[error] generator error. please check data format.')
                    break
                
                self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))  #进行模型的保存
                self.TestModel(self.datapath, str_dataset='train', data_count = 4) #进行训练集模型的测试
                self.TestModel(self.datapath, str_dataset='dev', data_count = 4) # 进行 验证集模型的测试
                
    # 加载模型函数
    def LoadModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName+'.model'):
        '''
        加载模型参数
        '''
        self._model.load_weights(filename)
        self.base_model.load_weights(filename + '.base')

    # 保存模型的函数
    def SaveModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName,comment=''): 
        '''
        保存模型参数
        '''
        self._model.save_weights(filename + comment + '.model')    #保存带ctc的模型
        self.base_model.save_weights(filename + comment + '.model.base')  #保存不带ctc的模型
        # 需要安装 hdf5 模块 (也是一种模型保存的方式) 
        self._model.save(filename + comment + '.h5')
        self.base_model.save(filename + comment + '.base.h5')

        f = open('step'+ModelName+'.txt','w')
        f.write(filename+comment)
        f.close()

    # 模型的测试
    def TestModel(self, datapath='', str_dataset='dev', data_count = 32, out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
        '''
        测试检验模型效果
        
        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现
        
        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现
        
        '''

        data=DataSpeech(self.datapath, str_dataset)   
        #data.LoadDataList(str_dataset) 
        num_data = data.GetDataNum() # 获取数据的数量
        if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试(正常使用的是32或者自己传递进来的需要测试的个数)
            data_count = num_data
        
        try:
            ran_num = random.randint(0,num_data - 1) # 获取一个随机数(0-num_data)
            
            words_num = 0  #总得单次数量
            word_error_num = 0 #错误的单次数量
            
            nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))   # '20190924_103104'  就是时间日期的一个字符串
            if(out_report == True):  #如果说输出开关打开的话
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入
            
            txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据
                
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_bias = 0
                while(data_input.shape[0] > self.AUDIO_LENGTH):
                    print('*[Error]','wave data lenghth of num',(ran_num + i) % num_data, 'is too long.','\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
                # 数据格式出错处理 结束
                
                pre = self.Predict(data_input, data_input.shape[0] // 8)  #预测的结果
                
                words_n = data_labels.shape[0] # 获取每个句子的字数
                words_num += words_n # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离(预测的结果跟真实的结果之间的编辑距离(整数))
                if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance # 使用编辑距离作为错误字数
                else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n # 就直接加句子本来的总字数就好了(错误率就是100%)
                
                if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
                    #print('测试进度：',i,'/',data_count)
                    print('Test Count: ',i,'/',data_count)
                
                
                if(out_report == True):
                    if(i % io_step_file == 0 or i == data_count - 1):
                        txt_obj.write(txt)
                        txt = ''
                    
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'
                    
                
            
            #print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
            if(out_report == True):  #将错误率进行写入
                txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt = ''
                txt_obj.close()
            
        except StopIteration:
            print('[Error] Model Test Error. please check data format.')
    
    # 预测函数(在训练过程中预测成拼音实际使用是用的下面的函数)
    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''
        
        batch_size = 1 # 每次只预测一个呗
        in_len = np.zeros((batch_size),dtype = np.int32)
        
        in_len[0] = input_len
        
        x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
        
        for i in range(batch_size):
            x_in[i,0:len(data_input)] = data_input
        
        
        base_pred = self.base_model.predict(x = x_in) #调用的是不带CTC的模型,因为带CTC的只是训练用的,解码使用CTC解码函数
        
        #print('base_pred:\n', base_pred)
        
        #y_p = base_pred
        #for j in range(200):
        #   mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
        #   print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
        #   print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
        #   count=0
        #   for i in range(y_p[0][j].shape[0]):
        #       if(y_p[0][j][i] < mean):
        #           count += 1
        #   print('count:',count)
        
        base_pred =base_pred[:, :, :]
        #base_pred =base_pred[:, 2:, :]
        
        r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1) #使用CTC进行解码
        
        #print('r', r)
        
        
        r1 = K.get_value(r[0][0])  #获取最后的解码的拼音
        #print('r1', r1)
        
        
        #r2 = K.get_value(r[1])
        #print(r2)
        
        r1=r1[0]
        
        return r1
        pass
    
    # 语音识别的函数
    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        不过这里现在还有bug
        '''
        
        #data = self.data
        #data = DataSpeech('E:\\dataset')
        #data.LoadDataList('dev')
        # 获取输入特征
        #data_input = GetMfccFeature(wavsignal, fs)
        #t0=time.time()
        data_input = GetFrequencyFeature3(wavsignal, fs)  #获取特征
        #t1=time.time()
        #print('time cost:',t1-t0)
        
        input_length = len(data_input)
        input_length = input_length // 8
        
        data_input = np.array(data_input, dtype = np.float)
        #print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #t2=time.time()
        r1 = self.Predict(data_input, input_length)  #还是数字
        #t3=time.time()
        #print('time cost:',t3-t2)
        list_symbol_dic = GetSymbolList(self.datapath) # 获取拼音列表
        
        
        r_str=[]
        for i in r1:
            r_str.append(list_symbol_dic[i])
        
        return r_str  #返回最终的拼音串
        pass
        
    def RecognizeSpeech_FromFile(self, filename):
        '''
        最终做语音识别用的函数，识别指定文件名的语音
        '''
        
        wavsignal,fs = read_wav_data(filename)
        
        r = self.RecognizeSpeech(wavsignal, fs)
        
        return r
        
        pass
        
    
        
    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model


if(__name__=='__main__'):
    
    #import tensorflow as tf
    #from keras.backend.tensorflow_backend import set_session
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #进行配置，使用95%的GPU
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    #config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    #set_session(tf.Session(config=config))
    
    
    datapath =  abspath + ''
    modelpath =  abspath + 'model_speech'
    
    
    if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
        os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
    
    system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
    if(system_type == 'Windows'):
        datapath = 'Z:\\dataset'
        modelpath = modelpath + '\\'
    elif(system_type == 'Linux'):
        datapath =  abspath + 'dataset'
        modelpath = modelpath + '/'
    else:
        print('*[Message] Unknown System\n')
        datapath = 'dataset'
        modelpath = modelpath + '/'
    
    ms = ModelSpeech(datapath)
    
    
    ms.LoadModel(modelpath + 'm251/speech_model251_e_0_step_625000.model')
#   ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500) 
    
    #下面是本文件的测试
    t1=time.time()
    ms.TestModel(datapath, str_dataset='train', data_count = 1280, out_report = True)
    #ms.TestModel(datapath, str_dataset='dev', data_count = 128, out_report = True)
    #ms.TestModel(datapath, str_dataset='test', data_count = 128, out_report = True)
    t2=time.time()
    print('Test Model Time Cost:',t2-t1,'s')
    #r = ms.RecognizeSpeech_FromFile('E:\\dataset\\ST-CMDS-20170001_1-OS\\20170001P00241I0053.wav')
    #r = ms.RecognizeSpeech_FromFile('E:\\dataset\\ST-CMDS-20170001_1-OS\\20170001P00020I0087.wav')
    #r = ms.RecognizeSpeech_FromFile('E:\\dataset\\wav\\train\\A11\\A11_167.WAV')
    #r = ms.RecognizeSpeech_FromFile('E:\\dataset\\wav\\test\\D4\\D4_750.wav')
    #print('*[提示] 语音识别结果：\n',r)
