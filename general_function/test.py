# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:07:45 2019

@author: zangz
"""

import os
import wave
import numpy as np
import matplotlib.pyplot as plt  
import math
import time

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft

filename = 'F:\\语音识别\\dataset\\data_thchs30\\data\\A2_0.wav'
wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
num_frame = wav.getnframes() # 获取帧数
num_channel=wav.getnchannels() # 获取声道数
framerate=wav.getframerate() # 获取帧速率
num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
str_data = wav.readframes(num_frame) # 读取全部的帧
wav.close() # 关闭流
wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
wave_data = wave_data.T # 将矩阵转置,就是个波形呗
    #wave_data = wave_data 
#return wave_data, framerate  