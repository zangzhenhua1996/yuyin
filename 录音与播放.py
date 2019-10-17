# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:40:35 2019

@author: zangz
"""

import wave
from pyaudio import PyAudio,paInt16
framerate=16000 #采样频率
NUM_SAMPLES=2000 # #pyaudio内置缓冲大小(每次的话采集的是2000个点,那么1秒钟需要8次才能采样16000个)
channels=1 #采样通道数
sampwidth=2 #采样位宽
TIME=10 ##录音时间，单位s

def save_wave_file(filename,data):
    '''音频文件的保存'''
    wf=wave.open(filename,'wb')  #打开文件
    wf.setnchannels(channels) #设置通道数
    wf.setsampwidth(sampwidth) #设置位宽
    wf.setframerate(framerate) #设置采样频率
    wf.writeframes(b"".join(data)) #写入数据(二进制格式)
    wf.close() #关闭文件

#这里要使用“pyaudio”这个包它会在内存中打开一个音频输入流，
#将从录音设备中采集到的声波信号持续读入内存，当一定时间录音结束后，
#就将录制好的声音写出到硬盘中。
def my_record():
    pa=PyAudio()  #创建一个录音对象
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)  #这里是input
    my_buf=[]
    count=0
    while count<TIME*8:#控制录音时间(16000/2000=8)
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
        print('.')
    save_wave_file('01.wav',my_buf)  #调用音频保存音频的函数
    stream.close()

"""
我们还可以写一个函数来播放刚才录制的音频文件，
其实就跟前面几步刚好反过来，前面是先用pyaudio录音，
再用wave写到文件，这里先用wave从文件中读入内存，
再用pyaudio写到输出流中，由音响设备来播放声音。
我们可以设置一个固定大小的chunk块来做缓冲。
播放完毕后，需要将文件的输入流和音频的输出流关闭，以解除资源占用。
"""
chunk=2014  #定大小的chunk块来做缓冲
def play():
    wf=wave.open(r"01.wav",'rb')  #以二进制的格式打开
    p=PyAudio() #创建一个对象
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
    wf.getnchannels(),rate=wf.getframerate(),output=True)  #这里是output
    while True:
        data=wf.readframes(chunk)  #进行读取
        if data=="":break
        stream.write(data) #进行输出
    stream.close() #关闭输出流
    p.terminate() # 关闭文件的输入流
    
if __name__ == '__main__':
    my_record()
    print('Over!') 
#    play()