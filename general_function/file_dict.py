#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
获取符号字典列表的程序
'''
import platform as plat
#返回平台架构。若无法确定，则返回空字符串。

def GetSymbolList(datapath):
    '''
    加载拼音符号列表，用于标记符号
    返回一个列表list类型变量
    '''
    if(datapath != ''):  #如果数据集文件路径不为空的话
        if(datapath[-1]!='/' or datapath[-1]!='\\'):
            datapath = datapath + '/' #将数据文件路径后面加一个 '/' 正斜杠是整个平台通用的符号
    
    txt_obj=open(datapath + 'dict.txt','r',encoding='UTF-8') # 打开拼音字典文件并读入
    txt_text=txt_obj.read()  #读取整个的文件
    txt_lines=txt_text.split('\n') # 文本分割,每一行
    list_symbol=[] # 初始化符号列表
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split('\t') #切割出拼音来
            list_symbol.append(txt_l[0])  #将拼音存到列表中
    txt_obj.close()  #关闭字典文件
    list_symbol.append('_')   #添加一个空格符
    #SymbolNum = len(list_symbol)
    return list_symbol #将拼音列表进行返回
    
def GetSymbolList_trash2(datapath):  #这个文件的作用是使用linux系统的时候读取的是整个训练文件中的dict,鬼知道现在是干什么用的
    
    '''
    加载拼音符号列表，用于标记符号
    返回一个列表list类型变量
    '''

    datapath_ = datapath.strip('dataset\\')   #

    system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
    if (system_type == 'Windows'):
        datapath_+='\\'
    elif (system_type == 'Linux'):
        datapath_ += '/'
    else:
        print('*[Message] Unknown System\n')
        datapath_ += '/'  
    
    txt_obj=open(datapath_ + 'dict.txt','r',encoding='UTF-8') # 打开文件并读入  
    txt_text=txt_obj.read()        
    txt_lines=txt_text.split('\n') # 文本分割    
    list_symbol=[] # 初始化符号列表
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split('\t')                     
            list_symbol.append(txt_l[0])            
    txt_obj.close()
    list_symbol.append('_')
    #SymbolNum = len(list_symbol)
    return list_symbol


if(__name__ == '__main__'):
    dict_list=GetSymbolList('Z:\\dataset\\')