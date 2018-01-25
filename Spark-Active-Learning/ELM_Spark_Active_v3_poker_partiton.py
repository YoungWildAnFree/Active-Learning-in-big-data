# -*- coding:utf-8 -*-
#coding:utf-8
""""
PYSPARK_PYTHON=C:\Python27\python.exe;
PYTHONPATH=D:\spark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python;
PYTHONUNBUFFERED=1;
SPARK_HOME=D:\spark\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from time import time
from elm import ELMClassifier
from random_hidden_layer import SimpleRandomHiddenLayer
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


def iris_label_transfor(s):
    if s == "Iris-setosa":
        return 0
    elif s=="Iris-versicolor":
        return 1
    elif s=="Iris-virginica":
        return 2


float_feature=lambda x:[float(i) for i in x]
def yuchuli_data(x):
    x=x.split(",")
    x[:feature_num]=[float(i) for i in x[0:feature_num]]
    x[class_index]=iris_label_transfor(x[class_index])# iris 使用
    # x[class_index]=int(x[class_index])  #pokerhand 使用
    return x


# 软最大化函数
ruanzuida = lambda x: np.exp(x) / sum((np.exp(x)))


def enry(x):
    tmp = []
    try:
        while True:
            a=x.next()
            tmp.append(a)
    except StopIteration:
        arr = np.array(tmp)
        b=elmc.predict(arr[:,0:feature_num])
        for i in range(len(b)):
            result = -sum(ruanzuida(b[i]) * np.log10(ruanzuida(b[i])))
            tmp[i]=(result,tmp[i])
        return tmp
        # x = x[0:feature_num]
    # x = elmc.predict(x)  # 对样本预测 得到ELM向量 生成的是二维嵌套列表 [[]]
    # x = [j for i in x for j in i]  # 转化为一维的 []
    # result = -sum(ruanzuida(x) * np.log10(ruanzuida(x)))  # 计算熵
    # return result



if __name__=="__main__":
    conf = SparkConf().setMaster("local[4]").setAppName("ActiveLearn")# spark://master:7077 local
    sc = SparkContext(conf=conf)

    all_data_path = "D:\\pysparktest\\ELM-Spark-ActiveLearning\\irisdata"
    # all_data_path = "hdfs://master:9000/Alspark/data/irisdata"

    #all_data_path = "D:\\pysparktest\\ELM-Spark-ActiveLearning\\pokerhandtesting.data"
    # all_data_path= "hdfs://master:9000/Alspark/data/pokerhandtesting.data"
    #save_file= ""
    hiddenLayer_num = 20  # ELM隐含层层数
    activation = "sigmoid"  #激活函数sigmoid hardlim  tribas  tanh  sine
    select_num = 5  #选择最小样例个数   iris选择10个  pokerhand 选择500
    feature_num =4  #数据集中属性个数 从0列到  iris4  pokerhand 10
    class_index =4#类标属性在第几列   iris 4  pokerhand 11
    iter_num = 5
    # frist_filter =0.7  # 初步过滤的概率值 小frist_filter值得将被选出
    #读入整个数据
    all_data = sc.textFile(all_data_path)
    print "all_data.first ：",all_data.first()
    print "feature_num:%d"%feature_num
    print "划分数据集"
    #划分数据集 将属性转化为float值  将类标转化为0 1 2   #[5.1, 3.5, 1.4, 0.2, 0]
    split_data=all_data.map(lambda x:yuchuli_data(x))
    test_data = split_data.sample(False, 0.1, 87)#从所有数据中随机采样  到做小的test_data
    test_array = test_data.map(lambda x:x[0:feature_num]).collect()  #属性集合
    test_label = test_data.map(lambda x:x[class_index]).collect() #类标集合
    print "test_data 1%：",test_data.count()
    # 将数据集划分为训练集 与 无类标的数据集  3比7

    train_data,unlabel_data= split_data.randomSplit([0.2,0.8],7)# train_data.first() [4.6, 3.1, 1.5, 0.2, 0]   unlabel_data.first() #[5.1, 3.5, 1.4, 0.2, 0]
    # unlabel_data.persist()
    # 训练集属性值不含类标 用于训练
    train_array = train_data.map(lambda x: x[0:feature_num]).collect()  # [4.6, 3.1, 1.5, 0.2]
    #     # 训练集 类标集合
    train_label =train_data.map(lambda x:x[class_index]).collect()

    start = time()

    # 创建隐含层
    srh = SimpleRandomHiddenLayer(activation_args=activation, n_hidden=hiddenLayer_num)
    # 创建ELM分类器
    elmc = ELMClassifier(hidden_layer=srh)
    for i in range(iter_num):
         print "-"*20+" %d train"%(i+1)+"-"*20
    #     ###############  ELM 训练  #############
         print "train_array_num:",len(train_array)
         #训练分类器
         elmc.fit(train_array,train_label)
         pred_class=elmc.predict_class(test_array)
         #分类精度
         soc = accuracy_score(pred_class,test_label)
         print "test_soc:",soc
         #对无类标的数据集进行预测 每个样例的到一个向量  然后进行软最大化处理 之后计算熵  按熵降序排序 取出前select_num 数的样例
         select_result = unlabel_data.mapPartitions(lambda x:enry(x)).takeOrdered(select_num,key=lambda (x,y):(-x,y))#.sortByKey(ascending=False).top(select_num)
         # print select_result #[(1.017120190624998, [6.1, 3.0, 4.6, 1.4, 1]), (1.016313951000489, [7.4, 2.8, 6.1, 1.9, 2]),
         train_array.extend([ i[1][:feature_num] for i in select_result])  #train_array 加入新属性[6.1, 3.0, 4.6, 1.4
         train_label.extend([ i[1][class_index] for i in select_result])   #train_label 加入新类标

    end = time()
    print "训练 %d 次时间："%iter_num,end-start
    sc.stop()




